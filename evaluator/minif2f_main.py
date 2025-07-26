import os
import os.path as osp
import sys
import subprocess
import json
import asyncio
import traceback
from datetime import datetime
from typing import Optional
from pathlib import Path
from typing import Dict, Set
import pickle

import vllm     # vLLM should be imported before any 3rd-party libraries, o.w. all Pantograph REPLs are scheduled at the same CPU core.
from openai import OpenAI, AsyncOpenAI, NOT_GIVEN
import argparse
from tqdm import tqdm
from termcolor import colored
from loguru import logger
logger = logger.opt(colors=True)
from fire import Fire

# TODO: Maybe more convenient to integrate running and closing the API server into one script
# from vllm.entrypoints.openai.api_server import run_server

from common.constants import CORE_OPTIONS, OPEN_HEADER
from common.pantograph.dataclasses import GoalState
from common.pantograph.server import Server, ServerError
from agent.proof_search import ProofSearchResult
from agent.proof_search import ProofSearchAgent, HammerProofSearchAgent, StepProver_NALP_LLMProofSearchAgent, StepProver_Critic_LLMProofSearchAgent, SFT_NALP_LLMProofSearchAgent, SFT_NALP_AVGGOAL_LLMProofSearchAgent


MODEL_DICT = {
    'hammer': HammerProofSearchAgent,
    'stepprover_vanilla': StepProver_NALP_LLMProofSearchAgent,
    'stepprover_critic': StepProver_Critic_LLMProofSearchAgent,
    'sft_vanilla': SFT_NALP_LLMProofSearchAgent,
    'sft_vanilla_avggoal' : SFT_NALP_AVGGOAL_LLMProofSearchAgent
}


def main(
        log_root: str,
        dataset_root: str,
        model: str,
        split: str,
        max_search_trials: int=100,
        num_samples_per_trial: int=32,
        temperature: float=0.7,
        max_tokens: int=256,
        num_concurrency: int=12,
        verbose: bool=False,
        dry_run: bool=False,
        gen_base_url: Optional[str]='',
        gen_api_key: Optional[str]='',
        gen_model_name: Optional[str]='',
        critic_base_url: Optional[str]='',
        critic_api_key: Optional[str]='',
        critic_model_name: Optional[str]=''
):
    saved_args = {**locals()}

    model = model.lower()
    split = split.lower()
    assert osp.exists(dataset_root)
    assert model.lower() in MODEL_DICT.keys()
    assert split.lower() in ['valid', 'test', 'all']

    os.makedirs(log_root, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level='INFO')
    logger.add(osp.join(log_root, datetime.now().strftime("%Y%m%d-%H%M%S")+'.log'), level='DEBUG')
    logger.info(f'Running MiniF2F proof search experiment with hyperparams: {saved_args}')
    log_debug = logger.info if verbose else logger.debug

    gen_client = AsyncOpenAI(
        base_url=gen_base_url,
        api_key=gen_api_key
    )
    critic_client = AsyncOpenAI(
        base_url=critic_base_url,
        api_key=critic_api_key
    )

    splits = [split] if split != 'all' else ['valid', 'test']
    samples = []
    # {
    #     "id": "mathd_algebra_478",
    #     "split": "test",
    #     "formal_statement": "theorem mathd_algebra_478\n  (b h v : \u211d)\n  (h\u2080 : 0 < b \u2227 0 < h \u2227 0 < v)\n  (h\u2081 : v = 1 / 3 * (b * h))\n  (h\u2082 : b = 30)\n  (h\u2083 : h = 13 / 2) :\n  v = 65 := sorry",
    #     "header": "import Mathlib.Algebra.BigOperators.Basic\nimport Mathlib.Data.Real.Basic\nimport Mathlib.Data.Complex.Basic\nimport Mathlib.Data.Nat.Log\nimport Mathlib.Data.Complex.Exponential\nimport Mathlib.NumberTheory.Divisors\nimport Mathlib.Data.ZMod.Defs\nimport Mathlib.Data.ZMod.Basic\nimport Mathlib.Topology.Basic\nimport Mathlib.Data.Nat.Digits\n\nopen BigOperators\nopen Real\nopen Nat\nopen Topology",
    #     "informal_stmt": "The volume of a cone is given by the formula $V = \\frac{1}{3}Bh$, where $B$ is the area of the base and $h$ is the height. The area of the base of a cone is 30 square units, and its height is 6.5 units. What is the number of cubic units in its volume? Show that it is 65.",
    #     "informal_proof": "We are given that $B = 30$ and $h = 6.5$ and asked to find $\\frac{1}{3}Bh$.  We find that \\[\\frac{1}{3}Bh = \\frac{1}{3}(30)(6.5) = (10)(6.5) = 65.\\]"
    # }

    for split in splits:
        with open(osp.join(dataset_root, split+'.jsonl'), 'r') as f:
            samples.extend([json.loads(l) for l in f.readlines()])
        log_debug(f"{split} data loaded.")
    finished = []

    async def search(sample: Dict) -> None:
        try:
            agent: ProofSearchAgent=MODEL_DICT[model](
                gen_client=gen_client,
                gen_model_name=gen_model_name,
                critic_client=critic_client,
                critic_model_name=critic_model_name,
                max_search_trials=max_search_trials,
                num_samples_per_trial=num_samples_per_trial,
                temperature=temperature,
                max_tokens=(max_tokens if max_tokens > 0 else NOT_GIVEN),
            )
            log_debug(f"search({sample['split']}-{sample['id']}): agent initialized.")
            server = await Server.create(
                imports=["Mathlib", "Aesop"],
                project_path=dataset_root,
                core_options=CORE_OPTIONS,
                timeout=300,
            )
            log_debug(f"search({sample['split']}-{sample['id']}): server initialized.")
            init_units = await server.load_sorry_async(OPEN_HEADER + sample['formal_statement'])
            if len(init_units) != 2 or init_units[0].goal_state != None or init_units[0].messages != [] or not isinstance(init_units[-1].goal_state, GoalState):
                logger.opt(colors=False).error(f"search({sample['split']}-{sample['id']}): error ```\n{str(init_units)}\n```.")
                return
            else:
                init_state = init_units[-1].goal_state
                logger.opt(colors=False).info(f"search({sample['split']}-{sample['id']}): initial state ```\n{str(init_state)}\n```.")
            if dry_run:
                return
            search_result = await agent.search_async(   # TODO: Async search
                server=server,
                init_state=init_state,
                tag=f"{sample['split']}-{sample['id']}",
                verbose=verbose,
            )
            sample['search_result'] = search_result
            logger.info(f"search({sample['split']}-{sample['id']}): " + ('<green>succeeded</green>' if search_result.success else '<yellow>failed</yellow>') + f' in {search_result.duration} (s)')
            finished.append(sample)
        except:
            logger.error(f"search({sample['split']}-{sample['id']}): failed because {traceback.format_exc()}")

    async def _async_main():
        pending_tasks: Set[asyncio.Task] = set()
        for i, sample in tqdm(enumerate(samples)):
            if len(pending_tasks) >= num_concurrency:
                done_tasks, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done_tasks:
                    if task.exception() is not None:
                        logger.opt(colors=False).error(f"Exception occurred: {task.exception()} {task.get_stack()}")
                        for pending_task in pending_tasks:
                            pending_task.cancel()
                        return
            pending_tasks.add(
                asyncio.create_task(
                    search(sample)
                )
            )
        if len(pending_tasks) > 0:
            await asyncio.wait(pending_tasks)
        await logger.complete()
    
    try:
        asyncio.run(_async_main())
    finally:
        try:
            logger.info(f"Finished search, saving at {osp.join(log_root, f'minif2f_search_results.(pkl|jsonl)')}")
            with open(osp.join(log_root, f'minif2f_search_results.pkl'), 'wb') as f:
                pickle.dump(finished, f)
            with open(osp.join(log_root, f'minif2f_search_results.jsonl'), 'w') as f:
                for sample in finished:
                    sample['search_result'] = sample['search_result'].serialize()
                    f.write(json.dumps(sample)+'\n')
        except Exception as e:
            logger.error(traceback.format_exc())
            import pdb; pdb.set_trace()


if __name__ == '__main__':
    Fire(main)
