import os
import os.path as osp
import sys
import json
import asyncio
import traceback
from datetime import datetime
from typing import Optional
from typing import Dict, Set, List, Any
import pickle
import regex as re
import itertools as I
import random

import vllm     # vLLM should be imported before any 3rd-party libraries, o.w. all Pantograph REPLs are scheduled at the same CPU core.
from openai import AsyncOpenAI, NOT_GIVEN
from tqdm import tqdm
from loguru import logger
from fire import Fire

from common.constants import FPS_GLOBAL_SETTING, CORE_OPTIONS
from common.pantograph.server import PersistentServer
from agent.problem_generation import ProblemGenerationAgent, SFT_LLMAutoregressiveProblemGenerationAgent

# FPS_GLOBAL_SETTING['TO_SYNC_ENABLED'] = True
AGENT_DICT: Dict[str, ProblemGenerationAgent] = {
    'sft_ar' : SFT_LLMAutoregressiveProblemGenerationAgent
}


def main(
    log_root: str,
    agent_name: str,
    base_url: str,
    api_key: str,
    model_name: str,
    project_root: str='/home/ma-user/workspace/fps_pantograph/formal_problem_solving/data/MiniF2F',
    num_generation_attempt: int=5,
    reassemble_trajectory: bool=False,
    temperature: float=0.7,
    max_search_trials: int=80,
    num_max_samples_per_trial: int=8,
    max_tokens: int=-1,
    num_concurrency: int=12,
    resume_from: Optional[str]=None,
):
    saved_args = {**locals()}
    
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_prefix = 'problem_generation'+'.'

    os.makedirs(log_root, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level='INFO')    # filter=lambda record: record["name"] != "agent.solution_autoformalization"
    logger.add(osp.join(log_root, log_prefix+now+'.log'), level='DEBUG')
    logger.info(f'Evaluating problem generator with hyperparams: {saved_args}')

    # In-domain
    problem_types = ['Algebra', 'Geometry', 'Combinatorics', 'Number Theory', 'Logic and Puzzles', 'Calculus', 'Inequalities', 'Other']
    sources = ['cn_k12', 'olympiads', 'aops_forum', 'cn_contest', 'number_theory', 'olympiads_ref', 'amc_aime', 'inequalities']
    
    # OOD
    problem_types_ood = ['Linear Algebra', 'Abstract Algebra', 'Real Analysis', 'Topology']
    sources_ood = ['Chinese Gaokao', 'IMO', 'Undergraduate Math Exam', 'Undergraduate Math Textbook', 'Graduate Math Exam', 'Graduate Math Textbook']
    
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )

    # Load data
    tasks = list(I.product(
        I.product(  # Condition
            I.chain(problem_types, problem_types_ood),
            I.chain(sources, sources_ood),
        ), 
        range(num_generation_attempt)
    ))
    random.shuffle(tasks)
    finished = dict()
    logger.info(f"Created {len(tasks)} tasks")
    
    # Resume from interrupted experiments
    if resume_from is not None:
        load_file = sorted([p for p in os.listdir(resume_from) if p.startswith(log_prefix) and p.endswith('.pkl')])
        if len(load_file) > 0:
            if len(load_file) > 1:
                logger.warning(f'Detected multiple checkpoints: {load_file}')
            load_file = load_file[-1]
            with open(osp.join(resume_from, load_file), 'rb') as f:
                finished = pickle.load(f)
            tasks = [k for k in tasks if k not in finished.keys()]
            logger.critical(f'Resumed {len(finished)} results from {osp.join(resume_from, load_file)}, now remaining {len(tasks)} tasks to evaluate.')

    async def generate_worker(condition: Any, key: Any, tag_i: int) -> None:
        problem_generator: ProblemGenerationAgent = AGENT_DICT[agent_name](
            gen_client=client,
            gen_model_name=model_name,
            max_search_trials=max_search_trials,
            num_max_samples_per_trial=num_max_samples_per_trial,
            temperature=temperature,
            max_tokens=(max_tokens if max_tokens > 0 else NOT_GIVEN)
        )
        result = await problem_generator.generate_async(
            conditions=condition,
            server=PersistentServer(
                is_state_based=True,
                tag=f'{tag_i}',
                _sync_init=False,
                imports=["Mathlib", "Aesop"],
                project_path=project_root,
                core_options=CORE_OPTIONS,
                timeout=300,
            ),
            reassemble_trajectory=reassemble_trajectory,
            tag=str(tag_i),
            verbose=False,
        )
        
        logger.info(f'generate_worker({tag_i}, {condition}): generation finished: {result.formal_statement}')
        finished[key] = result

    async def _async_main():
        pending_tasks: Set[asyncio.Task] = set()
        for i, (condition, i_generation) in enumerate(tasks):
            if len(pending_tasks) >= num_concurrency:
                done_tasks, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                with open(osp.join(log_root, log_prefix+now+'.pkl'), 'wb') as f:
                    pickle.dump(finished, f)
                for task in done_tasks:
                    if task.exception() is not None:
                        logger.error(f"Exception occurred: {task.exception()} {task.get_stack()}")
                        for pending_task in pending_tasks:
                            pending_task.cancel()
                        return
            pending_tasks.add(
                asyncio.create_task(
                    generate_worker(condition, (condition, i_generation), i)
                )
            )
        if len(pending_tasks) > 0:
            await asyncio.wait(pending_tasks)
        await logger.complete()
    
    try:
        asyncio.run(_async_main())
    finally:
        try:
            logger.info(f"Finished generation, saving at {osp.join(log_root, log_prefix+now+'.(pkl|jsonl)')}")
            with open(osp.join(log_root, log_prefix+now+'.pkl'), 'wb') as f:
                pickle.dump(finished, f)
        except Exception as e:
            logger.error(traceback.format_exc())
            import pdb; pdb.set_trace()


if __name__ == '__main__':
    Fire(main)
