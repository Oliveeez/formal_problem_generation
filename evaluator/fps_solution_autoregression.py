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
from typing import Dict, Set, List
import time
import pickle
import regex as re

import vllm     # vLLM should be imported before any 3rd-party libraries, o.w. all Pantograph REPLs are scheduled at the same CPU core.
from openai import OpenAI, AsyncOpenAI, NOT_GIVEN
from openai.types.chat import ChatCompletion
import argparse
from tqdm import tqdm
from termcolor import colored
from loguru import logger
from fire import Fire

from common.constants import CORE_OPTIONS, OPEN_HEADER, BANNED_TOKENS, MVAR_PATTERN
from common.pantograph.dataclasses import GoalState, FormalProblem, TacticInvocation, ProofSearchResult
from common.pantograph.server import Server, ServerError
from common.pantograph.solving_server import PropSolvingServer
from agent.proof_generation import ProofSearchResult, HammerProofSearchAgent, StepProver_NALP_LLMProofSearchAgent, StepProver_Critic_LLMProofSearchAgent, SFT_NALP_LLMProofSearchAgent, SFT_NALP_AVGGOAL_LLMProofSearchAgent
from agent.solution_autoregression import SFT_NALP_LLMSolutionAutoregressionAgent
from common.utils import remove_comments, normalize_spaces

SOLVER_MODEL_DICT = {
    'sft_vanilla': SFT_NALP_LLMSolutionAutoregressionAgent
}

PROVER_MODEL_DICT = {
    'hammer': HammerProofSearchAgent,
    'stepprover_vanilla': StepProver_NALP_LLMProofSearchAgent,
    'sft_vanilla': SFT_NALP_LLMProofSearchAgent,
}

def main(
        log_root: str,
        benchmark: str,
        solver_model: str,
        solver_base_url: str,
        solver_api_key: str,
        solver_model_name: str,
        prover_model: str,
        prover_base_url: Optional[str]=None,
        prover_api_key: Optional[str]=None,
        prover_model_name: Optional[str]=None,
        benchmark_root: str='/home/ma-user/workspace/fps_pantograph/formal_problem_solving/data/benchmark',
        project_root: str='/home/ma-user/workspace/fps_pantograph/formal_problem_solving/data/MiniF2F',
        try_num: int=1,
        temperature: float=0.7,
        solver_max_search_trials: int=80,
        solver_num_samples_per_trial: int=8,
        prover_max_search_trials: int=16,
        prover_num_samples_per_trial: int=4,
        max_tokens: int=2048,
        num_concurrency: int=12,
        verbose: bool=False,
        dry_run: bool=False,
        resume_from: Optional[str]=None,
):
    saved_args = {**locals()}
    
    benchmark = benchmark.lower()
    assert benchmark in ['formal_math500', 'minif2f_solving', 'putnam_solving']
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_prefix = benchmark+'.'+'solution_autoregression_dfps'+'.'

    os.makedirs(log_root, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level='INFO')    # filter=lambda record: record["name"] != "agent.solution_autoformalization"
    logger.add(osp.join(log_root, log_prefix+now+'.log'), level='DEBUG')
    logger.info(f'Evaluating solution autoregression on {benchmark} with hyperparams: {saved_args}')
    log_debug = logger.info if verbose else logger.debug

    prover_client = AsyncOpenAI(
        base_url=prover_base_url,
        api_key=prover_api_key
    )
    solver_client = AsyncOpenAI(
        base_url=solver_base_url,
        api_key=solver_api_key
    )

    # Load data
    samples = []
    with open(osp.join(benchmark_root, benchmark+'.json'), 'r') as f:
        samples = json.load(f)
    if benchmark == 'formal_math500':
        samples = [
            FormalProblem(
                informal_problem=s['informal_problem'],
                informal_answer=s['informal_answer'],
                informal_solution=s['informal_solution'],
                header=s['header'],
                formal_statement=f"example\n{s['formal_problem']}\n: {s['formal_answer']}\n:= sorry",
                metainfo=dict(
                    level=s['level'],
                    subject=s['subject'],
                    annotator=s['annotator']
                )
            ) for s in samples
        ]
    elif benchmark == 'putnam_solving':
        samples = [
            FormalProblem(
                informal_problem=s['informal_statement'],
                informal_answer=s['informal_answer'],
                informal_solution=s['informal_solution'],
                header=None if len(s['opens']) == 0 else 'open ' + ' '.join(s['opens']),
                formal_statement=f"example\n{s['formal_problem']}\n: {s['formal_answer']}\n:= sorry",
                metainfo=dict(
                    problem_name=s['problem_name'],
                    tags=s['tags'],
                    annotator=s['annotator']
                )
            ) for s in samples
        ]
    elif benchmark == 'minif2f_solving':
        samples = [
            FormalProblem(
                informal_problem=s['informal_stmt'],
                informal_answer=s['informal_answer'],
                informal_solution=s['informal_proof'],
                header=None,
                formal_statement=f"example\n{s['formal_problem']}\n: {s['formal_answer']}\n:= sorry",
                metainfo=dict(
                    id=s['id'],
                    split=s['split'],
                    annotator=s['annotator'],
                    formal_statement=s['formal_statement']
                )
            ) for s in samples
        ]
    
    finished = []
    logger.info(f"Loaded {len(samples)} samples for {benchmark} from {osp.join(benchmark_root, benchmark+'.json')}")
    
    # Resume from interrupted experiments
    if resume_from is not None:
        load_file = [p for p in os.listdir(resume_from) if p.startswith(log_prefix) and p.endswith('.pkl')]
        if len(load_file) != 1:
            logger.error(f'There should be only one loadable experiment log in "{resume_from}", but detected {load_file}')
            return
        load_file = load_file[0]
        with open(osp.join(resume_from, load_file), 'rb') as f:
            finished = pickle.load(f)
        samples_str_all = set([
            (d.informal_problem, d.informal_answer, d.informal_solution, d.formal_statement, d.formal_answer) for d in samples
        ])
        finished_str_all = set([
            (d.informal_problem, d.informal_answer, d.informal_solution, d.formal_statement, d.formal_answer) for d in finished
        ])
        assert len(samples_str_all) == len(samples)
        assert len(finished_str_all) == len(finished)
        assert finished_str_all.issubset(samples_str_all)
        samples = [d for d in samples if (d.informal_problem, d.informal_answer, d.informal_solution, d.formal_statement, d.formal_answer) not in finished_str_all]
        logger.critical(f'Resumed {len(finished)} samples from {osp.join(resume_from, load_file)}, now remaining {len(samples)} samples to evaluate.')

    async def search(sample: FormalProblem, tag_i: int) -> None:
        results = []
        for try_i in range(try_num):
            n_cost_attempts = 0
            while n_cost_attempts < solver_max_search_trials:
                try:
                    time_start = time.time()
                    solution_searcher = SOLVER_MODEL_DICT[solver_model](
                        gen_client=solver_client,
                        gen_model_name=solver_model_name,
                        proof_searcher=PROVER_MODEL_DICT[prover_model](
                            gen_client=prover_client,
                            gen_model_name=prover_model_name,
                            max_search_trials=prover_max_search_trials,
                            num_samples_per_trial=prover_num_samples_per_trial,
                            temperature=temperature,
                            max_tokens=(max_tokens if max_tokens > 0 else NOT_GIVEN),
                        ),
                        max_search_trials=solver_max_search_trials - n_cost_attempts,
                        num_samples_per_trial=solver_num_samples_per_trial,
                        temperature=temperature,
                        max_tokens=(max_tokens if max_tokens > 0 else NOT_GIVEN),
                    )
                    solving_server = PropSolvingServer(
                        imports=["Mathlib", "Aesop"],
                        project_path=project_root,
                        timeout=300,
                        tag=f'{tag_i}-{try_i}/{try_num}'
                    )

                    log_debug(f"search({tag_i}-{try_i}/{try_num}): server initialized.")
                    await solving_server.load_problem_async(sample, force_parse=True)
                    if dry_run:
                        return
                    init_forward_state = await solving_server.init_forward_reasoning_state_async()
                    init_solution_state = await solving_server.init_solving_state_async()
                    result = await solution_searcher.search_async(
                        solving_server,
                        init_forward_state,
                        init_solution_state,
                        tag=f'{tag_i}-{try_i}/{try_num}',
                        verbose=verbose,
                    )
                    n_cost_attempts += result.cost
                    results.append(result)
                    # if result.submission is not None: # Submitted
                    #     if result.success:
                    #         logger.opt(colors=True).info(f'<green>search({tag_i}): succeeded at the {try_i}-th attempt in {time.time() - time_start} (s).</green>')
                    #     else:
                    #         logger.opt(colors=True).info(f'<red>search({tag_i}): finished with incorrect submission at the {try_i}-th attempt in {time.time() - time_start} (s).</red>')
                    if result.success: # Success@Budget
                        logger.opt(colors=True).info(f'<green>search({tag_i}): succeeded at the {try_i}-th attempt in {time.time() - time_start} (s).</green>')
                        break
                    else:
                        logger.info(f'search({tag_i}-{try_i}/{try_num}): failed with {result.cost} attempts in {time.time() - time_start} (s).')
                except:
                    logger.error(f"search({tag_i}-{try_i}/{try_num}): failed because {traceback.format_exc()}")
                    break
        
        logger.info(f'search({tag_i}): search finished.')
        finished.append(results)

    async def _async_main():
        pending_tasks: Set[asyncio.Task] = set()
        for i, sample in tqdm(enumerate(samples)):
            if len(pending_tasks) >= num_concurrency:
                done_tasks, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done_tasks:
                    if task.exception() is not None:
                        logger.error(f"Exception occurred: {task.exception()} {task.get_stack()}")
                        for pending_task in pending_tasks:
                            pending_task.cancel()
                        return
            pending_tasks.add(
                asyncio.create_task(
                    search(sample, i)
                )
            )
        if len(pending_tasks) > 0:
            await asyncio.wait(pending_tasks)
        await logger.complete()
    
    try:
        asyncio.run(_async_main())
    finally:
        try:
            logger.info(f"Finished search, saving at {osp.join(log_root, log_prefix+now+'.(pkl|jsonl)')}")
            with open(osp.join(log_root, log_prefix+now+'.pkl'), 'wb') as f:
                pickle.dump(finished, f)
            with open(osp.join(log_root, log_prefix+now+'.jsonl'), 'w') as f:
                for attempts in finished:
                    f.write(json.dumps([s.serialize() for s in attempts])+'\n')
        except Exception as e:
            logger.error(traceback.format_exc())
            import pdb; pdb.set_trace()


if __name__ == '__main__':
    Fire(main)
