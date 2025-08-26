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
import aiofiles

from common.constants import FPS_GLOBAL_SETTING, CORE_OPTIONS
from common.utils import add_one_to_port
from common.pantograph.server import PersistentServer
from agent.problem_generation import LLMWholeProblemGenerationAgent, SFT_LLMWholeProblemGenerationAgent

# FPS_GLOBAL_SETTING['TO_SYNC_ENABLED'] = True
AGENT_DICT: Dict[str, LLMWholeProblemGenerationAgent] = {
    'sft_wg': SFT_LLMWholeProblemGenerationAgent
}


def main(
    log_root: str,
    agent_name: str,
    statement_gen_base_url: str,
    statement_gen_api_key: str,
    statement_gen_model_name: str,
    proof_gen_base_urls: List[str],
    proof_gen_api_keys: List[str],
    proof_gen_model_names: List[str],
    project_root: str='/home/ma-user/workspace/fps_pantograph/formal_problem_solving/data/MiniF2F',
    num_generation_attempt: int=5,
    temperature: float=0.7,
    num_max_samples_per_trial: int=1,
    max_tokens: int=-1,
    num_concurrency: int=12,
    resume_from: Optional[str]=None,
):
    saved_args = {**locals()}
    
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_prefix = 'problem_generation'+'.'

    os.makedirs(log_root, exist_ok=True)
    if num_concurrency > 1:
        logger.remove()
        logger.add(sys.stdout, level='INFO')    # filter=lambda record: record["name"] != "agent.solution_autoformalization"
    logger.add(osp.join(log_root, log_prefix+now+'.log'), level='DEBUG')
    logger.info(f'Evaluating problem generator with hyperparams: {saved_args}')
    assert len(proof_gen_base_urls) == len(proof_gen_api_keys), f'{len(proof_gen_base_urls)} != {len(proof_gen_api_keys)}'
    assert len(proof_gen_api_keys) == len(proof_gen_model_names), f'{len(proof_gen_api_keys)} != {len(proof_gen_model_names)}'

    # In-domain
    problem_types = ['Algebra', 'Number Theory', 'Precalculus', 'Trigonometry', 'Arithmetic', 'Functional Equations', 'Inequalities', 'Recursion Other', 'Calculus', 'Logic and Puzzles', 'Linear Algebra', 'Combinatorics', 'Other', 'Geometry', 'unknown', 'Intermediate Algebra', 'NaN']
    sources = ['unknown', 'number_theory', 'aops_forum', 'cn_k12', 'math_train', 'math_test', 'olympiads_ref', 'amc_aime', 'olympiads', 'inequalities', 'secondary_math', 'cn_contest', 'synthetic']
    
    # OOD
    problem_types_ood = ['Abstract Algebra', 'Real Analysis', 'Topology']
    sources_ood = ['Chinese Gaokao', 'IMO', 'Undergraduate Math Exam', 'Undergraduate Math Textbook', 'Graduate Math Exam', 'Graduate Math Textbook']
    
    statement_gen_client = AsyncOpenAI(
        base_url=statement_gen_base_url,
        api_key=statement_gen_api_key
    )
    proof_gen_clients = [
        AsyncOpenAI(
            base_url=proof_gen_base_url,
            api_key=proof_gen_api_key
        ) for (proof_gen_base_url, proof_gen_api_key) in zip(proof_gen_base_urls, proof_gen_api_keys)
    ]
    
    available_servers = [
        PersistentServer(
            is_state_based=True,
            tag='',
            _sync_init=False,
            imports=["Mathlib", "Aesop"],
            project_path=project_root,
            core_options=CORE_OPTIONS,
            timeout=300,
        ) for i in range(num_concurrency)
    ]

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
        server = available_servers.pop()
        try:
            server.tag = str(tag_i)
            problem_generator: LLMWholeProblemGenerationAgent = AGENT_DICT[agent_name](
                statement_gen_client=statement_gen_client,
                statement_gen_model=statement_gen_model_name,
                proof_gen_clients=proof_gen_clients,
                proof_gen_models=proof_gen_model_names,
                num_max_samples_per_trial=num_max_samples_per_trial,
                temperature=temperature,
                max_tokens=(max_tokens if max_tokens > 0 else NOT_GIVEN)
            )
            result = await problem_generator.generate_async(
                conditions=condition,
                server=server,
                decompose_steps=False,
                tag=str(tag_i),
                verbose=False,
            )
            
            logger.info(f'generate_worker({tag_i}, {condition}): generation finished: {result.formal_statement}')
            finished[key] = result
        except Exception as e:
            logger.info(f'generate_worker({tag_i}, {condition}): generation failed due to: {repr(e)}\n{traceback.format_exc()}')
        finally:
            server.tag = ''
            available_servers.insert(0, server)

    async def _async_main():
        pending_tasks: Set[asyncio.Task] = set()
        for i, (condition, i_generation) in enumerate(tasks):
            if len(pending_tasks) >= num_concurrency:
                done_tasks, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                async with aiofiles.open(osp.join(log_root, log_prefix+now+'.pkl'), 'wb') as f:
                    await f.write(pickle.dumps(finished))
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
