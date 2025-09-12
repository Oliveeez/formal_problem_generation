import os
import os.path as osp
import sys
import json
import asyncio
import traceback
from datetime import datetime
from typing import Dict, Set, List, Any, Tuple, Optional
import pickle
import regex as re
import itertools as I
import collections as C
import random

import vllm     # vLLM should be imported before any 3rd-party libraries, o.w. all Pantograph REPLs are scheduled at the same CPU core.
from openai import AsyncOpenAI, NOT_GIVEN
from tqdm import tqdm
from loguru import logger
from fire import Fire
import aiofiles

from common.constants import FPS_GLOBAL_SETTING, CORE_OPTIONS
from common.utils import add_one_to_port, starified_downsample
from common.pantograph.server import PersistentServer
from common.pantograph.parsing_server import PersistentParsingServer
from agent.problem_generation import ProblemFalsifier
from agent.problem_generation import AutoregressiveProblemGenerationAgent, SFT_LLMAutoregressiveProblemGenerationAgent, SFT_LLMAutoregressiveProblemGenerationAgentV2, SFT_LLMAutoregressiveProblemGenerationAgentV3

NEWLINE = '\n'
# FPS_GLOBAL_SETTING['TO_SYNC_ENABLED'] = True
AGENT_DICT: Dict[str, AutoregressiveProblemGenerationAgent] = {
    'sft_ar' : SFT_LLMAutoregressiveProblemGenerationAgent,
    'sft_ar_v2': SFT_LLMAutoregressiveProblemGenerationAgentV2,
    'sft_ar_v3': SFT_LLMAutoregressiveProblemGenerationAgentV3
}
CONDITION_FILE_DICT = {
    'fineleancorpus' : 'data/conditions.fineleancorpus.82438.json',
    'numina_lean' : 'data/conditions.numina_lean.39509.json'
}

def main(
    log_root: str,
    agent_name: str,
    num_generation_attempt: int,    # 10000
    condition_sources: List[str],   # fineleancorpus, numina_lean
    base_url: str,
    api_key: str,
    model_name: str,
    n_servers: int=1,   # Assuming all servers share the same ip and have consecutive ports
    falsify_base_url: Optional[str]=None,
    falsify_api_key: Optional[str]=None,
    falsify_model_name: Optional[str]=None,
    falsify_n_servers: int=0,   # Assuming all servers share the same ip and have consecutive ports
    project_root: str='/home/ma-user/workspace/fps_pantograph/formal_problem_solving/data/MiniF2F',
    reassemble_trajectory: bool=False,
    temperature: float=1.0,
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
    if num_concurrency > 1:
        logger.remove()
        logger.add(sys.stdout, level='INFO')    # filter=lambda record: record["name"] != "agent.solution_autoformalization"
        logger.add(osp.join(log_root, log_prefix+now+'.log'), level='DEBUG')
    logger.info(f'Evaluating problem generator with hyperparams: {saved_args}')

    # Resume from interrupted experiments
    if resume_from is not None:
        load_file = sorted([p for p in os.listdir(resume_from) if p.startswith(log_prefix) and p.endswith('.pkl')])
        if len(load_file) > 0:
            if len(load_file) > 1:
                logger.warning(f'Detected multiple checkpoints: {load_file}')
            load_file = load_file[-1]
            with open(osp.join(resume_from, load_file), 'rb') as f:
                (conditions_sampled, finished) = pickle.load(f)
            # tasks = [k for (k, v) in finished.items() if v is None]
            logger.critical(f'Resumed {len(finished)} results from {osp.join(resume_from, load_file)}.')
    else:
        conditions_total = []
        for c in condition_sources:
            with open(CONDITION_FILE_DICT[c], 'r') as f:
                conditions_total.extend([tuple(sorted([(k, str(v)) for k, v in c.items()])) for c in json.load(f)])
        conditions_sampled = starified_downsample(conditions_total, num_generation_attempt)
        finished = [None for _ in range(len(conditions_sampled))]
        
    logger.info(f"Created {len([v for v in finished if v is None])} tasks")

    base_urls = [base_url]
    for _ in range(n_servers-1):
        base_urls.append(add_one_to_port(base_urls[-1]))
    
    if falsify_n_servers > 0:
        falsifier_base_urls = [falsify_base_url]
        for _ in range(falsify_n_servers-1):
            falsifier_base_urls.append(add_one_to_port(falsifier_base_urls[-1]))
            
        available_falsifiers: List[ProblemFalsifier] = [
            ProblemFalsifier(
                clients=[
                    AsyncOpenAI(
                        base_url=falsifier_base_urls[i % falsify_n_servers],
                        api_key=falsify_api_key
                    )
                ],
                models=[falsify_model_name],
                server=PersistentServer(
                    is_state_based=True,
                    tag=f'Falsify-{i}',
                    _sync_init=False,
                    imports=["Mathlib", "Aesop"],
                    project_path=project_root,
                    core_options=CORE_OPTIONS,
                    timeout=60,
                ),
                temperature=0.0
            ) for i in range(num_concurrency)
        ]
    else:
        available_falsifiers = []
    
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
    available_parsers = [
        PersistentParsingServer(
            max_count=32,
            is_state_based=True,
            tag='',
            _sync_init=False,
            imports=["Mathlib", "Aesop"],
            project_path=project_root,
            core_options=CORE_OPTIONS,
            timeout=300
        ) for i in range(num_concurrency)
    ]

    async def generate_worker(condition: List[Tuple[str, Any]], tag_i: int) -> None:
        server = available_servers.pop()
        parser = available_parsers.pop()
        if len(available_falsifiers) > 0:
            falsifier = available_falsifiers.pop()
        else:
            falsifier = None
        
        try:
            server.tag = str(tag_i)
            parser.tag = str(tag_i)
            client = AsyncOpenAI(
                base_url=base_urls[tag_i % len(base_urls)],
                api_key=api_key
            )
            problem_generator: AutoregressiveProblemGenerationAgent = AGENT_DICT[agent_name](
                gen_client=client,
                gen_model_name=model_name,
                max_search_trials=max_search_trials,
                num_max_samples_per_trial=num_max_samples_per_trial,
                temperature=temperature,
                max_tokens=(max_tokens if max_tokens > 0 else NOT_GIVEN)
            )
            # breakpoint()
            if falsifier is not None:
                problem_generator.falsifiers.append(
                    falsifier.falsify_async
                )
            result = await problem_generator.generate_async(
                conditions={k : v for (k, v) in condition},
                server=server,
                parser=parser,
                reassemble_trajectory=reassemble_trajectory,
                tag=str(tag_i),
                verbose=False,
            )
            
            logger.info(f'generate_worker({tag_i}, {condition}): generation finished: {"" if result.header is None else (result.header.rstrip() + NEWLINE)}{result.formal_statement}')
            finished[tag_i] = result
        except Exception as e:
            logger.info(f'generate_worker({tag_i}, {condition}): generation failed due to: {repr(e)}\n{traceback.format_exc()}')
        finally:
            server.tag = ''
            parser.tag = ''
            available_servers.insert(0, server)
            available_parsers.insert(0, parser)
            if falsifier is not None:
                try:
                    result.metainfo = json.loads(result.metainfo)
                except:
                    pass
                result.metainfo['falsifier_token_usage'] = falsifier.token_usage
                result.metainfo = json.dumps(result.metainfo)
                falsifier.token_usage = C.defaultdict(list)
                available_falsifiers.insert(0, falsifier)

    async def _async_main():
        pending_tasks: Set[asyncio.Task] = set()
        for i, v in enumerate(finished):
            if v is not None:
                continue
            if len(pending_tasks) >= num_concurrency:
                done_tasks, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                # async with aiofiles.open(osp.join(log_root, log_prefix+now+'.pkl'), 'wb') as f:
                #     await f.write(pickle.dumps((conditions_sampled, finished)))
                for task in done_tasks:
                    if task.exception() is not None:
                        logger.error(f"Exception occurred: {task.exception()} {task.get_stack()}")
                        for pending_task in pending_tasks:
                            pending_task.cancel()
                        return
            pending_tasks.add(
                asyncio.create_task(
                    generate_worker(conditions_sampled[i], i)
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
                pickle.dump((conditions_sampled, finished), f)
            if len(available_falsifiers) > 0:
                breakpoint()
                with open(osp.join(log_root, log_prefix+'falsify_train.'+now+'.pkl'), 'wb') as f:
                    data_falsify_train = []
                    for agent in available_falsifiers:
                        data_falsify_train.extend(agent.data_train)
                        agent.data_train.clear()
                    pickle.dump(data_falsify_train, f)
        except Exception as e:
            logger.error(traceback.format_exc())
            import pdb; pdb.set_trace()


if __name__ == '__main__':
    Fire(main)
