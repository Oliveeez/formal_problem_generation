import sys
import os
import os.path as osp
from io import BufferedWriter
import collections as C
import itertools as I
import random
import pickle
from typing import List, Dict, Set, Tuple, Callable
import asyncio
import regex as re
from datetime import datetime
import traceback
import multiprocessing as mp
import time
import json

import dacite
import fire
import aiofiles
from aiofiles.threadpool.text import AsyncTextIOWrapper
from openai import AsyncOpenAI
import pyarrow
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm
import msgspec

from common.constants import OPEN_HEADER, CORE_OPTIONS, MVAR_PATTERN, BANNED_TOKENS
from common.utils import remove_comments, normalize_spaces, remove_spaces, normalize_draft, remove_min_whitespace, chunk_list, replace_sorry, replace_calc, remove_multiline_comments
from common.pantograph.dataclasses import TacticInvocation, Goal, GoalState, ProblemGenerationStep, ProblemGenerationProcess, TacticDraft
from common.pantograph.server import Server, PersistentServer, TacticFailure, ServerError
from common.pantograph.parsing_server import PersistentParsingServer
from agent.problem_generation import ProblemEvaluator

async def async_worker(
    result: ProblemGenerationProcess,
    idx: int,
    agent: ProblemEvaluator,
    available_servers: List[PersistentServer],
    results: List,
) -> None:
    server = available_servers.pop()
    server.tag = str(idx)
    try:
        result.metainfo = result.metainfo if isinstance(result.metainfo, Dict) else json.loads(result.metainfo)
        eval_result = await agent.evaluate_async(
            server=server,
            result=result,
            tag=str(idx)
        )
        result.metainfo['eval_result'] = eval_result
        result.metainfo = json.dumps(result.metainfo)
    except Exception as e:
        logger.debug(f'async_worker({idx}): failed due to traceback {traceback.format_exc()}')
        logger.warning(f'async_worker({idx}): failed due to exception {repr(e)}')
        return
    finally:
        result.metainfo = json.dumps(result.metainfo)
        results[idx] = result
        server.tag = ''
        available_servers.insert(0, server)

def main(
    load_path: str,
    log_root: str,
    proof_gen_base_urls: List[str],
    proof_gen_api_keys: List[str],
    proof_gen_model_names: List[str],
    project_root: str='/home/ma-user/workspace/fps_pantograph/formal_problem_solving/data/MiniF2F',
    temperature: float=1.0,
    max_tokens: int=-1,
    top_p: float=0.95,
    try_num: int=8,
    num_concurrency: int=12,
):
    saved_args = {**locals()}
    
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_prefix = 'evaluate_falsify_kc'+'.'

    os.makedirs(log_root, exist_ok=True)
    if num_concurrency > 1:
        logger.remove()
        logger.add(sys.stdout, level='INFO')    # filter=lambda record: record["name"] != "agent.solution_autoformalization"
    logger.add(osp.join(log_root, log_prefix+now+'.log'), level='DEBUG')
    logger.info(f'Evaluating problem generator with hyperparams: {saved_args}')
    assert len(proof_gen_base_urls) == len(proof_gen_api_keys), f'{len(proof_gen_base_urls)} != {len(proof_gen_api_keys)}'
    assert len(proof_gen_api_keys) == len(proof_gen_model_names), f'{len(proof_gen_api_keys)} != {len(proof_gen_model_names)}'
    
    with open(load_path, 'rb') as f:
        data_to_process = pickle.load(f)
    
    finished_list = [None for _ in range(len(data_to_process))]

    available_servers = [
        PersistentServer(
            max_count=64,
            is_state_based=True,
            tag='',
            _sync_init=False,
            imports=["Mathlib", "Aesop"],
            project_path=project_root,
            core_options=CORE_OPTIONS,
            timeout=300,
        ) for _ in range(num_concurrency)
    ]
    proof_gen_clients = [
        AsyncOpenAI(
            base_url=proof_gen_base_url,
            api_key=proof_gen_api_key
        ) for (proof_gen_base_url, proof_gen_api_key) in zip(proof_gen_base_urls, proof_gen_api_keys)
    ]
    agent = ProblemEvaluator(
        clients=proof_gen_clients,
        models=proof_gen_model_names,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        try_num=try_num,
    )

    tasks = [
        (i, d) for (i, d) in enumerate(data_to_process)
    ]
    logger.info(f'Evaluating {len(data_to_process)} samples.')

    async def _async_main():
        pending_tasks = set()
        for i, d in tasks:
            if len(pending_tasks) >= num_concurrency:
                done_tasks, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done_tasks:
                    if task.exception() is not None:
                        logger.error(f"Exception occurred: {task.exception()}")
                        for pending_task in pending_tasks:
                            pending_task.cancel()
                        # import ipdb; ipdb.set_trace()
            pending_tasks.add(
                asyncio.create_task(
                    async_worker(
                        result=d,
                        idx=i,
                        agent=agent,
                        available_servers=available_servers,
                        results=finished_list,
                    )
                )
            )
        if len(pending_tasks) > 0:
            await asyncio.wait(pending_tasks)
        await logger.complete()

    try:
        asyncio.run(_async_main())
        logger.opt(colors=True).info(f'<cyan>main(): All finished.</cyan>')
    except Exception as e:
        logger.error(f"main(): Failed due to Exception {e}\n{traceback.format_exc()}")
    finally:
        try:
            logger.info(f"Finished generation, saving at {osp.join(log_root, log_prefix+now+'.(pkl|jsonl)')}")
            with open(osp.join(log_root, log_prefix+now+'.pkl'), 'wb') as f:
                pickle.dump(finished_list, f)
        except Exception as e:
            logger.error(traceback.format_exc())
            import pdb; pdb.set_trace()

if __name__ == '__main__':
    fire.Fire(main)
