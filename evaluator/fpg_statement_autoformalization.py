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
import random

import vllm     # vLLM should be imported before any 3rd-party libraries, o.w. all Pantograph REPLs are scheduled at the same CPU core.
from openai import AsyncOpenAI, NOT_GIVEN
from tqdm import tqdm
from loguru import logger
from fire import Fire
import aiofiles

from common.constants import FPS_GLOBAL_SETTING, CORE_OPTIONS
from common.utils import add_one_to_port
from common.pantograph.dataclasses import ProblemGenerationProcess
from common.pantograph.server import PersistentServer
from agent.statement_autoformalization import VersatileLLMStatementAutoformalizationAgent

NEWLINE = '\n'

def main(
    log_root: str,
    base_url: str,
    api_key: str,
    model_name: str,
    n_servers: int,
    load_path: str,
    num_generation_attempt: int=5000,
    project_root: str='/home/ma-user/workspace/fps_pantograph/formal_problem_solving/data/MiniF2F',
    num_concurrency: int=12,
    seed: int=42,
):
    saved_args = {**locals()}
    
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_prefix = 'statement_autoformalization'+'.'

    os.makedirs(log_root, exist_ok=True)
    if num_concurrency > 1:
        logger.remove()
        logger.add(sys.stdout, level='INFO')    # filter=lambda record: record["name"] != "agent.solution_autoformalization"
        logger.add(osp.join(log_root, log_prefix+now+'.log'), level='DEBUG')
    logger.info(f'Evaluating problem generator with hyperparams: {saved_args}')

    with open(load_path, 'r') as f:
        data = [json.loads(l) for l in f.readlines()]
    data_chosen = random.sample(data, num_generation_attempt)
    finished = [None for _ in random.choice(data_chosen)]
    logger.info(f"Created {len([v for v in finished if v is None])} tasks")
        
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
    
    clients = []
    for _ in range(n_servers):
        clients.append(
            AsyncOpenAI(
                base_url=base_url,
                api_key=api_key
            )
        )
        base_url = add_one_to_port(base_url)
    
    available_agents = [
        VersatileLLMStatementAutoformalizationAgent(
            client=clients[i%n_servers],
            model_name=model_name,
            temperature=0.0
        ) for i in range(num_concurrency)
    ]
    
    async def generate_worker(tag_i: int) -> None:
        server = available_servers.pop()
        agent = available_agents.pop()
        informal_datapoint = data_chosen[tag_i]
        informal_statement = informal_datapoint['informal_problem'] + '\nShow that the answer is "' + informal_datapoint['informal_answer'].strip() + '".'
        header = None
        stmt_code = ''
        try:
            server.tag = str(tag_i)
            header, stmt_code = await agent.autoformalize_async(
                informal_statement=informal_statement,
                server=server,
                tag=str(tag_i),
            )
            if stmt_code is not None:
                logger.opt(colors=True).info(f'<green>generate_worker({tag_i}): generation succeeded.</green>')
                logger.info("" if header is None else (header.rstrip() + NEWLINE) + stmt_code)
        except Exception as e:
            logger.info(f'generate_worker({tag_i}): generation failed due to: {repr(e)}\n{traceback.format_exc()}')
        finally:
            finished[tag_i] = ProblemGenerationProcess(
                informal_problem=informal_datapoint['informal_problem'],
                informal_answer=informal_datapoint['informal_answer'],
                informal_solution=informal_datapoint['informal_solution'],
                header=header,
                formal_statement=stmt_code,
                formal_solution_draft=None,
                formal_proofs=[],
                steps=[],
                dependencies=[],
                trajectory=[],
                metainfo={k : v for (k, v) in informal_datapoint.items() if k not in ['informal_problem', 'informal_answer', 'informal_solution']} | {'token_usage:stmt_autoformalizer': agent.last_token_usage}
            )
            available_agents.insert(0, agent)
            available_servers.insert(0, server)

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
                    generate_worker(i)
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
