import sys
import os
import os.path as osp
from io import BufferedWriter
import collections as C
import itertools as I
import random
import pickle
from typing import List, Dict, Set, Tuple, Callable, Any
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
from common.utils import remove_comments, remove_spaces, rotate, generate_submission_name, decompose_statement
from common.pantograph.dataclasses import TacticInvocation, Goal, GoalState, ProblemGenerationStep, ProblemGenerationProcess, TacticDraft
from common.pantograph.server import Server, PersistentServer, TacticFailure, ServerError
from agent.proof_generation import AesopProofGenerationAgent, MultipleProvers


async def async_worker(
    result: ProblemGenerationProcess,
    key: Any,
    available_agents: List[MultipleProvers],
    available_servers: List[PersistentServer],
    finished_list: Dict,
    is_code: bool=False,
) -> None:
    agent = available_agents.pop()
    server = available_servers.pop()
    server.tag = str(key)
    try:
        result.metainfo = result.metainfo if isinstance(result.metainfo, Dict) else json.loads(result.metainfo)

        if is_code:
            provers, proofs, _ = await agent.prove_code_async(
                server=server,
                formal_statement=result.formal_statement,
                early_stop=True,
                tag=str(key)+'/prove'
            )
        
        else:
            variables = []
            context, target = decompose_statement(result.formal_statement)
            for declaration in context:
                if declaration[0] == '[':
                    try:
                        var_names, var_type = declaration[1:-1].split(':', 1)
                    except ValueError:
                        var_names = '_'
                        var_type = declaration[1:-1]
                    for name in var_names.strip().split():
                        # print(name, var_type)
                        variables.append((name.strip(), var_type))
                else:
                    assert '✝' not in declaration, f'declaration: {declaration}'
                    try:
                        var_names, var_type = declaration[1:-1].split(':', 1)
                    except ValueError:
                        var_names = declaration[1:-1]
                        var_type = None
                    for name in var_names.strip().split():
                        if '✝' in name:
                            name = '_'
                        variables.append((name.strip(), var_type))

            provers, proofs = await agent.prove_async(
                server=server,
                formal_statement='example\n' + (('\n'.join(context) + '\n: ') if len(context) > 0 else ': ') + target + ' := by\n  sorry',
                load_statement=(('∀ ' + '\n'.join(context) + '\n, ') if len(context) > 0 else '') + target,
                intros=[v[0] for v in variables],
                header=result.header,
                early_stop=True,
                tag=str(key)+'/prove'
            )
        
        is_trivial = proofs[-1] is not None
        logger.info(f'async_worker({key}): is_trivial={is_trivial}')
        result.metainfo['is_trivial'] = is_trivial
        logger.info(f'async_worker({key}): finished.')
    except Exception as e:
        logger.debug(f'async_worker({key}): failed due to traceback {traceback.format_exc()}')
        logger.warning(f'async_worker({key}): failed due to exception {repr(e)}')
    finally:
        result.metainfo = json.dumps(result.metainfo)
        finished_list[key] = result
        server.tag = ''
        available_servers.insert(0, server)
        available_agents.insert(0, agent)

def main(
    load_path: str,
    log_root: str,
    project_root: str='/home/ma-user/workspace/fps_pantograph/formal_problem_solving/data/MiniF2F',
    num_concurrency: int=12,
    debug: bool=False,
):
    saved_args = {**locals()}
    is_code = ('mustard' in load_path.lower())
    
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_prefix = 'fpg_evaluate_trivial'+'.'

    os.makedirs(log_root, exist_ok=True)
    if not debug:
        logger.remove()
        logger.add(sys.stdout, level='INFO')    # filter=lambda record: record["name"] != "agent.solution_autoformalization"
    logger.add(osp.join(log_root, log_prefix+now+'.log'), level='DEBUG')
    logger.info(f'Evaluating triviality with hyperparams: {saved_args}')
    if is_code:
        logger.warning('Evaluating MUSTARD in code mode.')
    
    with open(load_path, 'rb') as f:
        load_results = pickle.load(f)
        try:
            conditions_sampled, finished_list = load_results
        except:
            finished_list = load_results
            conditions_sampled = list(range(len(finished_list)))
            for i in conditions_sampled:
                assert isinstance(finished_list[i], object)
    
    for d in finished_list:
        d.metainfo = d.metainfo if isinstance(d.metainfo, Dict) else json.loads(d.metainfo)
    
    available_servers = [
        PersistentServer(
            max_count=64,
            is_state_based=not is_code,
            tag='',
            _sync_init=False,
            imports=["Mathlib", "Aesop"],
            project_path=project_root,
            core_options=CORE_OPTIONS,
            timeout=300,
        ) for _ in range(num_concurrency)
    ]
    
    available_agents = [
        MultipleProvers(clients=[], models=[]) for _ in range(num_concurrency)
    ]
    for a in available_agents:
        a.provers = [AesopProofGenerationAgent()]

    tasks = []
    for (i, sample) in enumerate(finished_list):
        if (len(sample.formal_statement or '') > 0):    # is submitted and proven
            tasks.append(i)
    
    logger.info(f'Evaluating {len(tasks)} samples.')

    async def _async_main():
        pending_tasks = set()
        for i, idx in enumerate(tasks):
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
                        result=finished_list[idx],
                        key=idx,
                        available_agents=available_agents,
                        available_servers=available_servers,
                        finished_list=finished_list,
                        is_code=is_code
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
                pickle.dump((conditions_sampled, finished_list), f)
        except Exception as e:
            logger.error(traceback.format_exc())
            import pdb; pdb.set_trace()

if __name__ == '__main__':
    fire.Fire(main)
