# Should be used after `numina-lean.parse+deductive_transform.py`
import sys
import os
import os.path as osp
from io import BufferedWriter
import json
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

import dacite
import fire
import aiofiles
from aiofiles.threadpool.text import AsyncTextIOWrapper
import pyarrow
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm
import msgspec

from common.constants import OPEN_HEADER, CORE_OPTIONS, MVAR_PATTERN, BANNED_TOKENS
from common.utils import remove_comments, normalize_spaces, remove_spaces, normalize_draft, remove_min_whitespace, chunk_list, replace_sorry, replace_calc, remove_multiline_comments, decompose_statement
from common.pantograph.dataclasses import TacticInvocation, Goal, GoalState, ProblemGenerationStep, ProblemGenerationProcess, TacticDraft, Variable
from common.pantograph.server import Server, PersistentServer, TacticFailure, ServerError
from agent.problem_generation import AutoregressiveProblemGenerationAgent

N_CONCURRENCY_PER_WORKER = 8

async def async_worker(
    result: ProblemGenerationProcess,
    base_cnt: int,
    idx: int,
    available_servers: List[PersistentServer],
    results: List,
) -> None:
    server = available_servers.pop()
    server.tag = str(idx)
    try:
        agent = AutoregressiveProblemGenerationAgent(0)
        time_start = time.time()
        try:
            if len(result.dependencies) == 0:
                states: List[GoalState] = []
                cur_problem_state = await server.load_statement_async('False')
                assert result.steps[-1].is_submitting
                
                for cur_step in result.steps[:-1]:
                    new_problem_state = await server.goal_tactic_async(cur_problem_state, 0, cur_step.step)
                    assert len(new_problem_state.goals) == 1 and new_problem_state.goals[0].target == 'False', str(new_problem_state)
                    cur_step.new_contexts = [
                        v for v in new_problem_state.goals[0].variables if
                            v.raw_name not in {vv.raw_name for vv in cur_problem_state.goals[0].variables}
                    ]
                    states.append(new_problem_state)
                
                is_analyzed = await agent.analyze_async(
                    result=result,
                    states=states,
                    server=server,
                    tag=f'{base_cnt+idx}',
                    reassemble_trajectory=False,
                )
            else:
                is_analyzed = True
            
            if is_analyzed:
                # Reassemble trajectory
                is_reassembled = await agent.reassemble_trajectory_async(
                    result=result,
                    server=server,
                    tag=f'{base_cnt+idx}',
                )
                result.metainfo = json.dumps(result.metainfo | {'time_consumption:reassemble_trajectory_async': time.time() - time_start})
            
            logger.debug(f'async_worker({base_cnt+idx}): succeeded.')
        except Exception as e:
            logger.warning(f'async_worker({base_cnt+idx}): Failed: {traceback.format_exc()}')
            # import ipdb; ipdb.set_trace()
            logger.debug(f'async_worker({base_cnt+idx}): Failed, traceback: {[traceback.format_exc()]}')
        
    finally:
        results[idx] = result
        server.tag = ''
        available_servers.insert(0, server)

def worker(args: Tuple) -> int:
    working_root, base_cnt = args
    
    if not osp.exists(osp.join(working_root, f'done_chunk_{base_cnt}.pkl')):
        logger.opt(colors=True).info(f'<green>worker({base_cnt}): done pkl does not exist, exiting....</green>')
        return
    if osp.exists(osp.join(working_root, f'reassembled_fixed_chunk_{base_cnt}.pkl')):
        logger.opt(colors=True).info(f'<green>worker({base_cnt}): done_v2 pkl already exists, exiting....</green>')
        return

    with open(osp.join(working_root, f'done_chunk_{base_cnt}.pkl'), 'rb') as f:
        data_to_process: List[ProblemGenerationProcess] = pickle.load(f)
    
    finished_list = [None for _ in range(len(data_to_process))]

    available_servers = [
        PersistentServer(
            max_count=32,
            is_state_based=True,
            tag='',
            _sync_init=False,
            imports=["Mathlib", "Aesop"],
            project_path='/home/ma-user/workspace/formal_problem_generation/formal_problem_generation/data/MiniF2F',
            core_options=CORE_OPTIONS,
            timeout=300,
        ) for _ in range(N_CONCURRENCY_PER_WORKER)
    ]

    for result in data_to_process:
        result.metainfo = json.loads(result.metainfo)
        if 'original_trajectory' in result.metainfo.keys():
            result.trajectory = [
                ([dacite.from_dict(Variable, v) for v in S], i_s) for (S, i_s) in result.metainfo['original_trajectory']
            ]
            result.metainfo.pop('original_trajectory')

    tasks = [
        (i, d) for (i, d) in enumerate(data_to_process) if 'falsified_model' not in d.metainfo.keys() and len(d.steps) > 0
    ]
    logger.info(f'worker({base_cnt}): Initialized, loaded {len(data_to_process)} samples, processing {len(tasks)} invocation-parsed samples.')

    async def _async_main():
        pending_tasks = set()
        for i, d in tasks:
            if len(pending_tasks) >= N_CONCURRENCY_PER_WORKER:
                done_tasks, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done_tasks:
                    if task.exception() is not None:
                        logger.error(f"worker({base_cnt}): Exception occurred: {task.exception()}")
                        for pending_task in pending_tasks:
                            pending_task.cancel()
                        # import ipdb; ipdb.set_trace()
            pending_tasks.add(
                asyncio.create_task(
                    async_worker(
                        result=d,
                        base_cnt=base_cnt,
                        idx=i,
                        available_servers=available_servers,
                        results=finished_list,
                    )
                )
            )
        if len(pending_tasks) > 0:
            await asyncio.wait(pending_tasks)
        await logger.complete()

    try:
        asyncio.get_event_loop().run_until_complete(_async_main())
        logger.opt(colors=True).info(f'<cyan>worker({base_cnt}): All finished.</cyan>')
    except Exception as e:
        logger.error(f"worker({base_cnt}): Failed due to Exception {e}\n{traceback.format_exc()}")
    
    with open(osp.join(working_root, f'reassembled_fixed_chunk_{base_cnt}.pkl'), 'wb') as f:
        pickle.dump(finished_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f'worker({base_cnt}): Exiting.')
    return base_cnt

def load_and_split(working_root: str, reverse_order: bool):
    all_splits = set(
            [
                int(n[len('done_chunk_'):-len('.pkl')]) for n in os.listdir(working_root) if n.startswith('done_chunk_') and n.endswith('.pkl')
            ]
        )
    done_splits = set(
            [
                int(n[len('reassembled_fixed_chunk_'):-len('.pkl')]) for n in os.listdir(working_root) if n.startswith('reassembled_fixed_chunk_') and n.endswith('.pkl')
            ]
        )
    assert done_splits.issubset(all_splits)
    remaining = sorted(all_splits - done_splits)
    if reverse_order:
        remaining.reverse()
    logger.info(f'load_and_split(): Loaded {len(remaining)} remaining splits.')
    return remaining

def main(
    working_root: str='/home/ma-user/workspace/formal_problem_generation/formal_problem_generation/data/MiniF2F/Numina-Lean',
    use_mp: bool=True,
    n_concurrency: int=8,
    reverse_order: bool=False,
) -> None:
    os.makedirs(osp.join(working_root, 'lean'), exist_ok=True)
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    if use_mp:
        logger.remove()
        logger.add(sys.stdout, level='INFO')
        logger.add(osp.join(working_root, now+'.log'), level='DEBUG')
    
    splits = load_and_split(working_root, reverse_order)
    try:
        if use_mp:
            futures = []
            with mp.Pool(processes=n_concurrency, maxtasksperchild=1) as pool:
                for base_cnt in splits:
                    futures.append((base_cnt, pool.apply_async(worker,
                        [(working_root, base_cnt)]
                    )))
                pool.close()
                pool.join()
            for f in futures:
                try:
                    f[1].get(timeout=60)
                except Exception as e:
                    logger.error(f"main(): Task {f[0]} failed with error: {repr(e)}")
        else:
            list(map(
                    worker, [(
                            working_root, base_cnt
                    ) for base_cnt in splits]
                ))
    except Exception as e:
        traceback.print_exc()
        import ipdb; ipdb.set_trace()
        _ = futures.get()
        logger.error(f'main(): Exception occurred: {e}')
    
    import pdb; pdb.set_trace()
    logger.opt(colors=True).info(f'<green>main(): All finished.</green>')


if __name__ == '__main__':
    fire.Fire(main)
