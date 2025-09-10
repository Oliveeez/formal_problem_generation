import sys
import os
import os.path as osp
import json
import collections as C
import itertools as I
import random
import pickle
from typing import List, Dict, Set
import asyncio
import regex as re
from datetime import datetime
import traceback

import fire
import aiofiles
from loguru import logger
from tqdm import tqdm

from common.utils import remove_comments, parse_idents
from common.constants import OPEN_HEADER, CORE_OPTIONS
from common.pantograph.dataclasses import TacticInvocation
from common.pantograph.server import Server

async def async_worker(
    d: Dict,
    idx: int,
    results: List,
    lean_file_root: str,
) -> None:
    try:
        server = await Server.create(
            imports=["Mathlib", "Aesop"],
            project_path='/home/ma-user/workspace/fps_pantograph/formal_problem_solving/data/MiniF2F',
            core_options=CORE_OPTIONS,
            timeout=300,
            start=True,
        )
        async with aiofiles.open(osp.join(lean_file_root, d['lean4_source_path']), 'r') as f:
            code = await f.read()

        units = await server.tactic_invocations_async(osp.join(lean_file_root, d['lean4_source_path']))
        assert len(units) >= 1 and 'error' not in str([x.messages for x in units]), str([x.messages for x in units])
        
        last_ptr = 0
        code_with_sorry = []
        parse_results = []
        for u in units:
            if len(u.invocations) > 0:
                # Add previous code
                code_with_sorry.append(code.encode()[last_ptr:u.i_begin].decode())
                last_ptr = u.i_end
                # Add theorem code and `sorry``
                theorem_and_proof = code.encode()[u.i_begin:u.i_end].decode()
                theorem, proof = theorem_and_proof.split(':=', maxsplit=1)
                theorem = theorem + ' := by\n  sorry'
                if proof.lstrip().startswith('by'):
                    proof = proof.lstrip()[len('by'):]
                code_with_sorry.append(theorem)
                parse_results.append(
                    dict(code=code.encode()[:u.i_begin].decode() + '\n' + theorem, proof=proof)
                )
        d |= {
            'code_with_sorry': '\n'.join(code_with_sorry),
            'parse_results': parse_results,
        }
        logger.info(f'worker({idx}): finished with {len(parse_results)} units')
    except Exception as e:
        logger.info(f'worker({idx}): Failed due to {repr(e)}:\n{traceback.format_exc()}')
        d |= {
            'exception': str(repr(e)), 
            'traceback': traceback.format_exc()
        }
    finally:
        results[idx] = d

def main(
    n_concurrency: int=128,
    load_path: str='/cache/data/fpg_informal_baselines/MUSTARDSauce.processed.jsonl',
    save_root: str='/cache/data/MUSTARDSauce_lean4_parsed',
    lean_file_root: str='/cache/data/MUSTARDSauce_lean4',
    reverse: bool=False,
) -> None:
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.remove()
    logger.add(sys.stdout, level='INFO')
    logger.add(osp.join(save_root, now+'.log'), level='DEBUG')
    
    # Load Data
    with open(load_path, 'r') as f:
        data = [json.loads(l) for l in f.readlines()]
    datapoints = [d for d in data if d['lean4_source_path'] is not None]
    logger.info(f'{len(datapoints)} to process in total.')
    
    finished = [None for _ in range(len(datapoints))]
    async def _async_main():
        pending_tasks: Set[asyncio.Task] = set()
        loop = enumerate(datapoints)
        if reverse:
            loop = reversed(list(loop))
        for i, d in tqdm(loop):
            if len(pending_tasks) >= n_concurrency:
                done_tasks, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done_tasks:
                    if task.exception() is not None:
                        logger.error(f"Exception occurred: {task.exception()} {task.get_stack()}")
                        for pending_task in pending_tasks:
                            pending_task.cancel()
                        return
            pending_tasks.add(
                asyncio.create_task(
                    async_worker(d, i, finished, lean_file_root)
                )
            )
        if len(pending_tasks) > 0:
            await asyncio.wait(pending_tasks)
        await logger.complete()
    
    try:
        asyncio.run(_async_main())
    finally:
        try:
            logger.info(f"Finished generation, saving at {osp.join(save_root, now+'.(pkl|jsonl)')}")
            with open(osp.join(save_root, now+'.pkl'), 'wb') as f:
                pickle.dump(finished, f)
        except Exception as e:
            logger.error(traceback.format_exc())
            import pdb; pdb.set_trace()


if __name__ == '__main__':
    fire.Fire(main)
