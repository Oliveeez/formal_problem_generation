
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
import pyarrow
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from common.utils import remove_comments, parse_idents
from common.constants import OPEN_HEADER, CORE_OPTIONS
from common.pantograph.dataclasses import TacticInvocation
from common.pantograph.server import Server
from common.pantograph.solving_server import PersistentBaseSolvingServer

async def worker(
    d: Dict,
    idx: int,
    results: List,
    save_root: str
) -> None:
    try:
        server = await Server.create(
            imports=["Mathlib", "Aesop"],
            project_path='/home/ma-user/workspace/fps_pantograph/formal_problem_solving/data/MiniF2F',
            core_options=CORE_OPTIONS,
            timeout=300,
            start=True,
        )
        p_raw = d['formal_code']
        p = remove_comments(p_raw).strip().replace('\nlemma ', '\ntheorem ').replace('\nexample ', '\ntheorem thm_example')
        start_pos = p.find('theorem')
        assert start_pos != -1
        intro, stmt = p[:start_pos], p[start_pos:]
        
        import_list = []
        open_scoped_list = []
        open_list = []
        option_list = []

        for l in intro.splitlines():
            if len(l.strip()) == 0:
                continue
            elif l.startswith('import '):
                import_list.append(l[len('import '):].strip())
            elif l.startswith('open scoped '):
                for t in l[len('open scoped '):].strip().split():
                    open_scoped_list.append(t)
            elif l.startswith('open '):
                for t in l[len('open '):].strip().split():
                    open_list.append(t)
            elif l.startswith('set_option '):
                option_list.append(l[len('set_option '):].strip())
            else:
                raise

        p_injected: List[str] = p_raw.splitlines()
        for (i, l) in reversed(list(enumerate(p_injected))):
            if l.startswith('import '):
                i += 1
                break
        p_injected = '\n'.join(p_injected[:i]) + '\n\n' + '\n'.join('set_option ' + t.replace('=', ' ') for t in CORE_OPTIONS) + '\n\n' + '\n'.join(p_injected[i:])

        async with aiofiles.open(osp.join(save_root, str(idx)+'.lean'), 'w') as f:
            await f.write(p_injected)

        units = await server.tactic_invocations_async(osp.join(save_root, str(idx)+'.lean'))
        assert len(units) >= 1 and 'error' not in str([x.messages for x in units]), str([x.messages for x in units])
        # units = [
        #     u for u in units if len(u.invocations or []) > 0
        # ]
        # assert len(units) == 1, str(len(units))
        
        results[idx] = {
            'import_list': import_list,
            'open_scoped_list': open_scoped_list,
            'open_list': open_list,
            'option_list': option_list,
            'units': [u.serialize() for u in units]
        }
        logger.debug(f'worker({idx}): finished.')
    except Exception as e:
        logger.debug(f'worker({idx}): Failed due to {repr(e)}:\n{traceback.format_exc()}')
        results[idx] = (repr(e), traceback.format_exc())

def main(
    n_concurrency: int=128,
    save_root: str='/home/ma-user/workspace/formal_problem_generation/formal_problem_generation/data/MiniF2F/Numina-Lean',
    reverse: bool=False,
) -> None:
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.remove()
    logger.add(sys.stdout, level='INFO')
    logger.add(osp.join(save_root, now+'.log'), level='DEBUG')
    
    # Load Data
    data = pq.read_table('/home/ma-user/workspace/formal_problem_generation/formal_problem_generation/train-00000-of-00001.parquet').to_pandas().to_dict(orient="records")
    for d in data:
        if isinstance(d['formal_ground_truth'], str) and len(d['formal_ground_truth']) == 0 or d['ground_truth_type'] != 'complete':
            d['formal_ground_truth'] = None
        if isinstance(d['formal_proof'], str) and len(d['formal_proof']) == 0:
            d['formal_proof'] = None
    logger.info(str(C.Counter(d['question_type'] for d in data)))
    logger.info(str(C.Counter(d['source'] for d in data)))
    logger.info(str(C.Counter(d['problem_type'] for d in data)))

    # Data to process
    datapoints = []
    for d in data:
        if d['formal_ground_truth'] is not None:
            datapoints.append({
                k : v for (k, v) in d.items() if k != 'formal_ground_truth' and k != 'formal_proof'
            } | {'formal_code' : d['formal_ground_truth']})
        if d['formal_proof'] is not None:
            datapoints.append({
                k : v for (k, v) in d.items() if k != 'formal_ground_truth' and k != 'formal_proof'
            } | {'formal_code' : d['formal_proof']})
    logger.info(f'{len(datapoints)}/{len(data)} to process in total.')
    
    import_cnt = C.Counter()
    open_cnt = C.Counter()
    open_scoped_cnt = C.Counter()
    option_cnt = C.Counter()

    ood_lines = C.Counter()
    parsed_datapoints = []
    failed_datapoints = []

    for d in datapoints:
        p_raw = d['formal_code']
        try:
            p = remove_comments(p_raw).strip().replace('\nlemma ', '\ntheorem ').replace('\nexample ', '\ntheorem thm_example')
            start_pos = p.find('theorem')
            assert start_pos != -1
            intro, stmt = p[:start_pos], p[start_pos:]
            for l in intro.splitlines():
                if len(l.strip()) == 0:
                    continue
                elif l.startswith('import '):
                    import_cnt[l[len('import '):].strip()] += 1
                elif l.startswith('open scoped '):
                    for t in l[len('open scoped '):].strip().split():
                        open_scoped_cnt[t] += 1
                elif l.startswith('open '):
                    for t in l[len('open '):].strip().split():
                        open_cnt[t] += 1
                elif l.startswith('set_option '):
                    option_cnt[l[len('set_option '):].strip()] += 1
                else:
                    raise
            parsed_datapoints.append(d)
        except:
            failed_datapoints.append(d)
            continue

    logger.info(f'Context Parsing: {len(parsed_datapoints)} successfully / {len(failed_datapoints)} failed')
    
    finished = [None for _ in range(len(parsed_datapoints))]
    working_root = osp.join(save_root, 'lean')
    os.makedirs(working_root, exist_ok=True)
    async def _async_main():
        pending_tasks: Set[asyncio.Task] = set()
        loop = enumerate(parsed_datapoints)
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
                    worker(d, i, finished, working_root)
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
