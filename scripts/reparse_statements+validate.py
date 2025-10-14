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
from agent.problem_generation import ProblemGenerationProcess

async def async_worker(
    result: ProblemGenerationProcess,
    idx: int,
    available_servers: List[PersistentServer],
    available_parsers: List[PersistentParsingServer],
    results: List,
) -> None:
    server = available_servers.pop()
    parser = available_parsers.pop()
    server.tag = str(idx)
    parser.tag = str(idx)
    try:
        # I. Reproduce steps on PersistentParsingServer
        try:
            # Parse `target` on PersistentParsingServer
            result.metainfo = json.loads(result.metainfo)
            result.metainfo['is_statement_validated'] = False
            result.metainfo['is_solution_validated'] = False
            steps = result.steps
            cur_problem_state = await parser.load_statement_async('False')
            for i_step, cur_step in enumerate(steps):
                if cur_step.is_submitting:
                    assert i_step == len(steps) - 1, f'(i_step == {i_step}) != (len(steps) - 1 == {len(steps) - 1})'
                    break
                new_problem_state = await parser.goal_tactic_async(cur_problem_state, 0, cur_step.step)
                assert len(new_problem_state.goals) == 1 and new_problem_state.goals[0].target == '(False : Prop)', str(new_problem_state)
                cur_problem_state = new_problem_state

            submission_name = steps[-1].step_code[len('submit_answer '):]
            submission_fvar = [v for v in cur_problem_state.goals[0].variables if v.name == submission_name]
            assert len(submission_fvar) == 1, f'submission_name={submission_name}, cur_problem_state={[str(cur_problem_state)]}'
            submission_fvar = submission_fvar[0]

            submission_fvar_pp = [v for v in result.trajectory[-1][0] if v.name == submission_name]
            assert len(submission_fvar_pp) == 1, f'submission_name={submission_name}, new_context={[v.name for v in steps[-1].new_contexts]}'
            submission_fvar_pp = submission_fvar_pp[0]
            # breakpoint()

            # Construct statement and proof
            problem_hypotheses = []
            for s in steps:
                if s.is_introducing:
                    step_code = s.step_code
                    assert step_code.startswith('have ') and step_code.endswith(' := sorry')
                    problem_hypotheses.append('(' + step_code[len('have '):-len(' := sorry')] + ')')
            
            formal_solution = '\n\n'.join([s.step for s in steps if s.is_deducing] + [steps[-1].step.replace('submit_answer ', 'exact ')])
            result.header = ''
            result.formal_statement = 'example\n' + '\n'.join(problem_hypotheses) + '\n: ' + submission_fvar.t + '\n:= sorry'
            result.formal_solution_draft = formal_solution
            
            assert '✝' not in submission_fvar.t, 'Submitted "✝"'
        except Exception as e:
            if isinstance(e, AssertionError) and 'Submitted "✝"' in str(e):
                logger.debug(f'async_worker({idx}): Initialization failed due to {repr(e)}')
            else:
                logger.warning(f'async_worker({idx}): Initialization failed due to {repr(e)}')
            return False
        
        # Validate statement and proof
        try:
            formal_statement = (('∀\n' + '\n'.join(problem_hypotheses) + '\n, ') if len(problem_hypotheses) > 0 else '') + submission_fvar.t
            try:
                init_validation_state = await server.load_statement_async(formal_statement, intros=[(v.name if '✝' not in v.name else '_') for s in steps if s.is_introducing for v in s.new_contexts])
            except TacticFailure:
                open_set = set()
                open_scoped_set = set()
                option_set = set()
                for s in steps:
                    if s.is_introducing:
                        for l in s.header.splitlines():
                            l = l.strip()
                            if l.endswith(' in'):
                                l = l[:-len('in')].strip()
                            if l.startswith('open scoped'):
                                for elem in l[len('open scoped'):].strip().split():
                                    open_scoped_set.add(elem)
                            elif l.startswith('open'):
                                for elem in l[len('open'):].strip().split():
                                    open_set.add(elem)
                            elif l.startswith('set_option'):
                                option_set.add(l[len('set_option'):].strip())
                result.header = \
                    (('open scoped ' + ' '.join(open_scoped_set) + '\n') if len(open_scoped_set) > 0 else '') + \
                    (('open ' + ' '.join(open_set) + '\n') if len(open_set) > 0 else '') + \
                    '\n'.join(['set_option ' + t for t in option_set])
                init_validation_state = await server.load_statement_async(formal_statement, intros=[(v.name if '✝' not in v.name else '_') for s in steps if s.is_introducing for v in s.new_contexts], header=result.header)
            result.metainfo['is_statement_validated'] = True
        except Exception as e:
            if isinstance(e, TacticFailure) and 'unknown identifier' in str(e): # Bug caused by `extract_goal`
                logger.debug(f'async_worker({idx}): Statement validation failed due to {repr(e)}: {formal_statement}, submission_fvar_pp={submission_fvar_pp}')
            else:
                logger.warning(f'async_worker({idx}): Statement validation failed due to {repr(e)}: {formal_statement}, submission_fvar_pp={submission_fvar_pp}')
            return
        try:
            final_validation_state = await server.goal_tactic_async(init_validation_state, 0, 'have h_submission := {\n' + formal_solution + '\n}\nexact h_submission')
            assert final_validation_state.is_solved, str(final_validation_state)
            result.metainfo['is_solution_validated'] = True
        except Exception as e:
            logger.debug(f'async_worker({idx}): Solution validation failed due to {repr(e)}: {formal_solution}')
            return
    finally:
        result.metainfo = json.dumps(result.metainfo)
        results[idx] = result
        server.tag = ''
        available_servers.insert(0, server)
        parser.tag = ''
        available_parsers.insert(0, parser)

def main(
    load_path: str='/home/ma-user/workspace/formal_problem_generation/formal_problem_generation/data/MiniF2F/Goedel-Prover-V2-8B.Numina-Lean.problem_generator.nopack/failed.pkl',
    save_path: str='/home/ma-user/workspace/formal_problem_generation/formal_problem_generation/data/MiniF2F/Goedel-Prover-V2-8B.Numina-Lean.problem_generator.nopack/failed-revalidated.0830.pkl',
    n_concurrency: int=8
):
    with open(load_path, 'rb') as f:
        data_to_process = pickle.load(f)
    
    logger.remove()
    logger.add(sys.stdout, level='INFO')
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
        ) for _ in range(n_concurrency)
    ]
    available_parsers = [
        PersistentParsingServer(
            max_count=32,
            is_state_based=True,
            tag='',
            _sync_init=False,
            imports=["Mathlib", "Aesop"],
            project_path='/home/ma-user/workspace/formal_problem_generation/formal_problem_generation/data/MiniF2F',
            core_options=CORE_OPTIONS,
            timeout=300
        ) for _ in range(n_concurrency)
    ]

    tasks = [
        (i, d) for (i, d) in enumerate(data_to_process)
    ]
    logger.info(f'Initialized, loaded {len(data_to_process)} samples, processing {len(tasks)} invocation-parsed samples.')

    async def _async_main():
        pending_tasks = set()
        for i, d in tasks:
            if len(pending_tasks) >= n_concurrency:
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
                        available_servers=available_servers,
                        available_parsers=available_parsers,
                        results=finished_list,
                    )
                )
            )
        if len(pending_tasks) > 0:
            await asyncio.wait(pending_tasks)
        await logger.complete()

    try:
        asyncio.get_event_loop().run_until_complete(_async_main())
        logger.opt(colors=True).info(f'<cyan>main(): All finished.</cyan>')
    except Exception as e:
        logger.error(f"main(): Failed due to Exception {e}\n{traceback.format_exc()}")
    
    with open(save_path, 'wb') as f:
        pickle.dump(finished_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f'main(): Exiting.')

if __name__ == '__main__':
    fire.Fire(main)
