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
from common.utils import remove_comments, normalize_spaces, remove_spaces, normalize_draft, remove_min_whitespace, chunk_list, replace_sorry, replace_calc, remove_multiline_comments
from common.pantograph.dataclasses import TacticInvocation, Goal, GoalState, ProblemGenerationStep, ProblemGenerationProcess, TacticDraft
from common.pantograph.server import Server, PersistentServer, TacticFailure, ServerError
from agent.problem_generation import AutoregressiveProblemGenerationAgent

N_CONCURRENCY_PER_WORKER = 8

superscript_to_digit = {
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
}

subscript_to_digit = {
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
}

digit_to_superscript = {v: k for k, v in superscript_to_digit.items()}
digit_to_subscript = {v: k for k, v in subscript_to_digit.items()}

allowed_prefices = ['h', 'h_']

bracket_pairings = {
    '(' : ')',
    '[' : ']',
    '{' : '}',
    '⦃' : '⦄'
}

def parse_variables(s : str) -> Tuple[str, str]:
    base = 0
    variables = []
    target = None
    while base < len(s):
        if s[base] in ['(', '[', '{', '⦃']:
            bracket_type = s[base]
            bracket_pairing = bracket_pairings[bracket_type]
        
            stack_cnt = 0
            start_end_positions = []

            for i, char in enumerate(s[base:]):
                if char == bracket_type:
                    if stack_cnt == 0:
                        start_position = i
                    stack_cnt += 1
                elif char == bracket_pairing:
                    if stack_cnt > 0:
                        stack_cnt -= 1
                        if stack_cnt == 0:
                            end_position = i
                            start_end_positions.append((start_position, end_position))
                            break
            
            start, end = start_end_positions[0]
            variables.append(s[base+start:base+end+1])
            base += i
        else:
            if s[base] == ',':
                target = s[base+1:].strip()
                break
            base += 1
    
    return variables, target

def generate_submission_name(name_list: List[str]) -> str:
    # Parse names
    numbers_existing = C.defaultdict(list)
    for n in name_list:
        for p in allowed_prefices:
            if n.startswith(p):
                num_str = n[len(p):]
                if num_str == '':
                    numbers_existing[-1].append((p, 'text'))
                elif all(c in superscript_to_digit for c in num_str):
                    num = int(''.join(superscript_to_digit[c] for c in num_str))
                    numbers_existing[num].append((p, 'sup'))
                elif all(c in subscript_to_digit for c in num_str):
                    num = int(''.join(subscript_to_digit[c] for c in num_str))
                    numbers_existing[num].append((p, 'sub'))
                elif all(c.isascii() and c.isdigit() for c in num_str):
                    num = int(num_str)
                    numbers_existing[num].append((p, 'text'))
                    
    if not numbers_existing:
        numbers_existing = C.defaultdict(list, {
            -1: [('h', 'text')]
        })
    # Generate new name
    max_number = sorted(numbers_existing.keys())[-1]
    number_chosen = max_number + 1
    prefix, format_type = random.choice(numbers_existing[max_number])
    
    if number_chosen == 0:
        formatted_num = ''
    else:
        num_str = str(number_chosen)
        if format_type == 'sup':
            formatted_num = ''.join(digit_to_superscript[c] for c in num_str)
        elif format_type == 'sub':
            formatted_num = ''.join(digit_to_subscript[c] for c in num_str)
        else:  # text
            formatted_num = num_str
    new_name = f"{prefix}{formatted_num}"
    logger.debug(f'numbers_existing={numbers_existing}, max_number={number_chosen}, new_name={new_name}')
    return new_name

async def async_worker(
    datapoint: Dict,
    base_cnt: int,
    idx: int,
    available_servers: List[PersistentServer],
    results: List,
) -> None:
    server = available_servers.pop()
    server.tag = str(idx)
    try:
        # I. Load parse results
        
        # datapoint['parse_result'] = {
        #     'import_list': import_list,
        #     'open_scoped_list': open_scoped_list,
        #     'open_list': open_list,
        #     'option_list': option_list,
        #     'units': [u.serialize() for u in units]
        # }
        import_list = datapoint['parse_result']['import_list']
        open_scoped_list = datapoint['parse_result']['open_scoped_list']
        open_list = datapoint['parse_result']['open_list']
        option_list = datapoint['parse_result']['option_list']
        units = datapoint['parse_result']['units']
        
        all_transformed_units = [i_u for i_u, u in enumerate(units) if 'deductive_steps' in units[i_u].keys()]
        remaining_units = [i_u for i_u in all_transformed_units if 'generation_process' not in units[i_u].keys()]
        logger.debug(f'async_worker({base_cnt+idx}): {len(remaining_units)}/{len(all_transformed_units)} units to reassemble.')
        if len(remaining_units) == 0:
            return

        tactic_header = ''
        load_header = ''
        if len(open_scoped_list):
            tactic_header += 'open scoped ' + ' '.join(t for t in open_scoped_list) + ' in\n'
            load_header += 'open scoped ' + ' '.join(t for t in open_scoped_list) + '\n'
        if len(open_list):
            tactic_header += 'open ' + ' '.join(t for t in open_list) + ' in\n'
            load_header += 'open ' + ' '.join(t for t in open_list) + '\n'
        if len(option_list):
            tactic_header += '\n'.join('set_option ' + t + ' in' for t in option_list) + '\n'
            load_header += '\n'.join('set_option ' + t for t in option_list) + '\n'
        
        # II. Reassemble trajectories
        agent = AutoregressiveProblemGenerationAgent(0)
        for i_p, i_u in enumerate(remaining_units):
            u = units[i_u]
            time_start = time.time()
            # units[i_u] |= {
            #     'deductive_steps' : deductive_steps,
            #     'deductive_states' : [[g.serialize()] for s in states for g in s],
            #     'formal_statement' : formal_statement,
            #     'intros' : intros,
            #     'load_header' : load_header,
            #     'whole_proof' : whole_proof
            # }
            try:
                deductive_steps: List[Tuple[str, str]] = u['deductive_steps']
                deductive_states: List[List[Dict]] = u['deductive_states']
                if len(deductive_states[-1]) != 0:  #* Caused by a bug in `numina-lean.deductive_transform-decompose.py` (fixed in `e657161`)
                    for i in reversed(list(range(1, len(deductive_states)))):
                        if deductive_states[i] == deductive_states[i-1]:
                            break
                    if deductive_states[i] == deductive_states[i-1]:
                        logger.debug(f'async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): Removing {i}-th state (duplicated)')
                        deductive_states.pop(i)
                    deductive_states.append([])
                if len(deductive_steps) + 1 != len(deductive_states):
                    logger.warning(f'async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): len(deductive_steps) + 1 != len(deductive_states): {len(deductive_steps)}, {len(deductive_states)}')
                
                states: List[GoalState] = []
                steps: List[ProblemGenerationStep] = []
                cur_problem_state = await server.load_statement_async('False')
                states.append(cur_problem_state)
                
                # Execute introducing steps
                assert len(deductive_states[0]) == 1
                
                init_parsed_goal = dacite.from_dict(Goal, deductive_states[0][0])
                var_type_dict = {
                    v.name : v.t for v in init_parsed_goal.variables
                }
                var_value_dict = {
                    v.name : v.v for v in init_parsed_goal.variables
                }
                
                # Break from formal statement
                formal_statement = u['formal_statement']
                variables = []
                if formal_statement.startswith('∀ '):
                    context, target = parse_variables(formal_statement[len('∀ '):])
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
                                    logger.critical(f"async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): '✝' in name: {[formal_statement]}")
                                    name = '_'
                                variables.append((name.strip(), var_type or var_type_dict[name.strip()]))
                else:
                    target = formal_statement.strip()
                
                for (name, var_type) in variables:
                    # name = v.name
                    # decl = v.t
                    # if '✝' in v.name:
                    #     assert v.name.replace('✝', '_') not in str(u['deductive_states'])
                    #     name = v.name.replace('✝', '_')
                    # if decl.startswith('Type u_'):
                    #     decl = 'Type*'
                    # elif decl.startswith('Sort u*'):
                    #     decl = 'Sort*'
                    cur_step = ProblemGenerationStep(   # ProblemGenerationStepCategory.Introduce
                        step_draft=normalize_spaces(f'have {name.strip()} : {var_type.strip()} := sorry'), # if var_value_dict[name] is None else f'let {name} : {var_type} := {var_value_dict[name]}'
                        proof=None,
                        new_contexts=[]
                    )
                    # TODO: rename_i 到底会不会改变fvarid?
                    try:
                        new_problem_state = await server.goal_tactic_async(cur_problem_state, 0, cur_step.step)
                    except (TacticFailure, ServerError):
                        cur_step.step_draft = tactic_header + cur_step.step_draft
                        new_problem_state = await server.goal_tactic_async(cur_problem_state, 0, cur_step.step)
                    assert len(new_problem_state.goals) == 1 and new_problem_state.goals[0].target == 'False', str(new_problem_state)
                    idents = set(cur_step.step.split())
                    for banned_token in BANNED_TOKENS[1:]:
                        if banned_token in idents:
                            if any(v.name == banned_token for v in cur_problem_state.goals[0].variables):
                                logger.warning(f'async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): Banned token "{banned_token}" in step "{[cur_step.step]}", but is also in context {[str(cur_problem_state)]}')
                            else:
                                raise RuntimeError(f'Banned token "{banned_token}" in step "{[cur_step.step]}"')
                    
                    cur_step.new_contexts = [
                        v for v in new_problem_state.goals[0].variables if
                            v.raw_name not in {vv.raw_name for vv in cur_problem_state.goals[0].variables}  # 缺陷: 存在Bug，有时Tactic application会导致无关的fvar被重新assign，且该tactic和原tactic无依赖关系
                            # v not in cur_problem_state.goals[0].variables   # 缺陷: 可能只是改名了... —— 没事，正好是rename_i的需求！
                            # (v not in cur_problem_state.goals[0].variables) and (v.raw_name not in {vv.raw_name for vv in cur_problem_state.goals[0].variables})  # 缺陷: 对rename_i不友好
                    ]
                    if len(cur_step.new_contexts) != 1:
                        logger.debug(f'async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): Introducing step potentially leading name change: {str(cur_step)}, {cur_step.new_contexts}')
                    
                    states.append(new_problem_state)
                    steps.append(cur_step)
                    cur_problem_state = new_problem_state
                
                if init_parsed_goal.variables != cur_problem_state.goals[0].variables:
                    logger.debug(f'async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): init_parsed_goal.variables != cur_problem_state.goals[0].variables: {[str(init_parsed_goal), str(cur_problem_state.goals[0])]}')
                
                # Execute deriving steps
                for ((step_header, step_code), next_parsed_state) in zip(deductive_steps[:-1], deductive_states[1:-1]):
                    assert len(next_parsed_state) == 1
                    next_parsed_goal = dacite.from_dict(Goal, next_parsed_state[0])
                    cur_step = ProblemGenerationStep(   # ProblemGenerationStepCategory.Derive
                        step_draft=normalize_spaces(step_header + step_code),
                        proof=[],
                        new_contexts=[]
                    )
                    
                    new_problem_state = await server.goal_tactic_async(cur_problem_state, 0, cur_step.step)
                    assert len(new_problem_state.goals) == 1 and new_problem_state.goals[0].target == 'False', str(new_problem_state)
                    idents = set(cur_step.step.split())
                    for banned_token in BANNED_TOKENS:
                        if banned_token in idents:
                            if any(v.name == banned_token for v in cur_problem_state.goals[0].variables):
                                logger.warning(f'async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): Banned token "{banned_token}" in step "{[cur_step.step]}", but is also in context {[str(cur_problem_state)]}')
                            else:
                                raise RuntimeError(f'Banned token "{banned_token}" in step "{[cur_step.step]}"')

                    cur_step.new_contexts = [
                        v for v in new_problem_state.goals[0].variables if
                            v.raw_name not in {vv.raw_name for vv in cur_problem_state.goals[0].variables}
                    ]
                    if len(cur_step.new_contexts) == 0:
                        logger.debug(f'async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): Unused step: {str(cur_step)}')
                    
                    states.append(new_problem_state)
                    steps.append(cur_step)
                    cur_problem_state = new_problem_state
                
                    if next_parsed_goal.variables != cur_problem_state.goals[0].variables:
                        logger.debug(f'async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): next_parsed_goal.variables != cur_problem_state.goals[0].variables: {[str(next_parsed_goal), str(cur_problem_state.goals[0])]}')
                
                # Execute submitting step
                step_code = remove_comments(deductive_steps[-1][-1]).strip()
                assert step_code.startswith('exact '), step_code
                submission_name = step_code[len('exact '):]
                
                if ' ' in submission_name or '.' in submission_name:
                    new_name = generate_submission_name([v.name for v in cur_problem_state.goals[0].variables if v.name is not None])
                    cur_step = ProblemGenerationStep(   # ProblemGenerationStepCategory.Derive
                        step_draft=normalize_spaces(f'have {new_name.strip()} : {init_parsed_goal.target.strip()} := {submission_name.strip()}'),
                        proof=[],
                        new_contexts=[]
                    )
                    submission_name = new_name
                    
                    try:
                        new_problem_state = await server.goal_tactic_async(cur_problem_state, 0, cur_step.step)
                    except (TacticFailure, ServerError):
                        cur_step.step_draft = tactic_header + cur_step.step_draft
                        new_problem_state = await server.goal_tactic_async(cur_problem_state, 0, cur_step.step)
                    assert len(new_problem_state.goals) == 1 and new_problem_state.goals[0].target == 'False', str(new_problem_state)
                    idents = set(cur_step.step.split())
                    for banned_token in BANNED_TOKENS:
                        if banned_token in idents:
                            if any(v.name == banned_token for v in cur_problem_state.goals[0].variables):
                                logger.warning(f'async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): Banned token "{banned_token}" in step "{[cur_step.step]}", but is also in context {[str(cur_problem_state)]}')
                            else:
                                raise RuntimeError(f'Banned token "{banned_token}" in step "{[cur_step.step]}"')
                    
                    cur_step.new_contexts = [
                        v for v in new_problem_state.goals[0].variables if
                            v.raw_name not in {vv.raw_name for vv in cur_problem_state.goals[0].variables}
                    ]
                    if len(cur_step.new_contexts) == 0:
                        logger.debug(f'async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): Unused step: {str(cur_step)}')
                    
                    states.append(new_problem_state)
                    steps.append(cur_step)
                    cur_problem_state = new_problem_state
                
                assert submission_name in [v.name for v in cur_problem_state.goals[0].variables], f'submission_name={submission_name}, cur_problem_state={cur_problem_state}'
                steps.append(
                    ProblemGenerationStep(   # ProblemGenerationStepCategory.Submit
                        step_draft=normalize_spaces(f'submit_answer {submission_name.strip()}'),
                        proof=None,
                        new_contexts=None
                    )
                )
                
                # Parsed trajectory
                result = ProblemGenerationProcess(
                    informal_problem='',
                    informal_answer='',
                    informal_solution='',
                    header=None,
                    formal_statement='',
                    formal_solution_draft='',
                    formal_proofs='',
                    steps=steps,
                    dependencies=[],
                    trajectory=[(S.goals[0].variables, i) for i, S in enumerate(states)],
                    metainfo=dict()
                )
                
                # Reassemble trajectory
                is_analyzed = await agent.analyze_async(
                    result=result,
                    states=states,
                    server=server,
                    tag=f'{base_cnt+idx}-{i_p}/{len(remaining_units)}',
                    reassemble_trajectory=True
                )
                result.metainfo = json.dumps(result.metainfo | {'time_consumption': time.time() - time_start})
                
                u['generation_process'] = result
                logger.debug(f'async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): succeeded.')
            except Exception as e:
                logger.warning(f'async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): Failed: {traceback.format_exc()}')
                # import ipdb; ipdb.set_trace()
                logger.debug(f'async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): Failed, traceback: {[traceback.format_exc()]}')
        
        logger.debug(f'async_worker({base_cnt+idx}): finished.')
    except Exception as e:
        logger.error(f'async_worker({base_cnt+idx}): Async worker failed, traceback: {[traceback.format_exc()]}')
        # import ipdb; ipdb.set_trace()
    finally:
        results[idx] = datapoint
        server.tag = ''
        available_servers.insert(0, server)

def worker(args: Tuple) -> int:
    working_root, base_cnt = args
    
    if not osp.exists(osp.join(working_root, f'done_v2_chunk_{base_cnt}.pkl')):
        logger.opt(colors=True).info(f'<green>worker({base_cnt}): done pkl does not exist, exiting....</green>')
        return
    if osp.exists(osp.join(working_root, f'reassembled_chunk_{base_cnt}.pkl')):
        logger.opt(colors=True).info(f'<green>worker({base_cnt}): done_v2 pkl already exists, exiting....</green>')
        return

    with open(osp.join(working_root, f'done_v2_chunk_{base_cnt}.pkl'), 'rb') as f:
        data_to_process = pickle.load(f)
    
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

    tasks = [
        (i, d) for (i, d) in enumerate(data_to_process) if d is not None and 'parse_result' in d.keys()
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
                        datapoint=d,
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
    
    with open(osp.join(working_root, f'reassembled_chunk_{base_cnt}.pkl'), 'wb') as f:
        pickle.dump(finished_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f'worker({base_cnt}): Exiting.')
    return base_cnt

def load_and_split(working_root: str, reverse_order: bool):
    all_splits = set(
            [
                int(n[len('done_v2_chunk_'):-len('.pkl')]) for n in os.listdir(working_root) if n.startswith('done_v2_chunk_') and n.endswith('.pkl')
            ]
        )
    done_splits = set(
            [
                int(n[len('reassembled_chunk_'):-len('.pkl')]) for n in os.listdir(working_root) if n.startswith('reassembled_chunk_') and n.endswith('.pkl')
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
