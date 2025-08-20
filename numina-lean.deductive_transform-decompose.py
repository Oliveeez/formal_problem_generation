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

import fire
import aiofiles
from aiofiles.threadpool.text import AsyncTextIOWrapper
import pyarrow
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from common.constants import OPEN_HEADER, CORE_OPTIONS, MVAR_PATTERN
from common.utils import remove_comments, normalize_spaces, remove_spaces, normalize_draft, remove_min_whitespace, chunk_list, replace_sorry, replace_calc, remove_multiline_comments
from common.pantograph.dataclasses import TacticInvocation, Goal, GoalState
from common.pantograph.server import Server, PersistentServer, TacticFailure, ServerError


N_CONCURRENCY_PER_WORKER = 8

def count_indent(s: str) -> int:
    count = 0
    for char in s:
        if char == ' ':
            count += 1
        else:
            break
    return count

def proof_decompose(formal_proof: str) -> list[str]:
    '''Decompose a formal solution draft into steps'''
    # Count the minimal indents of all tactics
    min_indents = float('inf')
    pure_proof_lines = replace_sorry(replace_calc(remove_comments(formal_proof))).split('\n')
    for l in pure_proof_lines:
        if l.strip() != '':
            min_indents = min(min_indents, count_indent(l))

    # Reset the minimal indents to zero
    levels = []
    raw_lines = []  # (n_indents, line)
    for l in replace_sorry(replace_calc(remove_multiline_comments(formal_proof))).rstrip().split('\n'):
        n_indent = count_indent(l)
        if n_indent < min_indents:
            assert len(remove_comments(l).strip()) == 0
        
        if len(remove_comments(l).strip()) == 0:
            level = float('inf')   # Comment
        else:
            level = n_indent - min(n_indent, min_indents)   # Tactic
        raw_lines.append(l[min(n_indent, min_indents):])
        levels.append(level)
    
    # print('\n'.join(raw_lines))
    is_first = True
    parse_result = []
    cur_block = []
    for (level, line) in zip(levels, raw_lines):
        # print(line)
        if len(line.strip()) == 0:
            continue
        if level != 0:
            cur_block.append(line)
        else:   # Root-level tactic
            if is_first:    # First tactic block: neglect and add
                is_first = False
                cur_block.append(line)
            else:   # Other tactic block: end and new
                parse_result.append('\n'.join(cur_block))
                # print('\n<begin>\n' + parse_result[-1], end='\n<end>\n')
                cur_block = [line]
    
    if len(cur_block) > 0:
        parse_result.append('\n'.join(cur_block))
        # print('\n<begin>\n' + parse_result[-1], end='\n<end>\n')
    
    return parse_result

def is_deductive(state_before: List[Goal], state_after: List[Goal]) -> bool:
    return len(state_before) == 1 and len(state_after) == 1 and state_before[0].target == state_after[0].target

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

def factory_nonhygienic_transformer(initial_goal_str: str) -> Callable[[str], str]:
    # Parse context
    context_str, target = initial_goal_str.split('⊢ ')
    lines = context_str.splitlines()
    context = []
    for l in lines:
        if not l.startswith(' '):   # Multi-line variable produced by pretty-printer
            context.append(l)
        else:
            context[-1] = context[-1] + '\n' + l
    
    # Parse var_names
    x_number_set = set()
    x_replacement_stack = []
    inst_replacement_stack = []
    for c in context:
        var_names, var_type = c.split(':', 1)
        for n in var_names.strip().split(' '):
            if '✝' in n:
                base, num_superscript = n.split('✝')
                num_digit = ''.join([superscript_to_digit[d] for d in num_superscript])
                if base == 'inst':
                    inst_replacement_stack.append((0 if len(num_digit) == 0 else int(num_digit), n))
                elif base == 'x':
                    x_replacement_stack.append((0 if len(num_digit) == 0 else int(num_digit), n))
                else:
                    logger.debug(f'Detected normal anonymous variable: ({n} : {var_type})')
                    continue
            elif n == 'x':
                x_number_set.add(0)
            elif n.startswith('x_'):
                digit: str = n[2:]
                if digit.isdigit():
                    x_number_set.add(int(digit))
        
    for i, (num_parsed, n) in enumerate(inst_replacement_stack):
        if i + num_parsed + 1 != len(inst_replacement_stack):
            logger.error(f'replacement_stack not sorted {inst_replacement_stack}')

    for i, (num_parsed, n) in enumerate(x_replacement_stack):
        if i + num_parsed + 1 != len(x_replacement_stack):
            logger.error(f'replacement_stack not sorted {x_replacement_stack}')

    def closure(s: str):
        for i, (_, n) in enumerate(inst_replacement_stack): # Assuming `inst` not in var names
            s = s.replace(n, 'inst' + ('' if i == 0 else f'_'+str(i)))
        
        i_x_key = 0
        for _, n in x_replacement_stack:
            while i_x_key in x_number_set:
                i_x_key += 1
            s = s.replace(n, 'x' + ('' if i_x_key == 0 else f'_'+str(i_x_key)))
        
        return s
                
    return closure

# ANONYMOUS_PATTERN = re.compile(r'✝([⁰¹²³⁴⁵⁶⁷⁸⁹]+)')
# superscript_to_digit = {
#     '⁰': '0',
#     '¹': '1',
#     '²': '2',
#     '³': '3',
#     '⁴': '4',
#     '⁵': '5',
#     '⁶': '6',
#     '⁷': '7',
#     '⁸': '8',
#     '⁹': '9'
# }

# # Regular expression to match "✝" followed by one or more superscript digits
# def superscript_to_digit_replacement(match):
#     """Helper function to perform the replacement."""
#     superscripts = match.group(1)
#     # Convert each superscript character to its corresponding digit
#     regular_digits = ''.join(superscript_to_digit[char] for char in superscripts)
#     return '_' + regular_digits

# def transform_string(input_string: str) -> str:
#     """
#     Transforms a string by replacing occurrences of "✝" followed by superscript digits
#     with an underscore followed by the corresponding regular digits.

#     Args:
#         input_string (str): The string to transform.

#     Returns:
#         str: The transformed string.
#     """
#     # Mapping of superscript characters to their corresponding regular digits
#     # Replace all occurrences of the pattern in the input string
#     transformed_string = ANONYMOUS_PATTERN.sub(superscript_to_digit_replacement, input_string)
#     return transformed_string

def match_wo_mvar(s_w_mvar: str, s_wo_mvar: str) -> str:
    s_w_mvar = normalize_spaces(s_w_mvar.replace('(', '').replace(')', ''))
    s_wo_mvar = normalize_spaces(s_wo_mvar.replace('(', '').replace(')', ''))
    parts = re.split(MVAR_PATTERN, s_w_mvar)
    pattern = '^' + '.*?'.join(re.escape(part) for part in parts) + '.*?$'
    match = re.match(pattern, s_wo_mvar)
    return bool(match)

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
            if s[base] == ':':
                target = s[base+1:]
                break
            base += 1
    
    return variables, target

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
        assert 'exception' not in datapoint.keys() and 'traceback' not in datapoint.keys()
        
        # I. Parse tactic invocation
        p_raw = datapoint['formal_code']

        import_list = datapoint['parse_result']['import_list']
        open_scoped_list = datapoint['parse_result']['open_scoped_list']
        open_list = datapoint['parse_result']['open_list']
        option_list = datapoint['parse_result']['option_list']
        units = datapoint['parse_result']['units']
        
        all_parsed_units = [i_u for i_u, u in enumerate(units) if len(u['invocations'] or []) > 0]
        remaining_units = [i_u for i_u in all_parsed_units if 'deductive_steps' not in units[i_u].keys()]
        logger.debug(f'async_worker({base_cnt+idx}): {len(remaining_units)}/{len(all_parsed_units)} units to transform')
        
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

        p_injected: List[str] = p_raw.splitlines()
        for (i, l) in reversed(list(enumerate(p_injected))):
            if l.startswith('import '):
                i += 1
                break
        p_injected = '\n'.join(p_injected[:i]) + '\n\n' + '\n'.join('set_option ' + t.replace('=', ' ') for t in CORE_OPTIONS) + '\n\n' + '\n'.join(p_injected[i:])

        # II. Deductive transform
        for i_p, i_u in enumerate(remaining_units):
            u = units[i_u]
            
            try:
                invocations = u['invocations']
                assert len(invocations[0]['before']) == 1, 'Initial state contains multiple goals'
                # nonhygienic_transformer = factory_nonhygienic_transformer(invocations[0]['before'][0])

                code_segment = remove_comments(p_injected.encode()[u['i_begin']:u['i_end']].decode())
                start_pos = None
                for start_pos in re.finditer(r':=\s*by', code_segment):
                    break
                assert start_pos is not None, '":= by" not found'
                statement_code, proof_code = code_segment[:start_pos.span(0)[0]], code_segment[start_pos.span(0)[1]:]

                # Preprocess steps (deprecated in Pantograph v0.3.5)
                # for ivc in invocations:
                #     ivc['tactic'] = ivc['tactic'].replace('native_decide', 'decide')
                # proof_code = proof_code.replace('native_decide', 'decide')

                # 1. Parse Context from statement code
                context, target = parse_variables(statement_code)
                assert target is not None, f'Target parsing failed: {statement_code}'
                
                # 2. Parse intros
                hypotheses = []
                intros = []
                for i_ctx, declaration in enumerate(context):
                    if declaration[0] == '[':
                        intros.append('_')
                        hypotheses.append(declaration)
                    else:
                        assert '✝' not in declaration, f'declaration: {declaration}'
                        try:
                            var_names, var_type = declaration[1:-1].split(':', 1)
                        except ValueError:
                            var_names = declaration[1:-1]
                        # var_names = [n if '✝' not in n else '_' for n in var_names.strip().split(' ')]
                        intros.extend(var_names.strip().split(' '))
                        hypotheses.append('(' + declaration[1:-1] + ')')    # Replace '{v : T}' into '(v : T)
                
                # 3. Load statement (before Pantograph v0.3.5)
                formal_statement = (('∀ ' + '\n'.join(hypotheses) + '\n, ') if len(hypotheses) > 0 else '') + target
                assert '⊢' not in formal_statement, '⊢ in formal_statement'
                try:
                    init_state = await server.load_statement_async(formal_statement, intros=intros, header=load_header)
                except Exception as e:
                    # Fall-back: parse from initial state
                    context_str, target = invocations[0]['before'][0].split('⊢ ')
                    lines = context_str.splitlines()
                    context = []
                    for l in lines:
                        if not l.startswith(' '):   # Multi-line variable produced by pretty-printer
                            context.append(l)
                        else:
                            context[-1] = context[-1] + '\n' + l
                    
                    # Parse intros
                    intros = []
                    for c in context:
                        var_names, var_type = c.split(' : ', 1)
                        var_names = [n if '✝' not in n else '_' for n in var_names.split(' ')]
                        intros.extend(var_names)
                    
                    # Load statement
                    hypotheses = ('∀ ' + '\n'.join('(' + c + ')' for c in context) + '\n, ') if len(context) > 0 else ''
                    formal_statement = hypotheses + target
                    assert '⊢' not in formal_statement
                    try:
                        init_state = await server.load_statement_async(formal_statement, intros=intros, header=load_header)
                    except Exception as e:
                        raise RuntimeError(e, formal_statement, intros, load_header)
                    
                # assert all(match_wo_mvar(nonhygienic_transformer(g_parsed), str(g_now)) for g_parsed, g_now in zip(invocations[0]['before'], init_state.goals)), 'initial state not equivalent w/ parse results'
                assert len(init_state.goals) == 1, 'Strange initial state: ' + str(init_state) #* Non-strict match

                # 3. Load statement (after Pantograph v0.3.5)
                # formal_statement = ('example\n' + '\n'.join(hypotheses) + '\n: ') + target + '\n:= sorry'
                # try:
                #     init_units = await server.load_sorry_async(tactic_header + formal_statement)
                #     assert all(m.severity != Severity.ERROR for u in init_units for m in u.messages), f'State initialization failed: {str([m for u in init_units for m in u.messages])}'
                #     init_state = init_units[-1].goal_state
                #     assert init_state is not None and len(init_state.goals) == 1, f'State initialization failed: {str(init_state)}' #* Non-strict match
                # except Exception as e:
                #     raise RuntimeError(*e.args, context, target, tactic_header)
                # assert all(match_wo_mvar(nonhygienic_transformer(g_parsed), str(g_now)) for g_parsed, g_now in zip(invocations[0]['before'], init_state.goals)), 'initial state not equivalent w/ parse results'
                
                # Start transforming
                raw_steps = proof_decompose(proof_code)
                
                states: List[List[Goal]] = [init_state.goals[:]]
                deductive_steps: List[Tuple[str, str]] = []
                cur_state = init_state

                while len(raw_steps) > 0:
                    # Execute cur_step
                    cur_step = raw_steps[0]
                    used_tactic_header = ''
                    try:
                        next_state = await server.goal_tactic_async(cur_state, 0, cur_step)
                    except (TacticFailure, ServerError):
                        used_tactic_header = tactic_header
                        next_state = await server.goal_tactic_async(cur_state, 0, tactic_header + cur_step)
                    
                    if next_state.is_solved:
                        if remove_comments(cur_step).strip().startswith('exact '):
                            # If (solved) and (the final step is `exact`): add cur_step and break
                            raw_steps = []
                            cur_state = next_state
                            states.append(cur_state.goals[:])
                            deductive_steps.append((used_tactic_header, cur_step))
                            logger.info(f"async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): Detected `exact` submission: {[remove_comments(cur_step).strip()]}")
                            break
                        else:
                            # If (solved) but (the final step is not `exact`): don't add cur_step, don't update state
                            raw_steps = [cur_step]
                            logger.info(f"async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): Detected non-`exact` submission: {[remove_comments(cur_step).strip()]}")
                            break   # If the final step is not `exact`, 1) do not add to `steps` - leave it for final submission; 2) do not update `cur_state`
                    else:
                        if not is_deductive(cur_state.goals, next_state.goals):
                            # If (not solved) but (not deductive): don't add cur_step, don't update state
                            break
                        else:
                            # If (not solved) and (is deductive): add cur_step and continue
                            raw_steps.pop(0)
                            cur_state = next_state
                            states.append(cur_state.goals[:])
                            deductive_steps.append((used_tactic_header, cur_step))

                # Remaining non-deductive steps
                if len(raw_steps) > 0:
                    proof_state = cur_state

                    submission_name = generate_submission_name([v.name for v in cur_state.goals[0].variables if v.name is not None])
                    have_step = f'have {submission_name}: {target} := by {{\n' + '\n'.join(raw_steps) + '\n}'
                    try:
                        proof_state = await server.goal_tactic_async(proof_state, 0, have_step)
                        assert (len(proof_state.goals) == 1 and proof_state.goals[0].target == cur_state.goals[0].target), f'`have {submission_name}` failed due to proof state: ' + str(proof_state)
                        deductive_steps.append(('', have_step))
                    except:
                        proof_state = await server.goal_tactic_async(proof_state, 0, tactic_header + have_step)
                        assert (len(proof_state.goals) == 1 and proof_state.goals[0].target == cur_state.goals[0].target), f'`have {submission_name}` failed due to proof state: ' + str(proof_state)
                        deductive_steps.append((tactic_header, have_step))
                    states.append(proof_state.goals[:])
                    
                    submit_step = f'exact {submission_name}'
                    try:
                        proof_state = await server.goal_tactic_async(proof_state, 0, submit_step)
                        assert proof_state.is_solved, f'`exact {submission_name}` failed due to proof state: ' + str(proof_state)
                        deductive_steps.append(('', submit_step))
                    except:
                        proof_state = await server.goal_tactic_async(proof_state, 0, tactic_header + submit_step)
                        assert proof_state.is_solved, f'`exact {submission_name}` failed due to proof state: ' + str(proof_state)
                        deductive_steps.append((tactic_header, submit_step))
                    states.append(proof_state.goals[:])

                # Validate whole proof
                whole_proof = ''
                for t, s in deductive_steps:
                    if len(t) > 0:
                        whole_proof += t
                    whole_proof += s + '\n\n'
                whole_proof = whole_proof.strip()
                
                try:
                    final_state = await server.goal_tactic_async(init_state, 0, '{\n' + whole_proof + '\n}')
                    assert final_state.is_solved, 'final_state.is_solved Failed'
                except Exception as e:
                    whole_proof = None
                    logger.debug(f'async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): Whole-proof validation failed, traceback: {[traceback.format_exc()]}')
                    logger.warning(f'async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): Whole-proof validation failed due to {repr(e)}')
                
                units[i_u] |= {
                    'deductive_steps' : deductive_steps,
                    'deductive_states' : [[g.serialize()] for s in states for g in s],
                    'formal_statement' : formal_statement,
                    'intros' : intros,
                    'load_header' : load_header,
                    'whole_proof' : whole_proof
                }
                logger.debug(f'async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): succeeded.')
            except Exception as e:
                if 'unknown identifier' in str(e) or 'unknown constant' in str(e):
                    pass
                else:
                    logger.warning(f'async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): Failed: {traceback.format_exc()}')
                    # import pdb; pdb.set_trace()
                logger.debug(f'async_worker({base_cnt+idx}-{i_p}/{len(remaining_units)}): Failed, traceback: {[traceback.format_exc()]}')
        
        logger.debug(f'async_worker({base_cnt+idx}): finished.')
    except Exception as e:
        logger.error(f'async_worker({base_cnt+idx}): Failed, traceback: {[traceback.format_exc()]}')
        # import pdb; pdb.set_trace()
    finally:
        results[idx] = datapoint
        server.tag = ''
        available_servers.insert(0, server)

def worker(args: Tuple) -> int:
    working_root, base_cnt = args
    
    if not osp.exists(osp.join(working_root, f'done_chunk_{base_cnt}.pkl')):
        logger.opt(colors=True).info(f'<green>worker({base_cnt}): done pkl does not exist, exiting....</green>')
        return
    if osp.exists(osp.join(working_root, f'done_v2_chunk_{base_cnt}.pkl')):
        logger.opt(colors=True).info(f'<green>worker({base_cnt}): done_v2 pkl already exists, exiting....</green>')
        return

    with open(osp.join(working_root, f'done_chunk_{base_cnt}.pkl'), 'rb') as f:
        data_to_process = pickle.load(f)
    
    finished_list = [None for _ in range(len(data_to_process))]

    available_servers = [
        PersistentServer(
            max_count=64,
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
        (i, d) for (i, d) in enumerate(data_to_process) if 'parse_result' in d.keys()
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
    
    with open(osp.join(working_root, f'done_v2_chunk_{base_cnt}.pkl'), 'wb') as f:
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
                int(n[len('done_v2_chunk_'):-len('.pkl')]) for n in os.listdir(working_root) if n.startswith('done_v2_chunk_') and n.endswith('.pkl')
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
