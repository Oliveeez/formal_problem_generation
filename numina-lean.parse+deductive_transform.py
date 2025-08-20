
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
from common.utils import remove_comments, normalize_spaces, remove_spaces, normalize_draft, remove_min_whitespace, chunk_list
from common.pantograph.dataclasses import TacticInvocation, Goal
from common.pantograph.server import Server, PersistentServer, TacticFailure


N_CONCURRENCY_PER_WORKER = 8

def is_deductive(ivc: dict) -> bool:
    return len(ivc['before']) == 1 and len(ivc['after']) == 1 and ivc['before'][0].split('⊢ ')[-1] == ivc['after'][0].split('⊢ ')[-1]

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
    available_parsers: List[PersistentServer],
    available_servers: List[PersistentServer],
    results: List,
    working_root: str,
) -> None:
    parser = available_parsers.pop()
    parser.tag = str(idx)
    server = available_servers.pop()
    server.tag = str(idx)
    try:
        # I. Parse tactic invocation
        p_raw = datapoint['formal_code']
        p = remove_comments(p_raw).strip().replace('\nlemma ', '\ntheorem ').replace('\nexample ', '\ntheorem thm_example')
        start_pos = p.find('theorem')
        assert start_pos != -1, 'Start pos not found'
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
                raise ValueError('Unexpected line in intro code: ' + l)

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

        async with aiofiles.open(osp.join(working_root, str(base_cnt+idx)+'.lean'), 'w') as f:
            await f.write(p_injected)

        units = await parser.tactic_invocations_async(osp.join(working_root, str(base_cnt+idx)+'.lean'))
        assert len(units) >= 1 and 'error' not in str([x.messages for x in units]), 'tactic_invocations_async failed: ' + str([x.messages for x in units])
        
        datapoint['parse_result'] = {
            'import_list': import_list,
            'open_scoped_list': open_scoped_list,
            'open_list': open_list,
            'option_list': option_list,
            'units': [u.serialize() for u in units]
        }

        # II. Deductive transform
        parsed_units = [i_u for i_u, u in enumerate(datapoint['parse_result']['units']) if len(u['invocations'] or []) > 0 and 'deductive_steps' not in u.keys()]
        logger.debug(f'async_worker({base_cnt+idx}): {len(parsed_units)} units to transform')

        for i_p, i_u in enumerate(parsed_units):
            u = datapoint['parse_result']['units'][i_u]
            
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
                    raise RuntimeError(e, context, target, load_header)
                # assert all(match_wo_mvar(nonhygienic_transformer(g_parsed), str(g_now)) for g_parsed, g_now in zip(invocations[0]['before'], init_state.goals)), 'initial state not equivalent w/ parse results'
                assert len(init_state.goals) == 1, 'deductive step execution failed' #* Non-strict match

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
                states: List[List[Goal]] = []
                steps: List[Tuple[str, str]] = []
                cur_state = init_state

                next_state = None
                # Deductive steps
                if is_deductive(invocations[0]):
                    deductive_unit_indices = [0]

                    for i, ivc in enumerate(invocations[1:], 1):
                        if is_deductive(ivc) and match_wo_mvar(ivc['before'][0], invocations[deductive_unit_indices[-1]]['after'][0]):
                            deductive_unit_indices.append(i)
                            if len(ivc['after'][0]) == 0:
                                break

                    for deductive_unit_idx in deductive_unit_indices:
                        ivc = invocations[deductive_unit_idx]
                        states.append(cur_state.goals[:])
                        try:
                            next_state = await server.goal_tactic_async(cur_state, 0, ivc['tactic'])
                            # assert all(match_wo_mvar(nonhygienic_transformer(g_parsed), str(g_now)) for g_parsed, g_now in zip(ivc['after'], cur_state.goals)), 'deductive step execution failed'
                            if next_state.is_solved:
                                assert deductive_unit_idx == deductive_unit_indices[-1], 'next_state.is_solved but deductive_unit_idx != deductive_unit_indices[-1]'
                                if remove_comments(ivc['tactic']).strip().startswith('exact '): # If the final step is `exact`, do nothing (add to `steps`, update `cur_state` and exit)
                                    logger.info(f"async_worker({base_cnt+idx}-{i_p}/{len(parsed_units)}): Detected `exact` submission: {[remove_comments(ivc['tactic']).strip()]}")
                                else:
                                    logger.info(f"async_worker({base_cnt+idx}-{i_p}/{len(parsed_units)}): Detected non-`exact` submission: {[remove_comments(ivc['tactic']).strip()]}")
                                    break   # If the final step is not `exact`, 1) do not add to `steps` - leave it for final submission; 2) do not update `cur_state`
                            else:
                                assert len(next_state.goals) == 1 and next_state.goals[0].target == init_state.goals[0].target, 'deductive step execution failed' #* Non-strict match
                            steps.append(('', ivc['tactic']))
                        except:
                            next_state = await server.goal_tactic_async(cur_state, 0, tactic_header + ivc['tactic'])
                            # assert all(match_wo_mvar(nonhygienic_transformer(g_parsed), str(g_now)) for g_parsed, g_now in zip(ivc['after'], cur_state.goals)), 'deductive step execution failed'
                            if next_state.is_solved:
                                assert deductive_unit_idx == deductive_unit_indices[-1], 'next_state.is_solved but deductive_unit_idx != deductive_unit_indices[-1]'
                                if remove_comments(ivc['tactic']).strip().startswith('exact '): # If the final step is `exact`, do nothing (add to `steps`, update `cur_state` and exit)
                                    logger.info(f"async_worker({base_cnt+idx}-{i_p}/{len(parsed_units)}): Detected `exact` submission: {[remove_comments(ivc['tactic']).strip()]}")
                                else:
                                    logger.info(f"async_worker({base_cnt+idx}-{i_p}/{len(parsed_units)}): Detected non-`exact` submission: {[remove_comments(ivc['tactic']).strip()]}")
                                    break   # If the final step is not `exact`, 1) do not add to `steps` - leave it for final submission; 2) do not update `cur_state`
                            else:
                                assert len(next_state.goals) == 1 and next_state.goals[0].target == init_state.goals[0].target, 'deductive step execution failed' #* Non-strict match
                            steps.append((tactic_header, ivc['tactic']))
                        cur_state = next_state
                else:
                    deductive_unit_indices = []

                # Remaining non-deductive steps
                
                # 1. Extract remaining steps
                if not cur_state.is_solved:
                    deductive_code_wo_space = remove_spaces(remove_comments('\n'.join(s[1] for s in steps)))
                    assert remove_spaces(proof_code).startswith(deductive_code_wo_space), 'remove_spaces(proof_code).startswith(deductive_code_wo_space) failed'

                    ptr_deductive_code = 0
                    ptr_proof_line = None
                    proof_lines = proof_code.splitlines()
                    for (ptr_proof_line, line) in enumerate(proof_lines):
                        line_wo_space = remove_spaces(line)
                        if deductive_code_wo_space[ptr_deductive_code:].startswith(line_wo_space):
                            ptr_deductive_code += len(line_wo_space)
                        else:
                            break

                    if ptr_deductive_code != len(deductive_code_wo_space):
                        logger.warning(f'async_worker({base_cnt+idx}-{i_p}/{len(parsed_units)}): ptr_deductive_code != len(deductive_code_wo_space), cur_line: {line}')

                    # 2. Execute remaining steps
                    proof_state = cur_state

                    submission_name = generate_submission_name([v.name for v in cur_state.goals[0].variables if v.name is not None])
                    have_step = f'have {submission_name}: {target} := by {{\n' + '\n'.join(proof_lines[ptr_proof_line:]) + '\n}'
                    states.append(proof_state.goals[:])
                    try:
                        proof_state = await server.goal_tactic_async(proof_state, 0, have_step)
                        assert (len(proof_state.goals) == 1 and proof_state.goals[0].target == cur_state.goals[0].target), f'`have {submission_name}` failed due to proof state: ' + str(proof_state)
                        steps.append(('', have_step))
                    except:
                        proof_state = await server.goal_tactic_async(proof_state, 0, tactic_header + have_step)
                        assert (len(proof_state.goals) == 1 and proof_state.goals[0].target == cur_state.goals[0].target), f'`have {submission_name}` failed due to proof state: ' + str(proof_state)
                        steps.append((tactic_header, have_step))

                    states.append(proof_state.goals[:])
                    submit_step = f'exact {submission_name}'
                    try:
                        proof_state = await server.goal_tactic_async(proof_state, 0, submit_step)
                        assert proof_state.is_solved, f'`exact {submission_name}` failed due to proof state: ' + str(proof_state)
                        steps.append(('', submit_step))
                    except:
                        proof_state = await server.goal_tactic_async(proof_state, 0, tactic_header + submit_step)
                        assert proof_state.is_solved, f'`exact {submission_name}` failed due to proof state: ' + str(proof_state)
                        steps.append((tactic_header, submit_step))

                # Validate whole proof
                whole_proof = ''
                for t, s in steps:
                    if len(t) > 0:
                        whole_proof += t
                    whole_proof += remove_min_whitespace(s) + '\n\n'
                whole_proof = whole_proof.strip()
                
                try:
                    final_state = await server.goal_tactic_async(init_state, 0, '{\n' + whole_proof + '\n}')
                    assert final_state.is_solved, 'final_state.is_solved Failed'
                except Exception as e:
                    whole_proof = None
                    logger.debug(f'async_worker({base_cnt+idx}-{i_p}/{len(parsed_units)}): Whole-proof validation failed, traceback: {[traceback.format_exc()]}')
                    logger.warning(f'async_worker({base_cnt+idx}-{i_p}/{len(parsed_units)}): Whole-proof validation failed due to {repr(e)}')
                
                datapoint['parse_result']['units'][i_u] |= {
                    'deductive_steps' : steps,
                    'deductive_states' : [[g.serialize()] for s in states for g in s],
                    'formal_statement' : formal_statement,
                    'intros' : intros,
                    'load_header' : load_header,
                    'whole_proof' : whole_proof
                }
                logger.debug(f'async_worker({base_cnt+idx}-{i_p}/{len(parsed_units)}): succeeded.')
            except Exception as e:
                if 'unknown identifier' in str(e) or 'unknown constant' in str(e):
                    pass
                elif 'remove_spaces(proof_code).startswith(deductive_code_wo_space) failed' in str(e):
                    pass
                elif 'expected end of input' in str(e) or "expected '{' or indented tactic sequence" in str(e):
                    pass
                else:
                    logger.warning(f'async_worker({base_cnt+idx}-{i_p}/{len(parsed_units)}): Failed: {traceback.format_exc()}')
                    # import pdb; pdb.set_trace()
                logger.debug(f'async_worker({base_cnt+idx}-{i_p}/{len(parsed_units)}): Failed, traceback: {[traceback.format_exc()]}')
        
        logger.debug(f'async_worker({base_cnt+idx}): finished.')
    except Exception as e:
        logger.debug(f'async_worker({base_cnt+idx}): Failed, traceback: {[traceback.format_exc()]}')
        # import pdb; pdb.set_trace()
        datapoint |= {
            'exception': repr(e),
            'traceback': traceback.format_exc()
        }
    finally:
        results[idx] = datapoint
        parser.tag = ''
        available_parsers.insert(0, parser)
        server.tag = ''
        available_servers.insert(0, server)

def worker(args: Tuple) -> int:
    working_root, base_cnt = args
    
    if not osp.exists(osp.join(working_root, f'raw_chunk_{base_cnt}.pkl')):
        logger.opt(colors=True).info(f'<green>worker({base_cnt}): raw pkl does not exist, exiting....</green>')
        return
    if osp.exists(osp.join(working_root, f'done_chunk_{base_cnt}.pkl')):
        logger.opt(colors=True).info(f'<green>worker({base_cnt}): done pkl already exists, exiting....</green>')
        return

    with open(osp.join(working_root, f'raw_chunk_{base_cnt}.pkl'), 'rb') as f:
        data_to_process = pickle.load(f)
    
    finished_list = [None for _ in range(len(data_to_process))]

    available_parsers = [
        PersistentServer(
            max_count=2,
            is_state_based=False,
            tag='',
            _sync_init=False,
            imports=["Mathlib", "Aesop"],
            project_path='/home/ma-user/workspace/formal_problem_generation/formal_problem_generation/data/MiniF2F',
            core_options=CORE_OPTIONS,
            timeout=300,
        ) for _ in range(N_CONCURRENCY_PER_WORKER)
    ]
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
    logger.info(f'worker({base_cnt}): Initialized, processing {len(data_to_process)} samples.')

    async def _async_main():
        pending_tasks = set()
        for i, d in enumerate(data_to_process):
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
                        available_parsers=available_parsers,
                        available_servers=available_servers,
                        results=finished_list,
                        working_root=osp.join(working_root, 'lean'),
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
    
    with open(osp.join(working_root, f'done_chunk_{base_cnt}.pkl'), 'wb') as f:
        pickle.dump(finished_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f'worker({base_cnt}): Exiting.')
    return base_cnt

def load_and_split(working_root: str, datapoints: list, chunksize: int, reverse_order: bool):
    chunks = list(chunk_list(datapoints, chunksize))
    all_splits = set(
            [
                int(n[len('raw_chunk_'):-len('.pkl')]) for n in os.listdir(working_root) if n.startswith('raw_chunk_') and n.endswith('.pkl')
            ]
        )
    if len(all_splits) == 0:
        all_splits = []
        for base_cnt, chunk in chunks:
            with open(osp.join(working_root, f'raw_chunk_{base_cnt}.pkl'), 'wb') as f:
                pickle.dump(chunk, f, pickle.HIGHEST_PROTOCOL)
                all_splits.append(base_cnt)
        all_splits = set(all_splits)
    elif len(all_splits) != len(chunks):
        logger.warning(f'len(all_splits) != 0 and len(all_splits) != len(chunks): {len(all_splits)} - {len(chunks)}')
    
    done_splits = set(
            [
                int(n[len('done_chunk_'):-len('.pkl')]) for n in os.listdir(working_root) if n.startswith('done_chunk_') and n.endswith('.pkl')
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
    chunksize: int=1024,
    reverse_order: bool=False,
) -> None:
    os.makedirs(osp.join(working_root, 'lean'), exist_ok=True)
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.remove()
    logger.add(sys.stdout, level='INFO')
    logger.add(osp.join(working_root, now+'.log'), level='DEBUG')
    
    # Load Data
    data = pq.read_table('/home/ma-user/workspace/formal_problem_generation/data/Numina-Lean/train-00000-of-00001.parquet').to_pandas().to_dict(orient="records")
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

    splits = load_and_split(working_root, datapoints, chunksize, reverse_order)

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
