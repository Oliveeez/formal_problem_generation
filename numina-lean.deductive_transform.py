
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

import fire
import aiofiles
from aiofiles.threadpool.text import AsyncTextIOWrapper
import pyarrow
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from common.constants import OPEN_HEADER, CORE_OPTIONS, MVAR_PATTERN
from common.utils import remove_comments, normalize_spaces, remove_spaces, normalize_draft
from common.pantograph.dataclasses import TacticInvocation, Goal
from common.pantograph.server import Server, PersistentServer, TacticFailure

def is_deductive(ivc: dict) -> bool:
    return len(ivc['before']) == 1 and len(ivc['after']) == 1 and ivc['before'][0].split('⊢ ')[-1] == ivc['after'][0].split('⊢ ')[-1]

superscript_to_digit = {
    '⁰': '0',
    '¹': '1',
    '²': '2',
    '³': '3',
    '⁴': '4',
    '⁵': '5',
    '⁶': '6',
    '⁷': '7',
    '⁸': '8',
    '⁹': '9'
}
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

async def worker(
    d: Dict,
    idx: int,
    available_servers: List[PersistentServer],
    results: List,
    save_dst: AsyncTextIOWrapper
) -> None:
    server = available_servers.pop()
    server.tag = str(idx)
    try:
        p_raw = d['formal_code']
        
        import_list = d['parse_result']['import_list']
        open_scoped_list = d['parse_result']['open_scoped_list']
        open_list = d['parse_result']['open_list']
        option_list = d['parse_result']['option_list']

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

        parsed_units = [i_u for i_u, u in enumerate(d['parse_result']['units']) if len(u['invocations'] or []) > 0 and 'deductive_steps' not in u.keys()]
        logger.debug(f'worker({idx}): {len(parsed_units)} units to parse')

        for i_p, i_u in enumerate(parsed_units):
            u = d['parse_result']['units'][i_u]
            
            try:
                invocations = u['invocations']
                assert len(invocations[0]['before']) == 1, 'Initial state contains multiple goals'
                nonhygienic_transformer = factory_nonhygienic_transformer(invocations[0]['before'][0])

                code_segment = remove_comments(p_injected.encode()[u['i_begin']:u['i_end']].decode())
                start_pos = None
                for start_pos in re.finditer(r':=\s*by', code_segment):
                    break
                assert start_pos is not None, '":= by" not found'
                statement_code, proof_code = code_segment[:start_pos.span(0)[0]], code_segment[start_pos.span(0)[1]:]

                # Preprocess steps
                for ivc in invocations:
                    ivc['tactic'] = ivc['tactic'].replace('native_decide', 'decide')
                proof_code = proof_code.replace('native_decide', 'decide')

                # Parse Context from statement code
                context, target = parse_variables(statement_code)
                assert target is not None, f'Target parsing failed: {statement_code}'
                
                # Parse intros
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

                # Load statement
                # hypotheses = ('∀ ' + '\n'.join(
                #     '(' + ' '.join(var_names) + ' : ' + var_type + ')' if var_names != ['_'] else f'[{var_type}]' for (var_names, var_type) in telescope
                # ) + '\n, ') if len(context) > 0 else ''
                formal_statement = (('∀ ' + '\n'.join(hypotheses) + '\n, ') if len(hypotheses) > 0 else '') + target
                assert '⊢' not in formal_statement, '⊢ in formal_statement'

                try:
                    init_state = await server.load_statement_async(formal_statement, intros=intros, header=load_header)
                except Exception as e:
                    raise RuntimeError(context, target, load_header) from e
                # assert all(match_wo_mvar(nonhygienic_transformer(g_parsed), str(g_now)) for g_parsed, g_now in zip(invocations[0]['before'], init_state.goals)), 'initial state not equivalent w/ parse results'
                assert len(init_state.goals) == 1, 'deductive step execution failed' #* Non-strict match
                
                # Start transforming
                states: List[List[Goal]] = []
                steps: List[Tuple[str, str]] = []
                cur_state = init_state

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
                            cur_state = await server.goal_tactic_async(cur_state, 0, ivc['tactic'])
                            # assert all(match_wo_mvar(nonhygienic_transformer(g_parsed), str(g_now)) for g_parsed, g_now in zip(ivc['after'], cur_state.goals)), 'deductive step execution failed'
                            assert len(cur_state.goals) == 1 and cur_state.goals[0].target == init_state.goals[0].target, 'deductive step execution failed' #* Non-strict match
                            steps.append(('', ivc['tactic']))
                        except:
                            cur_state = await server.goal_tactic_async(cur_state, 0, tactic_header + ivc['tactic'])
                            # assert all(match_wo_mvar(nonhygienic_transformer(g_parsed), str(g_now)) for g_parsed, g_now in zip(ivc['after'], cur_state.goals)), 'deductive step execution failed'
                            assert len(cur_state.goals) == 1 and cur_state.goals[0].target == init_state.goals[0].target, 'deductive step execution failed' #* Non-strict match
                            steps.append((tactic_header, ivc['tactic']))
                else:
                    deductive_unit_indices = []

                # Remaining non-deductive steps
                
                # 1. Extract remaining steps
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

                assert ptr_deductive_code == len(deductive_code_wo_space), 'ptr_deductive_code != len(deductive_code_wo_space)'

                # 2. Execute remaining steps
                proof_state = cur_state

                have_step = f'have h_submission: {target} := by {{\n' + '\n'.join(proof_lines[ptr_proof_line:]) + '\n}'
                states.append(proof_state.goals[:])
                try:
                    proof_state = await server.goal_tactic_async(proof_state, 0, have_step)
                    assert (len(proof_state.goals) == 1 and proof_state.goals[0].target == cur_state.goals[0].target), '`have h_submission` failed due to proof state: ' + str(proof_state)
                    steps.append(('', have_step))
                except:
                    proof_state = await server.goal_tactic_async(proof_state, 0, tactic_header + have_step)
                    assert (len(proof_state.goals) == 1 and proof_state.goals[0].target == cur_state.goals[0].target), '`have h_submission` failed due to proof state: ' + str(proof_state)
                    steps.append((tactic_header, have_step))

                states.append(proof_state.goals[:])
                submit_step = 'exact h_submission'
                try:
                    proof_state = await server.goal_tactic_async(proof_state, 0, submit_step)
                    assert proof_state.is_solved, '`exact h_submission` failed due to proof state: ' + str(proof_state)
                    steps.append(('', submit_step))
                except:
                    proof_state = await server.goal_tactic_async(proof_state, 0, tactic_header + submit_step)
                    assert proof_state.is_solved, '`exact h_submission` failed due to proof state: ' + str(proof_state)
                    steps.append((tactic_header, submit_step))
    
                d['parse_result']['units'][i_u] |= {
                    'deductive_steps' : steps,
                    'deductive_states' : [[g.serialize()] for s in states for g in s]
                }
                logger.info(f'worker({idx}-{i_p}/{len(parsed_units)}): succeeded.')
            except Exception as e:
                logger.debug(f'worker({idx}-{i_p}/{len(parsed_units)}): Failed, traceback: {[traceback.format_exc()]}')
                logger.warning(f'worker({idx}-{i_p}/{len(parsed_units)}): Failed due to {repr(e)}')
                # if isinstance(e, ValueError):
                #     import pdb; pdb.set_trace()
                #     print()
        
        results[idx] = d
        await save_dst.write(json.dumps(d) + '\n')
        logger.debug(f'worker({idx}): finished.')
    except Exception as e:
        logger.warning(f'worker({idx}): Exception {repr(e)}:\n{traceback.format_exc()}')
        results[idx] = (repr(e), traceback.format_exc())
    finally:
        server.tag = ''
        available_servers.insert(0, server)

def main(
    n_concurrency: int=64,
    load_from: str='/home/ma-user/workspace/formal_problem_generation/formal_problem_generation/data/MiniF2F/Numina-Lean/20250815-172942.pkl',
    resume_from: str='/home/ma-user/workspace/formal_problem_generation/formal_problem_generation/data/MiniF2F/Numina-Lean/deductive.20250818-003916.jsonl',
    save_root: str='/home/ma-user/workspace/formal_problem_generation/formal_problem_generation/data/MiniF2F/Numina-Lean',
    reverse: bool=False,
) -> None:
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.remove()
    logger.add(sys.stdout, level='INFO')
    logger.add(osp.join(save_root, now+'.log'), level='DEBUG')
    
    if len(resume_from) == 0:
        # Load Parse Results
        with open(load_from, 'rb') as f:
            invocations_all = pickle.load(f)
        
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
        

        # Ensure the order of `invocations_all` and `parsed_datapoints` matches
        assert len(invocations_all) == len(parsed_datapoints)

        data_success = []

        for idx, (ivc_all, d) in enumerate(zip(invocations_all, parsed_datapoints)):
            try:
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
                
                assert open_scoped_list == ivc_all['open_scoped_list']
                assert open_list == ivc_all['open_list']
                assert option_list == ivc_all['option_list']
                
                ivc_all['import_list'] = import_list    # A bug in `numina-lean.parse.py` (0815)
                
                data_success.append(
                    d | {'parse_result': ivc_all, 'index': idx}
                )
            except:
                continue

        logger.info(f'Deductive Transforming {len(data_success)} datapoints')

        finished = [d for d in data_success]
    else:
        with open(resume_from, 'r') as f:
            finished = [json.loads(l) for l in f.readlines()]
            logger.info(f'Loaded {len(finished)} datapoints from {resume_from}')
    
    available_servers = [
        PersistentServer(
            is_state_based=True,
            tag='test',
            _sync_init=False,
            imports=["Mathlib", "Aesop"],
            project_path='/home/ma-user/workspace/fps_pantograph/formal_problem_solving/data/MiniF2F',
            core_options=CORE_OPTIONS,
            timeout=300,
        ) for _ in range(n_concurrency)
    ]

    async def _async_main():
        async with aiofiles.open(osp.join(save_root, 'deductive.'+now+'.jsonl'), 'w') as f:
            pending_tasks: Set[asyncio.Task] = set()
            loop = list(enumerate(finished))
            if reverse:
                loop = list(reversed(loop))
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
                        worker(d, i, available_servers, finished, f)
                    )
                )
            if len(pending_tasks) > 0:
                await asyncio.wait(pending_tasks)
        await logger.complete()
    
    try:
        asyncio.run(_async_main())
    finally:
        try:
            logger.info(f"Finished transformation, saving at {osp.join(save_root, now+'.(pkl|jsonl)')}")
            with open(osp.join(save_root, 'deductive.'+now+'.pkl'), 'wb') as f:
                pickle.dump(finished, f)
        except Exception as e:
            logger.error(traceback.format_exc())
            import pdb; pdb.set_trace()


if __name__ == '__main__':
    fire.Fire(main)
