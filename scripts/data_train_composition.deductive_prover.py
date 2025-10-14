# %%
import os
import json
import pickle
import collections as C
import itertools as I
import random
import regex as re
import traceback
import asyncio

import dacite
from loguru import logger
from tqdm import tqdm

from common.constants import CORE_OPTIONS
from common.utils import remove_min_whitespace, remove_comments
from common.pantograph.dataclasses import Goal
from common.pantograph.server import PersistentServer

bracket_pairings = {
    '(' : ')',
    '[' : ']',
    '{' : '}',
    '⦃' : '⦄'
}

def parse_variables(s : str) -> tuple[str, str]:
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

with open('/home/ma-user/workspace/formal_problem_generation/formal_problem_generation/data/MiniF2F/Numina-Lean/deductive.20250818-162148.pkl', 'rb') as f:
    data = pickle.load(f)

header = ("""
import Mathlib
import Aesop

""" + '\n'.join('set_option ' + t.replace('=', ' ') for t in CORE_OPTIONS)).strip()
print(header)

for d in data:
    for u in d['parse_result']['units']:
        if 'deductive_steps' in u.keys():
            assert 'deductive_states' in u.keys()
            break

len(set([d['index'] for d in data])), len(data)

tasks = [
    (d, u) for d in data for u in d['parse_result']['units'] if 'deductive_steps' in u.keys()
]
print(len(tasks))

available_servers = [
    PersistentServer(
        is_state_based=True,
        tag='test',
        _sync_init=False,
        imports=["Mathlib", "Aesop"],
        project_path='/home/ma-user/workspace/fps_pantograph/formal_problem_solving/data/MiniF2F',
        core_options=CORE_OPTIONS,
        timeout=300,
    ) for _ in range(64)
]

data_deductive_proof_generation = [None for _ in range(len(tasks))]

async def worker(d: dict, u: dict, idx: int, available_servers: list[PersistentServer]):
    server = available_servers.pop()
    server.tag = str(idx)
    try:
        # Reorganize Code
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

        p_injected: list[str] = p_raw.splitlines()
        for (i, l) in reversed(list(enumerate(p_injected))):
            if l.startswith('import '):
                i += 1
                break
        p_injected = '\n'.join(p_injected[:i]) + '\n\n' + '\n'.join('set_option ' + t.replace('=', ' ') for t in CORE_OPTIONS) + '\n\n' + '\n'.join(p_injected[i:])

        # Initialize
        invocations = u['invocations']
        assert len(invocations[0]['before']) == 1, 'Initial state contains multiple goals'

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

        init_state = await server.load_statement_async(formal_statement, intros=intros, header=load_header)
        assert [g.serialize() for g in init_state.goals] == u['deductive_states'][0], "[g.serialize() for g in init_state.goals] != u['deductive_states'][0]"
        
        proof = ''
        for tactic_header, step in u['deductive_steps']:
            if len(tactic_header) > 0:
                proof += tactic_header
            proof += remove_min_whitespace(step) + '\n\n'
        proof = proof.strip()
        
        final_state = await server.goal_tactic_async(init_state, 0, '{\n' + proof + '\n}')
        assert final_state.is_solved, 'final_state.is_solved Failed'

        data_deductive_proof_generation[idx] = {
            "conversation":[
                {
                    "input": f"""
Assume the following header is executed:
```lean4
{header}
```

Generate a deductive proof for the following Lean 4 proof state:
```lean4
{str(init_state)}
```
""".strip(),
                    "output": proof
                }
            ]
        }
    except Exception as e:
        logger.warning(f'worker({idx}): Failed due to {repr(e)}:\n{traceback.format_exc()}')
    finally:
        server.tag = ''
        available_servers.insert(0, server)

async def _async_main():
    pending_tasks: set[asyncio.Task] = set()
    loop = list(enumerate(tasks))
    for i, (d, u) in tqdm(loop):
        if len(pending_tasks) >= 64:
            done_tasks, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done_tasks:
                if task.exception() is not None:
                    logger.error(f"Exception occurred: {task.exception()} {task.get_stack()}")
                    for pending_task in pending_tasks:
                        pending_task.cancel()
                    return
        pending_tasks.add(
            asyncio.create_task(
                worker(d, u, i, available_servers)
            )
        )
    if len(pending_tasks) > 0:
        await asyncio.wait(pending_tasks)
    await logger.complete()

try:
    asyncio.run(_async_main())
    with open('/home/ma-user/workspace/formal_problem_generation/formal_problem_generation/data/MiniF2F/Numina-Lean/train.deductive_prover.jsonl', 'w') as f:
        for d in data_deductive_proof_generation:
            if d is not None:
                f.write(json.dumps(d) + '\n')
    with open('/home/ma-user/workspace/formal_problem_generation/formal_problem_generation/data/MiniF2F/Numina-Lean/train.deductive_prover.pkl', 'wb') as f:
        pickle.dump(data_deductive_proof_generation, f)
    logger.info('All succeeded, saving at /home/ma-user/workspace/formal_problem_generation/formal_problem_generation/data/MiniF2F/Numina-Lean/train.deductive_prover.(jsonl|pkl)')
except Exception as e:
    logger.error(f'Failed due to {repr(e)}:\n{traceback.format_exc()}')
    import pdb; pdb.set_trace()
