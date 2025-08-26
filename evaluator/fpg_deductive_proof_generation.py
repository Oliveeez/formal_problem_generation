import os
import os.path as osp
import sys
import json
import asyncio
import traceback
from datetime import datetime
from typing import Dict, Set, List, Any, Tuple, Optional
import pickle
import regex as re
import itertools as I
import random
import multiprocessing as mp
import time

import vllm     # vLLM should be imported before any 3rd-party libraries, o.w. all Pantograph REPLs are scheduled at the same CPU core.
from openai import AsyncOpenAI, NOT_GIVEN
from tqdm import tqdm
from loguru import logger
from fire import Fire
import aiofiles

from common.constants import FPS_GLOBAL_SETTING, CORE_OPTIONS
from common.utils import add_k_to_port, remove_comments, replace_sorry, decompose_statement
from common.pantograph.dataclasses import ProblemGenerationProcess
from common.pantograph.server import PersistentServer
from agent.problem_generation import SFT_DeductiveProofGenerator

N_CONCURRENCY_PER_WORKER = 8

async def async_worker(
    datapoint: Dict,
    base_cnt: int,
    idx: int,
    agent: SFT_DeductiveProofGenerator,
    available_parsers: List[PersistentServer],
    available_servers: List[PersistentServer],
    results: List,
) -> None:
    parser = available_parsers.pop()
    parser.tag = str(idx)
    server = available_servers.pop()
    server.tag = str(idx)
    time_start = time.time()

    result = ProblemGenerationProcess(
        informal_problem=datapoint['statement'],
        informal_answer='',
        informal_solution='',
        header='',
        formal_statement='',
        formal_solution_draft='',
        formal_proofs=[],
        steps=[],
        dependencies=[],
        trajectory=[],
        metainfo={'id' : datapoint['id']}
    )
    try:
        # I. Parse tactic invocation
        p_raw = datapoint['lean_code']
        p = remove_comments(p_raw).strip().replace('\nlemma ', '\ntheorem ').replace('\nexample ', '\ntheorem thm_example')
        start_pos = p.find('theorem')
        assert start_pos != -1, 'Start pos not found'
        intro, stmt = p[:start_pos], p[start_pos:]
        
        result.formal_statement = 'example ' + replace_sorry(stmt.split(maxsplit=2)[-1].strip())
        assert result.formal_statement.endswith(':= sorry')
                
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
        result.header = load_header
        
        context, target = decompose_statement(result.formal_statement)
        assert target is not None, f'Target parsing failed: {result.formal_statement}'

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
        
        formal_statement_load = (('∀ ' + '\n'.join(hypotheses) + '\n, ') if len(hypotheses) > 0 else '') + target
        init_state = await server.load_statement_async(formal_statement_load, intros=intros, header=load_header)
        
        # Falsify
        falsifying_state = await server.goal_tactic_async(init_state, 0, 'exfalso')
        r = await agent.generate_proofs_async(falsifying_state, server)
        if r is not None and len(r) == 1:
            falsified_model, falifying_proof = r[0]
            
        
        # Generate deductive proofs, decompose, and reassemble
        proven_model = None
        r = await agent.generate_proofs_async(init_state, server)
        assert r is not None and len(r) == 1
        proven_model, proof = r[0]
        result.formal_solution_draft = proof,
        
        is_decomposed = agent.decompose_deductive_steps_async(
            result=result,
            server=server,
            tag=str(base_cnt+idx),
            reassemble_trajectory=True
        )
        # TODO: metainfo
        
        result.metainfo = json.dumps(
            result.metainfo | {
                'time_consumption': time.time() - time_start,
                'proven_model': proven_model
            }
        )
        logger.debug(f'async_worker({base_cnt+idx}): succeeded.')
    except Exception as e:
        logger.debug(f'async_worker({base_cnt+idx}): Failed, traceback: {[traceback.format_exc()]}')
        # import pdb; pdb.set_trace()
    finally:
        results[idx] = result
        parser.tag = ''
        available_parsers.insert(0, parser)
        server.tag = ''
        available_servers.insert(0, server)

def worker(args: Tuple) -> int:
    working_root, base_cnt, proof_gen_clients, proof_gen_base_urls, proof_gen_api_keys, proof_gen_model_names, n_servers = args
        
    if not osp.exists(osp.join(working_root, f'raw_chunk_{base_cnt}.pkl')):
        logger.opt(colors=True).info(f'<green>worker({base_cnt}): raw pkl does not exist, exiting....</green>')
        return
    if osp.exists(osp.join(working_root, f'done_chunk_{base_cnt}.pkl')):
        logger.opt(colors=True).info(f'<green>worker({base_cnt}): done pkl already exists, exiting....</green>')
        return

    with open(osp.join(working_root, f'raw_chunk_{base_cnt}.pkl'), 'rb') as f:
        data_to_process = pickle.load(f)
    
    finished_list = [None for _ in range(len(data_to_process))]

    problem_generators = []
    for k in range(n_servers):
        proof_gen_clients = [
            AsyncOpenAI(
                base_url=add_k_to_port(proof_gen_base_url, k),
                api_key=proof_gen_api_key
            ) for (proof_gen_base_url, proof_gen_api_key) in zip(proof_gen_base_urls, proof_gen_api_keys)
        ]
        problem_generators.append(SFT_DeductiveProofGenerator(
            statements=None,    # TODO
            proof_gen_clients=proof_gen_clients,
            proof_gen_models=proof_gen_model_names,
            num_max_samples_per_trial=1,
            temperature=0,
            max_tokens=NOT_GIVEN
        ))

    available_parsers = [
        PersistentServer(
            max_count=32,
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
    proof_gen_base_urls: List[str],
    proof_gen_api_keys: List[str],
    proof_gen_model_names: List[str],
    working_root: str='/home/ma-user/workspace/formal_problem_generation/data/FineLeanCorpus/raw',
    use_mp: bool=True,
    reverse_order: bool=False,
    n_concurrency: int=12,
):
    saved_args = {**locals()}
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_prefix = 'problem_generation'+'.'

    os.makedirs(working_root, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level='INFO')    # filter=lambda record: record["name"] != "agent.solution_autoformalization"
    logger.add(osp.join(working_root, log_prefix+now+'.log'), level='DEBUG')
    logger.info(f'Generating deductive steps hyperparams: {saved_args}')
    
    splits = load_and_split(working_root, reverse_order)
    
    assert len(proof_gen_base_urls) == len(proof_gen_api_keys), f'{len(proof_gen_base_urls)} != {len(proof_gen_api_keys)}'
    assert len(proof_gen_api_keys) == len(proof_gen_model_names), f'{len(proof_gen_api_keys)} != {len(proof_gen_model_names)}'

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
    Fire(main)
