import sys
import os
import os.path as osp
from io import BufferedWriter
import collections as C
import itertools as I
import random
import pickle
from typing import List, Dict, Set, Tuple, Callable, Any
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
from openai import AsyncOpenAI
import pyarrow
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm
import msgspec

from common.constants import OPEN_HEADER, CORE_OPTIONS, MVAR_PATTERN, BANNED_TOKENS
from common.utils import remove_comments, remove_spaces, rotate, generate_submission_name, decompose_statement
from common.pantograph.dataclasses import TacticInvocation, Goal, GoalState, ProblemGenerationStep, ProblemGenerationProcess, TacticDraft
from common.pantograph.server import Server, PersistentServer, TacticFailure, ServerError
from common.pantograph.parsing_server import PersistentParsingServer
from agent.problem_generation import ProblemEvaluator


async def async_worker(
    result: ProblemGenerationProcess,
    key: Any,
    available_agents: List[ProblemEvaluator],
    available_servers: List[PersistentServer],
    finished_list: Dict,
) -> None:
    agent = available_agents.pop()
    server = available_servers.pop()
    server.tag = str(key)
    try:
        result.metainfo = result.metainfo if isinstance(result.metainfo, Dict) else json.loads(result.metainfo)
        assert all(k not in result.metainfo['eval_result'].keys() for k in ['provers', 'proofs', 'KC', 'prove_token_usage']), f"result.metainfo['eval_result'].keys()={result.metainfo['eval_result'].keys()}"

        variables = []
        context, target = decompose_statement(result.formal_statement)
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
                        name = '_'
                    variables.append((name.strip(), var_type))
        new_varname = generate_submission_name([v[0] for v in variables])
        assert new_varname not in [v[0] for v in variables], f'new_varname={new_varname}, variables={[v[0] for v in variables]}'

        provers, proofs = await agent.prove_async(
            server=server,
            formal_statement='example\n' + (('\n'.join(context) + '\n: ') if len(context) > 0 else ': ') + target + ' := by\n  sorry',
            load_statement=(('∀ ' + '\n'.join(context) + '\n, ') if len(context) > 0 else '') + target,
            intros=[v[0] for v in variables],
            header=result.header,
            early_stop=True,
            tag=str(key)+'/prove'
        )
        
        if proofs[-1] is not None and len(result.formal_solution_draft or '') == 0:
            result.formal_solution_draft = proofs[-1]
        
        kc_eval_result = {
            'provers': provers,
            'proofs' : proofs,
            'KC': min([len(remove_spaces(remove_comments(p))) for p in proofs if p is not None] + [float('inf')]),
            'prove_token_usage': agent.last_token_usage
        }

        logger.info(f'async_worker({key}): Estimated KC = {kc_eval_result.get("KC", float("inf"))}')
        result.metainfo['eval_result'] |= kc_eval_result
        logger.info(f'async_worker({key}): finished.')
    except Exception as e:
        logger.debug(f'async_worker({key}): failed due to traceback {traceback.format_exc()}')
        logger.warning(f'async_worker({key}): failed due to exception {repr(e)}')
    finally:
        result.metainfo = json.dumps(result.metainfo)
        finished_list[key] = result
        server.tag = ''
        available_servers.insert(0, server)
        available_agents.insert(0, agent)

def main(
    load_path: str,
    log_root: str,
    proof_gen_base_urls: List[str],
    proof_gen_api_keys: List[str],
    proof_gen_model_names: List[str],
    project_root: str='/home/ma-user/workspace/fps_pantograph/formal_problem_solving/data/MiniF2F',
    temperature: float=1.0,
    max_tokens: int=-1,
    top_p: float=0.95,
    try_num: int=4,
    num_concurrency: int=12,
    kc_estimation_mode: str='none',
    debug: bool=False,
):
    saved_args = {**locals()}
    
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_prefix = 'fpg_evaluate_kc'+'.'

    os.makedirs(log_root, exist_ok=True)
    if not debug:
        logger.remove()
        logger.add(sys.stdout, level='INFO')    # filter=lambda record: record["name"] != "agent.solution_autoformalization"
    logger.add(osp.join(log_root, log_prefix+now+'.log'), level='DEBUG')
    logger.info(f'Evaluating problem generator with hyperparams: {saved_args}')
    assert num_concurrency % len(proof_gen_base_urls) == 0
    assert len(proof_gen_base_urls) == len(proof_gen_api_keys), f'{len(proof_gen_base_urls)} != {len(proof_gen_api_keys)}'
    assert len(proof_gen_api_keys) == len(proof_gen_model_names), f'{len(proof_gen_api_keys)} != {len(proof_gen_model_names)}'
    assert kc_estimation_mode.lower() in ['none', 'early_stop', 'full'], f'kc_estimation_mode={kc_estimation_mode}'
    
    with open(load_path, 'rb') as f:
        load_results = pickle.load(f)
        try:
            conditions_sampled, finished_list = load_results
        except:
            finished_list = load_results
            conditions_sampled = list(range(len(finished_list)))
            for i in conditions_sampled:
                assert isinstance(finished_list[i], object)
    
    for d in finished_list:
        d.metainfo = d.metainfo if isinstance(d.metainfo, Dict) else json.loads(d.metainfo)
    
    available_servers = [
        PersistentServer(
            max_count=64,
            is_state_based=True,
            tag='',
            _sync_init=False,
            imports=["Mathlib", "Aesop"],
            project_path=project_root,
            core_options=CORE_OPTIONS,
            timeout=300,
        ) for _ in range(num_concurrency)
    ]
    proof_gen_clients = [
        AsyncOpenAI(
            base_url=proof_gen_base_url,
            api_key=proof_gen_api_key
        ) for (proof_gen_base_url, proof_gen_api_key) in zip(proof_gen_base_urls, proof_gen_api_keys)
    ]
    
    available_agents = [
        ProblemEvaluator(
            clients=clients,
            models=models,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            try_num=try_num,
            kc_estimation_mode=kc_estimation_mode,
        ) for _ in range(num_concurrency // len(proof_gen_clients)) for clients, models in rotate(proof_gen_clients, proof_gen_model_names)
    ]

    tasks = []
    for (i, sample) in enumerate(finished_list):
        if (len(sample.formal_statement or '') > 0) and sample.metainfo.get('is_solution_validated', True):    # is submitted and proven
            if all(k not in sample.metainfo.get('eval_old_result', {}).keys() for k in ['provers', 'proofs', 'KC', 'prove_token_usage']) and all(k not in sample.metainfo.get('eval_result', {}).keys() for k in ['provers', 'proofs', 'KC', 'prove_token_usage']):   # KC not evaluated
                if all(p is None for p in sample.metainfo.get('eval_old_result', {}).get('falsify_proofs', [None])) and all(p is None for p in sample.metainfo.get('eval_result', {}).get('falsify_proofs', [None])):   # not falsified
                    tasks.append(i)
    
    logger.info(f'Evaluating {len(tasks)} samples.')

    async def _async_main():
        pending_tasks = set()
        for i, idx in enumerate(tasks):
            if len(pending_tasks) >= num_concurrency:
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
                        result=finished_list[idx],
                        key=idx,
                        available_agents=available_agents,
                        available_servers=available_servers,
                        finished_list=finished_list,
                    )
                )
            )
        if len(pending_tasks) > 0:
            await asyncio.wait(pending_tasks)
        await logger.complete()

    try:
        asyncio.run(_async_main())
        logger.opt(colors=True).info(f'<cyan>main(): All finished.</cyan>')
    except Exception as e:
        logger.error(f"main(): Failed due to Exception {e}\n{traceback.format_exc()}")
    finally:
        try:
            logger.info(f"Finished generation, saving at {osp.join(log_root, log_prefix+now+'.(pkl|jsonl)')}")
            with open(osp.join(log_root, log_prefix+now+'.pkl'), 'wb') as f:
                pickle.dump((conditions_sampled, finished_list), f)
        except Exception as e:
            logger.error(traceback.format_exc())
            import pdb; pdb.set_trace()

if __name__ == '__main__':
    fire.Fire(main)
