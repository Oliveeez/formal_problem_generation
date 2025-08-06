import sys
import os
import os.path as osp
import json
import pickle
import regex as re
import collections as C
import itertools as I
import random
import traceback
import asyncio
import multiprocessing as mp
from datetime import datetime
from typing import List, Tuple, Dict
import time

import msgspec
import fire
from openai import AsyncOpenAI, NOT_GIVEN
from loguru import logger
import networkx as nx
from networkx.readwrite import node_link_data

from common.constants import CORE_OPTIONS
from common.utils import chunk_list, add_one_to_port, remove_comments, normalize_draft, parse_idents, remove_min_whitespace, solution_decompose
from common.pantograph.dataclasses import TacticDraft, SolutionAutoformalizationResult, ProblemGenerationStep, ProblemGenerationProcess
from common.pantograph.server import PersistentServer, TacticFailure
from common.pantograph.solving_server import PersistentPropSolvingServer
from agent.proof_search import SFT_NALP_LLMProofSearchAgent
from agent.solution_autoformalization import SolutionAutoformalizer, SolutionAutoformalizationResult

N_CONCURRENCY_PER_WORKER = 8

async def async_worker(
        sample: SolutionAutoformalizationResult,
        split_base_cnt: int,
        idx: str,
        available_servers: List[PersistentPropSolvingServer],
        finished_list: List[ProblemGenerationProcess],
        exceptions_dict: Dict
    ):
    server = available_servers.pop()
    server.set_tag(str(split_base_cnt+idx))
    time_start = time.time()
    logger.info(f'async_worker({split_base_cnt+idx}): Start parsing...')
    
    try:
        # Load data point
        if 'solution_state_transition' in sample.metainfo.keys():   # Cycle 0, 1, 2
            solution_blocks = [s[1] for s in sample.metainfo['solution_state_transition']]
            sample.metainfo.pop('solution_state_transition')
        else:
            solution_blocks = solution_decompose(sample.formal_solution_draft)   # Cycle 3
        formal_proofs = sample.formal_proofs[:]
        # logger.debug(f'async_worker({split_base_cnt+idx}): sample.formal_statement:\n{sample.formal_statement}')

        # Sanity check
        solution_draft_normalized = normalize_draft('\n'.join([s for s in solution_blocks]))
        matches = list(re.finditer(':= sorry', solution_draft_normalized))
        assert len(matches) == len(formal_proofs), f'`len(matches) == len(formal_proofs)` failed because {len(matches)} != {len(formal_proofs)}, unable to prune'

        # Parse submission
        action = remove_comments(solution_blocks[-1]).strip()
        assert action.startswith('exact '), action
        submission_name = action[len('exact '):]
        # logger.debug(f'async_worker({split_base_cnt+idx}): submission_name: {submission_name}')

        # Initialize
        forward_state = await server.init_forward_reasoning_state_async(sample)
        assert len(forward_state.goals) == 1, str(forward_state)
        assert all('✝' not in v.name for v in forward_state.goals[0].variables), str(forward_state)

        dependency_graph = nx.DiGraph()
        hard_dependencies_global = []
        parsed_steps = [
            ProblemGenerationStep(
                step_draft=f'have {v.name} : {v.t} := sorry' if v.v is None else f'let {v.name} : {v.t} := {v.v}',
                proof=None,
                new_contexts=[v]
            ) for v in forward_state.goals[0].variables
        ]
        fvarid_to_istep = {
            v.raw_name : i for (i, v) in enumerate(forward_state.goals[0].variables)
        }
        i_proof = 0

        # Add dependencies between current `parsed_steps` (hypotheses)
        dependency_graph.add_nodes_from(range(len(parsed_steps)))
        for (i, v) in enumerate(parsed_steps):
            idents = parse_idents(v.new_contexts[0].t)
            for (j, u) in enumerate(parsed_steps[:i]):
                if u.new_contexts[0].name in idents or any(ident.startswith(u.new_contexts[0].name + '.') for ident in idents):
                    # edge (u, v): v depends on u
                    dependency_graph.add_edge(j, i)

        # Depednency between proof scripts
        for i_step, draft_step in enumerate(solution_blocks[:-1]):
            # 1. Execute current step
            normalized_draft_step = normalize_draft(draft_step)
            if 'sorry' in parse_idents(normalized_draft_step):
                new_forward_state = await server.tactic_server.goal_tactic_async(forward_state, 0, TacticDraft('by\n' + normalized_draft_step + '\nsorry'))
            else:
                new_forward_state = await server.tactic_server.goal_tactic_async(forward_state, 0, normalized_draft_step)

            assert new_forward_state.goals[-1].target == 'False', str(new_forward_state)
            n_sorries = len(new_forward_state.goals) - 1
            for p in sample.formal_proofs[i_proof:i_proof+n_sorries]:
                new_forward_state = await server.tactic_server.goal_tactic_async(new_forward_state, 0, '{\n' + '\n'.join([remove_min_whitespace(s[1]) for s in p.proof]) + '\n}')
            
            assert len(new_forward_state.goals) == 1 and new_forward_state.goals[0].target == 'False', str(new_forward_state)
            
            # 2. Analyze state difference
            new_contexts = [
                v for v in new_forward_state.goals[0].variables if
                    v.raw_name not in {vv.raw_name for vv in forward_state.goals[0].variables}
                    # v not in forward_state.goals[0].variables
            ]
            if len(new_contexts) == 0:
                logger.warning(f'async_worker({split_base_cnt+idx}): Unused step: {[normalized_draft_step]}')
            for v in new_contexts:
                # assert v.raw_name not in fvarid_to_istep.keys()
                fvarid_to_istep[v.raw_name] = len(parsed_steps) # Maybe override!
            
            # 3.1 Add parsed step
            cur_step = ProblemGenerationStep(
                step_draft=draft_step,
                proof=['\n'.join([remove_min_whitespace(s[1]) for s in p.proof]) for p in sample.formal_proofs[i_proof:i_proof+n_sorries]],
                new_contexts=new_contexts
            )
            # logger.debug(f'async_worker({split_base_cnt+idx}): Step: {cur_step.step}')
            parsed_steps.append(cur_step)
            dependency_graph.add_node(len(parsed_steps)-1)
            i_proof += n_sorries
            # 3.2 Coarse-grained dependency
            # - Case 1. types in new_contexts
            # - Case 2. proofs
            
            # 4. (Optional) Validate assumption: forward_state.goals[0].variables is topologically sorted
            tmp_parsing_state = forward_state
            while len(tmp_parsing_state.goals[0].variables) > 0:
                tmp_parsing_state = await server.tactic_server.goal_tactic_async(tmp_parsing_state, 0, f'clear! {tmp_parsing_state.goals[0].variables[-1].name}')
            assert str(tmp_parsing_state) == '⊢ False', str(tmp_parsing_state)
            
            # 5. Analyze dependency
            soft_dependencies = set()    # Set of fVarId. Removing which will corrupt other variables
            hard_dependencies = set()    # Set of fVarId. Removing which will make the current step unable to prove
            # Try removing `v` and re-executing cur_step
            # Assumption: tmp_parsing_state.goals[0].variables is topologically sorted
            tmp_parsing_state = forward_state
            
            for v in forward_state.goals[0].variables:
                assert v.raw_name not in soft_dependencies and v.raw_name not in hard_dependencies, f'v.raw_name={v.raw_name}, soft_dependencies={soft_dependencies}, hard_dependencies={hard_dependencies}'
                # 5.1. Find v
                v_to_remove = [vv for vv in tmp_parsing_state.goals[0].variables if vv.raw_name == v.raw_name]
                if len(v_to_remove) == 0:
                    continue
                assert len(v_to_remove) == 1, str(v_to_remove)    # `tmp_parsing_state` is constructed by iteratively removing variables in forward_state, thus must find exactly one
                v_to_remove = v_to_remove[0]
                
                # 5.2. Try removing `v`
                if '✝' not in v_to_remove.name:
                    try:
                        new_tmp_parsing_state = await server.tactic_server.goal_tactic_async(tmp_parsing_state, 0, f'clear! {v_to_remove.name}')
                    except TacticFailure as e:
                        soft_dependencies.add(v_to_remove.raw_name)
                        logger.warning(f'async_worker({split_base_cnt+idx}): Cannot remove {v_to_remove} ({[vv.name for vv in parsed_steps[fvarid_to_istep[v_to_remove.raw_name]].new_contexts]})')
                        continue
                else:
                    n_inaccessible_after = 0
                    for vv in reversed(tmp_parsing_state.goals[0].variables):
                        if vv.raw_name == v_to_remove.raw_name:
                            break
                        else:
                            if '✝' in vv.name:
                                n_inaccessible_after += 1
                    assert all(vv.name != '_TMP_NAME_TO_REMOVE' for vv in tmp_parsing_state.goals[0].variables), str(tmp_parsing_state)
                    
                    new_tmp_parsing_state = await server.tactic_server.goal_tactic_async(tmp_parsing_state, 0, f'rename_i _TMP_NAME_TO_REMOVE' + ' _' * n_inaccessible_after)
                    
                    all_to_temove = [vv for vv in new_tmp_parsing_state.goals[0].variables if vv.name == '_TMP_NAME_TO_REMOVE']
                    assert len(all_to_temove) == 1 and all_to_temove[0].raw_name == v_to_remove.raw_name, f'all_to_temove={all_to_temove}, v_to_remove={v_to_remove}'
                    
                    try:
                        new_tmp_parsing_state = await server.tactic_server.goal_tactic_async(new_tmp_parsing_state, 0, f'clear! _TMP_NAME_TO_REMOVE')
                        # Try clear!
                    except TacticFailure as e:
                        soft_dependencies.add(v_to_remove.raw_name)
                        logger.warning(f'async_worker({split_base_cnt+idx}): Cannot remove {v_to_remove} ({[vv.name for vv in parsed_steps[fvarid_to_istep[v_to_remove.raw_name]].new_contexts]})')
                        continue
                
                # 5.3. Try executing cur_step
                try:
                    test_tmp_parsing_state = await server.tactic_server.goal_tactic_async(new_tmp_parsing_state, 0, cur_step.step)
                    tmp_parsing_state = new_tmp_parsing_state
                except TacticFailure as e:
                    hard_dependencies.add(v_to_remove.raw_name)
                    hard_dependencies_global.append((parsed_steps[fvarid_to_istep[v_to_remove.raw_name]], cur_step))
                    # logger.debug(f'async_worker({split_base_cnt+idx}): {[vv.name for vv in cur_step.new_contexts]} depends on {[vv.name for vv in parsed_steps[fvarid_to_istep[v_to_remove.raw_name]].new_contexts]}')
                    continue
                # logger.info(f'Removed {v_to_remove} ({[vv.name for vv in parsed_steps[fvarid_to_istep[v_to_remove.raw_name]].new_contexts]})')
            # logger.info(f'Final removing state: {test_tmp_parsing_state}')
            
            # 6. Iteration end
            if len(soft_dependencies) > 0:
                logger.warning(f'async_worker({split_base_cnt+idx}): len(soft_dependencies) > 0: {soft_dependencies}')
            for d in I.chain(soft_dependencies, hard_dependencies):
                # edge (u, v): v depends on u
                # logger.info(f'async_worker({split_base_cnt+idx}): Adding dependency: {[vv.name for vv in parsed_steps[fvarid_to_istep[d]].new_contexts]} -> {[vv.name for vv in cur_step.new_contexts]}')
                dependency_graph.add_edge(fvarid_to_istep[d], len(parsed_steps)-1)
            
            forward_state = new_forward_state

        assert i_proof == len(formal_proofs), f'i_proof={i_proof}, len(formal_proofs)={len(formal_proofs)}'
        assert submission_name in [v.name for v in parsed_steps[-1].new_contexts], f'submission_name={submission_name}, new_context={[v.name for v in parsed_steps[-1].new_contexts]}'

        # Add submission step
        submission_step = ProblemGenerationStep(
            step_draft=f'submit_answer {submission_name}',
            proof=None,
            new_contexts=None
        )
        dependency_graph.add_node(len(parsed_steps))
        for (i, s) in reversed(list(enumerate(parsed_steps))):
            if submission_name in [v.name for v in s.new_contexts]:
                dependency_graph.add_edge(i, len(parsed_steps))
                break
        assert dependency_graph.in_degree(len(parsed_steps)) == 1, f'dependency_graph.in_degree(submission_step)={dependency_graph.in_degree(len(parsed_steps))}'
        parsed_steps.append(submission_step)

        # Reduce transitive edges; Compute depths
        reduced_dependency_graph = nx.algorithms.dag.transitive_reduction(dependency_graph)
        depth_dict = {n : 0 for n in range(len(parsed_steps))}
        for u in nx.topological_sort(reduced_dependency_graph):
            for v in reduced_dependency_graph.successors(u):
                depth_dict[v] = max(depth_dict[v], depth_dict[u]+1)

        # Reassemble trajectories
        reassembled_trajectory = []
        G = reduced_dependency_graph.copy()
        deductive_state = await server.tactic_server.load_statement_async('False')
        
        # TODO: Shall we conduct backward-dfs to collect all nodes that `answer` needs?
        # TODO: the current setting (depth-first) can encourage models to explore!
        # TODO: Ablation on this: Graph pruning

        while True:
            available_actions = sorted([n for (n, d) in G.in_degree() if d == 0], key=lambda n : (-depth_dict[n], parsed_steps[n].is_introducing))
            chosen_action = parsed_steps[available_actions[0]]
            reassembled_trajectory.append((deductive_state.goals[0].variables, available_actions[0]))
            if chosen_action.is_submitting:
                assert submission_name in [v.name for v in deductive_state.goals[0].variables], f'submission_name={submission_name}, deductive_state={deductive_state}'
                if not set(deductive_state.goals[0].variables).issubset(set(forward_state.goals[0].variables)):
                    logger.warning(f'¬(deductive_state ⊆ forward_state): {deductive_state.goals[0].variables}, {forward_state.goals[0].variables}')
                break
            try:
                deductive_state = await server.tactic_server.goal_tactic_async(deductive_state, 0, chosen_action.step)
            except:
                import pdb; pdb.set_trace()
            G.remove_node(available_actions[0])
        
        finished_list[idx] = ProblemGenerationProcess(
            informal_problem=sample.informal_problem,
            informal_answer=sample.informal_answer,
            informal_solution=sample.informal_solution,
            header=sample.header,
            formal_statement=sample.formal_statement,
            formal_solution_draft=sample.formal_solution_draft,
            formal_proofs=[
                '\n'.join([remove_min_whitespace(s[1]) for s in p.proof]) for p in sample.formal_proofs
            ],
            steps=parsed_steps,
            dependencies=[e for e in dependency_graph.edges],
            trajectory=reassembled_trajectory,
            metainfo=json.dumps(sample.metainfo)
        )
    except Exception as e:
        logger.error(f'async_worker({split_base_cnt+idx}): Exception {e}\n{traceback.format_exc()}')
        assert split_base_cnt+idx not in exceptions_dict.keys(), f'exceptions_dict[split_base_cnt+idx]={exceptions_dict[split_base_cnt+idx]}, e={e}'
        exceptions_dict[split_base_cnt+idx] = e
    
    if finished_list[idx] is not None:
        logger.opt(colors=True).info(f'<green>async_worker({split_base_cnt+idx}): Succeeded in {time.time() - time_start:.2f} s.</green>')
    else:
        logger.error(f'async_worker({split_base_cnt+idx}): Failed in {time.time() - time_start:.2f} s.')
    server.set_tag('')
    available_servers.append(server)


def worker(args: Tuple) -> int:
    project_path, working_root, split_base_cnt = args
    enc = msgspec.msgpack.Encoder()
    
    if not osp.exists(osp.join(working_root, f'raw_chunk_{split_base_cnt}.pkl')):
        logger.opt(colors=True).info(f'<green>worker({split_base_cnt}): raw pkl does not exist, exiting....</green>')
        return
    if osp.exists(osp.join(working_root, f'done_chunk_{split_base_cnt}.msgp')):
        logger.opt(colors=True).info(f'<green>worker({split_base_cnt}): done pkl already exists, exiting....</green>')
        return

    with open(osp.join(working_root, f'raw_chunk_{split_base_cnt}.pkl'), 'rb') as f:
        data_to_process = pickle.load(f)
    
    finished_list = [None for _ in range(len(data_to_process))]
    exceptions_dict = dict()

    available_parsers = [
        PersistentPropSolvingServer(
            imports=["Mathlib", "Aesop"],
            project_path=project_path,
            timeout=120,
            tag='',
            _sync_init=False,
        ) for _ in range(N_CONCURRENCY_PER_WORKER)
    ]
    logger.info(f'worker({split_base_cnt}): Initialized, processing {len(data_to_process)} samples.')

    async def _async_main():
        pending_tasks = set()
        for i, d in enumerate(data_to_process):
            if len(pending_tasks) >= N_CONCURRENCY_PER_WORKER:
                done_tasks, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done_tasks:
                    if task.exception() is not None:
                        logger.error(f"worker({split_base_cnt}): Exception occurred: {task.exception()}")
                        for pending_task in pending_tasks:
                            pending_task.cancel()
                        # import ipdb; ipdb.set_trace()
            pending_tasks.add(
                asyncio.create_task(
                    async_worker(
                        sample=d,
                        split_base_cnt=split_base_cnt,
                        idx=i,
                        available_servers=available_parsers,
                        finished_list=finished_list,
                        exceptions_dict=exceptions_dict,
                    )
                )
            )
        if len(pending_tasks) > 0:
            await asyncio.wait(pending_tasks)
        await logger.complete()

    try:
        asyncio.get_event_loop().run_until_complete(_async_main())
        logger.opt(colors=True).info(f'<cyan>worker({split_base_cnt}): All finished.</cyan>')
    except Exception as e:
        logger.error(f"worker({split_base_cnt}): Failed due to Exception {e}\n{traceback.format_exc()}")
    
    with open(osp.join(working_root, f'done_chunk_{split_base_cnt}.msgp'), 'wb') as f:
        f.write(enc.encode(finished_list))
    with open(osp.join(working_root, f'exceptions_chunk_{split_base_cnt}.pkl'), 'wb') as f:
        pickle.dump(exceptions_dict, f)
    logger.info(f'worker({split_base_cnt}): Exiting.')

    return len(exceptions_dict)

def load_and_split(working_root: str, reverse_order: bool):
    all_splits = set(
            [
                int(n[len('raw_chunk_'):-len('.pkl')]) for n in os.listdir(working_root) if n.startswith('raw_chunk_') and n.endswith('.pkl')
            ]
        )
    assert len(all_splits) != 0
    done_splits = set(
            [
                int(n[len('done_chunk_'):-len('.msgp')]) for n in os.listdir(working_root) if n.startswith('done_chunk_') and n.endswith('.msgp')
            ]
        )
    assert done_splits.issubset(all_splits)
    remaining = sorted(all_splits - done_splits)
    if reverse_order:
        remaining.reverse()
    logger.info(f'load_and_split(): Loaded {len(remaining)} remaining splits.')
    return remaining


def main(
    working_root: str='/cache/data/cycle0123_succeeded',
    project_path: str='/home/ma-user/workspace/fps_pantograph/formal_problem_solving/data/MiniF2F',
    use_mp: bool=True,
    dry_run: bool=False,
    num_processes: int=1,
    seed: int=42,
    reverse_order: bool=False,
):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    saved_args = {**locals()}
    random.seed(seed)
    os.makedirs(working_root, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level='INFO')
    logger.add(osp.join(working_root, f'{now}.log'))
    logger.info(f'main(): hyperparameters: {saved_args | dict(CORE_OPTIONS=CORE_OPTIONS)}')

    splits = load_and_split(working_root, reverse_order)

    if dry_run:
        return

    try:
        if use_mp:
            futures = []
            fail_cnts = []
            with mp.Pool(processes=num_processes, maxtasksperchild=1) as pool:
                for split_base_cnt in splits:
                    futures.append((split_base_cnt, pool.apply_async(worker,
                        [(project_path, working_root, split_base_cnt)]
                    )))
                pool.close()
                pool.join()
            for f in futures:
                try:
                    fail_cnts.append(f[1].get(timeout=0))
                except Exception as e:
                    print(f"main(): Task {f[0]} failed with error: {repr(e)}")
        else:
            fail_cnts = map(
                    worker, [(
                            project_path, working_root, split_base_cnt
                    ) for split_base_cnt in splits]
                )
    except Exception as e:
        traceback.print_exc()
        import ipdb; ipdb.set_trace()
        fail_cnts = futures.get()
        logger.error(f'main(): Exception occurred: {e}')
    
    logger.opt(colors=True).info(f'<green>main(): All finished with {sum(fail_cnts)} failed.</green>')

if __name__ == '__main__':
    fire.Fire(main)
