import os
import os.path as osp
import pickle
import json
import collections as C
import itertools as I
import random
import regex as re
import multiprocessing as mp
import dataclasses as D
from typing import List, Optional, Dict, Tuple, Set

import networkx as nx
from tqdm import tqdm
from dacite import from_dict
from loguru import logger
import matplotlib.pyplot as plt

from common.constants import BANNED_TOKENS_IN_ANSWER_TYPE, BANNED_TOKENS_IN_SOLVING_STATE, CORE_OPTIONS, FPS_GLOBAL_SETTING, OPEN_HEADER
from common.pantograph.dataclasses import Goal, GoalState, Variable, CompilationUnit, TacticDraft, FormalProblem, SolutionAutoformalizationResult, ProblemGenerationStep
from common.pantograph.server import Server, TacticFailure
from common.pantograph.solving_server import PersistentPropSolvingServer
from common.utils import remove_comments, normalize_spaces, format_forward_solution_step_prompt, replace_span, chunk_list, parse_idents, remove_min_whitespace, normalize_draft


async def worker(sample: SolutionAutoformalizationResult, tag: str):
    solution_transitions = sample.metainfo['solution_state_transition'][:]
    formal_proofs = sample.formal_proofs[:]
    logger.info(f'worker{tag}: sample.formal_statement:\n{sample.formal_statement}')

    state, action = solution_transitions[-1]
    action = remove_comments(action).strip()
    assert action.startswith('exact ')
    submission_name = action[len('exact '):]
    logger.info(f'worker{tag}: submission_name: {submission_name}')

    server = PersistentPropSolvingServer(
        imports=["Mathlib", "Aesop"],
        project_path='/home/ma-user/workspace/formal_problem_generation/formal_problem_generation/data/MiniF2F',
        timeout=300,
        _sync_init=False,
    )
    server.set_tag(f'server({tag})')

    forward_state = await server.init_forward_reasoning_state_async(sample)
    assert len(forward_state.goals) == 1
    assert all('✝' not in v.name for v in forward_state.goals[0].variables)

    solution_draft_normalized = normalize_draft('\n'.join([s[1] for s in solution_transitions]))
    matches = list(re.finditer(':= sorry', solution_draft_normalized))
    assert len(matches) == len(formal_proofs), f'`len(matches) == len(formal_proofs)` failed because {len(matches)} != {len(formal_proofs)}, unable to prune'

    dependency_graph = nx.DiGraph()
    hard_dependencies_global = []
    parsed_steps = [
        ProblemGenerationStep(
            step_draft=f'have {v.name} : {v.t} := sorry' if v.v is None else f'let {v.name} : {v.t} := {v.v}',
            proof=None,
            new_contexts=tuple([v])
        ) for v in forward_state.goals[0].variables
    ]
    fvarid_to_istep = {
        v.raw_name : i for (i, v) in enumerate(forward_state.goals[0].variables)
    }
    i_proof = 0

    # Add dependencies between current `parsed_steps` (hypotheses)
    dependency_graph.add_nodes_from(parsed_steps)
    for (i, v) in enumerate(parsed_steps):
        idents = parse_idents(v.new_contexts[0].t)
        for u in parsed_steps[:i]:
            if u.new_contexts[0].name in idents:
                # edge (u, v): v depends on u
                dependency_graph.add_edge(u, v)

    # Depednency between proof scripts
    for i_step, (_, draft_step) in enumerate(solution_transitions[:-1]):
        # 1. Execute current step
        normalized_draft_step = normalize_draft(draft_step)
        if 'sorry' in parse_idents(normalized_draft_step):
            new_forward_state = await server.tactic_server.goal_tactic_async(forward_state, 0, TacticDraft('by\n' + normalized_draft_step + '\nsorry'))
        else:
            new_forward_state = await server.tactic_server.goal_tactic_async(forward_state, 0, normalized_draft_step)

        assert new_forward_state.goals[-1].target == 'False'
        n_sorries = len(new_forward_state.goals) - 1
        for p in sample.formal_proofs[i_proof:i_proof+n_sorries]:
            new_forward_state = await server.tactic_server.goal_tactic_async(new_forward_state, 0, '{\n' + '\n'.join([remove_min_whitespace(s[1]) for s in p.proof]) + '\n}')
        
        assert len(new_forward_state.goals) == 1 and new_forward_state.goals[0].target == 'False'
        
        # 2. Analyze state difference
        new_contexts = [
            v for v in new_forward_state.goals[0].variables if
                v.raw_name not in {vv.raw_name for vv in forward_state.goals[0].variables}
                # v not in forward_state.goals[0].variables
        ]
        if len(new_contexts) == 0:
            logger.warning(f'Unused step: {[normalized_draft_step]}')
        for v in new_contexts:
            # assert v.raw_name not in fvarid_to_istep.keys()
            fvarid_to_istep[v.raw_name] = len(parsed_steps) # Maybe override!
        
        # 3.1 Add parsed step
        cur_step = ProblemGenerationStep(
            step_draft=draft_step,
            proof=tuple(['\n'.join([remove_min_whitespace(s[1]) for s in p.proof]) for p in sample.formal_proofs[i_proof:i_proof+n_sorries]]),
            new_contexts=tuple(new_contexts)
        )
        logger.info(f'Step: {cur_step.step}')
        parsed_steps.append(cur_step)
        dependency_graph.add_node(cur_step)
        i_proof += n_sorries
        # 3.2 Coarse-grained dependency
        # - Case 1. types in new_contexts
        # - Case 2. proofs
        
        # 4. (Optional) Validate assumption: forward_state.goals[0].variables is topologically sorted
        tmp_parsing_state = forward_state
        while len(tmp_parsing_state.goals[0].variables) > 0:
            tmp_parsing_state = await server.tactic_server.goal_tactic_async(tmp_parsing_state, 0, f'clear! {tmp_parsing_state.goals[0].variables[-1].name}')
        assert str(tmp_parsing_state) == '⊢ False'
        
        # 5. Analyze dependency
        soft_dependencies = set()    # Set of fVarId. Removing which will corrupt other variables
        hard_dependencies = set()    # Set of fVarId. Removing which will make the current step unable to prove
        # Try removing `v` and re-executing cur_step
        # Assumption: tmp_parsing_state.goals[0].variables is topologically sorted
        tmp_parsing_state = forward_state
        
        for v in forward_state.goals[0].variables:
            assert v.raw_name not in soft_dependencies and v.raw_name not in hard_dependencies
            
            # Shall we try clearing steps introducing `v` and all variables dependent on it?
            # No. Because this clearing is in reversed order. If some variable `u` is dependent on `v`
            # - Case 1. `s` does not depend on `u`: `u` is already removed
            # - Case 2. `s` depends on `u`: it does not matter if we still connect `v` with `u`.
            # TODO: 08.05 - Current impl. is not reversed order. try clear!

            # 5.1. Find v
            v_to_remove = [vv for vv in tmp_parsing_state.goals[0].variables if vv.raw_name == v.raw_name]
            if len(v_to_remove) == 0:
                continue
            assert len(v_to_remove) == 1    # `tmp_parsing_state` is constructed by iteratively removing variables in forward_state, thus must find exactly one
            v_to_remove = v_to_remove[0]
            
            # 5.1. Try removing `v`
            if '✝' not in v.name:
                try:
                    new_tmp_parsing_state = await server.tactic_server.goal_tactic_async(tmp_parsing_state, 0, f'clear! {v.name}')
                except TacticFailure:
                    soft_dependencies.add(v.raw_name)
                    logger.warning(f'Cannot remove {v} ({[vv.name for vv in parsed_steps[fvarid_to_istep[v.raw_name]].new_contexts]})')
                    continue
            else:
                n_inaccessible_after = 0
                for vv in reversed(tmp_parsing_state.goals[0].variables):
                    if vv.raw_name == v.raw_name:
                        break
                    else:
                        if '✝' in vv.name:
                            n_inaccessible_after += 1
                assert all(vv.name != '_TMP_NAME_TO_REMOVE' for vv in tmp_parsing_state.goals[0].variables)
                new_tmp_parsing_state = await server.tactic_server.goal_tactic_async(tmp_parsing_state, 0, f'rename_i _TMP_NAME_TO_REMOVE' + ' _' * n_inaccessible_after)
                
                all_to_temove = [vv for vv in new_tmp_parsing_state.goals[0].variables if vv.name == '_TMP_NAME_TO_REMOVE']
                assert len(all_to_temove) == 1 and all_to_temove[0].raw_name == v_to_remove.raw_name
                
                try:
                    new_tmp_parsing_state = await server.tactic_server.goal_tactic_async(tmp_parsing_state, 0, f'clear! _TMP_NAME_TO_REMOVE')
                    # Try clear!
                except TacticFailure:
                    soft_dependencies.add(v.raw_name)
                    logger.warning(f'Cannot remove {v} ({[vv.name for vv in parsed_steps[fvarid_to_istep[v.raw_name]].new_contexts]})')
                    continue
            
            # Step 2. Try executing cur_step
            try:
                test_tmp_parsing_state = await server.tactic_server.goal_tactic_async(new_tmp_parsing_state, 0, cur_step.step)
                tmp_parsing_state = new_tmp_parsing_state
            except TacticFailure:
                hard_dependencies.add(v.raw_name)
                hard_dependencies_global.append((parsed_steps[fvarid_to_istep[v.raw_name]], cur_step))
                logger.info(f'{[vv.name for vv in cur_step.new_contexts]} depends on {[vv.name for vv in parsed_steps[fvarid_to_istep[v.raw_name]].new_contexts]}')
                continue
            logger.info(f'Removed {v} ({[vv.name for vv in parsed_steps[fvarid_to_istep[v.raw_name]].new_contexts]})')
        logger.info(f'Final removing state: {test_tmp_parsing_state}')
        # 6. Iteration end
        if len(soft_dependencies) > 0:
            logger.warning(f'len(soft_dependencies) > 0: {soft_dependencies}')
        for d in I.chain(soft_dependencies, hard_dependencies):
            # edge (u, v): v depends on u
            logger.info(f'Adding dependency: {[vv.name for vv in parsed_steps[fvarid_to_istep[d]].new_contexts]} -> {[vv.name for vv in cur_step.new_contexts]}')
            dependency_graph.add_edge(parsed_steps[fvarid_to_istep[d]], cur_step)
        
        forward_state = new_forward_state

    assert i_proof == len(formal_proofs)

    assert submission_name in [v.name for v in parsed_steps[-1].new_contexts]

    submission_step = ProblemGenerationStep(
        step_draft=f'submit_answer {submission_name}',
        proof=None,
        new_contexts=None
    )
    dependency_graph.add_node(submission_step)
    for s in reversed(parsed_steps):
        if submission_name in [v.name for v in s.new_contexts]:
            dependency_graph.add_edge(s, submission_step)
            break
    assert dependency_graph.in_degree(submission_step) == 1
    parsed_steps.append(submission_step)

    len(dependency_graph.nodes), len(parsed_steps)

    reduced_dependency_graph = nx.algorithms.dag.transitive_reduction(dependency_graph)

    pos = nx.nx_agraph.graphviz_layout(reduced_dependency_graph, prog="dot", args="")
    plt.figure(figsize=(24, 16))

    # pos = nx.bfs_layout(dependency_graph, parsed_steps[0])
    # pos = forest_bfs_layout(dependency_graph, [node for node, in_degree in dependency_graph.in_degree() if in_degree == 0])

    color_map = ['orange' if node.is_submitting else 'cyan' if node.is_deducing else 'green' for node in reduced_dependency_graph.nodes]
    # edge_styles = [
    #     ('--' if e not in hard_dependencies_global else '-') for e in reduced_dependency_graph.edges
    # ]

    labels = {node: '\n'.join(str(v) for v in (node.new_contexts or [node.step])) for node in reduced_dependency_graph}

    nx.draw(reduced_dependency_graph, pos, with_labels=True, labels=labels, node_size=800, font_size=6, node_color=color_map)
    plt.tight_layout()
    plt.savefig(f'./reduced_dependency_graph.{tag}.pdf')
    nx.write_graphml(reduced_dependency_graph, f'./reduced_dependency_graph.{tag}.graphml')

    #* For visualization only
    direct_dependency_graph = nx.algorithms.dag.transitive_reduction(dependency_graph)
    direct_dependency_graph.add_edges_from(hard_dependencies_global) # Add this -> direct dependency graph

    pos = nx.nx_agraph.graphviz_layout(direct_dependency_graph, prog="dot", args="")
    plt.figure(figsize=(24, 16))

    # pos = nx.bfs_layout(dependency_graph, parsed_steps[0])
    # pos = forest_bfs_layout(dependency_graph, [node for node, in_degree in dependency_graph.in_degree() if in_degree == 0])

    color_map = ['orange' if node.is_submitting else 'cyan' if node.is_deducing else 'green' for node in reduced_dependency_graph.nodes]
    # edge_styles = [
    #     ('--' if e not in hard_dependencies_global else '-') for e in reduced_dependency_graph.edges
    # ]

    labels = {node: '\n'.join(str(v) for v in (node.new_contexts or [node.step])) for node in direct_dependency_graph}

    nx.draw(direct_dependency_graph, pos, with_labels=True, labels=labels, node_size=800, font_size=6, node_color=color_map)
    plt.tight_layout()
    plt.savefig(f'./direct_dependency_graph.{tag}.pdf')
    nx.write_graphml(direct_dependency_graph, f'./direct_dependency_graph.{tag}.graphml')
    # ## Exploratory Action Sequence Reassembling

    # TODO: Maybe ablation, rethinking whether the depth works.
    depth_dict = {n : 0 for n in parsed_steps}

    for u in nx.topological_sort(reduced_dependency_graph):
        for v in reduced_dependency_graph.successors(u):
            depth_dict[v] = max(depth_dict[v], depth_dict[u]+1)

    pos = nx.nx_agraph.graphviz_layout(reduced_dependency_graph, prog="dot", args="")
    plt.figure(figsize=(24, 16))


    color_map = ['orange' if node.is_submitting else 'cyan' if node.is_deducing else 'green' for node in reduced_dependency_graph.nodes]

    labels = {n: str(depth_dict[n]) + '\n' + '\n'.join(str(v) for v in (n.new_contexts or [n.step])) for n in reduced_dependency_graph.nodes}

    nx.draw(reduced_dependency_graph, pos, with_labels=True, labels=labels, node_size=800, font_size=6, node_color=color_map)
    plt.tight_layout()
    plt.savefig(f'./reduced_dependency_graph.{tag}.pdf')

    reassembled_trajectory = []
    G = reduced_dependency_graph.copy()
    deductive_state = await server.tactic_server.load_statement_async('False')

    with open(f'./reassembled_trajectory.{tag}.txt', 'w') as f:
        while True:
            available_actions = sorted([n for (n, d) in G.in_degree() if d == 0], key=lambda n : (-depth_dict[n], n.is_introducing))
            chosen_action = available_actions[0]
            reassembled_trajectory.append((deductive_state.goals[0].variables, chosen_action))
            f.write('### State\n```lean4\n' + str(deductive_state.goals[0]) + '\n```\n### Action\n```lean4\n' + chosen_action.step + '\n```\n\n\n')
            if chosen_action.is_submitting:
                assert submission_name in [v.name for v in deductive_state.goals[0].variables]
                if not set(deductive_state.goals[0].variables).issubset(set(forward_state.goals[0].variables)):
                    logger.warning(f'¬(deductive_state ⊆ forward_state): {deductive_state.goals[0].variables}, {forward_state.goals[0].variables}')
                break
            deductive_state = await server.tactic_server.goal_tactic_async(deductive_state, 0, chosen_action.step)
            G.remove_node(chosen_action)

    pos = nx.nx_agraph.graphviz_layout(reduced_dependency_graph, prog="dot", args="")
    plt.figure(figsize=(24, 16))


    color_map = ['orange' if node.is_submitting else 'cyan' if node.is_deducing else 'green' for node in reduced_dependency_graph.nodes]

    order_dict = {
        n : i for i, (s, n) in enumerate(reassembled_trajectory)
    }
    labels = {n: f'{order_dict[n]}, {depth_dict[n]}' + '\n' + '\n'.join(str(v) for v in (n.new_contexts or [n.step])) for n in reduced_dependency_graph.nodes}

    nx.draw(reduced_dependency_graph, pos, with_labels=True, labels=labels, node_size=800, font_size=6, node_color=color_map)
    plt.tight_layout()
    plt.savefig(f'./reduced_dependency_graph.{tag}.pdf')

async def main():
    server = PersistentPropSolvingServer(
        imports=["Mathlib", "Aesop"],
        project_path='/home/ma-user/workspace/formal_problem_generation/formal_problem_generation/data/MiniF2F',
        timeout=300,
        _sync_init=False,
    )
    server.set_tag(f'main()')
    
    if osp.exists('data/MiniF2F/example_deductive_proof.pkl'):
        with open('data/MiniF2F/example_deductive_proof.pkl', 'rb') as f:
            samples = pickle.load(f)
    else:
        with open('data/MiniF2F/example_deductive_proof.lean', 'r') as f:
            examples = f.read().split('example\n')[1:]
        samples = []

        for (i, ex) in enumerate(examples):
            elems = ex.split('\n\n')
            assert elems[0].endswith(':= by')
            formal_statement = 'example\n' + elems[0][:-len(':= by')] + ':= sorry'
            solution_steps = [
                remove_min_whitespace(e) for e in elems[1:] if len(remove_comments(e).strip()) > 0
            ]
            assert solution_steps[-1] == 'exact h_answer'
            sample = await server.load_problem_async(SolutionAutoformalizationResult(
                header=OPEN_HEADER,
                formal_statement=formal_statement
            ))
            sample.metainfo['solution_state_transition'] = [(None, s) for s in solution_steps]
            samples.append(
                sample
            )
            logger.info(f'load_examples({i}/{len(examples)}): formal_statement:\n{formal_statement}')
            print(f'{i}/{len(examples)}')
            print(formal_statement)

        with open('data/MiniF2F/example_deductive_proof.pkl', 'wb') as f:
            pickle.dump(samples, f)

    pending_tasks: Set[asyncio.Task] = set()
    for i_sample, sample in tqdm(enumerate(samples)):
        if len(pending_tasks) >= 16:
            done_tasks, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done_tasks:
                if task.exception() is not None:
                    logger.error(f"Exception occurred: {task.exception()} {task.get_stack()}")
                    # for pending_task in pending_tasks:
                    #     pending_task.cancel()
                    # return
        pending_tasks.add(
            asyncio.create_task(
                worker(sample, i_sample)
            )
        )
    if len(pending_tasks) > 0:
        await asyncio.wait(pending_tasks)
    await logger.complete()


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
