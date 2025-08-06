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
from typing import List, Optional, Dict, Tuple

import networkx as nx
from tqdm import tqdm
from dacite import from_dict
from loguru import logger

from common.constants import BANNED_TOKENS_IN_ANSWER_TYPE, BANNED_TOKENS_IN_SOLVING_STATE, CORE_OPTIONS, FPS_GLOBAL_SETTING, OPEN_HEADER
from common.pantograph.dataclasses import Goal, GoalState, Variable, CompilationUnit, TacticDraft, FormalProblem, SolutionAutoformalizationResult, ProblemGenerationStep, ProblemGenerationProcess
from common.pantograph.server import Server, TacticFailure
from common.pantograph.solving_server import PersistentPropSolvingServer
from common.utils import remove_comments, normalize_spaces, format_forward_solution_step_prompt, replace_span, chunk_list, parse_idents, remove_min_whitespace, normalize_draft

FPS_GLOBAL_SETTING['TO_SYNC_ENABLED'] = True

server = PersistentPropSolvingServer(
    imports=["Mathlib", "Aesop"],
    project_path='/home/ma-user/workspace/formal_problem_generation/formal_problem_generation/data/MiniF2F',
    timeout=120,
    _sync_init=False,
)

server.set_tag(f'test')

with open('data/MiniF2F/example_deductive_proof.pkl', 'rb') as f:
    samples = pickle.load(f)
for (idx, sample) in enumerate(samples):
    # Load data point
    assert 'solution_state_transition' in sample.metainfo.keys()   # Cycle 0, 1, 2
    solution_blocks = [s[1] for s in sample.metainfo['solution_state_transition']]
    sample.metainfo.pop('solution_state_transition')
    formal_proofs = sample.formal_proofs[:]
    # logger.debug(f'async_worker({0+idx}): sample.formal_statement:\n{sample.formal_statement}')

    # Sanity check
    solution_draft_normalized = normalize_draft('\n'.join([s for s in solution_blocks]))
    matches = list(re.finditer(':= sorry', solution_draft_normalized))
    assert len(matches) == len(formal_proofs), f'`len(matches) == len(formal_proofs)` failed because {len(matches)} != {len(formal_proofs)}, unable to prune'

    # Parse submission
    action = remove_comments(solution_blocks[-1]).strip()
    assert action.startswith('exact '), action
    submission_name = action[len('exact '):]
    # logger.debug(f'async_worker({0+idx}): submission_name: {submission_name}')

    # Initialize
    forward_state = server.init_forward_reasoning_state(sample)
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
            if u.new_contexts[0].name in idents:
                # edge (u, v): v depends on u
                dependency_graph.add_edge(j, i)

    # Depednency between proof scripts
    for i_step, draft_step in enumerate(solution_blocks[:-1]):
        # 1. Execute current step
        normalized_draft_step = normalize_draft(draft_step)
        if 'sorry' in parse_idents(normalized_draft_step):
            new_forward_state = server.tactic_server.goal_tactic(forward_state, 0, TacticDraft('by\n' + normalized_draft_step + '\nsorry'))
        else:
            new_forward_state = server.tactic_server.goal_tactic(forward_state, 0, normalized_draft_step)

        assert new_forward_state.goals[-1].target == 'False', str(new_forward_state)
        n_sorries = len(new_forward_state.goals) - 1
        for p in sample.formal_proofs[i_proof:i_proof+n_sorries]:
            new_forward_state = server.tactic_server.goal_tactic(new_forward_state, 0, '{\n' + '\n'.join([remove_min_whitespace(s[1]) for s in p.proof]) + '\n}')
        
        assert len(new_forward_state.goals) == 1 and new_forward_state.goals[0].target == 'False', str(new_forward_state)
        
        # 2. Analyze state difference
        new_contexts = [
            v for v in new_forward_state.goals[0].variables if
                v.raw_name not in {vv.raw_name for vv in forward_state.goals[0].variables}
                # v not in forward_state.goals[0].variables
        ]
        if len(new_contexts) == 0:
            logger.warning(f'async_worker({0+idx}): Unused step: {[normalized_draft_step]}')
        for v in new_contexts:
            # assert v.raw_name not in fvarid_to_istep.keys()
            fvarid_to_istep[v.raw_name] = len(parsed_steps) # Maybe override!
        
        # 3.1 Add parsed step
        cur_step = ProblemGenerationStep(
            step_draft=draft_step,
            proof=['\n'.join([remove_min_whitespace(s[1]) for s in p.proof]) for p in sample.formal_proofs[i_proof:i_proof+n_sorries]],
            new_contexts=new_contexts
        )
        # logger.debug(f'async_worker({0+idx}): Step: {cur_step.step}')
        parsed_steps.append(cur_step)
        dependency_graph.add_node(len(parsed_steps)-1)
        i_proof += n_sorries
        # 3.2 Coarse-grained dependency
        # - Case 1. types in new_contexts
        # - Case 2. proofs
        
        # 4. (Optional) Validate assumption: forward_state.goals[0].variables is topologically sorted
        tmp_parsing_state = forward_state
        while len(tmp_parsing_state.goals[0].variables) > 0:
            tmp_parsing_state = server.tactic_server.goal_tactic(tmp_parsing_state, 0, f'clear! {tmp_parsing_state.goals[0].variables[-1].name}')
        assert str(tmp_parsing_state) == '⊢ False', str(tmp_parsing_state)
        
        # 5. Analyze dependency
        soft_dependencies = set()    # Set of fVarId. Removing which will corrupt other variables
        hard_dependencies = set()    # Set of fVarId. Removing which will make the current step unable to prove
        # Try removing `v` and re-executing cur_step
        # Assumption: tmp_parsing_state.goals[0].variables is topologically sorted
        tmp_parsing_state = forward_state
        
        for v in forward_state.goals[0].variables:
            assert v.raw_name not in soft_dependencies and v.raw_name not in hard_dependencies, f'v.raw_name={v.raw_name}, soft_dependencies={soft_dependencies}, hard_dependencies={hard_dependencies}'
            
            # Shall we try clearing steps introducing `v` and all variables dependent on it?
            # No. Because this clearing is in reversed order. If some variable `u` is dependent on `v`
            # - Case 1. `s` does not depend on `u`: `u` is already removed
            # - Case 2. `s` depends on `u`: it does not matter if we still connect `v` with `u`.
            # TODO: 08.05 - Current impl. is not reversed order. try clear!

            # 5.1. Find v
            v_to_remove = [vv for vv in tmp_parsing_state.goals[0].variables if vv.raw_name == v.raw_name]
            if len(v_to_remove) == 0:
                continue
            assert len(v_to_remove) == 1, str(v_to_remove)    # `tmp_parsing_state` is constructed by iteratively removing variables in forward_state, thus must find exactly one
            v_to_remove = v_to_remove[0]
            
            # 5.2. Try removing `v`
            if '✝' not in v.name:
                try:
                    new_tmp_parsing_state = server.tactic_server.goal_tactic(tmp_parsing_state, 0, f'clear! {v.name}')
                except TacticFailure as e:
                    soft_dependencies.add(v.raw_name)
                    import pdb; pdb.set_trace()
                    logger.warning(f'async_worker({0+idx}): Cannot remove {v} ({[vv.name for vv in parsed_steps[fvarid_to_istep[v.raw_name]].new_contexts]})')
                    continue
            else:
                n_inaccessible_after = 0
                for vv in reversed(tmp_parsing_state.goals[0].variables):
                    if vv.raw_name == v.raw_name:
                        break
                    else:
                        if '✝' in vv.name:
                            n_inaccessible_after += 1
                assert all(vv.name != '_TMP_NAME_TO_REMOVE' for vv in tmp_parsing_state.goals[0].variables), str(tmp_parsing_state)
                new_tmp_parsing_state = server.tactic_server.goal_tactic(tmp_parsing_state, 0, f'rename_i _TMP_NAME_TO_REMOVE' + ' _' * n_inaccessible_after)
                
                all_to_temove = [vv for vv in new_tmp_parsing_state.goals[0].variables if vv.name == '_TMP_NAME_TO_REMOVE']
                assert len(all_to_temove) == 1 and all_to_temove[0].raw_name == v_to_remove.raw_name, f'all_to_temove={all_to_temove}, v_to_remove={v_to_remove}'
                
                try:
                    new_tmp_parsing_state = server.tactic_server.goal_tactic(new_tmp_parsing_state, 0, f'clear! _TMP_NAME_TO_REMOVE')
                    # Try clear!
                except TacticFailure as e:
                    soft_dependencies.add(v.raw_name)
                    import pdb; pdb.set_trace()
                    logger.warning(f'async_worker({0+idx}): Cannot remove {v} ({[vv.name for vv in parsed_steps[fvarid_to_istep[v.raw_name]].new_contexts]})')
                    continue
            
            # 5.3. Try executing cur_step
            try:
                test_tmp_parsing_state = server.tactic_server.goal_tactic(new_tmp_parsing_state, 0, cur_step.step)
                tmp_parsing_state = new_tmp_parsing_state
            except TacticFailure as e:
                hard_dependencies.add(v.raw_name)
                hard_dependencies_global.append((parsed_steps[fvarid_to_istep[v.raw_name]], cur_step))
                # logger.debug(f'async_worker({0+idx}): {[vv.name for vv in cur_step.new_contexts]} depends on {[vv.name for vv in parsed_steps[fvarid_to_istep[v.raw_name]].new_contexts]}')
                continue
            # logger.info(f'Removed {v} ({[vv.name for vv in parsed_steps[fvarid_to_istep[v.raw_name]].new_contexts]})')
        # logger.info(f'Final removing state: {test_tmp_parsing_state}')
        
        # 6. Iteration end
        if len(soft_dependencies) > 0:
            import pdb; pdb.set_trace()
            logger.warning(f'async_worker({0+idx}): len(soft_dependencies) > 0: {soft_dependencies}')
        for d in I.chain(soft_dependencies, hard_dependencies):
            # edge (u, v): v depends on u
            # logger.info(f'async_worker({0+idx}): Adding dependency: {[vv.name for vv in parsed_steps[fvarid_to_istep[d]].new_contexts]} -> {[vv.name for vv in cur_step.new_contexts]}')
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
    deductive_state = server.tactic_server.load_statement('False')

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
        deductive_state = server.tactic_server.goal_tactic(deductive_state, 0, chosen_action.step)
        G.remove_node(available_actions[0])

    ret = ProblemGenerationProcess(
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

    with open(f'./reassembled_trajectory_{i}.txt', 'w') as f:
        for state_vars, chosen_action in reassembled_trajectory:
            f.write('### State\n```lean4\n' + str(Goal(
                variables=state_vars,
                target=deductive_state.goals[0].target,
                sibling_dep=None,
                name=None)) + '\n```\n### Action\n```lean4\n' + parsed_steps[chosen_action].step + '\n```\n\n\n')