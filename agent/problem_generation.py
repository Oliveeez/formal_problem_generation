from abc import abstractmethod
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Dict, Set, Any, Optional
import collections, unittest
import heapq
import asyncio
import traceback
import json
import regex as re
import itertools as I
import collections as C

from loguru import logger
import networkx as nx
from openai import AsyncOpenAI, NOT_GIVEN
from openai.types.chat.chat_completion import Choice
from easydict import EasyDict
import vllm
from transformers import AutoTokenizer
import dacite

from common.constants import BANNED_TOKENS, SYSTEM_PROMPT_FPG, FALSIFY_TACTICS, BRACKET_PAIRINGS, CORE_OPTIONS
from common.pantograph.server import PersistentServer, TacticFailure, ServerError
from common.pantograph.dataclasses import ProblemGenerationStep, ProblemGenerationProcess, Goal, GoalState, TacticDraft, Variable, ProblemGenerationStepCategory
from common.utils import zip_strict, remove_comments, format_forward_solution_step_prompt, normalize_spaces, extract_code, normalize_draft, parse_idents, decompose_statement, proof_decompose, generate_submission_name, is_deductive_transition
from agent.proof_generation import ProofSearchResult, ProofSearchAgent

class ProblemGenerationAgent:
    """
    A template autoregessive problem generation agent.
    """

    def __init__(self, *args, **kwargs) -> None:
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning(f'Redundant arguments for {type(self)}: {args} {kwargs}')
    
    @staticmethod
    async def decompose_deductive_steps_async(
        result: ProblemGenerationProcess,
        init_state: GoalState,
        server: PersistentServer,
        tag: str='',
    ) -> Optional[Tuple[List[str], List[List[Goal]]]]:
        # Decompose deductive steps result.formal_solution_draft
        breakpoint()
        try:
            raw_steps = proof_decompose(result.formal_solution_draft)
            deductive_states: List[List[Goal]] = [init_state.goals[:]]
            deductive_steps: List[str] = []
            cur_state = init_state

            # Stage 1. Parse deductive steps
            while len(raw_steps) > 0:
                # Execute cur_step
                cur_step = raw_steps[0]
                next_state = await server.goal_tactic_async(cur_state, 0, cur_step)
                
                if next_state.is_solved:
                    if remove_comments(cur_step).strip().startswith('exact '):
                        # If (solved) and (the final step is `exact`): add cur_step and break
                        raw_steps = []
                        cur_state = next_state
                        deductive_states.append(cur_state.goals[:])
                        deductive_steps.append(cur_step)
                        logger.info(f"decompose_deductive_steps_async({tag}): Detected `exact` submission: {[remove_comments(cur_step).strip()]}")
                        break
                    else:
                        # If (solved) but (the final step is not `exact`): don't add cur_step, don't update state
                        raw_steps = [cur_step]
                        logger.info(f"decompose_deductive_steps_async({tag}): Detected non-`exact` submission: {[remove_comments(cur_step).strip()]}")
                        break   # If the final step is not `exact`, 1) do not add to `steps` - leave it for final submission; 2) do not update `cur_state`
                else:
                    if not is_deductive_transition(cur_state.goals, next_state.goals):
                        # If (not solved) but (not deductive): don't add cur_step, don't update state
                        break
                    else:
                        # If (not solved) and (is deductive): add cur_step and continue
                        raw_steps.pop(0)
                        cur_state = next_state
                        deductive_states.append(cur_state.goals[:])
                        deductive_steps.append(cur_step)

            # Remaining non-deductive steps
            if len(raw_steps) > 0:
                proof_state = cur_state

                submission_name = generate_submission_name([v.name for v in cur_state.goals[0].variables if v.name is not None])
                have_step = f'have {submission_name}: {cur_state.goals[0].target} := by {{\n' + '\n'.join(raw_steps) + '\n}'
                proof_state = await server.goal_tactic_async(proof_state, 0, have_step)
                assert is_deductive_transition(cur_state.goals, proof_state.goals), f'`have {submission_name}` failed due to proof state: ' + str(proof_state)
                deductive_steps.append(have_step)
                deductive_states.append(proof_state.goals[:])
                
                submit_step = f'exact {submission_name}'
                proof_state = await server.goal_tactic_async(proof_state, 0, submit_step)
                assert proof_state.is_solved, f'`exact {submission_name}` failed due to proof state: ' + str(proof_state)
                deductive_steps.append(submit_step)
                deductive_states.append(proof_state.goals[:])

            return deductive_steps, deductive_states
        except Exception as e:
            logger.error(f'decompose_deductive_steps_async({tag}): failed due to {repr(e)}, traceback: {[traceback.format_exc()]}')
            return None
    
    # Stage II. Validate problem generation steps
    @staticmethod
    async def validate_deductive_steps_async(
        result: ProblemGenerationProcess,
        deductive_steps: List[Tuple[str, str]],
        deductive_states: List[List[Goal]],
        server: PersistentServer,
        tag: str='',
        reassemble_trajectory: bool=False,
    ) -> bool:
        try:
            assert len(deductive_states[-1]) == 0 and len(deductive_steps) + 1 != len(deductive_states), f'len(deductive_states[-1])={len(deductive_states[-1])}, len(deductive_steps)={len(deductive_steps)}, len(deductive_states)={len(deductive_states)}'
            
            states: List[GoalState] = []
            steps: List[ProblemGenerationStep] = []
            cur_problem_state = await server.load_statement_async('False')
            states.append(cur_problem_state)
            
            # Execute introducing steps
            assert len(deductive_states[0]) == 1
            init_parsed_goal = deductive_states[0][0]
            var_type_dict = {
                v.name : v.t for v in init_parsed_goal.variables
            }
            var_value_dict = {
                v.name : v.v for v in init_parsed_goal.variables
            }
            
            # Break from formal statement
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
                            # logger.critical(f"validate_deductive_steps_async({tag}): '✝' in name: {[result.formal_statement]}")
                            name = '_'
                        variables.append((name.strip(), var_type or var_type_dict[name.strip()]))
            
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
                step_code = remove_comments(cur_step.step_code)
                # TODO: rename_i 到底会不会改变fvarid?
                try:
                    new_problem_state = await server.goal_tactic_async(cur_problem_state, 0, cur_step.step)
                except (TacticFailure, ServerError):
                    cur_step.step_draft = result.header + cur_step.step_draft
                    new_problem_state = await server.goal_tactic_async(cur_problem_state, 0, cur_step.step)
                assert len(new_problem_state.goals) == 1 and new_problem_state.goals[0].target == 'False', str(new_problem_state)
                idents = set(step_code.split()).union(parse_idents(step_code))
                for banned_token in BANNED_TOKENS[1:]:
                    assert banned_token not in idents, f'Banned token "{banned_token}" in step "{step_code}"'
                
                cur_step.new_contexts = [
                    v for v in new_problem_state.goals[0].variables if
                        v.raw_name not in {vv.raw_name for vv in cur_problem_state.goals[0].variables}  # 缺陷: 存在Bug，有时Tactic application会导致无关的fvar被重新assign，且该tactic和原tactic无依赖关系
                        # v not in cur_problem_state.goals[0].variables   # 缺陷: 可能只是改名了... —— 没事，正好是rename_i的需求！
                        # (v not in cur_problem_state.goals[0].variables) and (v.raw_name not in {vv.raw_name for vv in cur_problem_state.goals[0].variables})  # 缺陷: 对rename_i不友好
                ]
                if len(cur_step.new_contexts) != 1:
                    logger.debug(f'validate_deductive_steps_async({tag}): Introducing step potentially leading name change: {str(cur_step)}, {cur_step.new_contexts}')
                
                states.append(new_problem_state)
                steps.append(cur_step)
                cur_problem_state = new_problem_state
            
            if init_parsed_goal.variables != cur_problem_state.goals[0].variables:
                logger.warning(f'validate_deductive_steps_async({tag}): init_parsed_goal.variables != cur_problem_state.goals[0].variables: {[str(init_parsed_goal), str(cur_problem_state.goals[0])]}')
            
            # Execute deriving steps
            for (step_code, next_parsed_state) in zip(deductive_steps[:-1], deductive_states[1:-1]):
                assert len(next_parsed_state) == 1
                next_parsed_goal = dacite.from_dict(Goal, next_parsed_state[0])
                cur_step = ProblemGenerationStep(   # ProblemGenerationStepCategory.Derive
                    step_draft=normalize_spaces(step_code),
                    proof=[],
                    new_contexts=[]
                )
                
                new_problem_state = await server.goal_tactic_async(cur_problem_state, 0, cur_step.step)
                assert len(new_problem_state.goals) == 1 and new_problem_state.goals[0].target == 'False', str(new_problem_state)
                step_code = remove_comments(cur_step.step_code)
                idents = set(step_code.split()).union(parse_idents(step_code))
                for banned_token in BANNED_TOKENS:
                    assert banned_token not in idents, f'Banned token "{banned_token}" in step "{step_code}"'

                cur_step.new_contexts = [
                    v for v in new_problem_state.goals[0].variables if
                        v.raw_name not in {vv.raw_name for vv in cur_problem_state.goals[0].variables}
                ]
                if len(cur_step.new_contexts) == 0:
                    logger.debug(f'validate_deductive_steps_async({tag}): Unused step: {str(cur_step)}')
                
                states.append(new_problem_state)
                steps.append(cur_step)
                cur_problem_state = new_problem_state
            
                if next_parsed_goal.variables != cur_problem_state.goals[0].variables:
                    logger.debug(f'validate_deductive_steps_async({tag}): next_parsed_goal.variables != cur_problem_state.goals[0].variables: {[str(next_parsed_goal), str(cur_problem_state.goals[0])]}')
            
            # Execute submitting step
            step_code = remove_comments(deductive_steps[-1]).strip()
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
                
                new_problem_state = await server.goal_tactic_async(cur_problem_state, 0, cur_step.step)
                assert len(new_problem_state.goals) == 1 and new_problem_state.goals[0].target == 'False', str(new_problem_state)
                step_code = remove_comments(cur_step.step_code)
                idents = set(step_code.split()).union(parse_idents(step_code))
                for banned_token in BANNED_TOKENS:
                    assert banned_token not in idents, f'Banned token "{banned_token}" in step "{step_code}"'
                
                cur_step.new_contexts = [
                    v for v in new_problem_state.goals[0].variables if
                        v.raw_name not in {vv.raw_name for vv in cur_problem_state.goals[0].variables}
                ]
                if len(cur_step.new_contexts) == 0:
                    logger.debug(f'validate_deductive_steps_async({tag}): Unused step: {str(cur_step)}')
                
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
            is_analyzed = await ProblemGenerationAgent.analyze_async(
                result=result,
                states=states,
                server=server,
                tag=f'{tag}',
                reassemble_trajectory=reassemble_trajectory
            )
        except Exception as e:
            logger.error(f'validate_deductive_steps_async({tag}): failed due to {repr(e)}, traceback: {[traceback.format_exc()]}')
            return None

    @staticmethod
    async def analyze_async(
        result: ProblemGenerationProcess,
        states: List[GoalState],
        server: PersistentServer,
        tag: str='',
        reassemble_trajectory: bool=False,
    ) -> bool:
        # Dependency Analysis
        try:
            # Initialize
            steps = result.steps
            assert len(states) == len(steps)
            assert steps[-1].is_submitting and steps[-1].step_code.startswith('submit_answer ')
            for problem_state in states:
                assert len(problem_state.goals) == 1 and problem_state.goals[0].target == 'False', str(problem_state)

            submission_name = steps[-1].step_code[len('submit_answer '):]
            dependency_graph = nx.DiGraph()
            hard_dependencies_global = []
            fvarid_to_istep: Dict[str, int] = dict()
            # var_to_istep: Dict[Variable, int] = dict()
            # TODO: 要不要var_to_istep?

            # Depednency between proof scripts
            for i_step, cur_step in enumerate(steps[:-1]):
                # 1. Load current state and next state
                problem_state = states[i_step]
                new_problem_state = states[i_step + 1]
                
                assert len(new_problem_state.goals) == 1 and new_problem_state.goals[0].target == 'False', str(new_problem_state)
                
                # 2. Analyze state difference                
                dependency_graph.add_node(i_step)
                
                # 3. (Optional) Validate assumption: forward_state.goals[0].variables is topologically sorted
                # tmp_state = problem_state
                # while len(tmp_state.goals[0].variables) > 0:
                #     for v in tmp_state.goals[0].variables:
                #         if '✝' in v.name:
                #             continue
                #         tmp_state = await server.goal_tactic_async(tmp_state, 0, f'clear! {v.name}')
                #         break
                # assert len(tmp_state.goals) == 1 and len(tmp_state.goals[0].variables) == 0 and tmp_state.goals[0].target == 'False', str(tmp_state)
                
                # 4. Analyze dependency
                soft_dependencies = set()    # Set of fVarId. Removing which will corrupt other variables
                hard_dependencies = set()    # Set of fVarId. Removing which will make the current step unable to prove
                # Try removing `v` and re-executing cur_step
                # Assumption: tmp_parsing_state.goals[0].variables is topologically sorted
                tmp_state = problem_state
                
                for v in problem_state.goals[0].variables:
                    assert v.raw_name not in soft_dependencies and v.raw_name not in hard_dependencies, f'v.raw_name={v.raw_name}, soft_dependencies={soft_dependencies}, hard_dependencies={hard_dependencies}'
                    # 4.1. Find v
                    v_to_remove = [vv for vv in tmp_state.goals[0].variables if vv.raw_name == v.raw_name]
                    if len(v_to_remove) == 0:
                        continue
                    assert len(v_to_remove) == 1, str(v_to_remove)    # `tmp_parsing_state` is constructed by iteratively removing variables in forward_state, thus must find exactly one
                    v_to_remove = v_to_remove[0]
                    
                    # 4.2. Try removing `v`
                    if '✝' not in v_to_remove.name:
                        try:
                            new_tmp_state = await server.goal_tactic_async(tmp_state, 0, f'clear! {v_to_remove.name}')
                        except TacticFailure as e:
                            soft_dependencies.add(v_to_remove.raw_name)
                            logger.warning(f'analyze_async({tag}): Cannot remove {v_to_remove} ({[vv.name for vv in steps[fvarid_to_istep[v_to_remove.raw_name]].new_contexts]})')
                            continue
                    else:
                        n_inaccessible_after = 0
                        for vv in reversed(tmp_state.goals[0].variables):
                            if vv.raw_name == v_to_remove.raw_name:
                                break
                            else:
                                if '✝' in vv.name:
                                    n_inaccessible_after += 1
                        assert all(vv.name != '_TMP_NAME_TO_REMOVE' for vv in tmp_state.goals[0].variables), str(tmp_state)
                        
                        new_tmp_state = await server.goal_tactic_async(tmp_state, 0, f'rename_i _TMP_NAME_TO_REMOVE' + ' _' * n_inaccessible_after)
                        
                        all_to_temove = [vv for vv in new_tmp_state.goals[0].variables if vv.name == '_TMP_NAME_TO_REMOVE']
                        assert len(all_to_temove) == 1 and all_to_temove[0].raw_name == v_to_remove.raw_name, f'all_to_temove={all_to_temove}, v_to_remove={v_to_remove}'
                        
                        try:
                            new_tmp_state = await server.goal_tactic_async(new_tmp_state, 0, f'clear! _TMP_NAME_TO_REMOVE')
                            # Try clear!
                        except TacticFailure as e:
                            soft_dependencies.add(v_to_remove.raw_name)
                            logger.warning(f'analyze_async({tag}): Cannot remove {v_to_remove} ({[vv.name for vv in steps[fvarid_to_istep[v_to_remove.raw_name]].new_contexts]})')
                            continue
                    
                    # 4.3. Try executing cur_step
                    try:
                        test_tmp_state = await server.goal_tactic_async(new_tmp_state, 0, cur_step.step)    # Test whether removing `v_to_remove` will break `cur_step`
                        tmp_state = new_tmp_state
                    except TacticFailure as e:
                        hard_dependencies.add(v_to_remove.raw_name)
                        hard_dependencies_global.append((steps[fvarid_to_istep[v_to_remove.raw_name]], cur_step))
                        # logger.debug(f'analyze_async({tag}): {[vv.name for vv in cur_step.new_contexts]} depends on {[vv.name for vv in parsed_steps[fvarid_to_istep[v_to_remove.raw_name]].new_contexts]}')
                        continue
                    # logger.info(f'Removed {v_to_remove} ({[vv.name for vv in parsed_steps[fvarid_to_istep[v_to_remove.raw_name]].new_contexts]})')
                # logger.info(f'Final removing state: {test_tmp_state}')
                
                # 5. Iteration end
                if len(soft_dependencies) > 0:
                    logger.warning(f'analyze_async({tag}): len(soft_dependencies) > 0: {soft_dependencies}')
                for d in I.chain(soft_dependencies, hard_dependencies):
                    # edge (u, v): v depends on u
                    # logger.info(f'analyze_async({tag}): Adding dependency: {[vv.name for vv in parsed_steps[fvarid_to_istep[d]].new_contexts]} -> {[vv.name for vv in cur_step.new_contexts]}')
                    dependency_graph.add_edge(fvarid_to_istep[d], i_step)
                
                # Update fvar mapping
                for v in cur_step.new_contexts:
                    # assert v.raw_name not in fvarid_to_istep.keys()
                    fvarid_to_istep[v.raw_name] = i_step # Maybe override!
                # problem_state = new_problem_state

            submission_fvar = [v for v in result.trajectory[-1][0] if v.name == submission_name]
            assert len(submission_fvar) == 1, f'submission_name={submission_name}, new_context={[v.name for v in steps[-1].new_contexts]}'
            submission_fvar = submission_fvar[0]
            dependency_graph.add_node(len(steps)-1)
            
            for (i, s) in reversed(list(enumerate(steps))):
                if s.new_contexts is not None and submission_fvar.raw_name in [v.raw_name for v in s.new_contexts]:
                    dependency_graph.add_edge(i, len(steps)-1)
                    break
            assert dependency_graph.in_degree(len(steps)-1) == 1, f'dependency_graph.in_degree(submission_step)={dependency_graph.in_degree(len(steps))}'

            result.dependencies = [e for e in dependency_graph.edges]

            # (Optional) Reassemble trajectories
            if reassemble_trajectory:
                # Reduce transitive edges; Compute depths
                # dependency_graph = nx.algorithms.dag.transitive_reduction(dependency_graph)
                depth_dict = {n : 0 for n in range(len(steps))}
                for u in nx.topological_sort(dependency_graph):
                    for v in dependency_graph.successors(u):
                        depth_dict[v] = max(depth_dict[v], depth_dict[u]+1)
                
                reassembled_trajectory = []
                G = dependency_graph.copy()
                deductive_state = await server.load_statement_async('False')
                
                # TODO: Shall we conduct backward-dfs to collect all nodes that `answer` needs?
                # TODO: the current setting (depth-first) can encourage models to explore!
                # TODO: Ablation on this: Graph pruning (`extract_goal`?)

                while True:
                    available_actions = sorted([n for (n, d) in G.in_degree() if d == 0], key=lambda n : (-depth_dict[n], steps[n].is_introducing))
                    is_success = False
                    for i, action_id in enumerate(available_actions):   # If fail, fall back to other available actions
                        try:
                            chosen_action = steps[action_id]
                            if chosen_action.is_submitting:
                                submission_fvar_re = [v for v in deductive_state.goals[0].variables if v.name == submission_name]
                                assert len(submission_fvar_re) == 1, f'submission_name={submission_name}, deductive_state={deductive_state}'
                                submission_fvar_re = submission_fvar_re[0]
                                assert submission_fvar_re.t == submission_fvar.t, f'submission_fvar_re.t != submission_fvar.t: {submission_fvar_re.t} != {submission_fvar.t}'
                                reassembled_trajectory.append((deductive_state.goals[0].variables, action_id))
                                if not set(deductive_state.goals[0].variables).issubset(set(states[-1].goals[0].variables)):
                                    logger.warning(f'analyze_async({tag}): ¬(deductive_state ⊆ states[-1]): {[str(deductive_state.goals[0])], str(states[-1].goals[0])}')
                                result.metainfo['original_trajectory'] = [([v.serialize() for v in S], i_s) for (S, i_s) in result.trajectory]
                                result.trajectory = reassembled_trajectory
                                return True
                            new_deductive_state = await server.goal_tactic_async(deductive_state, 0, chosen_action.step)
                            reassembled_trajectory.append((deductive_state.goals[0].variables, action_id))
                            is_success = True
                            G.remove_node(action_id)
                            deductive_state = new_deductive_state
                        except Exception as e:
                            logger.debug(f'analyze_async({tag}): [{i}/{len(available_actions)}] available actions failed due to {repr(e)}')
                    if not is_success:
                        raise RuntimeError('All available actions failed.')
                
        except Exception as e:
            logger.warning(f'analyze_async({tag}): Failed due to {repr(e)}')
            logger.debug(f'analyze_async({tag}): Failed traceback {[traceback.format_exc()]}')
            # # reduced_dependency_graph = nx.algorithms.dag.transitive_reduction(dependency_graph)
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(24, 16))
            # pos = nx.nx_agraph.graphviz_layout(dependency_graph, prog="dot", args="")
            # color_map = ['orange' if steps[int(node)].is_submitting else 'cyan' if steps[int(node)].is_deducing else 'green' for node in dependency_graph.nodes]
            # labels = {n: f'{depth_dict[n]}' + '\n' + '\n'.join(str(v) for v in (steps[n].new_contexts or [steps[n].step])) for n in dependency_graph.nodes}
            # # labels = {node: '\n'.join(str(v) for v in (steps[int(node)].new_contexts or [steps[int(node)].step])) for node in reduced_dependency_graph.nodes}
            # # order_dict = {n : i for i, (s, n) in enumerate(reassembled_trajectory)}
            # # labels = {n: f'{order_dict.get(n, "∞")}, {depth_dict[n]}' + '\n' + '\n'.join(str(v) for v in (steps[n].new_contexts or [steps[n].step])) for n in dependency_graph.nodes}
            # nx.draw(dependency_graph, pos, with_labels=True, labels=labels, node_size=800, font_size=6, node_color=color_map)
            # plt.tight_layout()
            # plt.savefig(f'/home/ma-user/workspace/formal_problem_generation/formal_problem_generation/direct_dependency_graph.pdf')
            # import ipdb; ipdb.set_trace()
            return False
        
        return True
    
    @staticmethod
    async def validate_async(
        result: ProblemGenerationProcess,
        server: PersistentServer,
        tag: str='',
    ) -> bool:
        # Compose result.formal_statement and result.formal_solution_draft from result.steps, and validate them
        try:
            assert (result.formal_statement == '') and (result.formal_solution_draft is None)
            steps = result.steps
            submission_name = steps[-1].step_code[len('submit_answer '):]
            submission_fvar = [v for v in result.trajectory[-1][0] if v.name == submission_name]
            assert len(submission_fvar) == 1, f'submission_name={submission_name}, new_context={[v.name for v in steps[-1].new_contexts]}'
            submission_fvar = submission_fvar[0]
            
            # Construct statement and proof
            problem_hypotheses = []
            for s in steps:
                if s.is_introducing:
                    step_code = s.step_code
                    assert step_code.startswith('have ') and step_code.endswith(' := sorry')
                    problem_hypotheses.append('(' + step_code[len('have '):-len(' := sorry')] + ')')
            
            formal_solution = '\n\n'.join([s.step for s in steps if s.is_deducing] + [steps[-1].step.replace('submit_answer ', 'exact ')])
            result.header = ''
            result.formal_statement = 'example\n' + '\n'.join(problem_hypotheses) + '\n: ' + submission_fvar.t + '\n:= sorry'
            result.formal_solution_draft = formal_solution
        except Exception as e:
            logger.warning(f'validate_async({tag}): Initialization failed due to {repr(e)}')
            return False

        # Validate statement and proof
        try:
            formal_statement = '∀\n' + '\n'.join(problem_hypotheses) + '\n, ' + submission_fvar.t
            try:
                init_validation_state = await server.load_statement_async(formal_statement, intros=[v.name for s in steps if s.is_introducing for v in s.new_contexts])
            except TacticFailure:
                open_set = set()
                open_scoped_set = set()
                option_set = set()
                for s in steps:
                    if s.is_introducing:
                        for l in s.header.splitlines():
                            if l.startswith('open scoped'):
                                for elem in l[len('open scoped'):].strip().split():
                                    open_scoped_set.add(elem)
                            elif l.startswith('open'):
                                for elem in l[len('open'):].strip().split():
                                    open_set.add(elem)
                            elif l.startswith('set_option'):
                                option_set.add(l[len('set_option'):].strip())
                result.header = \
                    'open scoped ' + ' '.join(open_scoped_set) + \
                    'open ' + ' '.join(open_set) + \
                    '\n'.join(['set_option ' + t for t in option_set])
                init_validation_state = await server.load_statement_async(formal_statement, intros=[v.name for s in steps if s.is_introducing for v in s.new_contexts], header=result.header)
            result.metainfo['is_statement_validated'] = True
        except Exception as e:
            logger.warning(f'validate_async({tag}): Statement validation failed due to {repr(e)}: {formal_statement}')
            return False
        try:
            final_validation_state = await server.goal_tactic_async(init_validation_state, 0, '{\n' + formal_solution + '\n}')
            assert final_validation_state.is_solved, str(final_validation_state)
            result.metainfo['is_solution_validated'] = True
        except Exception as e:
            logger.warning(f'validate_async({tag}): Solution validation failed due to {repr(e)}: {formal_solution}')
            return False
        
        return True
    
    @abstractmethod
    async def generate_async(
            self,
            conditions: Any,
            server: PersistentServer,
            reassemble_trajectory: bool=False,
            tag: str='',
            verbose: bool=False,
        ) -> ProblemGenerationProcess:
        """
        Problem generation.
        """
    
    async def reset_async(self):
        """
        Clean garbabge
        """
        await logger.complete()

class AutoregressiveProblemGenerationAgent(ProblemGenerationAgent):
    """
    A template autoregessive problem generation agent.
    """

    def __init__(self, max_search_trials: int, *args, **kwargs) -> None:
        super().__init__()
        self.max_search_trials = max_search_trials
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning(f'Redundant arguments for {type(self)}: {args} {kwargs}')

    @abstractmethod
    async def gen_step_async(
            self,
            state: GoalState,
            step_history: List[ProblemGenerationStep],
            conditions: Any,
        ) -> ProblemGenerationStep:
        """
        Given the current problem state and conditions, generate the next step for exploration.
        The returned `new_contexts` field should be empty (`None` for submitting and `[]` for introducing / deducing)
        """
    
    async def generate_async(
            self,
            conditions: Any,
            server: PersistentServer,
            reassemble_trajectory: bool=False,
            tag: str='',
            verbose: bool=False,
        ) -> ProblemGenerationProcess:
        """
        Autoregressive problem generation.
        """
        # Initialize
        assert server.is_automatic(), "Search must be run in automatic mode"
        
        time_start = time.time()
        states: List[GoalState] = []
        steps: List[ProblemGenerationStep] = []
        
        cur_problem_state = await server.load_statement_async('False')
        states.append(cur_problem_state)
        log = logger.info if verbose else logger.debug
        
        # Search
        try:
            i_trial = 0
            while i_trial < self.max_search_trials:
                i_trial += 1
                # assert [(g.name, g.target) for g in cur_problem_state.goals] == [(None, 'False')], 'Error: Strange cur_problem_state: ```' + json.dumps(cur_problem_state.serialize()) + '```'
                
                cur_step = await self.gen_step_async(cur_problem_state, steps, conditions)
                log(f'generate_async({tag}): {i_trial}/{self.max_search_trials}, Condition {conditions}, State\n{cur_problem_state}\nStep {str(cur_step)}')
                
                if cur_step.is_submitting:
                    try:
                        step_code = remove_comments(cur_step.step_code).strip()
                        assert step_code.startswith('submit_answer '), step_code
                        submission_name = step_code[len('submit_answer '):]
                        assert submission_name in [v.name for v in cur_problem_state.goals[0].variables], f'submission_name={submission_name}, cur_problem_state={cur_problem_state}'
                    except:
                        logger.debug(f'generate_async({tag}): {i_trial}/{self.max_search_trials}, step {cur_step.category} failed due to {repr(e)}')
                    
                    steps.append(cur_step)
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
                    is_valid = await self.validate_async(
                        result=result,
                        server=server,
                        tag=tag,
                    )
                    is_analyzed = await self.analyze_async(
                        result=result,
                        states=states,
                        server=server,
                        tag=tag,
                        reassemble_trajectory=reassemble_trajectory
                    )
                    result.metainfo = json.dumps(result.metainfo | {'time_consumption': time.time() - time_start})
                    return result
                
                # Not submitting: deducing or introducing
                try:
                    step_code = remove_comments(cur_step.step_code)
                    idents = set(step_code.split()).union(parse_idents(step_code))
                    if cur_step.is_deducing:
                        # Validate step: 'deducing' should contain no sorries.
                        for banned_token in BANNED_TOKENS:
                            assert banned_token not in idents, f'Banned token "{banned_token}" in step "{step_code}"'
                        new_problem_state = await server.goal_tactic_async(cur_problem_state, 0, cur_step.step)   # Preventing sorry by `TacticDraft('by\n' + cur_step.step + '\nsorry')` may hinder some steps.
                    elif cur_step.is_introducing:
                        for banned_token in BANNED_TOKENS[1:]:
                            assert banned_token not in idents, f'Banned token "{banned_token}" in step "{step_code}"'
                        new_problem_state = await server.goal_tactic_async(cur_problem_state, 0, cur_step.step)
                        for tac in FALSIFY_TACTICS: # TODO: Stricter falsify check? (e.g. using prover model) - Maybe final check?
                            falsify_problem_state = await server.goal_tactic_async(new_problem_state, 0, 'try ' + tac)
                            assert not falsify_problem_state.is_solved, 'Introduced contradiction'
                    else:
                        raise RuntimeError(cur_step)
                    assert len(new_problem_state.goals) == 1 and new_problem_state.goals[0].target == 'False', str(new_problem_state)
                except Exception as e:
                    logger.debug(f'generate_async({tag}): {i_trial}/{self.max_search_trials}, step {cur_step.category} failed due to {repr(e)}')
                    continue


                # If cur_step is successfully executed, add it.
                cur_step.new_contexts = [
                    v for v in new_problem_state.goals[0].variables if
                        v.raw_name not in {vv.raw_name for vv in cur_problem_state.goals[0].variables}
                        # v not in forward_state.goals[0].variables
                ]
                if len(cur_step.new_contexts) == 0:
                    logger.warning(f'generate_async({tag}): Unused step: {str(cur_step)}')

                states.append(new_problem_state)
                steps.append(cur_step)
                cur_problem_state = new_problem_state
        
        except Exception as e:
            logger.error(f'generate_async({tag}): {i_trial}/{self.max_search_trials}, fatal error```{[traceback.format_exc()]}```')

        logger.info(f'generate_async({tag}): search finished with {i_trial} expansions.')
        await self.reset_async()

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
            metainfo=json.dumps({'time_consumption': time.time() - time_start})
        )

        return result


class LLMAutoregressiveProblemGenerationAgent(AutoregressiveProblemGenerationAgent):
    def __init__(
        self,
        gen_client: AsyncOpenAI,
        gen_model_name: str,
        *args,
        max_search_trials: int=100,
        num_max_samples_per_trial: int=32,
        temperature: Optional[float]=None,
        max_tokens: int=NOT_GIVEN,
        **kwargs
    ) -> None:
        super().__init__(max_search_trials)
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning(f'Redundant arguments for {type(self)}: {args} {kwargs}')

        self.gen_client = gen_client
        self.gen_model_name = gen_model_name
        self.num_max_samples_per_trial = num_max_samples_per_trial
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def gen_prompt(
        self,
        state: GoalState,
        step_history: List[ProblemGenerationStep],
        conditions: Any,
    ) -> List[Dict[str, str]]:
        """
        Generate a prompt for the generator
        """

    @abstractmethod
    def parse_step(
        self,
        response: str
    ) -> ProblemGenerationStep:
        """
        Parse the step from generation results
        """

    async def gen_step_async(
            self,
            state: GoalState,
            step_history: List[ProblemGenerationStep],
            conditions: Any,
        ) -> str:
        """
        Given the current state and conditions, try at most `self.num_max_samples_per_trial` times to generate one valid step.
        """
        # Generate tactics
        prompt = self.gen_prompt(state=state, step_history=step_history, conditions=conditions)
        for _ in range(self.num_max_samples_per_trial):
            try:
                if 'internlm' in self.gen_model_name.lower():
                    outputs = (await self.gen_client.chat.completions.create(
                        model=self.gen_model_name,
                        messages=prompt,
                        max_tokens=(self.max_tokens if (self.max_tokens != NOT_GIVEN and self.max_tokens > 0) else NOT_GIVEN),
                        stream=False,
                        temperature=self.temperature,
                        n=1,
                        stop='<|im_end|>'
                    )).choices
                else:
                    outputs = (await self.gen_client.chat.completions.create(
                        model=self.gen_model_name,
                        messages=prompt,
                        max_tokens=(self.max_tokens if (self.max_tokens != NOT_GIVEN and self.max_tokens > 0) else NOT_GIVEN),
                        stream=False,
                        temperature=self.temperature,
                        n=1,
                    )).choices
            except Exception as e:
                logger.debug(f'gen_steps_async(): Failed to generate tactics due to {repr(e)}')
                continue
            
            # Neglect failed generations
            if not outputs[0].finish_reason == 'stop':
                logger.debug(f'gen_steps_async(): Tactic rejected due to abnormal finishing: {outputs[0].finish_reason}')
                continue

            try:
                step = self.parse_step(outputs[0].message.content)
                step_code = remove_comments(step.step_code)
                
                if step.is_deducing:
                    idents = set(step_code.split()).union(parse_idents(step_code))
                    for banned_token in BANNED_TOKENS:
                        assert banned_token not in idents, f'Banned token "{banned_token}" in step "{step_code}"'
                elif step.is_introducing:
                    assert step_code.startswith('have ') and step_code.endswith(' := sorry')
                    idents = set(step_code.split()).union(parse_idents(step_code))
                    for banned_token in BANNED_TOKENS[1:]:  # Assuming the first banned token is `sorry`
                        assert banned_token not in idents, f'Banned token "{banned_token}" in step "{step_code}"'
                else:
                    assert step_code.startswith('submit_answer '), f'Invalid submission step: {step_code}'
                    submission_name = step_code[len('submit_answer '):].strip()
                    assert ' ' not in submission_name and '.' not in submission_name, f'Invalid submission name: {submission_name}'
            except Exception as e:
                logger.debug(f'parse_step(): Failed due to {repr(e)}')
                continue

            return step
        raise RuntimeError('LLM calling budget exceeded')

class SFT_LLMAutoregressiveProblemGenerationAgent(LLMAutoregressiveProblemGenerationAgent):
    def gen_prompt(
        self,
        state: GoalState,
        step_history: List[ProblemGenerationStep],
        conditions: Tuple[str, str],
    ) -> List[Dict[str, str]]:
        """
        Generate a prompt for the generator
        """
        problem_type, source = conditions
        context = ''
        vars_to_format = [v for v in state.goals[0].variables]
        while len(vars_to_format) > 0:
            for i in range(len(vars_to_format)):
                if i + 1 == len(vars_to_format) or not (vars_to_format[i].t == vars_to_format[i+1].t and vars_to_format[i].v is None and vars_to_format[i+1].v is None):
                    break
            if i == 0:
                context += str(vars_to_format[0]) + '\n'
                vars_to_format.pop(0)
            else:
                context += ' '.join([v.name if v.name is not None else "_" for v in vars_to_format[:i+1]]) + f' : {vars_to_format[0].t}\n'
                vars_to_format = vars_to_format[i+1:]
        
        prompt = f'''Given a Lean 4 context, propose the single most natural next step to explore toward a beautiful conclusion — either
- derive a new intermediate fact,
- introduce a fresh variable or hypothesis, or
- submit one of the local facts as the final answer.

Requirements
1. Flavoured {problem_type} and suitable for posting on forums about {source}.
2. Fully formal Lean 4 code (inline comments in natural language are fine for planning and reasoning). Assume `import Mathlib`.


# Lean 4 Context
```lean4
{context.rstrip()}
```
'''
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_FPG
            },
            {
                "role": "user",
                "content": prompt
            }
        ]


    def parse_step(
        self,
        response: str
    ) -> ProblemGenerationStep:
        """
        Parse the step from generation results
        """
        start_pos = max(0, response.find('# Step'))
        step_category, step_code = response[start_pos:].strip().split('\n', 1)
        assert step_category.startswith('# Step ') and step_code.startswith('```') and step_code.endswith('```'), f'Unable to parse step: {response}'
        step_category = ProblemGenerationStepCategory(step_category[len('# Step '):])
        step_code = '\n'.join(step_code.splitlines()[1:-1])

        if step_category == ProblemGenerationStepCategory.Derive:
            normalized_step_draft = normalize_draft(step_code)
            matches = list(re.finditer(':= sorry', normalized_step_draft))
            assert len(matches) == 0, normalized_step_draft
            return ProblemGenerationStep(
                step_draft=step_code,
                proof=[],
                new_contexts=[]
            )
        elif step_category == ProblemGenerationStepCategory.Introduce:
            return ProblemGenerationStep(
                step_draft=step_code,
                proof=None,
                new_contexts=[]
            )
        elif step_category == ProblemGenerationStepCategory.Submit:
                return ProblemGenerationStep(
                    step_draft=step_code,
                    proof=None,
                    new_contexts=None
                )
        else:
            raise RuntimeError(step_category)

        # match step_category:
        #     case ProblemGenerationStepCategory.Derive:
        #         return ProblemGenerationStep(
        #             step_draft=step_code,
        #             proof=[],
        #             new_contexts=[]
        #         )
        #     case ProblemGenerationStepCategory.Introduce:
        #         return ProblemGenerationStep(
        #             step_draft=step_code,
        #             proof=None,
        #             new_contexts=[]
        #         )
        #     case ProblemGenerationStepCategory.Submit:
        #             return ProblemGenerationStep(
        #                 step_draft=step_code,
        #                 proof=None,
        #                 new_contexts=None
        #             )

class SFT_LLMAutoregressiveProblemGenerationAgentV2(SFT_LLMAutoregressiveProblemGenerationAgent):
    def gen_prompt(
        self,
        state: GoalState,
        step_history: List[ProblemGenerationStep],
        conditions: Tuple[str, str],
    ) -> List[Dict[str, str]]:
        """
        Generate a prompt for the generator
        """
        problem_type, source = conditions
        context = ''
        vars_to_format = [v for v in state.goals[0].variables]
        while len(vars_to_format) > 0:
            for i in range(len(vars_to_format)):
                if i + 1 == len(vars_to_format) or not (vars_to_format[i].t == vars_to_format[i+1].t and vars_to_format[i].v is None and vars_to_format[i+1].v is None):
                    break
            if i == 0:
                context += str(vars_to_format[0]) + '\n'
                vars_to_format.pop(0)
            else:
                context += ' '.join([v.name if v.name is not None else "_" for v in vars_to_format[:i+1]]) + f' : {vars_to_format[0].t}\n'
                vars_to_format = vars_to_format[i+1:]
        
        introduced_fvars = []
        for step in step_history:
            if step.is_introducing:
                lines = [l for l in step.step_draft.splitlines() if l != '']
                while len(lines) > 0 and lines[0].split()[0] in ['open', 'set_option']:
                    lines.pop(0)
                step_code = '\n'.join(lines)
                assert step_code.startswith('have ') and step_code.endswith(' := sorry')
                introduced_fvars.append(step_code[len('have '):-len(' := sorry')])
        
        introduced_fvars = '\n'.join(introduced_fvars)
        prompt = f'''Given the introduced variables/hypotheses and the current context in Lean 4, propose the single most natural next step to explore toward a beautiful conclusion — either
- derive a new intermediate fact,
- introduce a fresh variable or hypothesis, or
- submit one of the local facts as the final answer.

Requirements
1. Flavoured {problem_type} and suitable for posting on forums about {source}.
2. Fully formal Lean 4 code (inline comments in natural language are fine for planning and reasoning). Assume `import Mathlib`.

# Intorduced Variables/Hypotheses
```lean4
{introduced_fvars}
```

# Lean 4 Context
```lean4
{context.rstrip()}
```
'''.strip()
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_FPG
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

class LLMWholeProblemGenerationAgent(ProblemGenerationAgent):
    def __init__(
        self,
        statement_gen_client: AsyncOpenAI,
        statement_gen_model: str,
        proof_gen_clients: List[AsyncOpenAI],
        proof_gen_models: List[str],
        *args,
        num_max_samples_per_trial: int=1,
        temperature: Optional[float]=None,
        max_tokens: int=NOT_GIVEN,
        **kwargs
    ) -> None:
        super().__init__()
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning(f'Redundant arguments for {type(self)}: {args} {kwargs}')
        
        assert len(proof_gen_clients) > 0 and len(proof_gen_clients) == len(proof_gen_models)
        
        self.statement_gen_client = statement_gen_client
        self.statement_gen_model = statement_gen_model
        self.proof_gen_clients = proof_gen_clients
        self.proof_gen_models = proof_gen_models
        self.num_max_samples_per_trial = num_max_samples_per_trial
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def format_statement_gen_prompt(self, condition: Any) -> str:
        pass

    @abstractmethod
    def parse_statement_gen_result(self, output: str) -> str:
        pass
    
    async def generate_statement_async(self, conditions: Any) -> str:
        # Generate and parse
        outputs = (await self.statement_gen_client.chat.completions.create(
            model=self.statement_gen_model,
            messages=self.format_statement_gen_prompt(conditions),
            max_tokens=self.max_tokens,
            stream=False,
            temperature=self.temperature,
            n=1,
        )).choices
        return self.parse_statement_gen_result(outputs[0].message.content)
    
    @abstractmethod
    def format_proof_gen_prompt(self, init_state: GoalState) -> str:
        pass
    
    @abstractmethod
    def parse_proof_gen_result(self, output: str) -> str:
        pass
    
    async def generate_one_proof_async(self, init_state: GoalState, server: PersistentServer, client: AsyncOpenAI, model: str) -> Optional[str]:
        try:
            # Generate, parse, and validate one proof
            outputs = (await client.chat.completions.create(
                model=model,
                messages=self.format_proof_gen_prompt(init_state),
                max_tokens=self.max_tokens,
                stream=False,
                temperature=self.temperature,
                n=1,
            )).choices
            proof = self.parse_proof_gen_result(outputs[0].message.content)
            proof_code = remove_comments(proof)
            idents = set(proof_code.split()).union(parse_idents(proof_code))
            # No banned token allowed
            for banned_token in BANNED_TOKENS:
                assert banned_token not in idents, f'Banned token "{banned_token}" in proof "{proof_code}"'
            # Goal should be solved
            final_state = await server.goal_tactic_async(init_state, 0, '{\n' + proof + '\n}')
            assert final_state.is_solved, '!final_state.is_solved'
            return proof_code
        except:
            return None
    
    async def generate_proofs_async(self, init_state: GoalState, server: PersistentServer) -> List[Tuple[str, str]]:
        tasks = [(client, model) for _ in range(self.num_max_samples_per_trial) for (client, model) in zip(self.proof_gen_clients, self.proof_gen_models)]
        results = await asyncio.gather(*[
            self.generate_one_proof_async(
                init_state=init_state,
                server=server,
                client=client,
                model=model
            ) for (client, model) in tasks
        ])
        assert len(tasks) == len(results)
        return [(model, proof) for ((client, model), proof) in zip(tasks, results) if proof is not None]
        
    
    async def generate_async(
            self,
            conditions: Any,
            server: PersistentServer,
            decompose_steps: bool=False,
            reassemble_trajectory: bool=False,
            tag: str='',
            verbose: bool=False,
        ) -> ProblemGenerationProcess:
        """
        Autoregressive problem generation.
        """
        # Initialize
        assert server.is_automatic(), "Search must be run in automatic mode"
        
        time_start = time.time()
        log = logger.info if verbose else logger.debug
        
        # Search
        try:
            # assert [(g.name, g.target) for g in cur_problem_state.goals] == [(None, 'False')], 'Error: Strange cur_problem_state: ```' + json.dumps(cur_problem_state.serialize()) + '```'
            raw_code = await self.generate_statement_async(conditions)
            lines = [l for l in raw_code.strip().splitlines() if l != '']
            load_header = []
            while len(lines) > 0 and lines[0].split()[0] in ['open', 'set_option']:
                load_header.append(lines.pop(0))
            
            load_header = '\n'.join(load_header)
            formal_statement = '\n'.join(lines)
            assert formal_statement.startswith('example\n') and formal_statement.endswith('\n:= sorry')
            
            variables = []
            context, target = decompose_statement(formal_statement)
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
            
            init_state = await server.load_statement_async(
                statement=(('∀ ' + '\n'.join(context) + '\n, ') if len(context) > 0 else '') + target,
                intros=[v[0] for v in variables],
                header=load_header
            )
            formal_proofs = await self.generate_proofs_async(init_state, server)
            formal_proofs.sort(key=lambda s : len(s[-1]))   # Ascending order of proof length (Kolmogorov Complexity)
            assert len(formal_proofs) > 0, 'len(formal_proofs) == 0'
            
            result = ProblemGenerationProcess(
                informal_problem='',
                informal_answer='',
                informal_solution='',
                header=load_header,
                formal_statement=formal_statement,
                formal_solution_draft=formal_proofs[0][-1],
                formal_proofs=[],
                steps=[],
                dependencies=[],
                trajectory=[],
                metainfo=dict()
            )
            
            if decompose_steps:
                is_decomposed = self.decompose_deductive_steps_async(
                    result=result,
                    server=server,
                    tag=tag,
                    reassemble_trajectory=reassemble_trajectory
                )
            
            result.metainfo = json.dumps(
                json.loads(result.metainfo) | {
                    'time_consumption': time.time() - time_start,  
                    'all_formal_proofs': formal_proofs
                }
            )
            logger.info(f'generate_async({tag}): generation succeeded.')
        except Exception as e:
            logger.error(f'generate_async({tag}): generation failed due to{[traceback.format_exc()]}')
            result = ProblemGenerationProcess(
                informal_problem='',
                informal_answer='',
                informal_solution='',
                header='',
                formal_statement='',
                formal_solution_draft='',
                formal_proofs=[],
                steps=[],
                dependencies=[],
                trajectory=[],
                metainfo=dict()
            )

        await self.reset_async()

        return result

class SFT_LLMWholeProblemGenerationAgent(LLMWholeProblemGenerationAgent):
    def format_statement_gen_prompt(self, condition: Tuple[str, str]) -> str:
        problem_type, source = condition
        prompt = f'''Propose a Lean 4 statement that explores toward a beautiful conclusion.

Requirements
1. Flavoured {problem_type} and suitable for posting on forums about {source}.
2. Fully formal Lean 4 code (inline comments in natural language are fine for planning and reasoning). Assume `import Mathlib`.
'''.strip()
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_FPG
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

    def parse_statement_gen_result(self, output: str) -> str:
        return extract_code(output)
    
    def format_proof_gen_prompt(self, init_state: GoalState) -> str:
        header = ("""
import Mathlib
import Aesop

""" + '\n'.join('set_option ' + t.replace('=', ' ') for t in CORE_OPTIONS)).strip()
        prompt = f"""
Assume the following header is executed:
```lean4
{header}
```

Generate a deductive proof for the following Lean 4 proof state:
```lean4
{str(init_state)}
```
""".strip()
        return [
            {
                "role": "user",
                "content": prompt
            }
        ]

    def parse_proof_gen_result(self, output: str) -> str:
        return extract_code(output)

class SFT_DeductiveProofGenerator(SFT_LLMWholeProblemGenerationAgent):
    def __init__(
        self,
        statements: List[str],  # Statements to generate proof
        proof_gen_clients: List[AsyncOpenAI],
        proof_gen_models: List[str],
        *args,
        num_max_samples_per_trial: int=1,
        temperature: Optional[float]=None,
        max_tokens: int=NOT_GIVEN,
        **kwargs
    ) -> None:
        super().__init__(None, None, proof_gen_clients, proof_gen_models, *args, num_max_samples_per_trial=num_max_samples_per_trial, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.statements = statements
        
    async def generate_statement_async(self, conditions: int) -> str:
        return self.statements[conditions]

    async def generate_proofs_async(self, init_state: GoalState, server: PersistentServer) -> List[Tuple[str, str]]:
        for (client, model) in zip(self.proof_gen_clients, self.proof_gen_models):
            proof = await self.generate_one_proof_async(
                init_state=init_state,
                server=server,
                client=client,
                model=model
            )
            if proof is not None:
                return [(model, proof)]
        return None
