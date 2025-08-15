from abc import abstractmethod
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Dict, Set, Any
import collections, unittest
import heapq
import asyncio
import traceback
import json
import regex as re
import itertools as I

from loguru import logger
import networkx as nx
from openai import AsyncOpenAI, NOT_GIVEN
from openai.types.chat.chat_completion import Choice
from easydict import EasyDict
import vllm
from transformers import AutoTokenizer

from common.constants import BANNED_TOKENS, CODEBLOCK_PATTERN, SYSTEM_PROMPT_FPG, FALSIFY_TACTICS
from common.pantograph.server import PersistentServer, TacticFailure, ServerError
from common.pantograph.dataclasses import ProblemGenerationStep, ProblemGenerationProcess, GoalState, TacticDraft, Variable, ProblemGenerationStepCategory
from common.utils import zip_strict, remove_comments, format_forward_solution_step_prompt, normalize_spaces, extract_code, normalize_draft
from agent.proof_generation import ProofSearchResult, ProofSearchAgent

class ProblemGenerationAgent:
    """
    A template autoregessive problem generation agent.
    """

    def __init__(self, *args, **kwargs) -> None:
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning(f'Redundant arguments for {type(self)}: {args} {kwargs}')
    
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
            conditions: Any,
        ) -> ProblemGenerationStep:
        """
        Given the current problem state and conditions, generate the next step for exploration.
        The returned `new_contexts` field should be empty (`None` for submitting and `[]` for introducing / deducing)
        """

    async def reset_async(self):
        """
        Clean garbabge
        """
        await logger.complete()
    
    async def analyze_async(
        self,
        result: ProblemGenerationProcess,
        states: List[GoalState],
        server: PersistentServer,
        tag: str='',
        reassemble_trajectory: bool=False,
    ) -> ProblemGenerationProcess:
        try:
            # Initialize
            # breakpoint()
            steps = result.steps
            assert len(states) == len(steps)
            assert steps[-1].is_submitting and steps[-1].step.startswith('submit_answer ')
            for problem_state in states:
                assert len(problem_state.goals) == 1 and problem_state.goals[0].target == 'False', str(problem_state)

            submission_name = steps[-1].step[len('submit_answer '):]
            dependency_graph = nx.DiGraph()
            hard_dependencies_global = []
            fvarid_to_istep: Dict[str, int] = dict()

            # Depednency between proof scripts
            for i_step, cur_step in enumerate(steps[:-1]):
                # 1. Load current state and next state
                problem_state = states[i_step]
                new_problem_state = states[i_step + 1]
                
                assert len(new_problem_state.goals) == 1 and new_problem_state.goals[0].target == 'False', str(new_problem_state)
                
                # 2. Analyze state difference
                for v in cur_step.new_contexts:
                    # assert v.raw_name not in fvarid_to_istep.keys()
                    fvarid_to_istep[v.raw_name] = i_step # Maybe override!
                
                dependency_graph.add_node(i_step)
                
                # 3. (Optional) Validate assumption: forward_state.goals[0].variables is topologically sorted
                tmp_state = problem_state
                while len(tmp_state.goals[0].variables) > 0:
                    tmp_state = await server.goal_tactic_async(tmp_state, 0, f'clear! {tmp_state.goals[0].variables[-1].name}')
                assert str(tmp_state) == '⊢ False', str(tmp_state)
                
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
                        test_tmp_state = await server.goal_tactic_async(new_tmp_state, 0, cur_step.step)
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
                reduced_dependency_graph = nx.algorithms.dag.transitive_reduction(dependency_graph)
                depth_dict = {n : 0 for n in range(len(steps))}
                for u in nx.topological_sort(reduced_dependency_graph):
                    for v in reduced_dependency_graph.successors(u):
                        depth_dict[v] = max(depth_dict[v], depth_dict[u]+1)
                
                reassembled_trajectory = []
                G = reduced_dependency_graph.copy()
                deductive_state = await server.load_statement_async('False')
                
                # TODO: Shall we conduct backward-dfs to collect all nodes that `answer` needs?
                # TODO: the current setting (depth-first) can encourage models to explore!
                # TODO: Ablation on this: Graph pruning (`extract_goal`?)

                while True:
                    available_actions = sorted([n for (n, d) in G.in_degree() if d == 0], key=lambda n : (-depth_dict[n], steps[n].is_introducing))
                    chosen_action = steps[available_actions[0]]
                    reassembled_trajectory.append((deductive_state.goals[0].variables, available_actions[0]))
                    if chosen_action.is_submitting:
                        assert submission_name in [v.name for v in deductive_state.goals[0].variables], f'submission_name={submission_name}, deductive_state={deductive_state}'
                        if not set(deductive_state.goals[0].variables).issubset(set(problem_state.goals[0].variables)):
                            logger.warning(f'analyze_async({tag}): ¬(deductive_state ⊆ forward_state): {deductive_state.goals[0].variables}, {problem_state.goals[0].variables}')
                        break
                    deductive_state = await server.goal_tactic_async(deductive_state, 0, chosen_action.step)
                    G.remove_node(available_actions[0])
                
                result.metainfo['original_trajectory'] = [([v.serialize() for v in S], i_s) for (S, i_s) in result.trajectory]
                result.trajectory = reassembled_trajectory
        except Exception as e:
            logger.warning(f'analyze_async({tag}): Failed due to {repr(e)}')
            return False
        
        return True
        
    async def validate_async(
        self,
        result: ProblemGenerationProcess,
        server: PersistentServer,
        tag: str='',
    ) -> bool:
        try:
            # breakpoint()
            steps = result.steps
            submission_name = steps[-1].step[len('submit_answer '):]
            submission_fvar = [v for v in result.trajectory[-1][0] if v.name == submission_name]
            assert len(submission_fvar) == 1, f'submission_name={submission_name}, new_context={[v.name for v in steps[-1].new_contexts]}'
            submission_fvar = submission_fvar[0]
            
            # Construct statement and proof
            # TODO: Reassemble solution; When reassembling formal solution, shall we avoid implicit variable renaming? 
            #* Maybe we can use fvarid and new_context to locate!
            problem_hypotheses = [f'  ({v.name} : {v.t})' for s in steps if s.is_introducing for v in s.new_contexts]
            formal_solution = '\n\n'.join([s.step for s in steps if s.is_deducing] + [steps[-1].step.replace('submit_answer ', 'exact ')])
            result.formal_statement = 'example\n' + '\n'.join(problem_hypotheses) + '\n: ' + submission_fvar.t + '\n:= sorry'
            result.formal_solution_draft = formal_solution
        except Exception as e:
            logger.warning(f'validate_async({tag}): Initialization failed due to {repr(e)}')
            return False

        # Validate statement and proof
        try:
            formal_statement = '∀\n' + '\n'.join(problem_hypotheses) + '\n, ' + submission_fvar.t
            init_validation_state = await server.load_statement_async(formal_statement, intros=[v.name for s in steps if s.is_introducing for v in s.new_contexts])
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
                assert [(g.name, g.target) for g in cur_problem_state.goals] == [(None, 'False')], 'Error: Strange cur_problem_state: ```' + json.dumps(cur_problem_state.serialize()) + '```'
                
                cur_step = await self.gen_step_async(cur_problem_state, conditions)
                log(f'Search({tag}): {i_trial}/{self.max_search_trials}, Condition {conditions}, State\n{cur_problem_state}\nStep {str(cur_step)}')
                
                if cur_step.is_submitting:
                    try:
                        step_code = remove_comments(cur_step.step).strip()
                        assert step_code.startswith('submit_answer '), step_code
                        submission_name = step_code[len('submit_answer '):]
                        assert submission_name in [v.name for v in cur_problem_state.goals[0].variables], f'submission_name={submission_name}, cur_problem_state={cur_problem_state}'
                    except:
                        logger.debug(f'Search({tag}): {i_trial}/{self.max_search_trials}, step {cur_step.category} failed due to {repr(e)}')
                    
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
                    if cur_step.is_deducing:
                        # Validate step: 'deducing' should contain no sorries.
                        new_problem_state = await server.goal_tactic_async(cur_problem_state, 0, TacticDraft('by\n' + cur_step.step + '\nsorry'))
                    elif cur_step.is_introducing:
                        new_problem_state = await server.goal_tactic_async(cur_problem_state, 0, cur_step.step)
                        for tac in FALSIFY_TACTICS: # TODO: Stricter falsify check? (e.g. using prover model) - Maybe final check?
                            falsify_problem_state = await server.goal_tactic_async(new_problem_state, 0, 'try ' + tac)
                            assert not falsify_problem_state.is_solved, 'Introduced contradiction'
                    else:
                        raise RuntimeError(cur_step)
                    assert len(new_problem_state.goals) == 1 and new_problem_state.goals[0].target == 'False', str(new_problem_state)
                except Exception as e:
                    logger.debug(f'Search({tag}): {i_trial}/{self.max_search_trials}, step {cur_step.category} failed due to {repr(e)}')
                    continue


                # If cur_step is successfully executed, add it.
                cur_step.new_contexts = [
                    v for v in new_problem_state.goals[0].variables if
                        v.raw_name not in {vv.raw_name for vv in cur_problem_state.goals[0].variables}
                        # v not in forward_state.goals[0].variables
                ]
                if len(cur_step.new_contexts) == 0:
                    logger.warning(f'Search({tag}): Unused step: {str(cur_step)}')

                states.append(new_problem_state)
                steps.append(cur_step)
                cur_problem_state = new_problem_state
        
        except Exception as e:
            logger.error(f'Search({tag}): {i_trial}/{self.max_search_trials}, fatal error```{[traceback.format_exc()]}```')

        logger.info(f'Search({tag}): search finished with {i_trial} expansions.')
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
            conditions: Any,
        ) -> str:
        """
        Given the current state and conditions, try at most `self.num_max_samples_per_trial` times to generate one step.
        """
        # Generate tactics
        prompt = self.gen_prompt(state=state, conditions=conditions)
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
            except Exception as e:
                logger.debug(f'parse_step(): Failed due to {repr(e)}')
                continue
            step_code = step.step
            if any(banned_token in step_code for banned_token in BANNED_TOKENS[1:]):   # Assuming the first banned token is `sorry`
                logger.warning(f'gen_steps_async(): Tactic `{step_code}` rejected due to bannded token.')
                continue
            return step
        raise RuntimeError('LLM calling budget exceeded')

class SFT_LLMAutoregressiveProblemGenerationAgent(LLMAutoregressiveProblemGenerationAgent):
    def gen_prompt(
        self,
        state: GoalState,
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
