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
from openai.types.chat.chat_completion import ChatCompletion, Choice
from easydict import EasyDict
import vllm

from common.pantograph.server import PersistentServer
from common.utils import remove_comments, extract_code, replace_sorry, decompose_statement

class StatementAutoformalizationAgent:
    """
    A template statement autoformalizer.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning(f'Redundant arguments for {type(self)}: {args} {kwargs}')
    
    @abstractmethod
    async def autoformalize_async(
        informal_statement: str,
        server: Optional[PersistentServer]
    ) -> Optional[Tuple[str, str]]:
        """
        Given an informal statement, generate a formal statement, and optionally validate its syntactical correctness.
        """
        
class LLMStatementAutoformalizationAgent(StatementAutoformalizationAgent):
    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        temperature: Optional[float]=None,
        max_tokens: int=NOT_GIVEN,
        top_k: Optional[int]=None,
        top_p: Optional[float]=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.token_usage = C.defaultdict(int)
        
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.top_p = top_p

    @abstractmethod
    def format_prompt(
        self,
        informal_statement: str,
    ) -> List[Dict[str, str]]:
        """
        Generate a prompt for the autoformalizer
        """

    @abstractmethod
    def parse_result(
        self,
        response: str
    ) -> Tuple[str, str]:
        """
        Parse the autoformalization results
        """

    async def autoformalize_async(
        self,
        informal_statement: str,
        server: Optional[PersistentServer],
        tag: str=''
    ) -> Optional[Tuple[str, str]]:
        breakpoint()
        messages = self.format_prompt(informal_statement=informal_statement)
        try:
            response: ChatCompletion = (await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                stream=False,
                temperature=self.temperature,
                top_p=self.top_p,
                n=1,
                extra_body={'top_k': self.top_k} if self.top_k is not None else None
            ))
        except Exception as e:
            logger.warning(f'autoformalize_async({tag}): Exception {repr(e)}')
            return None
        self.token_usage['completion_tokens'] += response.usage.completion_tokens
        self.token_usage['prompt_tokens'] += response.usage.prompt_tokens
        breakpoint()
        
        try:
            header, stmt_code = self.parse_result(response.choices[0].message.content)
            assert stmt_code.startswith('example') and stmt_code.endswith('\n:= sorry')
            
            if server is not None:
                variables = []
                context, target = decompose_statement(stmt_code)
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
                breakpoint()
                init_state = await server.load_statement_async(
                    statement=(('∀ ' + '\n'.join(context) + '\n, ') if len(context) > 0 else '') + target,
                    intros=[v[0] for v in variables],
                    header=header
                )
                assert not init_state.is_solved, f'init_state={[str(init_state)]}'
        except Exception as e:
            logger.warning(f'autoformalize_async({tag}): Statement parsing/validation failed due to {repr(e)}')
            return None
        
        breakpoint()
        return header, stmt_code
        
class Goedel_LLMStatementAutoformalizationAgent(LLMStatementAutoformalizationAgent):
    THEOREM_NAME = "test_problem"
    
    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        temperature: Optional[float]=0.9,
        max_tokens: int=NOT_GIVEN,
        top_k: Optional[int]=20,
        top_p: Optional[float]=0.95,
        *args,
        **kwargs
    ) -> None:
        super().__init__(client, model_name, temperature, max_tokens, top_k, top_p, *args, **kwargs)

    def format_prompt(
        self,
        informal_statement: str,
    ) -> List[Dict[str, str]]:
        user_prompt_content = (
            f"Please autoformalize the following natural language problem statement in Lean 4. "
            f"Use the following theorem name: {self.THEOREM_NAME}\n"
            f"The natural language statement is: \n"
            f"{informal_statement}"
            f"Think before you provide the lean statement."
        )
        return [
            {"role": "user", "content": user_prompt_content},
        ]

    def parse_result(
        self,
        response: str
    ) -> Tuple[str, str]:
        body = response.split('</think>')[-1]
        code = extract_code(body)
        
        p = remove_comments(code).strip().replace('\nlemma ', '\ntheorem ').replace('\nexample ', '\ntheorem test_problem')
        start_pos = p.find('theorem test_problem')
        assert start_pos != -1, 'Start pos not found'
        intro_code, stmt_code = p[:start_pos], p[start_pos:]
        
        # Parse header
        import_list = []
        open_scoped_list = []
        open_list = []
        option_list = []

        for l in intro_code.splitlines():
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
                logger.debug('Neglecting unexpected line in intro code: ' + l)
        
        header = ''
        if len(open_scoped_list):
            header += 'open scoped ' + ' '.join(t for t in open_scoped_list) + '\n'
        if len(open_list):
            header += 'open ' + ' '.join(t for t in open_list) + '\n'
        if len(option_list):
            header += '\n'.join('set_option ' + t for t in option_list) + '\n'

        # Parse statement
        stmt_code = replace_sorry('example' + stmt_code)
        assert stmt_code.endswith(':= sorry'), f'stmt_code={[stmt_code]}'
        
        return header, stmt_code

class VersatileLLMStatementAutoformalizationAgent(LLMStatementAutoformalizationAgent):
    MODEL_STR_TO_CLS: List[Tuple[str, LLMStatementAutoformalizationAgent]] = [
        ('goedel', Goedel_LLMStatementAutoformalizationAgent),
    ]
    
    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        temperature: Optional[float]=None,
        max_tokens: int=NOT_GIVEN,
        top_k: Optional[int]=None,
        top_p: Optional[float]=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(client, model_name, temperature, max_tokens, top_k, top_p, *args, **kwargs)
        
        model_name = self.model_name[:-1] if self.model_name.endswith('/') else self.model_name
        model_name = model_name.split('/')[-1].lower()
        for (k, v) in self.MODEL_STR_TO_CLS:
            if k in model_name:
                assert all(kk not in model_name for (kk, _) in self.MODEL_STR_TO_CLS if kk != k), f'Ambiguous model: {model_name}'
                logger.info(f'Dispatching {v.__name__} to {model_name}')
                self.format_prompt = v(None, None).format_prompt
                self.parse_result = v(None, None).parse_result
                return
        
        assert False, f'Unable to parse model: {model_name}'
