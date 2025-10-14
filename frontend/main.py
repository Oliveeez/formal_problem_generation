import os
import os.path as osp
import sys
import json
import asyncio
import traceback
from datetime import datetime
from typing import Dict, Set, List, Any, Tuple, Optional, AsyncGenerator, Callable
import pickle
import regex as re
import itertools as I
import collections as C
import uuid
import random
import threading

import vllm     # vLLM should be imported before any 3rd-party libraries, o.w. all Pantograph REPLs are scheduled at the same CPU core.
from openai import AsyncOpenAI, NOT_GIVEN
from tqdm import tqdm
from loguru import logger
from fire import Fire
import aiofiles
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from common.constants import FPS_GLOBAL_SETTING, CORE_OPTIONS
from common.utils import add_one_to_port, starified_downsample
from common.pantograph.server import PersistentServer
from common.pantograph.parsing_server import PersistentParsingServer
from agent.problem_generation import ProblemFalsifier, ProblemGenerationProcess
from agent.problem_generation import AutoregressiveProblemGenerationAgent, SFT_LLMAutoregressiveProblemGenerationAgent, SFT_LLMAutoregressiveProblemGenerationAgentV2, SFT_LLMAutoregressiveProblemGenerationAgentV3
from frontend.informalizer import informalize

AGENT_DICT: Dict[str, AutoregressiveProblemGenerationAgent] = {
    'sft_ar' : SFT_LLMAutoregressiveProblemGenerationAgent,
    'sft_ar_v2': SFT_LLMAutoregressiveProblemGenerationAgentV2,
    'sft_ar_v3': SFT_LLMAutoregressiveProblemGenerationAgentV3
}

app = FastAPI(title="数学问题生成可视化系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境用 *，生产环境指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置静态文件和模板
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# 存储活跃的生成任务
active_tasks: Dict[str, AsyncGenerator] = {}

class GenerateRequest(BaseModel):
    subject: str
    source: str

n_calling = 0
n_calling_lock = threading.Lock()
generate_worker: Callable[[Tuple[str, str, int]], AsyncGenerator] = None

@app.get("/")
async def get_home(request: Request):
    """返回主页面"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket端点，处理实时通信"""
    # logger.info(f'Incoming websocket client `{client_id}`')
    await websocket.accept()
    # logger.info(f'Handling websocket client `{client_id}`')
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["action"] == "start_generation":
                # 开始生成问题
                subject = data["subject"]
                source = data["source"]
                
                # 创建生成器
                global n_calling, generate_worker
                assert generate_worker is not None, "generate_worker is None"
                with n_calling_lock:
                    current_tag = n_calling
                    n_calling += 1
                generator = generate_worker(subject, source, current_tag)
                task_id = str(uuid.uuid4())
                active_tasks[task_id] = generator
                
                # 发送任务ID
                await websocket.send_json({
                    "type": "task_started",
                    "task_id": task_id
                })
                
                # 处理生成过程
                # conditions, conclusion, solution = None, None, None
                cur_state = '    '
                try:
                    async for content in generator:
                        match content:
                            case str(message):
                                # Fatal error
                                await websocket.send_json({
                                    "type": "error",
                                    "message": message
                                })
                            case (str(operation_type), (str(step_content), str(message))):
                                # Step failed
                                await websocket.send_json({
                                    "type": "operation",
                                    "operation_type": operation_type,
                                    "content": '(Rejected)\n' + step_content,
                                    "introduced_context": message,
                                    "current_context": cur_state,
                                    "task_id": task_id
                                })
                            case (str(operation_type), (str(step_content), str(introduced_context), str(cur_problem_state))):
                                # Step succeeded
                                assert cur_problem_state.strip().endswith('⊢ False')
                                cur_state = cur_problem_state.strip()[:-len('⊢ False')].strip()
                                await websocket.send_json({
                                    "type": "operation",
                                    "operation_type": operation_type,
                                    "content": step_content,
                                    "introduced_context": introduced_context,
                                    "current_context": cur_state,
                                    "task_id": task_id
                                })
                            case (str(operation_type), dict(data)):
                                # Generation finished
                                assert operation_type == 'Return' and isinstance(data, dict)
                                
                                match data.get('problem_type'):
                                    case None:
                                        informal_problem_and_answer = ''
                                        informal_solution = ''
                                    case 'Problem-Solving Question':
                                        informal_problem_and_answer = data['informal_problem'] + '\n\nAnswer: ' + data['informal_answer']
                                        informal_solution = data['informal_solution']
                                    case 'Proof Question':
                                        informal_problem_and_answer = data['informal_problem']
                                        informal_solution = data['informal_solution']
                                    case _:
                                        raise ValueError('Unparsable informalization: ' + repr(data))
                                # TODO: is_valid
                                
                                await websocket.send_json({
                                    "type": "result",
                                    "formal_code": data['formal_code'],
                                    "informal_problem_and_answer": informal_problem_and_answer,
                                    "informal_solution": informal_solution,
                                    "task_id": task_id
                                })
                                break
                            case _:
                                raise ValueError('Unparsable response: ' + repr(content))
                except:
                    logger.error(f'websocket_endpoint({current_tag}): Failed due to {traceback.format_exc()}')

                # 清理任务
                if task_id in active_tasks:
                    del active_tasks[task_id]
            
            # TODO: 这个功能还没做好... 点击"取消生成"没有用
            elif data["action"] == "cancel_generation":
                # 取消生成任务
                task_id = data["task_id"]
                if task_id in active_tasks:
                    del active_tasks[task_id]
                    await websocket.send_json({
                        "type": "task_canceled",
                        "task_id": task_id
                    })
                    
    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected")
    except Exception as e:
        print(f"Error: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })

def initialize(
    agent_name: str,
    
    base_url: str,
    api_key: str,
    model_name: str,
    n_servers: int=1,   # Assuming all servers share the same ip and have consecutive ports
    
    frontend_port: int=8000,
    
    falsify_base_url: Optional[str]=None,
    falsify_api_key: Optional[str]=None,
    falsify_model_name: Optional[str]=None,
    falsify_n_servers: int=0,   # Assuming all servers share the same ip and have consecutive ports
    
    project_root: str='/home/ma-user/workspace/fps_pantograph/formal_problem_solving/data/MiniF2F',
    reassemble_trajectory: bool=False,
    staged: bool=False,
    temperature: float=1.0,
    max_search_trials: int=80,
    num_max_samples_per_trial: int=8,
    max_tokens: int=-1,
):
    saved_args = {**locals()}
    
    now = datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.remove()
    logger.add(sys.stdout, level='INFO')    # filter=lambda record: record["name"] != "agent.solution_autoformalization"
    logger.add(now+'.log', level='DEBUG')
    logger.info(f'Initializing with hyperparams: {saved_args}')
    if staged:
        logger.warning('Staged intro-deducing generation.')

    base_urls = [base_url]
    for _ in range(n_servers-1):
        base_urls.append(add_one_to_port(base_urls[-1]))
    
    if falsify_n_servers > 0:
        falsifier_base_urls = [falsify_base_url]
        for _ in range(falsify_n_servers-1):
            falsifier_base_urls.append(add_one_to_port(falsifier_base_urls[-1]))

    async def _generate_worker(
        subject: str,
        source: str,
        tag_i: int,
    ) -> AsyncGenerator[Tuple[str, Any], None]:
        logger.info(f'generate_worker({tag_i}): Initializing.')
        server = PersistentServer(
            is_state_based=True,
            tag=f'Server-{tag_i}',
            _sync_init=False,
            imports=["Mathlib", "Aesop"],
            project_path=project_root,
            core_options=CORE_OPTIONS,
            timeout=300,
        )
        parser = PersistentParsingServer(
            max_count=32,
            is_state_based=True,
            tag=f'Parser-{tag_i}',
            _sync_init=False,
            imports=["Mathlib", "Aesop"],
            project_path=project_root,
            core_options=CORE_OPTIONS,
            timeout=300
        )
        falsifier = None if falsify_n_servers <= 0 else ProblemFalsifier(
            clients=[
                AsyncOpenAI(
                    base_url=falsifier_base_urls[tag_i % falsify_n_servers],
                    api_key=falsify_api_key
                )
            ],
            models=[falsify_model_name],
            server=PersistentServer(
                is_state_based=True,
                tag=f'FalsifierServer-{tag_i}',
                _sync_init=False,
                imports=["Mathlib", "Aesop"],
                project_path=project_root,
                core_options=CORE_OPTIONS,
                timeout=60,
            ),
            temperature=0.0
        )
        client = AsyncOpenAI(
            base_url=base_urls[tag_i % len(base_urls)],
            api_key=api_key
        )
        problem_generator: AutoregressiveProblemGenerationAgent = AGENT_DICT[agent_name](
            gen_client=client,
            gen_model_name=model_name,
            max_search_trials=max_search_trials,
            num_max_samples_per_trial=num_max_samples_per_trial,
            temperature=temperature,
            max_tokens=(max_tokens if max_tokens > 0 else NOT_GIVEN)
        )
        if falsifier is not None:
            problem_generator.falsifiers.append(
                falsifier.falsify_async
            )
        logger.info(f'generate_worker({tag_i}): Initialized')
        
        try:
            g: AsyncGenerator[Tuple[str, Any], None] = problem_generator.detailed_generate_async(
                conditions={'subject': subject, 'source': source},
                server=server,
                parser=parser,
                reassemble_trajectory=reassemble_trajectory,
                tag=str(tag_i),
                staged=staged,
            )
            
            result = None
            async for content in g:
                match content:
                    case str(message):  # Fatal Error
                        logger.error(f'generate_worker({tag_i}): Fatal Error: content={content}')
                        yield message
                        return
                    case (str(operation_type), content):
                        logger.info(f'generate_worker({tag_i}): Streaming content: operation_type={operation_type}')
                        logger.debug(f'generate_worker({tag_i}): Streaming content: operation_type={operation_type}, content={repr(content)}')
                        if operation_type != 'Final':
                            yield (operation_type, content)
                        else:
                            result: ProblemGenerationProcess = content
                            break
            
            if result is not None:
                assert result.formal_statement.endswith(':= sorry')
                formal_code = '\n'.join(l + ' in' for l in result.header.splitlines()) + '\n' + result.formal_statement[:-len(':= sorry')] + '= by\n' + result.formal_solution_draft + '\n\n'
                
                logger.info(f'generate_worker({tag_i}): Generation succeeded, informalizing')
                informalization = await informalize(
                    header=result.header,
                    formal_statement=result.formal_statement,
                    formal_proof=result.formal_solution_draft,
                    tag_i=tag_i
                )
                if informalization is None:
                    logger.warning(f'generate_worker({tag_i}): Informalization failed.')
                    yield('Return', dict(formal_code=formal_code))
                else:
                    logger.info(f'generate_worker({tag_i}): Informalization succeeded: {repr(informalization)}')
                    informalization['formal_code'] = formal_code
                    yield('Return', informalization)
            
        except Exception as e:
            yield 'Fatal Error:\n' + repr(e) + '\n' + traceback.format_exc()
        finally:
            falsifier.data_train = []

    global generate_worker
    generate_worker = _generate_worker
    logger.info("Main process initialized.")
    
    import uvicorn
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=frontend_port,
        reload=False,
        workers=1,
    )

logger.info('Process starting.')

if __name__ == "__main__":
    Fire(initialize)
