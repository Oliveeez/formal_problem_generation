import os
import os.path as osp
import time
import errno
from pathlib import Path
import regex as re
from typing import Union, Tuple, List, Dict, Any, Optional, Generator, AsyncGenerator, Set, Iterable
from contextlib import contextmanager, asynccontextmanager
import asyncio
import collections as C
import itertools as I
import functools as F
import inspect
import signal

import ipdb
import aiofiles
from loguru import logger
import pexpect

from common.constants import Expr, _SPACES_REGEX, LOCK_WAIT_TIMEOUT, LOCK_TRY_DELAY, BANNED_TOKENS, FPS_GLOBAL_SETTING, CODEBLOCK_PATTERN


replace_calc = lambda s: re.sub(r'by\s+calc', r'calc', s)
replace_sorry = lambda s: re.sub(r'by\s+sorry', r'sorry', s)

def format_variable_sequence(s : Iterable['Variable']) -> str:
    return ' '.join([f'({v.name} : {v.t})' if v.name not in [None, '_'] else f'[{v.t}]' for v in s])

def inplace_add(s: str, ss: List[str]) -> str:
    ss.append(s)
    return s

def extract_code(s: str) -> str:
    parse_result = re.findall(CODEBLOCK_PATTERN, s)
    if len(parse_result) > 0:
        step = parse_result[0].strip()
    else:
        split_cnt = len(re.findall('```', s))
        if split_cnt == 0:
            step = s.strip()
        if split_cnt == 1:
            step = s[:s.find('```')].strip()
        else:
            step = s

    return step

def format_forward_solution_step_prompt(informal_problem: str, solution_goal: Union['Goal', str]) -> str:
    return f'''Given a natural language math problem and the current solution state, please generate the next solution step.
Please use comments to plan and reason in natural language and deductive reasoning to derive the answer.
Assume `Mathlib` is imported.
# Informal Problem
"""
{informal_problem}
"""
# Current Solution State
```lean4
{str(solution_goal)}
```
'''

def format_whole_solution_generation_prompt(informal_problem: str, initial_solution_state: Union['Goal', str]) -> str:
    g_str = str(initial_solution_state)
    assert g_str.startswith('case ')
    g_str = 'case h.mp\n' + '\n'.join(g_str.splitlines()[1:])
    prompt = f'''Given a natural language math problem and the initial solution state, please generate a Lean 4 formal solution.
You can use Lean 4 comments to conduct natural language reasoning.
Please only use forward reasoning; do not use tactics that modify the final goal.
Please assume the following header code has already been executed, and do not add any imports or openings.
```lean4
import Mathlib
```

# Problem
"""
{informal_problem}
"""

# Initial Solution State
```
{g_str}
```
'''
    return prompt

def format_solution_draft_prompt(informal_problem: str, initial_solution_state: Union['Goal', str]) -> str:
    g_str = str(initial_solution_state)
    assert g_str.startswith('case ')
    g_str = 'case h.mp\n' + '\n'.join(g_str.splitlines()[1:])
    prompt = f'''Given a natural language math problem and the initial solution state, please generate a Lean 4 solution sketch.
You can use Lean 4 comments to conduct natural language reasoning.
Please only use forward reasoning; do not use tactics that modify the final goal.
For new hypotheses, please do not prove them and use `sorry` to close them.
Please assume the following header code has already been executed, and do not add any imports or openings.
```lean4
import Mathlib
```

# Problem
"""
{informal_problem}
"""

# Initial Solution State
```
{g_str}
```
'''
    return prompt

def solution_decompose(formal_solution_draft: str) -> list[str]:
    '''Decompose a formal solution draft into steps (blocks)'''
    raw_lines = replace_sorry(replace_calc(remove_multiline_comments(formal_solution_draft))).rstrip().split('\n')
    if len(remove_comments(formal_solution_draft).strip()) == 0:
        return []

    line_stack = list(reversed(raw_lines))
    parse_result = []

    while len(line_stack) > 0:
        line = line_stack.pop().rstrip()
        if line.strip() == '':
            # Current line is empty: skip
            continue
        
        # If submitted, add the rest and exit
        if line.startswith('exact '):
            cur_block = line + '\n' + '\n'.join(reversed(line_stack))
            parse_result.append(cur_block)
            break
        
        cur_block = line
        if remove_singleline_comments(line) == '':
            # Current line is a root-level comment: Add following lines, until empty line or another root-level comment
            
            # 1. Add consecutive comments
            while len(line_stack) > 0 and remove_singleline_comments(line_stack[-1]).strip() == '':
                cur_block += '\n' + line_stack.pop().rstrip()
            # 2. When encounter tactics, add them and following lines, until empty line or another root-level comment
            assert len(line_stack) > 0 and remove_singleline_comments(line_stack[-1]).strip() != '', 'Comments-probe failed'
            # Add tactic and structures
            
            # If submitted, add the rest and exit
            if line_stack[-1].startswith('exact '):
                cur_block += '\n' + '\n'.join(reversed(line_stack))
                parse_result.append(cur_block)
                break
            
            # Else: add the current structure
            cur_block += '\n' + line_stack.pop().rstrip()
            while len(line_stack) > 0 and (line_stack[-1].strip() == '' or remove_singleline_comments(line_stack[-1]).startswith(' ')):
                cur_block += '\n' + line_stack.pop().rstrip()
            parse_result.append(cur_block)
        else:
            # Tactic
            while len(line_stack) > 0 and (line_stack[-1].strip() == '' or remove_singleline_comments(line_stack[-1]).startswith(' ')):
                cur_block += '\n' + line_stack.pop().rstrip()
            parse_result.append(cur_block)

    assert '\n'.join(I.chain(*[[l.rstrip() for l in r.split('\n') if l.strip() != ''] for r in parse_result])) == '\n'.join(l.rstrip() for l in raw_lines if l.strip() != ''), 'Reconstruction failed'
    # except Exception as e:
        # print(e)
        # import ipdb; ipdb.set_trace()
        # logger.error
    return parse_result

def split_list(lst: list, n: int) -> list[tuple[int, list]]:
    k, m = divmod(len(lst), n)
    return [(i * k + min(i, m), lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]) for i in range(n)]

def chunk_list(data: list, chunksize: int):
    for i in range(0, len(data), chunksize):
        # logger.info(f'chunk_list(): Yielding chunk {i}...')
        yield (i, data[i:i + chunksize])

def add_one_to_port(input_string):
    match = re.findall(r'\d+', input_string)
    
    if match:
        port_number = match[-2]
        incremented_number = str(int(port_number) + 1)
        return input_string.replace(port_number, incremented_number, 1)
    else:
        return input_string

class Spawn(pexpect.spawn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delayafterclose = 0.5
        self.delayafterterminate = 0.5
        self.ptyproc.delayafterclose = 0.5
        self.ptyproc.delayafterterminate = 0.5
        # self.delaybeforesend = None

    async def send_async(self, s):
        if self.delaybeforesend is not None:
            await asyncio.sleep(self.delaybeforesend)

        s = self._coerce_send_string(s)
        self._log(s, 'send')

        b = self._encoder.encode(s, final=False)

        return os.write(self.child_fd, b)        
        # while b:
        #     try:
        #         bytes_written = os.write(self.child_fd, b)
        #         b = b[bytes_written:]
        #     except BlockingIOError:
        #         await asyncio.sleep(0)
        #         pass
        
        # return bytes_written  

    async def sendline_async(self, s=''):
        '''Wraps send(), sending string ``s`` to child process, with
        ``os.linesep`` automatically appended. Returns number of bytes
        written.  Only a limited number of bytes may be sent for each
        line in the default terminal mode, see docstring of :meth:`send`.
        '''
        s = self._coerce_send_string(s)
        return await self.send_async(s + self.linesep)

    async def read_async(self, size=-1):
        '''This reads at most "size" bytes from the file (less if the read hits
        EOF before obtaining size bytes). If the size argument is negative or
        omitted, read all data until EOF is reached. The bytes are returned as
        a string object. An empty string is returned when EOF is encountered
        immediately. '''

        if size == 0:
            return self.string_type()
        if size < 0:
            # delimiter default is EOF
            await self.expect(self.delimiter, async_=True)
            return self.before

        # I could have done this more directly by not using expect(), but
        # I deliberately decided to couple read() to expect() so that
        # I would catch any bugs early and ensure consistent behavior.
        # It's a little less efficient, but there is less for me to
        # worry about if I have to later modify read() or expect().
        # Note, it's OK if size==-1 in the regex. That just means it
        # will never match anything in which case we stop only on EOF.
        cre = re.compile(self._coerce_expect_string('.{%d}' % size), re.DOTALL)
        # delimiter default is EOF
        index = await self.expect([cre, self.delimiter], async_=True)
        if index == 0:
            ### FIXME self.before should be ''. Should I assert this?
            return self.after
        return self.before

    async def readline_async(self, size=-1):
        '''This reads and returns one entire line. The newline at the end of
        line is returned as part of the string, unless the file ends without a
        newline. An empty string is returned if EOF is encountered immediately.
        This looks for a newline as a CR/LF pair (\\r\\n) even on UNIX because
        this is what the pseudotty device returns. So contrary to what you may
        expect you will receive newlines as \\r\\n.

        If the size argument is 0 then an empty string is returned. In all
        other cases the size argument is ignored, which is not standard
        behavior for a file-like object. '''

        if size == 0:
            return self.string_type()
        # delimiter default is EOF
        index = await self.expect([self.crlf, self.delimiter], async_=True)
        if index == 0:
            return self.before + self.crlf
        else:
            return self.before

    async def terminate_async(self, force=False):
        '''This forces a child process to terminate. It starts nicely with
        SIGHUP and SIGINT. If "force" is True then moves onto SIGKILL. This
        returns True if the child was terminated. This returns False if the
        child could not be terminated. '''

        if not self.isalive():
            return True
        try:
            self.kill(signal.SIGHUP)
            await asyncio.sleep(self.delayafterterminate)
            if not self.isalive():
                return True
            self.kill(signal.SIGCONT)
            await asyncio.sleep(self.delayafterterminate)
            if not self.isalive():
                return True
            self.kill(signal.SIGINT)
            await asyncio.sleep(self.delayafterterminate)
            if not self.isalive():
                return True
            if force:
                self.kill(signal.SIGKILL)
                await asyncio.sleep(self.delayafterterminate)
                if not self.isalive():
                    return True
                else:
                    return False
            return False
        except OSError:
            # I think there are kernel timing issues that sometimes cause
            # this to happen. I think isalive() reports True, but the
            # process is dead to the kernel.
            # Make one last attempt to see if the kernel is up to date.
            await asyncio.sleep(self.delayafterterminate)
            if not self.isalive():
                return True
            else:
                return False

class Profiler:
    def __init__(self):
        self.stats = C.defaultdict(lambda : (float(), int()))
        self.start_time = None
        self.cur_tag = None

    def __call__(self, tag: str):
        assert self.cur_tag is None
        self.cur_tag = tag
        return self

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.cur_tag is not None
        end_time = time.perf_counter()
        old_total_time, old_count = self.stats[self.cur_tag]
        self.stats[self.cur_tag] = (old_total_time + end_time - self.start_time, old_count + 1)
        self.cur_tag = None

    @property
    def results(self) -> List[Tuple[str, float]]:
        return [(k, total_time / count) for k, (total_time, count) in self.stats.items()]


class FileLockException(Exception):
    pass

class FileLock(object):
    """ A file locking mechanism that has context-manager support so 
        you can use it in a with statement. This should be relatively cross
        compatible as it doesn't rely on msvcrt or fcntl for the locking.

        Mainly borrowed from https://github.com/dmfrey/FileLock/blob/master/filelock/filelock.py
    """

    def __init__(self, path, timeout=10, delay=.05):
        """ Prepare the file locker. Specify the file to lock and optionally
            the maximum timeout and the delay between each attempt to lock.
        """
        if timeout is not None and delay is None:
            raise ValueError(
                "If timeout is not None, then delay must not be None.")
        self.is_locked = False
        self.path = path
        self.lockfile = os.path.join(
            os.path.expanduser('~'), "%s.lock" % self.path.replace('/', '--'))
        self.timeout = timeout
        self.delay = delay

    def acquire(self):
        """ Acquire the lock, if possible. If the lock is in use, it check again
            every `wait` seconds. It does this until it either gets the lock or
            exceeds `timeout` number of seconds, in which case it throws 
            an exception.
        """
        start_time = time.time()
        while True:
            try:
                self.fd = os.open(self.lockfile, os.O_CREAT |
                                  os.O_EXCL | os.O_RDWR)
                self.is_locked = True  # moved to ensure tag only when locked
                break
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                if self.timeout is None:
                    raise FileLockException(
                        f"Could not acquire lock on {self.path}")
                if (time.time() - start_time) >= self.timeout:
                    raise FileLockException(
                        f"Could not acquire lock on {self.path}: timeout occured.")
                time.sleep(self.delay)

    async def async_acquire(self):
        """ Acquire the lock, if possible. If the lock is in use, it check again
            every `wait` seconds. It does this until it either gets the lock or
            exceeds `timeout` number of seconds, in which case it throws 
            an exception.
        """
        start_time = time.time()
        while True:
            try:
                self.fd = os.open(self.lockfile, os.O_CREAT |
                                  os.O_EXCL | os.O_RDWR)
                self.is_locked = True  # moved to ensure tag only when locked
                break
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                if self.timeout is None:
                    raise FileLockException(
                        f"Could not acquire lock on {self.path}")
                if (time.time() - start_time) >= self.timeout:
                    raise FileLockException(
                        f"Could not acquire lock on {self.path}: timeout occured.")
                await asyncio.sleep(self.delay)

    def release(self):
        """ Get rid of the lock by deleting the lockfile. 
            When working in a `with` statement, this gets automatically 
            called at the end.
        """
        if self.is_locked:
            os.close(self.fd)
            os.unlink(self.lockfile)
            self.is_locked = False

    def __enter__(self):
        """ Activated when used in the with statement. 
            Should automatically acquire a lock to be used in the with block.
        """
        if not self.is_locked:
            self.acquire()
        return self

    def __exit__(self, type, value, traceback):
        """ Activated at the end of the with statement.
            It automatically releases the lock if it isn't locked.
        """
        if self.is_locked:
            self.release()

    async def __aenter__(self):
        """ Activated when used in the async with statement. 
            Should automatically acquire a lock to be used in the with block.
        """
        if not self.is_locked:
            await self.async_acquire()
        return self


    async def __aexit__(self, type, value, traceback):
        """ Activated at the end of the async with statement.
            It automatically releases the lock if it isn't locked.
        """
        if self.is_locked:
            self.release()

    def __del__(self):
        """ Make sure that the FileLock instance doesn't leave a lockfile
            lying around.
        """
        self.release()

def unique(r : List) -> List:
    s = []
    for i in r:
        if i not in s:
            s.append(i)
    return s

def parse_expr(payload: Dict) -> Expr:
    """
    :meta private:
    """
    return payload["pp"]

PLACEHOLDER_REGEX = re.compile(r'\?([A-Za-z])(\.\d*)+')
PLACEHOLDER_STRING = r'<\1_PLACEHOLDER>'
PLACEHOLDER_REGEX_u = re.compile(r'u_\d+')
PLACEHOLDER_STRING_u = r'<u_PLACEHOLDER>'
def simplify_state(goal: str) -> str:
    return re.sub(r'\s+', '', PLACEHOLDER_REGEX.sub(PLACEHOLDER_STRING, PLACEHOLDER_REGEX_u.sub(PLACEHOLDER_STRING_u, goal)))

def eliminate_split_tokens(s: str) -> str:
    return s.replace('🛈🛈', '🛈').replace('🗟🛈', '').replace('\n🛈', '\n').replace('🛈 :', ' :').replace('🗟', '').replace('🛈', ' ')

def normalize_spaces(s: str) -> str:
    """Repalce any consecutive block of whitespace characters in ``s`` with a single whitespace."""
    return _SPACES_REGEX.sub(" ", s).strip()

def remove_spaces(s: str) -> str:
    """Repalce any consecutive block of whitespace characters in ``s`` with a single whitespace."""
    return _SPACES_REGEX.sub("", s).strip()

def remove_comments(code):
    code = re.sub(r'/-(.|\n)*?-/', '', code)
    code = re.sub(r'--.*', '', code)
    return code

def remove_singleline_comments(code):
    code = re.sub(r'--.*', '', code)
    return code

def remove_multiline_comments(code):
    code = re.sub(r'/-(.|\n)*?-/', '', code)
    return code

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

@contextmanager
def suppress_module_log(module_name: str):
    try:
        logger.disable(module_name)
        yield
    finally:
        logger.enable(module_name)

def unique_sorted(texts, scores):
    texts_ = []
    scores_ = []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_

def replace_span(span: Tuple[int, int], replacement: str, input_string: str) -> str:
    start, end = span
    return input_string[:start] + replacement + input_string[end:]

def zip_strict(*args):
    assert len(args) > 1 and all(len(args[0]) == len(a) for a in args[1:])
    return zip(*args)

def pdb_on_assertion_error(func):
    @F.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AssertionError as e:
            ipdb.set_trace()
            raise e
    return wrapper

@contextmanager
def temporarily_modify_code(
    path: Union[str, Path]
) -> Generator[str, None, None]:
    """Context manager for temporarily editing a lean code file.
    The file content is restored after the context manager exits.
    """
    with FileLock(path, LOCK_WAIT_TIMEOUT, LOCK_TRY_DELAY):
        with open(path, 'r') as f:
            content_backup = f.read()
        try:
            yield content_backup
        finally:
            with open(path, 'w') as f:
                f.write(content_backup)

@asynccontextmanager
async def temporarily_modify_code_async(
    path: Union[str, Path]
) -> AsyncGenerator[str, None]:
    """Context manager for temporarily editing a lean code file.
    The file content is restored after the context manager exits.
    """
    async with FileLock(path, LOCK_WAIT_TIMEOUT, LOCK_TRY_DELAY):
        async with aiofiles.open(path, 'r') as f:
            content_backup = await f.read()
        try:
            yield content_backup
        finally:
            async with aiofiles.open(path, 'w') as f:
                await f.write(content_backup)

def to_sync(func):
    @F.wraps(func)
    def wrapper(*args, **kwargs):
        if not FPS_GLOBAL_SETTING['TO_SYNC_ENABLED']:
            raise RuntimeError('to_sync() is not enabled in common.constants.FPS_GLOBAL_SETTING')
        return asyncio.get_event_loop().run_until_complete(func(*args, **kwargs))
    return wrapper

def timeit(func):
    if inspect.iscoroutinefunction(func):
        @F.wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not (hasattr(self, 'timing_history') and isinstance(self.timing_history, C.defaultdict)):
                self.timing_history = C.defaultdict(lambda : list())
            start_time = time.perf_counter()
            try:
                result = await func(self, *args, **kwargs)
                return result
            except Exception as e:
                raise e
            finally:
                end_time = time.perf_counter()
                self.timing_history[func.__name__].append(end_time - start_time)
    else:
        @F.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not (hasattr(self, 'timing_history') and isinstance(self.timing_history, C.defaultdict)):
                self.timing_history = C.defaultdict(lambda : list())
            start_time = time.perf_counter()
            try:
                result = func(self, *args, **kwargs)
                return result
            except Exception as e:
                raise e
            finally:
                end_time = time.perf_counter()
                self.timing_history[func.__name__].append(end_time - start_time)
    return wrapper

def flip_options(s: List[str]) -> List[str]:
    ret = []
    for o in s:
        if o.endswith('=true'):
            ret.append(o[:-len('=true')] + '=false')
        elif o.endswith('=false'):
            ret.append(o[:-len('=false')] + '=true')
        else:
            raise ValueError(o)
    return ret

def to_comment(s: str) -> str:
    return '\n'.join(['-- ' + line for line in s.split('\n')])

class IdentifierRegex:
    greek = (
        '[\\u03b1-\\u03ba\\u03bc-\\u03c9\\u0391-\\u039f\\u03a1-\\u03a2'
        '\\u03a4-\\u03a9\\u1f00-\\u1ffe]'
    )
    coptic = '[\\u03ca-\\u03fb]'
    letterlike_symbols = '[\\u2100-\\u214f]'
    letterlike = f'([a-zA-Z]|{greek}|{coptic}|{letterlike_symbols})'
    escaped_ident_part = (
        '\\xab([\\x00-\\x08][\x0b-\x0c]|[\\x0e-\\xaa\\xac-\\xba'
        '\\xbc-\\U0010ffff])*\\xbb'
    )
    atomic_ident_start = f'({letterlike}|_|{escaped_ident_part})'
    subscript = '[\\u2080-\\u2089\\u2090-\\u209c\\u1d62-\\u1d6a]'
    superscript = '[\\u2070\\xb9\\xb2-\\xb3\\u2074-\\u2079]'
    atomic_ident_rest = (
        f"({atomic_ident_start}|[0-9'\\u207f]|{subscript}|"
        f'\\u271d({superscript})*)'
    )
    atomic_ident = f'{atomic_ident_start}({atomic_ident_rest})*'
    ident = f'{atomic_ident}(\\.{atomic_ident})*'

    ident_pattern = re.compile(ident)
    
def parse_idents(s: str) -> List[str]:
    return [m.group() for m in re.finditer(IdentifierRegex.ident_pattern, s)]

def exists_bannded_token(code: str) -> List[str]:
    # TODO: clean comments, parse idents, then filter
    raise NotImplementedError

def remove_min_whitespace(s: str) -> str:
    lines = s.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return s
    min_whitespace = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
    result = []
    for line in lines:
        if line.strip():
            result.append(line[min_whitespace:])
        else:
            result.append(line)
    return '\n'.join(result)

def normalize_draft(s: str) -> str:
    s_normalized = re.sub(
        r':=\s*sorry', r':= sorry',
            re.sub(
        r':=\s*by\s+sorry', r':= sorry',
        remove_comments(s)
    )).strip()
    s_filled = re.sub(r'have\s+:', r'have this :', s_normalized)
    return '\n'.join(l for l in s_filled.splitlines() if l.strip() != '')
