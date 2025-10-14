import os
import os.path as osp
import itertools as I
import collections as C
import json
import asyncio
from typing import Dict, Optional

from loguru import logger
import fire
import openai
from openai import AsyncOpenAI

from common.utils import replace_sorry, remove_comments


RETRY_WAIT_TIME = 5
N_RETRY = 5
demonstrations = {"-- Define the transformation function\n-- Define the transformation function\ndef transformation (p : ℝ × ℝ) : ℝ × ℝ :=\n  (-p.2, p.1)\n\n-- Define the points\ndef a : ℝ × ℝ :=\n  (0, 0)\n\ndef b : ℝ × ℝ :=\n  (4, 0)\n\ndef c : ℝ × ℝ :=\n  (4, 3)\n\n-- Apply the transformation\ndef a' :=\n  transformation a\n\ndef b' :=\n  transformation b\n\ndef c' :=\n  transformation c\n\n-- Verify the results\nexample : a' = (0, 0) := by unfold a' transformation a <;> simp\n\n\nexample : b' = (0, 4)  := by\n unfold b' transformation b <;> simp": {'problem_type': 'Problem-Solving Question', 'informal_problem': "Given a transformation defined by T(x, y) = (-y, x) and a point B = (4, 0), find the coordinates of its image B' under this transformation.", 'informal_answer': '\\boxed{(0, 4)}', 'informal_solution': "The transformation is defined as T(x, y) = (-y, x). Applying this to point B = (4, 0):\n- B' = T(4, 0) = (-0, 4) = (0, 4)\n\nThus, the image of point B under the transformation is \\boxed{(0, 4)}."}, '-- Define the equation\n-- Define the equation\ndef Eqn (x : ℝ) :=\n  7 * x - 3 = 2 * x + 9\n\n-- Proof\n\nexample : ∃ x, Eqn x  := by\n  use 12 / 5\n  unfold Eqn\n  norm_num': {'problem_type': 'Problem-Solving Question', 'informal_problem': 'Find a real number \\( x \\) such that \\( 7x - 3 = 2x + 9 \\).', 'informal_answer': '\\boxed{\\dfrac{12}{5}}', 'informal_solution': 'We are to solve the equation \\( 7x - 3 = 2x + 9 \\) for \\( x \\in \\mathbb{R} \\).\n\nFirst, subtract \\( 2x \\) from both sides to get:\n\\[\n5x - 3 = 9\n\\]\nNext, add 3 to both sides to get:\n\\[\n5x = 12\n\\]\nFinally, divide both sides by 5 to solve for \\( x \\):\n\\[\nx = \\frac{12}{5}\n\\]\nSubstituting \\( x = \\frac{12}{5} \\) back into the original equation confirms that it is a solution. Therefore, the answer is \\( \\boxed{\\dfrac{12}{5}} \\).'}, '-- defining the variables\n-- defining the variables\nnoncomputable def r :=\n  (70 : ℝ) / 100\n\n-- percentage of students who identified the real part correctly\nnoncomputable def i :=\n  (60 : ℝ) / 100\n\n-- percentage of students who identified the imaginary part correctly\nnoncomputable def b :=\n  (50 : ℝ) / 100\n\n-- percentage of students who identified both parts correctly\n-- defining the principle of inclusion and exclusion\nnoncomputable def principleOfInclusionAndExclusion :=\n  r + i - b\n\n-- showing that the principle of inclusion and exclusion gives the percentage of students who identified either part correctly\n\nexample : principleOfInclusionAndExclusion = 0.8  := by\n  unfold principleOfInclusionAndExclusion r i b\n  norm_num\n\n-- defining the percentage of students who could not identify either part correctly': {'problem_type': 'Problem-Solving Question', 'informal_problem': 'In a mathematics assessment, 70% of students correctly identified the real part of a complex number, 60% correctly identified the imaginary part, and 50% correctly identified both parts. What percentage of students correctly identified at least one of the two parts?', 'informal_answer': '\\boxed{80\\%}', 'informal_solution': 'To find the percentage of students who identified at least one part correctly, we apply the principle of inclusion-exclusion. This principle states that for two sets, the percentage in the union is equal to the sum of the percentages in each individual set minus the percentage in their intersection.\n\nThus, the calculation is:\n\\[\n70\\% + 60\\% - 50\\% = 80\\%\n\\]\n\nTherefore, 80% of the students identified at least one part correctly.'}, 'noncomputable def fracOne : ℝ :=\n  1 / 2\n\nnoncomputable def fracTwo : ℝ :=\n  2 / 4\n\n-- proof that the two fractions are equal\n\ntheorem equivalent_fractions : fracOne = fracTwo  := by\n  unfold fracOne fracTwo\n  norm_num': {'problem_type': 'Proof Question', 'informal_problem': 'The real numbers $\\frac{1}{2}$ and $\\frac{2}{4}$ are equal.', 'informal_answer': None, 'informal_solution': 'We begin by observing that $\\frac{2}{4}$ simplifies to $\\frac{1}{2}$ by dividing both numerator and denominator by their greatest common divisor, which is 2. Since both expressions represent the same rational number, we conclude that:\n\n$$\\frac{1}{2} = \\frac{2}{4}$$\n\nThis completes the proof.'}, '-- First, we define a function to represent the quadratic expression 6x^2 - x - 2\n-- First, we define a function to represent the quadratic expression 6x^2 - x - 2\ndef f (x : ℝ) : ℝ :=\n  6 * x ^ 2 - x - 2\n\n-- Then, we define a function to represent the original form (3x - 2)(2x + 1)\ndef g (x : ℝ) : ℝ :=\n  (3 * x - 2) * (2 * x + 1)\n\n-- Now, we prove that these two functions are equal for all real numbers x\n\nexample : ∀ x : ℝ, f x = g x  := by\n  intro x\n  -- Expand both sides\n  rw [f]\n  rw [g]\n  -- Simplify and group terms\n  ring': {'problem_type': 'Proof Question', 'informal_problem': 'For all real numbers \\( x \\), the expression \\( 6x^2 - x - 2 \\) is equal to the expression \\( (3x - 2)(2x + 1) \\).', 'informal_answer': None, 'informal_solution': 'Let \\( x \\) be an arbitrary real number. Define the functions:\n\\[\nf(x) = 6x^2 - x - 2\n\\]\nand\n\\[\ng(x) = (3x - 2)(2x + 1).\n\\]\nTo show \\( f(x) = g(x) \\), we expand the right-hand side:\n\\[\ng(x) = (3x - 2)(2x + 1) = 3x \\cdot 2x + 3x \\cdot 1 - 2 \\cdot 2x - 2 \\cdot 1 = 6x^2 + 3x - 4x - 2.\n\\]\nCombining like terms yields:\n\\[\ng(x) = 6x^2 - x - 2.\n\\]\nThis is exactly \\( f(x) \\). Therefore, \\( f(x) = g(x) \\) for all \\( x \\in \\mathbb{R} \\). ∎'}}


def format_informalization_response(
    problem_type: str,
    informal_problem: str,
    informal_solution: str,
    informal_answer: Optional[str]
) -> str:
    assert all(split not in field
        for split in [f'## Problem-Solving Question', '## Proof Question', '## Answer', '## Solution', '## Proof']
        for field in [problem_type, informal_problem, informal_solution, (informal_answer or '')]
    )
    response = f'## {problem_type.strip()}\n{informal_problem.strip()}\n\n'
    if problem_type == 'Problem-Solving Question':
        assert informal_answer is not None
        response += f'## Answer\n{informal_answer.strip()}\n\n'
        response += f'## Solution\n{informal_solution.strip()}\n\n'
    elif problem_type == 'Proof Question':
        response += f'## Proof\n{informal_solution.strip()}\n\n'
    else:
        raise ValueError(f'Invalid problem_type: "{problem_type}"')
    return response


DEMONSTRATION_TEXT = ''
for code, d in demonstrations.items():
    DEMONSTRATION_TEXT += f'''## Formal Code
```lean4
{remove_comments(code).strip()}
```

{format_informalization_response(problem_type=d['problem_type'], informal_problem=(d.get('informal_problem') or d.get('informal_statement')), informal_solution=d.get('informal_solution') or d.get('informal_proof'), informal_answer=d.get('informal_answer')).strip()}

---

'''

def format_informalization_prompt(header: Optional[str], formal_statement: str, formal_proof: str) -> str:
    header = (header or '').strip()
    formal_statement = replace_sorry(formal_statement)
    assert formal_statement.endswith(':= sorry')
    if formal_proof.startswith('\n'):
        formal_proof = formal_proof[1:]
    formal_code = header + '\n' + formal_statement[:-len('sorry')] + 'by\n' + formal_proof
    return f'''Given a Lean 4 formal statement and its proof, please translate them into natural language.
1. Determine its question type: "Problem-Solving Question" or "Proof Question". Prefer "Problem-Solving Question" if possible.
2. If it is a "Problem-Solving Question", please translate the formal code into a natural language question, its answer, and its solution. The answer should not appear in the question text. The solution should wrap the final answer with "\\boxed{{}}".
3. If it is a "Proof Question", please translate it into a natural language proposition and its proof.
4. Please maintain the semantic equivalence between the natural language question+answer/proposition and the formal statement, and between the solution/proof and the formal proof.
5. Please reply in markdown format with level-2 headers such as "## Problem-Solving Question", "## Answer" and "## Solution".
6. If the Lean 4 code contains multiple statement-proof pairs, only translate the last one.

The following are some examples:

{DEMONSTRATION_TEXT}Now, please translate the following formal statement and its proof.

## Formal Code
```lean4
{remove_comments(formal_code).strip()}
```
'''

def parse_response(response: str) -> dict:
    if '## Problem-Solving Question' in response:
        assert '\n## Answer' in response
        assert '\n## Solution' in response
        assert '## Proof Question' not in response
        assert '\n## Proof' not in response
        intro, body = response.split('## Problem-Solving Question')
        informal_problem, informal_answer_solution = body.split('\n## Answer')
        informal_answer, informal_solution = informal_answer_solution.split('\n## Solution')
        return dict(
            problem_type='Problem-Solving Question',
            informal_problem=informal_problem.strip(),
            informal_answer=informal_answer.strip(),
            informal_solution=informal_solution.strip()
        )
    else:
        assert '## Proof Question' in response
        assert '\n## Proof' in response
        assert '## Problem-Solving Question' not in response
        assert '\n## Answer' not in response
        assert '\n## Solution' not in response
        intro, body = response.split('## Proof Question')
        informal_statement, informal_proof = body.split('\n## Proof')
        return dict(
            problem_type='Proof Question',
            informal_problem=informal_statement.strip(),
            informal_answer=None,
            informal_solution=informal_proof.strip()
        )

def main(
    load_root: str,
    save_root: str,
    file_name: str,
    n_concurrency: int=1
) -> None:
    with open(osp.join(load_root, file_name), 'r') as f:
        data = [json.loads(l) for l in f.readlines()]
    logger.info(f'Informalizing {len(data)} samples.')

    client = AsyncOpenAI(
        base_url='https://api.deepseek.com',
        api_key='sk-a08c67b565e84a1cbd50b1557db4a633'
    )

    async def informalize(index: int, d: Dict) -> None:
        d['responses'] = []
        d['n_prompt_tokens'] = []
        d['n_completion_tokens'] = []
        d['n_cached_tokens'] = []
        prompt = format_informalization_prompt(
            header=d['header'],
            formal_statement=d['formal_statement'],
            formal_proof=d['formal_proof']
        )
        for i_retry in range(N_RETRY):
            try:
                # breakpoint()
                responses = (await client.chat.completions.create(
                    model='deepseek-chat',
                    messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant."
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                    stream=False,
                    temperature=0.2 * i_retry,
                ))
                # breakpoint()
                n_prompt_tokens = None if responses.usage is None else responses.usage.prompt_tokens
                n_completion_tokens = None if responses.usage is None else responses.usage.completion_tokens
                n_cached_tokens = None if responses.usage is None or responses.usage.prompt_tokens_details is None else responses.usage.prompt_tokens_details.cached_tokens
                d['responses'].append(responses.choices[0].message.content)
                d['n_prompt_tokens'].append(n_prompt_tokens)
                d['n_completion_tokens'].append(n_completion_tokens)
                d['n_cached_tokens'].append(n_cached_tokens)
                
                d['informalization'] = parse_response(responses.choices[0].message.content)
                logger.opt(colors=True).info(f'<green>informalize({index}): succeeded</green>')
                return
            except Exception as e:
                logger.debug(f'informalize({index}): {i_retry}-th try failed due to exception {repr(e)}')
                await asyncio.sleep(RETRY_WAIT_TIME)
        logger.opt(colors=True).error(f'informalize({index}): failed')

    async def _async_main(data):
        pending_tasks = set()
        for i, d in enumerate(data):
            if len(pending_tasks) >= n_concurrency:
                done_tasks, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done_tasks:
                    if task.exception() is not None:
                        logger.error(f"Exception occurred: {task.exception()} {task.get_stack()}")
                        for pending_task in pending_tasks:
                            pending_task.cancel()
                        return
            pending_tasks.add(
                asyncio.create_task(
                    informalize(i, d)
                )
            )
        if len(pending_tasks) > 0:
            await asyncio.wait(pending_tasks)
        await logger.complete()


    try:
        asyncio.run(_async_main(data))
    finally:
        logger.info('Finished')
        with open(osp.join(save_root, file_name), 'w') as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        # import pdb; pdb.set_trace()
        print()


if __name__ == '__main__':
    fire.Fire(main)
