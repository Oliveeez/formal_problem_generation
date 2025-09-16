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

from common.utils import replace_sorry


RETRY_WAIT_TIME = 5
N_RETRY = 5
demonstrations = {
    r'''example
(x : ℝ)
(hx : 0 < x)
(y : ℝ)
(hy : 0 < y)
(z : ℝ)
(hz : 0 < z)
(k : ℝ)
(h : x * y * z * k = 1)
(hk : 0 < k)
: x / (y * z * k) + y / (z * x * k) + z / (x * y * k) + k / (x * y * z) ≥ 4
:= by
have h1 : 0 < x * y * z + y * z * k + z * x * k + x * y * k := by positivity

have h2 : x / (y * z * k) + y / (z * x * k) + z / (x * y * k) + k / (x * y * z) = (x ^ 2 + y ^ 2 + z ^ 2 + k ^ 2) / (x * y * z * k) :=
  by
  field_simp
  ring

have h3: 
    x / (y * z * k) + y / (z * x * k) + z / (x * y * k) + k / (x * y * z) ≥ 4  := by {
  rw [h2]
  rw [h]
  nlinarith [sq_nonneg (x - y), sq_nonneg (x - z), sq_nonneg (x - k), sq_nonneg (y - z), sq_nonneg (y - k), sq_nonneg (z - k),
    mul_pos hx hy, mul_pos hx hz, mul_pos hx hk, mul_pos hy hz, mul_pos hy hk, mul_pos hz hk]
}

exact h3''' : 
    {
        'problem_type': 'Proof Question',
        'informal_statement': r'''Let \(x, y, z, k\) be positive real numbers such that \(x \cdot y \cdot z \cdot k = 1\). Prove that:

\[
\frac{x}{y \cdot z \cdot k} + \frac{y}{z \cdot x \cdot k} + \frac{z}{x \cdot y \cdot k} + \frac{k}{x \cdot y \cdot z} \geq 4.
\]''',
        'informal_proof': r'''**Step 1.** Since \(x, y, z, k > 0\), the denominator \(x \cdot y \cdot z + y \cdot z \cdot k + z \cdot x \cdot k + x \cdot y \cdot k\) is positive (by the positivity of the terms).

**Step 2.** We simplify the left-hand side by combining the fractions:

\[
\begin{aligned}
&\frac{x}{y \cdot z \cdot k} + \frac{y}{z \cdot x \cdot k} + \frac{z}{x \cdot y \cdot k} + \frac{k}{x \cdot y \cdot z} \\
&= \frac{x^2 + y^2 + z^2 + k^2}{x \cdot y \cdot z \cdot k}.
\end{aligned}
\]

This is verified by using a common denominator \(x \cdot y \cdot z \cdot k\) and simplifying the numerator.

**Step 3.** Since \(x \cdot y \cdot z \cdot k = 1\), the expression becomes:

\[
x^2 + y^2 + z^2 + k^2.
\]

**Step 4.** To prove \(x^2 + y^2 + z^2 + k^2 \geq 4\), we use the non-negativity of squares:

\[
(x - y)^2 \geq 0, \quad (x - z)^2 \geq 0, \quad (x - k)^2 \geq 0,
\]
\[
(y - z)^2 \geq 0, \quad (y - k)^2 \geq 0, \quad (z - k)^2 \geq 0.
\]

Expanding and summing these inequalities (along with using the fact that \(x, y, z, k > 0\)) leads to:

\[
3(x^2 + y^2 + z^2 + k^2) \geq 2(xy + xz + xk + yz + yk + zk).
\]

By the AM–GM inequality (or further algebraic manipulation), we conclude:

\[
x^2 + y^2 + z^2 + k^2 \geq 4.
\]

Thus, the original inequality holds.'''
    },
    
    r'''example
(z : ℂ)
(hz : z = 2 + 3 * .I)
(w : ℂ)
(hw : w = 2 - 3 * .I)
: z * w = 13
:= by
have h: 
    z * w = 13  := by {
  rw [hz, hw]
  simp [Complex.ext_iff, Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im, Complex.I_re, Complex.I_im]
  all_goals norm_num
}

exact h''' :
    {
        'problem_type': 'Problem-Solving Question',
        'informal_problem': r'''Let \( z \) and \( w \) be complex numbers such that \( z = 2 + 3i \) and \( w = 2 - 3i \). Compute the product \( z \times w \).''',
        'informal_answer': r'''13''',
        'informal_solution': r'''**Step 1: Write down the given expressions.**  
Given:  
\[
z = 2 + 3i, \quad w = 2 - 3i.
\]

**Step 2: Compute the product \( z \times w \).**  
\[
z \times w = (2 + 3i)(2 - 3i).
\]

**Step 3: Expand the product using the distributive property (FOIL method).**  
\[
(2 + 3i)(2 - 3i) = 2 \cdot 2 + 2 \cdot (-3i) + 3i \cdot 2 + 3i \cdot (-3i).
\]  
Simplify each term:  
\[
= 4 - 6i + 6i - 9i^2.
\]

**Step 4: Combine like terms and simplify using \( i^2 = -1 \).**  
The imaginary terms \(-6i\) and \(+6i\) cancel each other:  
\[
= 4 + 0i - 9i^2 = 4 - 9i^2.
\]  
Substitute \( i^2 = -1 \):  
\[
= 4 - 9(-1) = 4 + 9 = 13.
\]

**Step 5: Conclude the result.**  
Thus,  
\[
z \times w = 13.
\]

**Final Answer:**  
\[
\boxed{13}
\]'''
    },
    r'''example
(a : ℕ → ℤ)
(ha1 : a 1 = 2)
(han : ∀ n ≥ 1, a (n + 1) = a n + 2 * n)
: a 100 = 9902
:= by
have h: 
    a 100 = 9902  := by {
  norm_num [ha1, han] at *
  <;> try { all_goals omega }
}

exact h''' :
    {
        'problem_type': 'Problem-Solving Question',
        'informal_problem': r'''Define a sequence \( a_1, a_2, a_3, \ldots \) of integers by:
- \( a_1 = 2 \), and
- For every integer \( n \geq 1 \), \( a_{n+1} = a_n + 2n \).

Calculate the value of \( a_{100} \).''',
        'informal_answer': r'''9902''',
        'informal_solution': r'''1.  **Understand the Recurrence Relation:**
    The sequence is defined by:
    \[
    a_1 = 2
    \]
    \[
    a_{n+1} = a_n + 2n \quad \text{for all } n \geq 1
    \]
    This means each term is obtained by adding \( 2n \) (which is twice the current index \( n \)) to the previous term.

2.  **Write Out the First Few Terms:**
    Let's compute the first few terms to identify a pattern.
    \[
    \begin{align*}
    a_1 &= 2 \\
    a_2 &= a_1 + 2 \cdot 1 = 2 + 2 = 4 \\
    a_3 &= a_2 + 2 \cdot 2 = 4 + 4 = 8 \\
    a_4 &= a_3 + 2 \cdot 3 = 8 + 6 = 14 \\
    a_5 &= a_4 + 2 \cdot 4 = 14 + 8 = 22 \\
    \end{align*}
    \]
    The sequence begins: \( 2, 4, 8, 14, 22, \ldots \)

3.  **Look for a Pattern or Closed Form:**
    Observe the differences between consecutive terms:
    \[
    a_2 - a_1 = 2 = 2 \cdot 1 \\
    a_3 - a_2 = 4 = 2 \cdot 2 \\
    a_4 - a_3 = 6 = 2 \cdot 3 \\
    a_5 - a_4 = 8 = 2 \cdot 4 \\
    \]
    This confirms the recurrence \( a_{n+1} - a_n = 2n \).

    To find a closed form for \( a_n \), we can express \( a_n \) as the sum of the first term and all the subsequent differences:
    \[
    \begin{align*}
    a_n &= a_1 + \sum_{k=1}^{n-1} (a_{k+1} - a_k) \\
        &= 2 + \sum_{k=1}^{n-1} 2k \\
        &= 2 + 2 \sum_{k=1}^{n-1} k
    \end{align*}
    \]

4.  **Apply the Formula for the Sum of the First \( m \) Integers:**
    The well-known formula is \( \sum_{k=1}^{m} k = \frac{m(m+1)}{2} \). Here, the upper limit is \( m = n-1 \), so:
    \[
    \sum_{k=1}^{n-1} k = \frac{(n-1)n}{2}
    \]
    Substituting this back into our expression for \( a_n \):
    \[
    \begin{align*}
    a_n &= 2 + 2 \cdot \left( \frac{(n-1)n}{2} \right) \\
        &= 2 + (n-1)n \\
        &= n^2 - n + 2
    \end{align*}
    \]
    We have derived the closed-form formula:
    \[
    a_n = n^2 - n + 2
    \]
    Let's verify this with the initial terms:
    \[
    a_1 = 1^2 - 1 + 2 = 2 \quad \checkmark \\
    a_2 = 4 - 2 + 2 = 4 \quad \checkmark \\
    a_3 = 9 - 3 + 2 = 8 \quad \checkmark \\
    a_4 = 16 - 4 + 2 = 14 \quad \checkmark \\
    \]
    The formula holds.

5.  **Calculate \( a_{100} \):**
    Substitute \( n = 100 \) into the closed-form formula:
    \[
    \begin{align*}
    a_{100} &= 100^2 - 100 + 2 \\
            &= 10000 - 100 + 2 \\
            &= 9900 + 2 \\
            &= 9902
    \end{align*}
    \]

**Conclusion:** The 100th term of the sequence is \( 9902 \).

\[
\boxed{9902}
\]'''
    },
    r'''example
(a : ℝ)
(ha : 0 < a)
(b : ℝ)
(hb : 0 < b)
(f : ℝ → ℝ)
(hf : ∀ x, f x = a * x ^ 4 - b * x ^ 2 + x + 7)
(h : f (-5) = 3)
: f 5 = 13
:= by
have eq1 : a * (-5) ^ 4 - b * (-5) ^ 2 + (-5) + 7 = 3 := by
  rw [hf] at h
  linarith

have eq2 : 625 * a - 25 * b = 1 := by nlinarith

have eq3 : 25 * b = 625 * a - 1 := by linarith [eq2]

have eq4 : b = 25 * a - 1 / 25 := by linarith

have h:  f 5 = 13  := by {
  rw [hf 5]
  rw [show b = 25 * a - 1 / 25 by linarith [eq4] ]
  nlinarith [ sq_nonneg (a), sq_nonneg (b), 
          sq_nonneg (a - 1), sq_nonneg (b - 1), 
          sq_nonneg (a * b), sq_nonneg (a * b - 1)]
}

exact h''' : 
    {
        'problem_type': 'Problem-Solving Question',
        'informal_problem': r'''Let \( a \) and \( b \) be positive real numbers. Define a function \( f : \mathbb{R} \to \mathbb{R} \) by  
\[
f(x) = a \cdot x^4 - b \cdot x^2 + x + 7.
\]  
Given that \( f(-5) = 3 \), what is the value of \( f(5) \)?''',
        'informal_answer': r'''13''',
        'informal_solution': r'''**Step 1:** Substitute \( x = -5 \) into \( f(x) \).  
Since \( f(-5) = 3 \), we have:  
\[
a (-5)^4 - b (-5)^2 + (-5) + 7 = 3.
\]  
Simplifying the powers:  
\[
625a - 25b - 5 + 7 = 3.
\]  
Combine constants:  
\[
625a - 25b + 2 = 3.
\]  
Subtract 2 from both sides:  
\[
625a - 25b = 1. \tag{1}
\]

**Step 2:** Solve equation (1) for \( b \).  
Rewriting:  
\[
25b = 625a - 1.
\]  
Divide by 25:  
\[
b = 25a - \frac{1}{25}. \tag{2}
\]

**Step 3:** Compute \( f(5) \).  
\[
f(5) = a (5)^4 - b (5)^2 + 5 + 7 = 625a - 25b + 12.
\]  
Substitute \( b \) from equation (2):  
\[
f(5) = 625a - 25\left(25a - \frac{1}{25}\right) + 12.
\]  
Simplify:  
\[
f(5) = 625a - 625a + 1 + 12 = 13.
\]

Therefore, \( f(5) = 13 \), as required.

---

### Final Answer:
\[
\boxed{13}
\]'''
    },
    r'''example
(g : ℕ → ℕ)
(hg : ∀ n, 0 < g n)
(h1 : ∀ m n, g (m + n) = g m * g n)
(h2 : ∀ n, g n ∣ n)
: ∀ (n : ℕ), g n = 1
:= by
have h3 := h2 0

have h7 : g 0 = 1 := by
  have h10 : g 0 ∣ 0 := by
    use 0
    all_goals
      simp
  have h11 : g 0 = 1 := by
    have h6 : 0 < g 0 := by
      apply hg 0
    have h12 : g 0 ∣ 0 := h10
    have h13 : g 0 < 0 ∨ g 0 = 0 ∨ 0 < g 0 := by
      omega
    rcases h13 with (h | h | h)
    · 
      exfalso
      have h14 : 0 < g 0 := by
        apply hg 0
      omega
    · 
      exfalso
      have h14 : 0 < g 0 := by
        apply hg 0
      omega
    · 
      have h14 : 0 < g 0 := h
      have h15 : g 0 = 1 := by
        by_contra h17
        push_neg at h17
        have h18 : g 0 > 1 := by
          omega
        have h19 : ∃ x, g 0 = x + 1 := by
          refine ⟨g 0 - 1, by omega⟩
        rcases h19 with ⟨x, h18⟩
        have h20 : g 0 ∣ 0 := h10
        have h21 : ∃ k, g 0 * k = 0 := by
          rcases h20 with ⟨k, hk⟩
          use k
          linarith
        rcases h21 with ⟨k, hk⟩
        have h22 : g 0 * k = 0 := hk
        have h23 : g 0 > 0 := by
          linarith
        have h24 : k = 0 := by
          nlinarith
        have h25 : g (0 + 0) = g 0 * g 0 := h1 0 0
        have h26 : g (0 + 0) = g 0 := by
          simp
        rw [h26] at h25
        rw [h24] at h22
        nlinarith
      exact h15
  exact h11

have h4 : g 1 ∣ 1 := h2 1

have h8 : g 1 = 1 := by
  have h9 : g 1 ∣ 1 := h4
  have h10 : 0 < g 1 := hg 1
  have h11 : g 1 ≤ 1 := by exact Nat.le_of_dvd (by norm_num) h9
  have h12 : g 1 ≥ 1 := by linarith [h10]
  linarith

have h9: 
    ∀ n, g n = 1  := by {
  intro n 
  have h30 : ∀ n, g n = 1 := by 
    intro n 
    induction' n using Nat.strongRecOn with n ih
    cases n with
    | zero =>
      exact h7 
    | succ n =>
      cases n with
      | zero =>
        exact h8 
      | succ n =>
        have h9 : g (n + 1 + 1) = g (n + 1) * g 1 := by 
          specialize h1 (n + 1) 1 
          exact h1 
        rw [h9]
        rw [show g (n + 1) = 1 by apply ih (n + 1) (by omega)]
        rw [show g 1 = 1 by exact h8]
        all_goals
          norm_num
  apply h30 n
}

exact h9''' : 
    {
        'problem_type': 'Problem-Solving Question',
        'informal_problem': r'''Let \( g \) be a function from natural numbers to natural numbers such that:
1. For every \( n \), \( g(n) > 0 \).
2. For every \( m \) and \( n \), \( g(m + n) = g(m) \cdot g(n) \).
3. For every \( n \), \( g(n) \) divides \( n \).

Determine the value of \( g(n) \) for every natural number \( n \).''',
        'informal_answer': r'''1''',
        'informal_solution': r'''#### Step 1: Show \( g(0) = 1 \)
From condition 3, \( g(0) \mid 0 \), so there exists \( k \) such that \( g(0) \cdot k = 0 \). Since \( g(0) > 0 \) by condition 1, we must have \( k = 0 \), so \( g(0) \cdot 0 = 0 \).  
Now, using condition 2 with \( m = 0 \) and \( n = 0 \):
\[
g(0 + 0) = g(0) \cdot g(0) \implies g(0) = g(0)^2.
\]
Since \( g(0) > 0 \), we conclude \( g(0) = 1 \).

#### Step 2: Show \( g(1) = 1 \)
From condition 3, \( g(1) \mid 1 \), so \( g(1) = 1 \) (since \( g(1) > 0 \)).

#### Step 3: Prove \( g(n) = 1 \) for all \( n \) by strong induction
- **Base cases**:  
  \( g(0) = 1 \) and \( g(1) = 1 \) as shown.
- **Inductive step**:  
  Assume \( g(k) = 1 \) for all \( k < n \). We show \( g(n) = 1 \).  
  Write \( n = (n-1) + 1 \). Then by condition 2:
  \[
  g(n) = g(n-1) \cdot g(1) = g(n-1) \cdot 1 = g(n-1).
  \]
  Since \( n-1 < n \), by the inductive hypothesis \( g(n-1) = 1 \), so \( g(n) = 1 \).

Thus, by strong induction, \( g(n) = 1 \) for all \( n \).

\[
\boxed{1}
\]
'''
    }
}


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
{code}
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

The following are some examples:

{DEMONSTRATION_TEXT}Now, please translate the following formal statement and its proof.

## Formal Code
```lean4
{formal_code.strip()}
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
