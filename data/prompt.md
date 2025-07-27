You are an expert math teacher and IMO coach.

There are basically two methods of writing math problems:
- **Backwards**: It involves taking some basic idea, technique, or method X, and constructing a problem specifically using X. Examples include problems like "Not like a G6" (reverse-engineered from a complex number identity) and "2012-2013 Fall OMO #28" (using silly algebraic manipulations). This method can relatively quickly generate many "acceptable" questions, but the resulting problems often feel artificial.
- **Forwards**: It is less about intentionally trying to write problems and more about genuinely exploring math. It involves actions such as generalizing existing results, finding analogs of familiar ideas in new contexts, refining hypotheses, or focusing on a particular aspect of a proof. Examples include "2012-2013 Fall OMO #29", "2012-2013 Fall OMO #25", "2013-2014 USA December TST #2", and "2013 ELMO #3". This method is considered the most natural way to generate interesting and motivated problems.

Given a problem and a reference answer, please
1) Clearly restate variables, hypotheses, and the final conclusion (final answer) of the problem in Lean 4;
2) Try to solve the problem in natural language;
3) Identify which pattern the problem is constructed in, express the problem design process as a Markov decision process in Lean 4, and summarize the action sequence.
- **State**: the "problem design" state maintains variables, hypotheses, and conclusions.
- **Action**: two actions: *introduce* and *deduce*.
    - *Introduce*: add new variables or hypotheses into the state (`have`-statement by `sorry`);
    - *Deduce*: deduce new conclusions from the state (`have`-statement by a concrete proof).

The following is an example
# Problem
In $\triangle ABC$, the sides opposite to angles $A$, $B$, $C$ are denoted as $a$, $b$, $c$ respectively, and $\overrightarrow{m}=(\sqrt{3}b-c,\cos C)$, $\overrightarrow{n}=(a,\cos A)$. Given that $\overrightarrow{m} \parallel \overrightarrow{n}$, determine the value of $\cos A$.
# Answer
\dfrac{\sqrt{3}}{3}
# Analysis
## Problem Statement
```lean4
import Mathlib

example
  -- $ABC$ is a triangle
  (A B C : ℝ)
  (h_triangle_sum : A + B + C = Real.pi)
  (h_triangle_A : 0 < A ∧ A < Real.pi)
  (h_triangle_B : 0 < B ∧ B < Real.pi)
  (h_triangle_C : 0 < C ∧ C < Real.pi)
  -- the sides opposite to angles $A$, $B$, $C$ are denoted as $a$, $b$, $c$
  (a b c : ℝ)
  (h_triangle_ab : a / Real.sin A = b / Real.sin B)
  (h_triangle_bc : b / Real.sin B = c / Real.sin C)
  -- $\overrightarrow{m}=(\sqrt{3}b-c,\cos C)$, $\overrightarrow{n}=(a,\cos A)$
  (m n : ℝ × ℝ)
  (hm : m = (Real.sqrt 3 * b - c, Real.cos C))
  (hn : n = (a, Real.cos A))
  -- $\overrightarrow{m} \parallel \overrightarrow{n}$
  (h_parallel : m.1 * n.2 = m.2 * n.1)
  -- determine the value of $\cos A$.
  (answer : ℝ)
  (h_answer : answer = Real.cos A)
  -- # Answer \dfrac{\sqrt{3}}{3}
: (answer = Real.sqrt 3 / 3)
:= by sorry
```
## Action Sequence
```


```