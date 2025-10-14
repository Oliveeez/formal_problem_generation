## Index 453, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(m : ℝ)
(M : ℝ)
(hm : m < M)
(hM : M < 0)
(m' : ℝ)
(M' : ℝ)
(hpair : (m + m') / 2 = (M + M') / 2 ∧ √(m * m') = √(M * M'))
(hM' : 0 < m')
: 10 * |m' - M'| = 10 * |M - m|
:= sorry
```
### Informal Problem
Let \( m, M, m', M' \) be real numbers such that:
- \( m < M < 0 \),
- \( 0 < m' \),
- The average of \( m \) and \( m' \) equals the average of \( M \) and \( M' \), i.e., \( \frac{m + m'}{2} = \frac{M + M'}{2} \),
- The geometric mean of \( m \) and \( m' \) equals the geometric mean of \( M \) and \( M' \), i.e., \( \sqrt{m \cdot m'} = \sqrt{M \cdot M'} \).

Compute the value of \( 10 \cdot |m' - M'| \).
### Informal Answer
\( 10 \cdot |M - m| \)


## Index 3241, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℕ)
(hx : 0 < x)
(y : ℕ)
(hxy : x ≤ y)
(hy : 0 < y)
(z : ℕ)
(hyz : y ≤ z)
(hz : 0 < z)
(t : ℕ)
(h : x * y * z * t = 2016)
(hzt : z ≤ t)
(ht : 0 < t)
: x + y + z + t ≥ 21
:= sorry
```
### Informal Problem
Let \( x, y, z, t \) be positive integers such that:
- \( x \leq y \leq z \leq t \),
- \( x \cdot y \cdot z \cdot t = 2016 \).

What is the minimum possible value of \( x + y + z + t \)?
### Informal Answer
21


## Index 1014, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(p : ℝ → ℝ)
(hp : ∃ a b, p = fun x => a * x + b)
(h2p1 : p (2 * 1 - 1) = 3)
(q : ℝ → ℝ)
(h : ∀ x, p (2 * x - 1) = 4 * q x + 5)
(hq : ∃ c d, q = fun x => c * x + d)
: p 1 = 3
:= sorry
```
### Informal Problem
Let \( p \) and \( q \) be real-valued functions such that:
- \( p \) is linear, i.e., \( p(x) = a \cdot x + b \) for some real numbers \( a \) and \( b \),
- \( p(2 \cdot 1 - 1) = 3 \),
- For every real \( x \), \( p(2x - 1) = 4 \cdot q(x) + 5 \),
- \( q \) is linear, i.e., \( q(x) = c \cdot x + d \) for some real numbers \( c \) and \( d \).

Find the value of \( p(1) \).
### Informal Answer
3


## Index 2472, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(A : ℂ)
(hA : A = 1 - .I)
(B : ℂ)
(hB : B = 2 + 3 * .I)
(C : ℂ)
(hC : C = -4 + .I)
(D : ℂ)
(hD : D = 2 - 3 * .I)
(E : ℂ)
(hE : E = -2 - .I)
(F : ℂ)
(hF : F = -8 - 3 * .I)
(G : ℂ)
(hG : G = -2 + 3 * .I)
(H : ℂ)
(hH : H = 2 + 3 * .I)
(M : ℂ)
(hM : M = (A + B) / 2)
(N : ℂ)
(hN : N = (C + D) / 2)
(P : ℂ)
(hP : P = (E + F) / 2)
(Q : ℂ)
(hQ : Q = (G + H) / 2)
(a : ℝ)
(b : ℝ)
(h : (M + N + P + Q) / 4 = a + b * .I)
: a = -9 / 8 ∧ b = 1 / 4
:= sorry
```
### Informal Problem
Let \( A, B, C, D, E, F, G, H \) be complex numbers defined as follows:
- \( A = 1 - i \)
- \( B = 2 + 3i \)
- \( C = -4 + i \)
- \( D = 2 - 3i \)
- \( E = -2 - i \)
- \( F = -8 - 3i \)
- \( G = -2 + 3i \)
- \( H = 2 + 3i \)

Define the midpoints:
- \( M = \frac{A + B}{2} \)
- \( N = \frac{C + D}{2} \)
- \( P = \frac{E + F}{2} \)
- \( Q = \frac{G + H}{2} \)

Let \( a \) and \( b \) be real numbers such that the average of \( M, N, P, Q \) is \( a + b i \), i.e.,
\[
\frac{M + N + P + Q}{4} = a + b i.
\]
Find the values of \( a \) and \( b \).
### Informal Answer
\( a = -\frac{9}{8} \), \( b = \frac{1}{4} \)


## Index 1549, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(hn : n ≠ 0)
(m : ℕ)
(h1 : (n + m) % 201 = 0)
(h2 : (2 * n + m) % 201 = 0)
: (n + 3 * m) % 201 = 0
:= sorry
```
### Informal Problem
Let \( n \) and \( m \) be natural numbers with \( n \neq 0 \). Suppose that:
- \( (n + m) \mod 201 = 0 \), and
- \( (2n + m) \mod 201 = 0 \).

Find the value of \( (n + 3m) \mod 201 \).
### Informal Answer
0


## Index 3077, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(y : ℝ)
(z : ℝ)
(a : ℝ)
(h₀ : x * y = a)
(h₃ : a ≠ 0)
(b : ℝ)
(h₁ : x * z = b)
(h₄ : b ≠ 0)
(c : ℝ)
(h₂ : y * z = c)
(h₅ : c ≠ 0)
: ∃ (A : ℝ), A = x ^ 2 + y ^ 2 + z ^ 2 ∧ A = a * b / c + a * c / b + b * c / a
:= sorry
```
### Informal Problem
Let \( x, y, z, a, b, c \) be real numbers such that:
- \( x \cdot y = a \), with \( a \neq 0 \),
- \( x \cdot z = b \), with \( b \neq 0 \),
- \( y \cdot z = c \), with \( c \neq 0 \).

Express \( x^2 + y^2 + z^2 \) in terms of \( a, b, c \).
### Informal Answer
\[
a \cdot \frac{b}{c} + a \cdot \frac{c}{b} + b \cdot \frac{c}{a}
\]


## Index 2258, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x = Real.pi)
: x - 3 * Real.pi / 3 = 0
:= sorry
```
### Informal Problem
Let \( x \) be a real number such that \( x = \pi \). Compute the value of \( x - \frac{3\pi}{3} \).
### Informal Answer
0


## Index 4552, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(hn : n ≠ 1)
(h : 319 % n = 0 ∧ 232 % n = 0)
: n = 29
:= sorry
```
### Informal Problem
Let \( n \) be a natural number with \( n \ne 1 \). Suppose \( n \) divides both 319 and 232. What is the value of \( n \)?
### Informal Answer
29


## Index 2684, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℕ)
(ha : 0 < a)
(m : ℕ)
(n : ℕ)
(h1 : a * m + 2 * n = 18)
(h2 : 2 * m + 3 * n = 16)
(hn : 0 < n)
: a = 11 ∨ a = 5
:= sorry
```
### Informal Problem
Let \( a \) be a positive integer, and let \( m \) and \( n \) be nonnegative integers with \( n > 0 \). Suppose:
\[
a \cdot m + 2 \cdot n = 18,
\]
\[
2 \cdot m + 3 \cdot n = 16.
\]
Determine the possible values of \( a \).
### Informal Answer
11 or 5


## Index 1786, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(hn : n = 120)
(m : ℕ)
(hm : m = 3)
(c : ℕ)
(hc : c = 5)
: n * m * c = 1800
:= sorry
```
### Informal Problem
Let \( n = 120 \), \( m = 3 \), and \( c = 5 \). Compute the product \( n \times m \times c \).
### Informal Answer
1800


## Index 3739, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(b : ℝ)
(ha : a ≠ b)
(h1 : a^2 = 20*a-43)
(h2 : b^2 = 20*b-43)
: a * b = 43
:= sorry
```
### Informal Problem
Let \( a \) and \( b \) be distinct real numbers such that:
\[
a^2 = 20a - 43 \quad \text{and} \quad b^2 = 20b - 43.
\]
Compute the product \( a \times b \).
### Informal Answer
43


## Index 3999, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℤ)
(hx : x * (x + 1) * (x + 2) = 120)
: x + (x + 1) + (x + 2) = 15
:= sorry
```
### Informal Problem
Let \( x \) be an integer such that \( x(x + 1)(x + 2) = 120 \). Find the value of \( x + (x + 1) + (x + 2) \).
### Informal Answer
15


## Index 3122, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example

: 9 ^ 1995 % 7 = 1
:= sorry
```
### Informal Problem
Compute the remainder when \( 9^{1995} \) is divided by 7.
### Informal Answer
1


## Index 1924, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℝ → ℝ)
(h : ∀ x, f x = - f (-x))
(h' : ∀ x < 0, f x = -x^2 + 2 * x)
: ∀ x > 0, f x = x ^ 2 + 2 * x
:= sorry
```
### Informal Problem
Let \( f \) be a function from real numbers to real numbers such that:
- For every real number \( x \), \( f(x) = -f(-x) \).
- For every real number \( x < 0 \), \( f(x) = -x^2 + 2x \).

Find the value of \( f(x) \) for every real number \( x > 0 \).
### Informal Answer
\( x^2 + 2x \)


## Index 2013, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℝ → ℝ)
(hf : f = λ x => sqrt (x + 3) + sqrt (3 * x - 2) - 7)
(x : ℝ)
(hx : x ∈ Icc (2 / 3) 11)
(h : f x = 0)
: x = 6 ∨ x = 102 / 11
:= sorry
```
### Informal Problem
Let \( f \) be a function from real numbers to real numbers defined by  
\[
f(x) = \sqrt{x + 3} + \sqrt{3x - 2} - 7.
\]  
Suppose \( x \) is in the interval \( \left[\frac{2}{3}, 11\right] \) and satisfies \( f(x) = 0 \). Find all possible values of \( x \).
### Informal Answer
\( x = 6 \) or \( x = \frac{102}{11} \)


## Index 1627, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x = Real.sqrt 2)
(y : ℝ)
(hy : y = Real.sqrt 3)
(z : ℝ)
(hz : z = Real.sqrt 6)
: x * y * z = 6
:= sorry
```
### Informal Problem
Let \( x = \sqrt{2} \), \( y = \sqrt{3} \), and \( z = \sqrt{6} \). Compute the product \( x \cdot y \cdot z \).
### Informal Answer
6


## Index 825, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(s : ℕ → ℕ → ℕ)
(hs1 : ∀m, s m 0 = m)
(hs2 : ∀m n, s m (n + 1) = s m n + m)
: ∀ (m n : ℕ), s m n = m * (n + 1)
:= sorry
```
### Informal Problem
Let \( s \) be a function from \( \mathbb{N} \times \mathbb{N} \) to \( \mathbb{N} \) defined recursively by:
- \( s(m, 0) = m \) for all \( m \in \mathbb{N} \),
- \( s(m, n+1) = s(m, n) + m \) for all \( m, n \in \mathbb{N} \).

Find a closed-form expression for \( s(m, n) \) in terms of \( m \) and \( n \).
### Informal Answer
\( m \cdot (n + 1) \)


## Index 2236, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(h : n = 10)
: 40 ≤ n * (n - 1) * (n - 2) * (n - 3)
:= sorry
```
### Informal Problem
Let \( n \) be a natural number such that \( n = 10 \). Evaluate the expression \( n \cdot (n - 1) \cdot (n - 2) \cdot (n - 3) \), and determine if it is at least 40.
### Informal Answer
Yes, it is at least 40.


## Index 1401, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℂ → ℂ)
(hf : ∀ x, f x = (x ^ 4 + x ^ 2) / (x + 1))
: f Complex.I = 0
:= sorry
```
### Informal Problem
Let \( f \) be a function from complex numbers to complex numbers defined by  
\[
f(x) = \frac{x^4 + x^2}{x + 1}.
\]  
Compute the value of \( f(i) \), where \( i \) is the imaginary unit.
### Informal Answer
0


## Index 2364, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(ha : 0 ≤ a)
(b : ℝ)
(h1 : Real.sqrt a + Real.sqrt b = 5)
(h2 : a + b = 13)
(hb : 0 ≤ b)
: a = 9 ∧ b = 4 ∨ a = 4 ∧ b = 9
:= sorry
```
### Informal Problem
Let \( a \) and \( b \) be non-negative real numbers such that:
- \( \sqrt{a} + \sqrt{b} = 5 \), and
- \( a + b = 13 \).

Determine the possible values of \( a \) and \( b \).
### Informal Answer
Either \( a = 9 \) and \( b = 4 \), or \( a = 4 \) and \( b = 9 \).


## Index 4441, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(z : ℂ)
(hz : z = a ^ 2 + a - 2 + (a ^ 2 - 3 * a + 2) * Complex.I)
(h : z.re = 0 ∧ z.im ≠ 0)
: a = -2
:= sorry
```
### Informal Problem
Let \( a \) be a real number and \( z \) be a complex number defined by  
\[
z = (a^2 + a - 2) + (a^2 - 3a + 2)i.
\]  
Given that the real part of \( z \) is zero and the imaginary part is nonzero, determine the value of \( a \).
### Informal Answer
-2


## Index 849, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℤ)
(hx : 0 < x)
(y : ℤ)
(h : x * y = x + y)
(hy : 0 < y)
: (x, y) = (2, 2)
:= sorry
```
### Informal Problem
Let \( x \) and \( y \) be positive integers such that \( x \cdot y = x + y \). Find the ordered pair \( (x, y) \).
### Informal Answer
(2, 2)


## Index 890, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(h : n + (Nat.digits 10 n).sum = 125)
: n = 121
:= sorry
```
### Informal Problem
Let \( n \) be a natural number such that the sum of \( n \) and the sum of its base-10 digits equals 125. Find the value of \( n \).
### Informal Answer
121


## Index 4571, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a1 : ℝ)
(d : ℝ)
(h : d ≠ 0)
(n : ℕ)
(hn : n > 0)
(Sodd : ℝ)
(Seven : ℝ)
(hS : Sodd = Seven + 60)
(m : ℕ)
(h_a1 : a1 = (n + 2 * m) * d / 2)
(hSeven : Seven = n * a1 / 2)
(hSodd : Sodd = (n + m) * d)
(hm : m > 0)
: Sodd - Seven = 60
:= sorry
```
### Informal Problem
Let \( a_1 \) and \( d \) be real numbers with \( d \neq 0 \), and let \( n \) and \( m \) be positive integers. Define the following sums:
- \( S_{\text{odd}} = (n + m) \cdot d \)
- \( S_{\text{even}} = \frac{n \cdot a_1}{2} \)
where \( a_1 = \frac{(n + 2m) \cdot d}{2} \).  
Given that \( S_{\text{odd}} = S_{\text{even}} + 60 \), compute the value of \( S_{\text{odd}} - S_{\text{even}} \).
### Informal Answer
60


## Index 4982, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(y : ℝ)
(z : ℝ)
(h1 : x^2 - y * z = -23)
(h2 : y^2 - x * z = -4)
(h3 : z^2 - x * y = 34)
: x = 5 ∨ x = -5
:= sorry
```
### Informal Problem
Let \( x, y, z \) be real numbers satisfying the following equations:
\[
x^2 - y \cdot z = -23, \quad y^2 - x \cdot z = -4, \quad z^2 - x \cdot y = 34.
\]
Determine the possible values of \( x \).
### Informal Answer
\( x = 5 \) or \( x = -5 \)


## Index 1621, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(hn : n > 0)
(h : (n - 1) + (n - 2) + (n - 3) + (n - 4) + (n - 5) = 100)
: n = 23
:= sorry
```
### Informal Problem
Let \( n \) be a positive integer. If the sum  
\[
(n - 1) + (n - 2) + (n - 3) + (n - 4) + (n - 5) = 100,
\]  
what is the value of \( n \)?
### Informal Answer
23


## Index 2492, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(h₀ : n > 0)
(h₁ : ∑ i ∈ Finset.range n, (i + 1) = 2016)
: (Nat.digits 10 n).sum = 9
:= sorry
```
### Informal Problem
Let \( n \) be a positive natural number. Suppose that the sum of the first \( n \) positive integers is 2016, i.e.,
\[
\sum_{i=0}^{n-1} (i + 1) = 2016.
\]
Find the sum of the digits of \( n \) in base 10.
### Informal Answer
9


## Index 251, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(hn : n > 0)
(h1 : 10 % n = 1)
(h2 : 14 % n = 2)
: n = 3 ∨ n = 6 ∨ n = 9
:= sorry
```
### Informal Problem
Let \( n \) be a positive integer such that:
- The remainder when 10 is divided by \( n \) is 1.
- The remainder when 14 is divided by \( n \) is 2.

Find all possible values of \( n \).
### Informal Answer
3, 6, or 9


## Index 844, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(ha : 2 * a = Real.sqrt 2)
(b : ℝ)
(hb : 2 * b = Real.sqrt 3)
(c : ℝ)
(hc : 2 * c = Real.sqrt 6)
: a * b * c = 3 / 4
:= sorry
```
### Informal Problem
Let \( a, b, c \) be real numbers such that:
- \( 2a = \sqrt{2} \),
- \( 2b = \sqrt{3} \),
- \( 2c = \sqrt{6} \).

Compute the product \( a \cdot b \cdot c \).
### Informal Answer
\[
\boxed{\frac{3}{4}}
\]


## Index 2043, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(y : ℝ)
(z : ℝ)
(h1 : x - y + z - 2 = 0)
(h2 : x + y - z + 2 = 0)
(h3 : 2 * y * z - y ^ 2 - 9 = 0)
: (x, y, z) = (0, -2 - √13, -√13) ∨ (x, y, z) = (0, -2 + √13, √13)
:= sorry
```
### Informal Problem
Let \( x, y, z \) be real numbers satisfying the following system of equations:
1. \( x - y + z - 2 = 0 \),
2. \( x + y - z + 2 = 0 \),
3. \( 2yz - y^2 - 9 = 0 \).

Find all possible values of \( (x, y, z) \).
### Informal Answer
The solutions are \( (x, y, z) = (0, -2 - \sqrt{13}, -\sqrt{13}) \) and \( (x, y, z) = (0, -2 + \sqrt{13}, \sqrt{13}) \).


## Index 4315, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℝ → ℝ)
(hf : ∀ x, x ≥ 1 → f x = x ^ 2 - 2 * x)
(hf' : ∀ x, x < 1 → f x = 2 - 2 * x)
(x : ℝ)
(hx : f x = f (Real.sqrt 2 / 2))
(h : 0 < x)
(h' : x < 1)
: x = √2 / 2
:= sorry
```
### Informal Problem
Let \( f \) be a real-valued function defined piecewise as:
- \( f(x) = x^2 - 2x \) for \( x \geq 1 \),
- \( f(x) = 2 - 2x \) for \( x < 1 \).

Suppose \( x \) is a real number such that \( 0 < x < 1 \) and \( f(x) = f\left(\frac{\sqrt{2}}{2}\right) \). Find the value of \( x \).
### Informal Answer
\[
\boxed{\frac{\sqrt{2}}{2}}
\]


## Index 335, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x > 0)
(y : ℝ)
(hxy : y = x + 27)
(h : x * 90 = y * 30)
: x = 13.5
:= sorry
```
### Informal Problem
Let \( x \) be a positive real number and define \( y = x + 27 \). Given that \( x \cdot 90 = y \cdot 30 \), find the value of \( x \).
### Informal Answer
13.5


## Index 212, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(b : ℝ)
(h1 : 6*a=5*b)
(c : ℝ)
(h2 : 3*b=11*c)
: 36 * a = 110 * c
:= sorry
```
### Informal Problem
Let \( a \), \( b \), and \( c \) be real numbers such that \( 6a = 5b \) and \( 3b = 11c \). Find the value of \( 36a \) in terms of \( c \).
### Informal Answer
\( 110c \)


## Index 1186, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(c : ℝ)
(l : ℝ)
(t : ℝ)
(h₀ : 0 < c ∧ 0 < l ∧ 0 < t)
(h₂ : l + t = 2 * c)
(r : ℝ)
(h₃ : c + t = 2 * r)
(h₁ : 0 < r ∧ 2 * r > c)
(s : ℝ)
(h₄ : c + l = 2 * s)
(h₅ : r + s = 2 * 10)
: c = 10
:= sorry
```
### Informal Problem
Let \( c, l, t, r, s \) be positive real numbers such that:
- \( l + t = 2c \),
- \( c + t = 2r \),
- \( c + l = 2s \),
- \( r + s = 20 \),
- \( r > 0 \) and \( 2r > c \).

Find the value of \( c \).
### Informal Answer
10


## Index 4375, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example

: GCDMonoid.lcm (GCDMonoid.lcm 51 68) 85 = 1020
:= sorry
```
### Informal Problem
Compute the least common multiple (LCM) of the LCM of 51 and 68, and 85.
### Informal Answer
1020


## Index 1732, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℕ)
(y : ℕ)
(z : ℕ)
(h : 13 * x + 8 * y = 5 * z)
: 39 * x + 24 * y = 15 * z
:= sorry
```
### Informal Problem
Let \( x \), \( y \), and \( z \) be natural numbers such that \( 13x + 8y = 5z \). Compute the value of \( 39x + 24y \) in terms of \( z \).
### Informal Answer
\( 15z \)


## Index 3824, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℝ → ℝ → ℝ)
(hf : f = fun x y => x + y)
(g : ℝ → ℝ → ℝ)
(hg : g = fun x y => x * y)
(A : ℝ × ℝ)
(hA : A.1 = f 1 2 ∧ A.2 = g 1 2)
(B : ℝ × ℝ)
(hB : B.1 = f 2 3 ∧ B.2 = g 2 3)
(C : ℝ × ℝ)
(hC : C.1 = f A.1 B.2 ∧ C.2 = g A.1 B.2)
: C = (9, 18)
:= sorry
```
### Informal Problem
Let \( f \) and \( g \) be functions from \( \mathbb{R} \times \mathbb{R} \) to \( \mathbb{R} \) defined by \( f(x, y) = x + y \) and \( g(x, y) = x \cdot y \). Define points \( A \), \( B \), and \( C \) in \( \mathbb{R} \times \mathbb{R} \) as follows:
- \( A = (f(1, 2), g(1, 2)) \),
- \( B = (f(2, 3), g(2, 3)) \),
- \( C = (f(A_1, B_2), g(A_1, B_2)) \), where \( A_1 \) is the first coordinate of \( A \) and \( B_2 \) is the second coordinate of \( B \).

Find the coordinates of \( C \).
### Informal Answer
(9, 18)


## Index 4271, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(y : ℝ)
(hxy : x / y = (x + y) / (2 * x))
(hy : y ≠ 0)
(hx : x ≠ 0)
(hxy' : x / y ≠ 1)
(r : ℝ)
(h : x / y = r)
: r = -1 / 2
:= sorry
```
### Informal Problem
Let \( x \) and \( y \) be nonzero real numbers such that \( x / y \neq 1 \). Suppose that \( x / y = (x + y) / (2x) \). If \( x / y = r \), find the value of \( r \).
### Informal Answer
\( -\frac{1}{2} \)


## Index 2161, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℕ)
(hx : 3*x+2*6=42)
: x = 10
:= sorry
```
### Informal Problem
Let \( x \) be a natural number such that \( 3x + 2 \times 6 = 42 \). Find the value of \( x \).
### Informal Answer
10


## Index 4132, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(b : ℝ)
(c : ℝ)
(f : ℝ → ℝ)
(hf : ∀ x, f x = a * x ^ 7 + b * x ^ 3 + c * x - 5)
(h : f (-7) = 7)
: f 7 = -17
:= sorry
```
### Informal Problem
Let \( a, b, c \) be real numbers. Define a function \( f : \mathbb{R} \to \mathbb{R} \) by  
\[
f(x) = a \cdot x^7 + b \cdot x^3 + c \cdot x - 5.
\]  
Given that \( f(-7) = 7 \), what is the value of \( f(7) \)?
### Informal Answer
-17


## Index 1028, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(y : ℝ)
(h1 : 5 * abs x + 3 * abs y = 20)
(h2 : 3 * abs x + 5 * abs y = 20)
: x = 5 / 2 ∧ y = 5 / 2 ∨ x = 5 / 2 ∧ y = -(5 / 2) ∨ x = -(5 / 2) ∧ y = 5 / 2 ∨ x = -(5 / 2) ∧ y = -(5 / 2)
:= sorry
```
### Informal Problem
Let \( x \) and \( y \) be real numbers such that:
\[
5|x| + 3|y| = 20 \quad \text{and} \quad 3|x| + 5|y| = 20.
\]
Find all possible values of \( x \) and \( y \).
### Informal Answer
The possible pairs are:
- \( x = \frac{5}{2} \) and \( y = \frac{5}{2} \)
- \( x = \frac{5}{2} \) and \( y = -\frac{5}{2} \)
- \( x = -\frac{5}{2} \) and \( y = \frac{5}{2} \)
- \( x = -\frac{5}{2} \) and \( y = -\frac{5}{2} \)


## Index 1532, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(S : ℕ → ℝ)
(n : ℕ)
(hS : S n = ∑ i in (Finset.Icc 1 n), (1 / ((i * (i + 1)) : ℝ)))
(h : n = 100)
: S n * 101 = 100
:= sorry
```
### Informal Problem
Let \( S(n) \) be defined as the sum:
\[
S(n) = \sum_{i=1}^{n} \frac{1}{i \cdot (i + 1)}.
\]
Given that \( n = 100 \), compute the value of \( S(n) \times 101 \).
### Informal Answer
100


## Index 3910, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℕ)
(y : ℕ)
(z : ℕ)
(h : 1*x+10*y+25*z=99)
: y = 0 ∨ y = 1 ∨ y = 2 ∨ y = 3 ∨ y = 4 ∨ y = 5 ∨ y = 6 ∨ y = 7 ∨ y = 8 ∨ y = 9
:= sorry
```
### Informal Problem
Let \( x \), \( y \), and \( z \) be natural numbers such that:
\[
1 \cdot x + 10 \cdot y + 25 \cdot z = 99.
\]
Determine all possible values of \( y \).
### Informal Answer
0, 1, 2, 3, 4, 5, 6, 7, 8, or 9


## Index 4578, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(b : ℝ)
(c : ℝ)
(f : ℝ → ℝ)
(hf : ∀ x, f x = x ^ 2 + b * x + c)
(h1 : 3 = f 2)
(h2 : 3 = f 4)
: c = 11
:= sorry
```
### Informal Problem
Let \( b \) and \( c \) be real numbers. Define a quadratic function \( f : \mathbb{R} \to \mathbb{R} \) by  
\[
f(x) = x^2 + b \cdot x + c.
\]  
Given that \( f(2) = 3 \) and \( f(4) = 3 \), find the value of \( c \).
### Informal Answer
11


## Index 4508, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(t : ℝ)
(ht : t > 0)
(h : (100 + t) * 0.02 = 100 * 0.03)
: t = 50
:= sorry
```
### Informal Problem
Suppose \( t \) is a positive real number. If the equation  
\[
(100 + t) \times 0.02 = 100 \times 0.03
\]  
holds, what is the value of \( t \)?
### Informal Answer
50


## Index 4402, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(y : ℝ)
(z : ℝ)
(h : x * y * z ≠ 0)
(a : ℝ)
(h1 : (x + y - z) / z = a)
(h2 : (x - y + z) / y = a)
(h3 : (-x + y + z) / x = a)
(ha : a ≠ 1)
: (x ^ 2 + y ^ 2 + z ^ 2) / (x * y + y * z + z * x) = -2
:= sorry
```
### Informal Problem
Let \( x, y, z \) be nonzero real numbers. Suppose there exists a real number \( a \ne 1 \) such that:
\[
\frac{x + y - z}{z} = a, \quad \frac{x - y + z}{y} = a, \quad \frac{-x + y + z}{x} = a.
\]
Find the value of:
\[
\frac{x^2 + y^2 + z^2}{x y + y z + z x}.
\]
### Informal Answer
-2


## Index 3650, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℕ)
(b : ℕ)
(c : ℕ)
(h : 2^a * 3^b * 5^c = 36000)
: 4 * a + 3 * b + 5 * c ≤ 62
:= sorry
```
### Informal Problem
Let \( a, b, c \) be natural numbers such that  
\[
2^a \cdot 3^b \cdot 5^c = 36000.
\]  
Find the maximum possible value of \( 4a + 3b + 5c \).
### Informal Answer
62


## Index 130, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℕ)
(hx : x ∈ Finset.Icc 100 999)
(y : ℕ)
(hxy : 1000 * x + y = 7 * (x * y))
(hy : y ∈ Finset.Icc 100 999)
: x = 143 ∧ y = 143
:= sorry
```
### Informal Problem
Let \( x \) and \( y \) be three-digit natural numbers (i.e., \( 100 \leq x \leq 999 \) and \( 100 \leq y \leq 999 \)) such that  
\[
1000 \cdot x + y = 7 \cdot (x \cdot y).
\]  
Find the values of \( x \) and \( y \).
### Informal Answer
\( x = 143 \) and \( y = 143 \)


## Index 2919, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(c : ℝ)
(f : ℝ → ℝ)
(hf : f = fun x => x^2 + c * x + 4)
(h : ∃ x1 x2, x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0 ∧ x1^2 + x2^2 = 13)
: c ^ 2 = 21
:= sorry
```
### Informal Problem
Let \( c \) be a real number and define a function \( f : \mathbb{R} \to \mathbb{R} \) by  
\[
f(x) = x^2 + c \cdot x + 4.
\]  
Suppose there exist two distinct real numbers \( x_1 \) and \( x_2 \) such that \( f(x_1) = 0 \), \( f(x_2) = 0 \), and \( x_1^2 + x_2^2 = 13 \).  
Find the value of \( c^2 \).
### Informal Answer
21


## Index 3722, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example

: (Real.cos (5 / 11 * Real.pi) - 1) / Real.sin (6 / 11 * Real.pi) +
    (Real.cos (6 / 11 * Real.pi) - 1 / 2) / Real.sin (6 / 11 * Real.pi) =
  -3 / (2 * Real.sin (6 / 11 * Real.pi))
:= sorry
```
### Informal Problem
Simplify the expression:
\[
\frac{\cos\left(\frac{5}{11}\pi\right) - 1}{\sin\left(\frac{6}{11}\pi\right)} + \frac{\cos\left(\frac{6}{11}\pi\right) - \frac{1}{2}}{\sin\left(\frac{6}{11}\pi\right)}.
\]
### Informal Answer
\[
-\frac{3}{2 \sin\left(\frac{6}{11}\pi\right)}
\]


## Index 1826, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(hn : n > 0)
(h : n / 2 + 10 = n)
: n = 20 ∨ n = 19
:= sorry
```
### Informal Problem
Let \( n \) be a positive integer. If \( \frac{n}{2} + 10 = n \), what are the possible values of \( n \)?
### Informal Answer
19 or 20


## Index 2517, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(u : ℤ)
(v : ℤ)
(h : u^2 = v^2 + 7 * v + 6)
: 2 * v + 7 - 2 * u = 1 ∨
  2 * v + 7 - 2 * u = 5 ∨
    2 * v + 7 - 2 * u = 25 ∨ 2 * v + 7 - 2 * u = -1 ∨ 2 * v + 7 - 2 * u = -5 ∨ 2 * v + 7 - 2 * u = -25
:= sorry
```
### Informal Problem
Let \( u \) and \( v \) be integers such that \( u^2 = v^2 + 7v + 6 \). Determine all possible integer values of the expression \( 2v + 7 - 2u \).
### Informal Answer
The possible values are \( 1, 5, 25, -1, -5, -25 \).


## Index 631, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(θ : ℝ)
(hθ : θ ∈ Set.Ioo 0 (Real.pi / 2))
(h : 1 / Real.tan θ + Real.tan θ = 4)
(m : ℝ)
(hm : 0 < m)
(n : ℝ)
(hmn : Real.sin θ - Real.cos θ = m * Real.sqrt n)
(hn : 0 < n)
: m ^ 2 * n = 1 / 2
:= sorry
```
### Informal Problem
Let \( \theta \) be a real number in the interval \( (0, \frac{\pi}{2}) \). Suppose that  
\[
\frac{1}{\tan \theta} + \tan \theta = 4,
\]  
and let \( m \) and \( n \) be positive real numbers such that  
\[
\sin \theta - \cos \theta = m \cdot \sqrt{n}.
\]  
Find the value of \( m^2 \cdot n \).
### Informal Answer
\( \frac{1}{2} \)


## Index 1775, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : 0 < x)
(y : ℝ)
(h1 : x - y^2 = 3)
(h2 : x^2 + y^4 = 13)
: x = (3 + √17) / 2
:= sorry
```
### Informal Problem
Let \( x \) be a positive real number and \( y \) be a real number such that:
- \( x - y^2 = 3 \), and
- \( x^2 + y^4 = 13 \).

Find the value of \( x \).
### Informal Answer
\[
\boxed{\frac{3 + \sqrt{17}}{2}}
\]


## Index 3207, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℕ)
(b : ℕ)
(ha : a + b = 315)
(c : ℕ)
(hb : b + c = 340)
(d : ℕ)
(hc : c + d = 310)
(e : ℕ)
(hd : d + e = 320)
(f : ℕ)
(he : e + f = 285)
(hf : e + f + a = 305)
: a = 20 ∧ b = 295 ∧ c = 45 ∧ d = 265 ∧ e = 55 ∧ f = 230
:= sorry
```
### Informal Problem
Let \( a, b, c, d, e, f \) be natural numbers satisfying the following equations:
- \( a + b = 315 \)
- \( b + c = 340 \)
- \( c + d = 310 \)
- \( d + e = 320 \)
- \( e + f = 285 \)
- \( e + f + a = 305 \)

Find the values of \( a, b, c, d, e, f \).
### Informal Answer
\( a = 20 \), \( b = 295 \), \( c = 45 \), \( d = 265 \), \( e = 55 \), \( f = 230 \)


## Index 899, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(h : (1 + x / 100) * (1 - x / 100) = 84 / 100)
: |x| = 40
:= sorry
```
### Informal Problem
Let \( x \) be a real number such that  
\[
\left(1 + \frac{x}{100}\right) \left(1 - \frac{x}{100}\right) = \frac{84}{100}.
\]  
Find the value of \( |x| \).
### Informal Answer
40


## Index 46, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x^2 - 49 = 0)
(a : ℝ)
(ha : 0 < a)
(b : ℝ)
(hroots : x^2 - 49 = (x - a) * (x - b))
(hb : b < 0)
: x = -7 ∨ x = 7
:= sorry
```
### Informal Problem
Let \( x \) be a real number such that \( x^2 - 49 = 0 \). Let \( a \) and \( b \) be real numbers with \( a > 0 \) and \( b < 0 \), and suppose that \( x^2 - 49 = (x - a)(x - b) \). Determine the possible values of \( x \).
### Informal Answer
\( x = -7 \) or \( x = 7 \)


## Index 280, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℝ → ℝ)
(hf : f = fun x => x ^ 2)
(g : ℝ → ℝ)
(hg : g = fun x => 2 * x ^ 2)
: ∃ (c : ℝ), ∀ (x : ℝ), g x = c * f x
:= sorry
```
### Informal Problem
Let \( f \) and \( g \) be functions from the real numbers to the real numbers defined by \( f(x) = x^2 \) and \( g(x) = 2x^2 \). Find a real number \( c \) such that for every real number \( x \), \( g(x) = c \cdot f(x) \).
### Informal Answer
2


## Index 2221, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℕ)
(ha : a < 7)
(b : ℕ)
(hb : b < 7)
(h3 : 6 * b ≡ 3 + b [MOD 7])
(c : ℕ)
(h1 : a * b * c ≡ 1 [MOD 7])
(h2 : 5 * c ≡ 2 [MOD 7])
(hc : c < 7)
: (a + b + c) % 7 = 4
:= sorry
```
### Informal Problem
Let \( a, b, c \) be natural numbers less than 7. Suppose:
- \( 6 \cdot b \equiv 3 + b \pmod{7} \),
- \( a \cdot b \cdot c \equiv 1 \pmod{7} \),
- \( 5 \cdot c \equiv 2 \pmod{7} \).

Find the value of \( (a + b + c) \mod 7 \).
### Informal Answer
4


## Index 1165, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(cedric : ℝ)
(hcedric : cedric = 12000 * (1 + 0.05)^15)
(daniel : ℝ)
(hdaniel : daniel = 12000 * (1 + 0.07 * 15))
: round (cedric - daniel) = 347
:= sorry
```
### Informal Problem
Cedric invests \$12,000 at an annual interest rate of 5% compounded annually for 15 years. Daniel invests \$12,000 at an annual simple interest rate of 7% for 15 years. Calculate the difference between Cedric's final amount and Daniel's final amount, rounded to the nearest integer.
### Informal Answer
347


## Index 1297, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℕ)
(hx : x ≠ 0)
(h : x / 3 + 5 = 15 - x / 3)
: x / 3 = 5
:= sorry
```
### Informal Problem
Let \( x \) be a nonzero natural number. If \( \frac{x}{3} + 5 = 15 - \frac{x}{3} \), what is the value of \( \frac{x}{3} \)?
### Informal Answer
5


## Index 4760, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(c : ℝ)
(p : ℝ → ℝ)
(h : ∀ x, x^2 - 2 * x + c ≠ 0 → p x = x^4 - 3 * x^3 + 4 * x^2 - 3 * x + 1)
(hp : ∀ x, x^2 - 2 * x + c = 0 → p x = 0)
(m : ℝ)
(hm : p m = 0)
(hmp : m > 1)
(n : ℝ)
(hn : p n = 0)
(hmn : m > n)
(hnp : n < 0)
: m - n = 2 * (m - 1)
:= sorry
```
### Informal Problem
Let \( c \) be a real number and \( p : \mathbb{R} \to \mathbb{R} \) be a function such that:
- For all \( x \), if \( x^2 - 2x + c \neq 0 \), then \( p(x) = x^4 - 3x^3 + 4x^2 - 3x + 1 \).
- For all \( x \), if \( x^2 - 2x + c = 0 \), then \( p(x) = 0 \).

Suppose \( m \) and \( n \) are real numbers with \( p(m) = 0 \), \( p(n) = 0 \), \( m > 1 \), \( m > n \), and \( n < 0 \).  
Express \( m - n \) in terms of \( m \).
### Informal Answer
\( 2(m - 1) \)


## Index 1716, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℝ → ℝ)
(g : ℝ → ℝ)
(hf : Function.LeftInverse g f)
(hg : Function.RightInverse g f)
(h₁ : g (-16) = 0)
(h₂ : g 0 = 2)
(h₃ : g 2 = 16)
: f 16 = 2
:= sorry
```
### Informal Problem
Let \( f \) and \( g \) be functions from the real numbers to the real numbers such that:
- \( g \) is a left inverse of \( f \), i.e., \( g(f(x)) = x \) for all \( x \),
- \( g \) is a right inverse of \( f \), i.e., \( f(g(x)) = x \) for all \( x \),
- \( g(-16) = 0 \),
- \( g(0) = 2 \),
- \( g(2) = 16 \).

Find the value of \( f(16) \).
### Informal Answer
2


## Index 1960, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℝ → ℝ)
(h : ∀ x y, f x + f y ≥ x * y)
: f 1 ≥ 1 / 2
:= sorry
```
### Informal Problem
Let \( f \) be a function from real numbers to real numbers such that for all real numbers \( x \) and \( y \),  
\[
f(x) + f(y) \geq x \cdot y.
\]  
Determine a lower bound for \( f(1) \).
### Informal Answer
\( \frac{1}{2} \)


## Index 3964, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x1 : ℤ)
(x2 : ℤ)
(y1 : ℤ)
(h1 : x1 * 3 = y1 - 2)
(h3 : y1 = 17)
(y2 : ℤ)
(h2 : x2 * 5 = y2 - 2)
(h4 : y2 = 27)
: x1 + x2 = 10
:= sorry
```
### Informal Problem
Let \( x_1, x_2, y_1, y_2 \) be integers such that:
- \( 3x_1 = y_1 - 2 \),
- \( y_1 = 17 \),
- \( 5x_2 = y_2 - 2 \),
- \( y_2 = 27 \).

Find the value of \( x_1 + x_2 \).
### Informal Answer
10


## Index 243, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : 0 < x)
(y : ℝ)
(h1 : x^2 * y^3 = 16)
(h2 : x^3 * y^2 = 2)
(hy : 0 < y)
: x = 1 / 2 ∧ y = 4
:= sorry
```
### Informal Problem
Let \( x \) and \( y \) be positive real numbers such that:
\[
x^2 \cdot y^3 = 16 \quad \text{and} \quad x^3 \cdot y^2 = 2.
\]
Find the values of \( x \) and \( y \).
### Informal Answer
\( x = \frac{1}{2} \) and \( y = 4 \)


## Index 751, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : 0 < x)
(y : ℝ)
(hy : 0 < y)
(b : ℝ)
(hb : b > x ∧ b > y)
(h : (b - x) * (b - y) = x * y)
(hb' : 2 * b = 18)
: x + y = 9
:= sorry
```
### Informal Problem
Let \( x \) and \( y \) be positive real numbers. Suppose there exists a real number \( b \) such that \( b > x \), \( b > y \), and \( (b - x)(b - y) = x \cdot y \). If \( 2b = 18 \), what is the value of \( x + y \)?
### Informal Answer
9


## Index 4396, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(z : ℂ)
(w : ℂ)
(h₀ : z + w = 105)
(h₁ : z - w = 21)
: z.re + w.re = 105
:= sorry
```
### Informal Problem
Let \( z \) and \( w \) be complex numbers such that \( z + w = 105 \) and \( z - w = 21 \). Compute the sum of the real parts of \( z \) and \( w \), i.e., \( \text{Re}(z) + \text{Re}(w) \).
### Informal Answer
105


## Index 29, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(k : ℕ)
(a : ℕ)
(ha : 1 < a)
(b : ℕ)
(hb : 1 < b)
(c : ℕ)
(h1 : a * b = c)
(hc : 1 < c)
(d : ℕ)
(h2 : b * c = d)
(hd : 1 < d)
(e : ℕ)
(h3 : c * d = e)
(he : 1 < e)
(f : ℕ)
(h4 : d * e = f)
(h5 : e * f = k)
(hf : 1 < f)
: k ≥ 36
:= sorry
```
### Informal Problem
Let \( a, b, c, d, e, f, k \) be natural numbers such that:
- \( a > 1 \), \( b > 1 \), \( c > 1 \), \( d > 1 \), \( e > 1 \), \( f > 1 \),
- \( a \cdot b = c \),
- \( b \cdot c = d \),
- \( c \cdot d = e \),
- \( d \cdot e = f \),
- \( e \cdot f = k \).

Determine the minimum possible value of \( k \).
### Informal Answer
36


## Index 60, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(M : Set ℝ)
(hM : M = {x | x ^ 2 - 2 * x - 3 ≤ 0})
(N : Set ℝ)
(hN : N = {x | abs x < 2})
: M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2}
:= sorry
```
### Informal Problem
Let \( M \) be the set of real numbers \( x \) such that \( x^2 - 2x - 3 \leq 0 \), and let \( N \) be the set of real numbers \( x \) such that \( |x| < 2 \). Find the intersection \( M \cap N \).
### Informal Answer
\( \{x \in \mathbb{R} \mid -1 \leq x < 2\} \)


## Index 603, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(ha : 0 < a)
(b : ℝ)
(hb : 0 < b)
(f : ℝ → ℝ)
(hf : ∀ x, f x = a * x ^ 4 - b * x ^ 2 + x + 7)
(h : f (-5) = 3)
: f 5 = 13
:= sorry
```
### Informal Problem
Let \( a \) and \( b \) be positive real numbers. Define a function \( f : \mathbb{R} \to \mathbb{R} \) by  
\[
f(x) = a \cdot x^4 - b \cdot x^2 + x + 7.
\]  
Given that \( f(-5) = 3 \), what is the value of \( f(5) \)?
### Informal Answer
13


## Index 3690, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℝ → ℝ)
(h0 : f 0 = 1)
(hf : ∀ x, f (x + 4) = f x + 5)
: f 4 + f 8 + f 12 + f 16 = 54
:= sorry
```
### Informal Problem
Let \( f \) be a function from the real numbers to the real numbers such that:
- \( f(0) = 1 \), and
- For every real number \( x \), \( f(x + 4) = f(x) + 5 \).

Compute the value of \( f(4) + f(8) + f(12) + f(16) \).
### Informal Answer
54


## Index 1215, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(hn : n ≠ 0)
(h1 : Nat.lcm n 18 = 180)
(h2 : Nat.gcd n 45 = 15)
: n = 60
:= sorry
```
### Informal Problem
Let \( n \) be a positive natural number. Suppose that the least common multiple of \( n \) and 18 is 180, and the greatest common divisor of \( n \) and 45 is 15. Find the value of \( n \).
### Informal Answer
60


## Index 1441, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : 0 ≤ x)
(y : ℝ)
(hy : 0 ≤ y)
(z : ℝ)
(h1 : Real.sqrt x + Real.sqrt y = z - 1)
(h2 : Real.sqrt (x * y) = (z - 1) ^ 2 / 2)
(hz : 1 ≤ z)
: x + y = (z - 1) ^ 2 / 2
:= sorry
```
### Informal Problem
Let \( x \) and \( y \) be non-negative real numbers, and let \( z \) be a real number such that \( z \geq 1 \). Suppose:
- \( \sqrt{x} + \sqrt{y} = z - 1 \), and
- \( \sqrt{x \cdot y} = \frac{(z - 1)^2}{2} \).

Find the value of \( x + y \).
### Informal Answer
\[
\frac{(z - 1)^2}{2}
\]


## Index 2555, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(p : ℝ)
(hp : p^2 + 2 * p - 1 = 0)
(q : ℝ)
(hne : p ≠ q)
(hq : q^2 + 2 * q - 1 = 0)
: 1 / p ^ 2 + 1 / q ^ 2 = 6
:= sorry
```
### Informal Problem
Let \( p \) and \( q \) be distinct real numbers that both satisfy the equation \( x^2 + 2x - 1 = 0 \). Compute the value of \( \frac{1}{p^2} + \frac{1}{q^2} \).
### Informal Answer
6


## Index 3137, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(P : ℝ)
(hP : P = 4000000)
(r : ℝ)
(hr : r = 0.1)
(t : ℝ)
(ht : t = 5)
: P * (1 + r * t) = 6000000
:= sorry
```
### Informal Problem
An investment has a principal amount \( P \), an annual interest rate \( r \), and a time period \( t \) (in years). Given that \( P = 4,000,000 \), \( r = 0.1 \), and \( t = 5 \), compute the total amount after \( t \) years using simple interest, i.e., \( P \times (1 + r \times t) \).
### Informal Answer
6000000


## Index 1082, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℕ)
(b : ℕ)
(h₀ : a / b = 3)
(h₁ : a % b = 4)
(h₂ : a + b + 3 + 4 = 91)
: a = 64 ∧ b = 20
:= sorry
```
### Informal Problem
Let \( a \) and \( b \) be natural numbers such that:
- The quotient of \( a \) divided by \( b \) is 3,
- The remainder of \( a \) divided by \( b \) is 4, and
- The sum \( a + b + 3 + 4 = 91 \).

Find the values of \( a \) and \( b \).
### Informal Answer
\( a = 64 \) and \( b = 20 \)


## Index 4628, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(ha : a ≠ 0)
(b : ℝ)
(h1 : a^2 + a = b^2)
(hb : b ≠ 0)
(c : ℝ)
(h3 : c^2 + c = a^2)
(h2 : b^2 + b = c^2)
(hc : c ≠ 0)
: a + b + c - 1991 = -1991
:= sorry
```
### Informal Problem
Let \( a, b, c \) be nonzero real numbers satisfying the following system of equations:
\[
\begin{aligned}
a^2 + a &= b^2, \\
b^2 + b &= c^2, \\
c^2 + c &= a^2.
\end{aligned}
\]
Find the value of \( a + b + c - 1991 \).
### Informal Answer
-1991


## Index 4559, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(P : ℝ)
(hP : P = 10000)
(r : ℝ)
(hr : r = 0.1)
: ⌊P * (1 + r) ^ 10 - P * (1 + r / 4) ^ 20 / 2 * (1 + (1 + r / 4) ^ 20) + 1 / 2⌋ = 4319
:= sorry
```
### Informal Problem
Let \( P = 10000 \) and \( r = 0.1 \). Compute the floor of the expression:
\[
P \cdot (1 + r)^{10} - \frac{P \cdot (1 + \frac{r}{4})^{20}}{2} \cdot \left(1 + (1 + \frac{r}{4})^{20}\right) + \frac{1}{2}.
\]
### Informal Answer
4319


## Index 1322, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(y : ℝ)
(z : ℝ)
(h₀ : x + y + z = 1)
(h₁ : x - y + 2 * z = 3)
(h₂ : 2 * x - 2 * y + 3 * z = 5)
: x - y + z = 2
:= sorry
```
### Informal Problem
Let \( x, y, z \) be real numbers satisfying the following system of equations:
\[
\begin{aligned}
x + y + z &= 1, \\
x - y + 2z &= 3, \\
2x - 2y + 3z &= 5.
\end{aligned}
\]
Find the value of \( x - y + z \).
### Informal Answer
2


## Index 2685, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(hn : n ≥ 1000 ∧ n < 2000)
(h1 : n % 2 = 1)
(h2 : n % 3 = 1)
(h3 : n % 4 = 1)
(h4 : n % 5 = 1)
(h5 : n % 6 = 1)
(h6 : n % 7 = 0)
: n = 1001 ∨
  n = 1071 ∨
    n = 1141 ∨
      n = 1211 ∨
        n = 1281 ∨
          n = 1351 ∨ n = 1421 ∨ n = 1491 ∨ n = 1561 ∨ n = 1631 ∨ n = 1701 ∨ n = 1771 ∨ n = 1841 ∨ n = 1911 ∨ n = 1981
:= sorry
```
### Informal Problem
Find all natural numbers \( n \) between 1000 and 1999 (inclusive) that satisfy the following conditions:
- \( n \) is odd.
- \( n \) leaves a remainder of 1 when divided by 2, 3, 4, 5, and 6.
- \( n \) is divisible by 7.
### Informal Answer
1001, 1071, 1141, 1211, 1281, 1351, 1421, 1491, 1561, 1631, 1701, 1771, 1841, 1911, 1981


## Index 839, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(d : ℝ)
(h : d > 0)
(h1 : 20 * d / 100 = 10)
: 10 * d / 100 = 5
:= sorry
```
### Informal Problem
Let \( d \) be a positive real number. Given that \( 20 \times d / 100 = 10 \), what is the value of \( 10 \times d / 100 \)?
### Informal Answer
5


## Index 908, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x ≠ -3)
(h : 2 / (x + 3) = 1 / 6)
: x = 9
:= sorry
```
### Informal Problem
Let \( x \) be a real number such that \( x \neq -3 \). If
\[
\frac{2}{x + 3} = \frac{1}{6},
\]
what is the value of \( x \)?
### Informal Answer
9


## Index 2997, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℕ)
(hx : x > 0)
(h : 10 * x = 2 * x + 64)
: x = 8
:= sorry
```
### Informal Problem
Let \( x \) be a positive natural number. If \( 10x = 2x + 64 \), what is the value of \( x \)?
### Informal Answer
8


## Index 3832, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(S : ℕ → ℝ)
(T : ℕ → ℝ)
(hST : ∀ n, S n / T n = (7 * n + 1) / (4 * n + 27))
(s : ℕ → ℝ)
(hs : ∀ n, S (n + 1) = S n + s n)
(t : ℕ → ℝ)
(hT : ∀ n, T (n + 1) = T n + t n)
(h₀ : s 0 = t 0)
(h₁ : s 1 = t 1)
(h₂ : s 2 ≠ t 2)
(hs₂ : s 2 = 34)
(ht₂ : t 2 = 35)
: S 3 / T 3 = 22 / 39
:= sorry
```
### Informal Problem
Let \( S \) and \( T \) be sequences of real numbers such that for every natural number \( n \), the ratio \( \frac{S(n)}{T(n)} = \frac{7n + 1}{4n + 27} \). Additionally, define sequences \( s \) and \( t \) where for every \( n \), \( S(n+1) = S(n) + s(n) \) and \( T(n+1) = T(n) + t(n) \). Given that \( s(0) = t(0) \), \( s(1) = t(1) \), \( s(2) \ne t(2) \), \( s(2) = 34 \), and \( t(2) = 35 \), compute the value of \( \frac{S(3)}{T(3)} \).
### Informal Answer
\[
\frac{22}{39}
\]


## Index 552, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(h₀ : Odd n)
(h₁ : ∏ i ∈ Finset.range 6, (n + 2 * i) = 135135)
: n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) = 48
:= sorry
```
### Informal Problem
Let \( n \) be an odd natural number. The product of the six consecutive even numbers starting from \( n \) is 135135, i.e.,
\[
n(n+2)(n+4)(n+6)(n+8)(n+10) = 135135.
\]
Find the value of the sum:
\[
n + (n+2) + (n+4) + (n+6) + (n+8) + (n+10).
\]
### Informal Answer
48


## Index 28, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(h₀ : n < 100)
(h₁ : n > 9)
(h₂ : 5 * (7 * (n % 10) + (n / 10)) + 2 * n = 262)
: n = 91
:= sorry
```
### Informal Problem
Let \( n \) be a natural number such that:
- \( 9 < n < 100 \), and
- \( 5 \cdot (7 \cdot (n \bmod 10) + (n / 10)) + 2 \cdot n = 262 \).

Find the value of \( n \).
### Informal Answer
91


## Index 3018, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℕ → ℝ)
(ha1 : a 1 = 1)
(b : ℕ → ℝ)
(hroots : ∀ n ≥ 1, a n + a (n + 1) = -3 * n ∧ a n * a (n + 1) = b n)
: ∑ i ∈ Finset.Icc 1 20, b i = 6385
:= sorry
```
### Informal Problem
Let \( a \) and \( b \) be sequences of real numbers defined for natural numbers \( n \geq 1 \), with the following properties:
- \( a_1 = 1 \),
- For every \( n \geq 1 \), the sum \( a_n + a_{n+1} = -3n \), and the product \( a_n \cdot a_{n+1} = b_n \).

Compute the sum \( \sum_{i=1}^{20} b_i \).
### Informal Answer
6385


## Index 3837, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(p : ℝ)
(q : ℝ)
(r : ℝ)
(s : ℝ)
(h : p * s - q * r ≠ 0)
(h1 : p * s^3 + q * r^3 = 0)
(h2 : p * s^2 + q * r^2 = 0)
(h3 : p * s + q * r = 0)
: (p * s ^ 3 + q * r ^ 3) / (p * s - q * r) + (p * s ^ 2 + q * r ^ 2) / (p * s - q * r) +
    (p * s + q * r) / (p * s - q * r) =
  0
:= sorry
```
### Informal Problem
Let \( p, q, r, s \) be real numbers such that \( p \cdot s - q \cdot r \neq 0 \), and suppose:
\[
\begin{aligned}
p \cdot s^3 + q \cdot r^3 &= 0, \\
p \cdot s^2 + q \cdot r^2 &= 0, \\
p \cdot s + q \cdot r &= 0.
\end{aligned}
\]
Evaluate the expression:
\[
\frac{p \cdot s^3 + q \cdot r^3}{p \cdot s - q \cdot r} + \frac{p \cdot s^2 + q \cdot r^2}{p \cdot s - q \cdot r} + \frac{p \cdot s + q \cdot r}{p \cdot s - q \cdot r}.
\]
### Informal Answer
0


## Index 1609, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x > 1)
(h₁ : logb x (x - 1) + logb x (x + 1) = 1 + logb x 5)
(h₂ : (x - 1) * (x + 1) = 5 * x)
: x = (5 + √29) / 2
:= sorry
```
### Informal Problem
Let \( x \) be a real number greater than 1. Suppose that:

\[
\log_x (x - 1) + \log_x (x + 1) = 1 + \log_x 5
\]

and

\[
(x - 1)(x + 1) = 5x.
\]

Find the value of \( x \).
### Informal Answer
\[
\frac{5 + \sqrt{29}}{2}
\]


## Index 2152, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(ha : a ≠ 0)
(b : ℝ)
(c : ℝ)
(d : ℝ)
(e : ℝ)
(h : 2 * (a - 5) * (b - 3) * (c - 2) * (d - 4) * (e - 6) = a * b * c * d * e - 36)
(h1 : (a - 5) * (b - 3) * (c - 2) * (d - 4) * (e - 6) = 12)
: a * b * c * d * e = 60
:= sorry
```
### Informal Problem
Let \( a, b, c, d, e \) be real numbers with \( a \neq 0 \). Suppose that:
\[
2 \cdot (a - 5) \cdot (b - 3) \cdot (c - 2) \cdot (d - 4) \cdot (e - 6) = a \cdot b \cdot c \cdot d \cdot e - 36,
\]
and
\[
(a - 5) \cdot (b - 3) \cdot (c - 2) \cdot (d - 4) \cdot (e - 6) = 12.
\]
Find the value of \( a \cdot b \cdot c \cdot d \cdot e \).
### Informal Answer
60


## Index 1745, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x ≥ 2015)
(y : ℝ)
(hy : y = (x - 2015) * (x - 2016) / (x - 2015))
(h : y = 2016)
: x = 4032
:= sorry
```
### Informal Problem
Let \( x \) be a real number such that \( x \geq 2015 \). Define \( y = \frac{(x - 2015)(x - 2016)}{x - 2015} \). Given that \( y = 2016 \), find the value of \( x \).
### Informal Answer
4032


## Index 3626, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(p : ℕ)
(q : ℕ)
(r : ℕ)
(hpqr : p.Prime ∧ q.Prime ∧ r.Prime)
(hqr : r - q = 2 * p)
(hsum : p + q + r = 31)
(hqr' : r > q)
(hqp : q > p)
: p = 3
:= sorry
```
### Informal Problem
Let \( p, q, r \) be prime numbers such that:
- \( r - q = 2p \),
- \( p + q + r = 31 \),
- \( r > q \),
- \( q > p \).

Find the value of \( p \).
### Informal Answer
3


## Index 2916, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(y : ℝ)
(z : ℝ)
(h₀ : 4 * (x + y + z) = 1)
(h₁ : 5 * (x + y + z) + 0.1 * (2 * y + 4 * z) = 1)
(h₃ : 5 * (x + y + z) + 0.1 * (2 * y + 4 * z) + 2 * (y + z) = 1)
: x = 0.25
:= sorry
```
### Informal Problem
Let \( x, y, z \) be real numbers satisfying the following system of equations:
1. \( 4(x + y + z) = 1 \)
2. \( 5(x + y + z) + 0.1(2y + 4z) = 1 \)
3. \( 5(x + y + z) + 0.1(2y + 4z) + 2(y + z) = 1 \)

Find the value of \( x \).
### Informal Answer
0.25


## Index 1712, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(b : ℝ)
(h : 2 * a ≠ b)
(p : ℝ → ℝ)
(hp : p = fun x => x ^ 2 + a * x + b)
(h1 : p (2 * a) = 0)
(h2 : p b = 0)
: a + b = -1
:= sorry
```
### Informal Problem
Let \( a \) and \( b \) be real numbers with \( 2a \ne b \). Define a quadratic polynomial \( p(x) = x^2 + a x + b \). Suppose \( p(2a) = 0 \) and \( p(b) = 0 \). Find the value of \( a + b \).
### Informal Answer
-1


## Index 2342, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(y : ℝ)
(h : abs (x - 1) + abs y = 1)
: y ^ 2 + 2 * x - x ^ 2 ≤ 7 / 3
:= sorry
```
### Informal Problem
Let \( x \) and \( y \) be real numbers such that \( |x - 1| + |y| = 1 \). Find the maximum value of \( y^2 + 2x - x^2 \).
### Informal Answer
\(\frac{7}{3}\)


## Index 520, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(y : ℝ)
(z : ℝ)
(h1 : 1988 * (x - y) + 1989 * (y - z) + 1990 * (z - x) = 0)
(h2 : 1988^2 * (x - y) + 1989^2 * (y - z) + 1990^2 * (z - x) = 1989)
: y - z = -1989
:= sorry
```
### Informal Problem
Let \( x, y, z \) be real numbers satisfying the following equations:
1. \( 1988(x - y) + 1989(y - z) + 1990(z - x) = 0 \)
2. \( 1988^2(x - y) + 1989^2(y - z) + 1990^2(z - x) = 1989 \)

Find the value of \( y - z \).
### Informal Answer
-1989


## Index 3805, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℝ → ℝ)
(h : ∀ x y, f x + f (x * y + y) = f (x * y) + f x + y)
(h' : f 1 = 1)
: f 2011 = 2011
:= sorry
```
### Informal Problem
Let \( f \) be a function from real numbers to real numbers such that for all real numbers \( x \) and \( y \),
\[
f(x) + f(x \cdot y + y) = f(x \cdot y) + f(x) + y,
\]
and \( f(1) = 1 \). Determine the value of \( f(2011) \).
### Informal Answer
2011


## Index 1343, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℝ → ℝ)
(h : ∀ x, x ≠ 0 → 3 * f x - 2 * f (1 / x) = x)
: f 4 = 5 / 2
:= sorry
```
### Informal Problem
Let \( f \) be a function from the real numbers to the real numbers such that for every nonzero real number \( x \),
\[
3 f(x) - 2 f\left(\frac{1}{x}\right) = x.
\]
Find the value of \( f(4) \).
### Informal Answer
\(\frac{5}{2}\)


## Index 435, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(b : ℝ)
(c : ℝ)
(h₀ : a + b + c = 1)
(h₁ : a^2 + b^2 + c^2 = 1)
: 3 / 5 ≥ (a * b + 1) / 5 + (b * c + 1) / 5 + (c * a + 1) / 5 ∧
  (a * b + 1) / 5 + (b * c + 1) / 5 + (c * a + 1) / 5 ≥ 1 / 5
:= sorry
```
### Informal Problem
Let \( a, b, c \) be real numbers such that:
- \( a + b + c = 1 \), and
- \( a^2 + b^2 + c^2 = 1 \).

Evaluate the expression:
\[
\frac{a \cdot b + 1}{5} + \frac{b \cdot c + 1}{5} + \frac{c \cdot a + 1}{5}
\]
and determine the tightest bounds it satisfies.
### Informal Answer
The expression equals \( \frac{3}{5} \), so it satisfies:
\[
\frac{1}{5} \leq \frac{3}{5} \leq \frac{3}{5}.
\]


