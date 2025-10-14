## Index 4952, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℝ → ℝ)
(hf : f = fun x => x ^ 2)
(g : ℝ → ℝ)
(hfg : ∀ x, f (g x) = g (f x))
(hg2 : g 2 = 5)
: g 4 = 25
:= sorry
```
### Informal Problem
Let \( f : \mathbb{R} \to \mathbb{R} \) be defined by \( f(x) = x^2 \), and let \( g : \mathbb{R} \to \mathbb{R} \) be a function such that:
- For every real number \( x \), \( f(g(x)) = g(f(x)) \), and
- \( g(2) = 5 \).

Find the value of \( g(4) \).
### Informal Answer
25


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
\frac{3}{4}


## Index 217, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x ≠ 2)
(hx' : x + 1 ≠ 0)
(h : (x - 2) / (x + 1) = 3 / 4)
: 5 / (x - 2) = 5 / 9
:= sorry
```
### Informal Problem
Let \( x \) be a real number such that \( x \ne 2 \) and \( x + 1 \ne 0 \). If  
\[
\frac{x - 2}{x + 1} = \frac{3}{4},
\]  
what is the value of \( \frac{5}{x - 2} \)?
### Informal Answer
\frac{5}{9}


## Index 2026, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(y : ℝ)
(z : ℝ)
(h : x ≠ y ∧ y ≠ z ∧ z ≠ x)
(h1 : 2 * (y - z) / (x - y) = (z + x) / (y - z))
(h2 : (z + x) / (y - z) = 3 * (x - z) / (z - x))
: (x + 3 * y - 2 * z) / ((x ^ 2 + y ^ 2 + z ^ 2) / 2) = 0
:= sorry
```
### Informal Problem
Let \( x, y, z \) be distinct real numbers (i.e., \( x \ne y \), \( y \ne z \), and \( z \ne x \)) satisfying the following two equations:
1. \( \frac{2(y - z)}{x - y} = \frac{z + x}{y - z} \),
2. \( \frac{z + x}{y - z} = \frac{3(x - z)}{z - x} \).

Find the value of \( \frac{x + 3y - 2z}{\frac{x^2 + y^2 + z^2}{2}} \).
### Informal Answer
0


## Index 1787, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x > 0)
(a : ℕ → ℝ)
(h : ∀ (n : ℕ), a n + a (n + 1) = x)
(ha1 : a 1 = 1)
(m : ℝ)
(h1 : a 9 * a 10 = m + 9)
(n : ℝ)
(h2 : a 18 * a 19 = n + 900)
: m - n = 891
:= sorry
```
### Informal Problem
Let \( x \) be a positive real number. Define a sequence \( a_0, a_1, a_2, \ldots \) of real numbers such that for every natural number \( n \),
\[
a_n + a_{n+1} = x.
\]
Given that \( a_1 = 1 \), and that:
- \( a_9 \cdot a_{10} = m + 9 \),
- \( a_{18} \cdot a_{19} = n + 900 \),
where \( m \) and \( n \) are real numbers, find the value of \( m - n \).
### Informal Answer
891


## Index 1632, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(y : ℝ)
(h₀ : x / y = 4 / 5)
(z : ℝ)
(h₁ : y / z = 3 / 10)
(w : ℝ)
(h₂ : z / w = 6 / 7)
: (x + y + w) / z = 128 / 75
:= sorry
```
### Informal Problem
Let \( x, y, z, w \) be real numbers such that:
- \( \frac{x}{y} = \frac{4}{5} \),
- \( \frac{y}{z} = \frac{3}{10} \),
- \( \frac{z}{w} = \frac{6}{7} \).

Assuming all denominators are nonzero, compute the value of \( \frac{x + y + w}{z} \).
### Informal Answer
\frac{128}{75}


## Index 1042, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℝ → ℝ)
(hf : ∀ x, 0 < x ∧ x < 1 → f x = (x + 1) / (x - 1))
(g : ℕ → ℝ → ℝ)
(hg : ∀ n, ∀ x, 0 < x ∧ x < 1 → g n x = f^[n] x)
(a : ℕ → ℝ)
(ha : ∀ n, a n = g n (1 / 2))
: a 0 = 1 / 2 ∧ a 1 = -3
:= sorry
```
### Informal Problem
Let \( f : \mathbb{R} \to \mathbb{R} \) be a function such that for all \( x \) with \( 0 < x < 1 \),  
\[
f(x) = \frac{x + 1}{x - 1}.
\]  
Define a sequence of functions \( g_n : \mathbb{R} \to \mathbb{R} \) (for \( n \in \mathbb{N} \)) by  
\[
g_n(x) = f^n(x) \quad \text{for } 0 < x < 1,
\]  
where \( f^n \) denotes the \( n \)-th iterate of \( f \).  
Define a sequence \( (a_n) \) of real numbers by  
\[
a_n = g_n\left(\frac{1}{2}\right).
\]  
Find the values of \( a_0 \) and \( a_1 \).
### Informal Answer
a_0 = \frac{1}{2} and a_1 = -3


## Index 776, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x > 0)
(y : ℝ)
(h1 : 52 * x = 48 * y)
(h2 : 48 * y = 50 * x + 20)
: x = 10
:= sorry
```
### Informal Problem
Let \( x \) and \( y \) be real numbers with \( x > 0 \). Given the equations:
1. \( 52x = 48y \)
2. \( 48y = 50x + 20 \)

Find the value of \( x \).
### Informal Answer
10


## Index 4214, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x = -20 / 100)
(y : ℝ)
(hy : y = 30 / 100)
: (1 + x) * (1 + y) - 1 = 4 / 100
:= sorry
```
### Informal Problem
Let \( x = -\frac{20}{100} \) and \( y = \frac{30}{100} \). Compute the value of \( (1 + x)(1 + y) - 1 \).
### Informal Answer
\frac{4}{100}


## Index 661, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℕ)
(ha : 0 < a)
(b : ℕ)
(h : a * 17 + b * 19 = 500)
(hb : 0 < b)
: a + b = 28 ∨ a + b = 30
:= sorry
```
### Informal Problem
Let \( a \) and \( b \) be positive integers such that  
\[
17a + 19b = 500.
\]  
Determine the possible values of \( a + b \).
### Informal Answer
30


## Index 4568, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(h₀ : n ≥ 200)
(h₁ : n % 12 = 1)
(h₂ : n % 13 = 1)
: n ≥ 313
:= sorry
```
### Informal Problem
Let \( n \) be a natural number such that:
- \( n \geq 200 \),
- \( n \) leaves a remainder of 1 when divided by 12, and
- \( n \) leaves a remainder of 1 when divided by 13.

Determine the smallest possible value of \( n \) that satisfies these conditions.
### Informal Answer
313


## Index 3260, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(y : ℝ)
(hx : x^2 - 2000 * x = y^2 - 2000 * y)
(hxy : x ≠ y)
: x + y = 2000
:= sorry
```
### Informal Problem
Let \( x \) and \( y \) be real numbers such that \( x^2 - 2000x = y^2 - 2000y \) and \( x \ne y \). Find the value of \( x + y \).
### Informal Answer
2000


## Index 277, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℤ)
(y : ℤ)
(z : ℤ)
(t : ℤ)
(a : ℤ)
(h1 : x * z - 2 * y * t = a)
(h2 : x * t + y * z = 0)
: (x ^ 2 + 2 * y ^ 2) * (z ^ 2 + 2 * t ^ 2) = a ^ 2
:= sorry
```
### Informal Problem
Let \( x, y, z, t, a \) be integers such that:
- \( x \cdot z - 2 \cdot y \cdot t = a \), and
- \( x \cdot t + y \cdot z = 0 \).

Compute the value of \( (x^2 + 2 \cdot y^2) \cdot (z^2 + 2 \cdot t^2) \).
### Informal Answer
a^2


## Index 258, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx3 : x^3 = 11)
: x ^ 6 = 121
:= sorry
```
### Informal Problem
Let \( x \) be a real number such that \( x^3 = 11 \). Compute the value of \( x^6 \).
### Informal Answer
121


## Index 701, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(ha : 0 < a)
(b : ℝ)
(h : a ^ 2 + b ^ 2 = 7 * a * b)
(hb : 0 < b)
: ((a + b) ^ 2 / (a * b)) ^ 2 - 4 * ((a - b) ^ 2 / (a * b)) = 61
:= sorry
```
### Informal Problem
Let \( a \) and \( b \) be positive real numbers such that \( a^2 + b^2 = 7ab \). Compute the value of the expression:
\[
\left( \frac{(a + b)^2}{a \cdot b} \right)^2 - 4 \cdot \left( \frac{(a - b)^2}{a \cdot b} \right).
\]
### Informal Answer
61


## Index 1589, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(b : ℝ)
(ha : a = 5 / 6)
(hb : b ≠ 0)
(c : ℝ)
(h1 : a / b = c / 100)
(h2 : b / c = 60 / 100)
(hc : c ≠ 0)
: c ^ 2 = 1250 / 9
:= sorry
```
### Informal Problem
Let \( a, b, c \) be real numbers such that:
- \( a = \frac{5}{6} \),
- \( b \neq 0 \),
- \( c \neq 0 \),
- \( \frac{a}{b} = \frac{c}{100} \), and
- \( \frac{b}{c} = \frac{60}{100} \).

Find the value of \( c^2 \).
### Informal Answer
\frac{1250}{9}


## Index 1705, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x ≠ 0)
(y : ℝ)
(hy : y ≠ 0)
(z : ℝ)
(h1 : 1 / x + 1 / y + 1 / z = 1 / 4)
(h2 : 1 / x ^ 2 + 1 / y ^ 2 + 1 / z ^ 2 = 1 / 16)
(hz : z ≠ 0)
(n : ℕ)
(hn : Odd n)
(k : ℤ)
(h3 : x ^ n + y ^ n + z ^ n = k)
(h4 : k = -1 ∨ k = 0 ∨ k = 1)
(m : ℕ)
(h5 : x ^ (n + 2 * m) + y ^ (n + 2 * m) + z ^ (n + 2 * m) = 39366)
(hm : m > 0)
: x ^ (n + 2 * m) + y ^ (n + 2 * m) + z ^ (n + 2 * m) < 28000 ∨
  x ^ (n + 2 * m) + y ^ (n + 2 * m) + z ^ (n + 2 * m) > 30000
:= sorry
```
### Informal Problem
Let \( x, y, z \) be nonzero real numbers such that:
\[
\frac{1}{x} + \frac{1}{y} + \frac{1}{z} = \frac{1}{4}
\]
and
\[
\frac{1}{x^2} + \frac{1}{y^2} + \frac{1}{z^2} = \frac{1}{16}.
\]
Let \( n \) be an odd natural number, and let \( k \) be an integer such that \( x^n + y^n + z^n = k \), where \( k \) is either \(-1\), \(0\), or \(1\). For a positive natural number \( m \), suppose that:
\[
x^{n + 2m} + y^{n + 2m} + z^{n + 2m} = 39366.
\]
Determine whether this sum is less than 28000 or greater than 30000.
### Informal Answer
30000


## Index 3940, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(S : ℕ)
(hS : S = 120)
(k : ℕ)
(h : 10 * 5 ≤ k * 8)
(hk : k ≤ 12)
: k = 7 ∨ k = 8 ∨ k = 9 ∨ k = 10 ∨ k = 11 ∨ k = 12
:= sorry
```
### Informal Problem
Let \( S = 120 \). Let \( k \) be a natural number such that \( 10 \times 5 \leq k \times 8 \) and \( k \leq 12 \). Determine all possible values of \( k \).
### Informal Answer
12


## Index 4649, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(y : ℝ)
(z : ℝ)
(h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
(h1 : x / y + y / z + z / x = 14)
(h2 : y / x + z / y + x / z = 20)
(k : ℝ)
(h3 : x / y + y / z + z / x + y / x + z / y + x / z = k)
: k = 34
:= sorry
```
### Informal Problem
Let \( x, y, z \) be nonzero real numbers such that:
\[
\frac{x}{y} + \frac{y}{z} + \frac{z}{x} = 14 \quad \text{and} \quad \frac{y}{x} + \frac{z}{y} + \frac{x}{z} = 20.
\]
Define \( k = \frac{x}{y} + \frac{y}{z} + \frac{z}{x} + \frac{y}{x} + \frac{z}{y} + \frac{x}{z} \). Compute the value of \( k \).
### Informal Answer
34


## Index 238, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x ≠ -11)
(hx' : x ≠ -7)
(hx'' : x ≠ -4)
(hx''' : x ≠ -3)
(h : ((x + 8) * (x + 11) * (x + 7) * (x + 4)) / ((x + 11) * (x + 7)) - ((x + 8) * (x + 11) * (x + 3)) / ((x + 11)) = (x - 3) * (x + 4))
: x ^ 2 = 20
:= sorry
```
### Informal Problem
Let \( x \) be a real number such that \( x \neq -11 \), \( x \neq -7 \), \( x \neq -4 \), and \( x \neq -3 \). Suppose the following equation holds:
\[
\frac{(x + 8)(x + 11)(x + 7)(x + 4)}{(x + 11)(x + 7)} - \frac{(x + 8)(x + 11)(x + 3)}{x + 11} = (x - 3)(x + 4).
\]
Find the value of \( x^2 \).
### Informal Answer
20


## Index 4335, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x * 1.07 = 27.82)
: x = 26
:= sorry
```
### Informal Problem
Let \( x \) be a real number such that \( x \times 1.07 = 27.82 \). Find the value of \( x \).
### Informal Answer
26


## Index 1431, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℝ)
(M : ℝ)
(hM : M = 60.0)
(N : ℝ)
(hN : N = 80.0)
(X : ℝ)
(h : (M * (1 - 20 / 100) + N * (1 - 25 / 100) + X * (1 - 50 / 100)) / (3 + (20 / 100) + (25 / 100) + (50 / 100))
    = n )
(hX : X = 70.0)
: round n = 36
:= sorry
```
### Informal Problem
Let \( M = 60.0 \), \( N = 80.0 \), and \( X = 70.0 \). Suppose \( n \) is a real number satisfying the equation:
\[
\frac{M \cdot (1 - 20\%) + N \cdot (1 - 25\%) + X \cdot (1 - 50\%)}{3 + 20\% + 25\% + 50\%} = n.
\]
Compute the value of \( \text{round}(n) \), where "round" denotes rounding to the nearest integer.
### Informal Answer
36


## Index 3242, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(d : ℝ)
(h₀ : ∑ i ∈ Finset.range 5, (a + d * i) = 70)
(h₁ : ∑ i ∈ Finset.range 10, (a + d * i) = 210)
: a = 42 / 5
:= sorry
```
### Informal Problem
Let \( a \) and \( d \) be real numbers. The sum of the first 5 terms of the arithmetic sequence \( a, a + d, a + 2d, \ldots \) is 70, and the sum of the first 10 terms is 210. Find the value of \( a \).
### Informal Answer
\frac{42}{5}


## Index 1608, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℕ)
(b : ℕ)
(c : ℕ)
(d : ℕ)
(e : ℕ)
(f : ℕ)
(h₀ : a + b + c + d + e + f = 2014)
(h₁ : ¬(3 ∣ a ∨ 3 ∣ b ∨ 3 ∣ c ∨ 3 ∣ d ∨ 3 ∣ e ∨ 3 ∣ f))
: (a + b + c + d + e + f) % 3 = 1
:= sorry
```
### Informal Problem
Let \( a, b, c, d, e, f \) be natural numbers such that:
- Their sum is 2014: \( a + b + c + d + e + f = 2014 \).
- None of them is divisible by 3: \( 3 \nmid a \), \( 3 \nmid b \), \( 3 \nmid c \), \( 3 \nmid d \), \( 3 \nmid e \), \( 3 \nmid f \).

Find the remainder when \( a + b + c + d + e + f \) is divided by 3.
### Informal Answer
1


## Index 3514, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(r1 : ℝ)
(h1 : r1 = 1)
(r2 : ℝ)
(h2 : r2 = 2)
(r3 : ℝ)
(h3 : r3 = 3)
(r4 : ℝ)
(h4 : r4 = 4)
(r5 : ℝ)
(h5 : r5 = 5)
(r6 : ℝ)
(h6 : r6 = 6)
(r7 : ℝ)
(h7 : r7 = 7)
(r8 : ℝ)
(h8 : r8 = 8)
(R : ℝ)
(h : (r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + R) * (R - (r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8)) = 28)
: R ^ 2 = 1324
:= sorry
```
### Informal Problem
Let \( r_1 = 1 \), \( r_2 = 2 \), \( r_3 = 3 \), \( r_4 = 4 \), \( r_5 = 5 \), \( r_6 = 6 \), \( r_7 = 7 \), \( r_8 = 8 \), and let \( R \) be a real number such that  
\[
(r_1 + r_2 + r_3 + r_4 + r_5 + r_6 + r_7 + r_8 + R) \cdot (R - (r_1 + r_2 + r_3 + r_4 + r_5 + r_6 + r_7 + r_8)) = 28.
\]  
Find the value of \( R^2 \).
### Informal Answer
1324


## Index 4561, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(u : ℝ → ℝ)
(hu : u = λ φ => 5 * sin φ)
(φ : ℝ)
(hφ : φ = Real.pi / 3)
: deriv u φ = 5 * 1 / 2
:= sorry
```
### Informal Problem
Let \( u \) be a real-valued function defined by \( u(\phi) = 5 \sin \phi \). If \( \phi = \frac{\pi}{3} \), compute the derivative of \( u \) at \( \phi \).
### Informal Answer
\frac{5}{2}


## Index 2044, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℤ)
(lx : 0 ≤ x)
(rx : x < 10)
(y : ℤ)
(h : 11 * x + 2 * y = 13)
(ly : 0 ≤ y)
(ry : y < 10)
: x = 1 ∧ y = 1
:= sorry
```
### Informal Problem
Let \( x \) and \( y \) be integers such that:
- \( 0 \leq x < 10 \),
- \( 0 \leq y < 10 \),
- and \( 11x + 2y = 13 \).

Find the values of \( x \) and \( y \).
### Informal Answer
x = 1 and y = 1


## Index 53, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x = 25)
(y : ℝ)
(hy : y = (1 - 20 / 100) * x)
(z : ℝ)
(hz : z = x + y)
: z = 45
:= sorry
```
### Informal Problem
Let \( x \) be a real number equal to 25. Define \( y \) as \( (1 - \frac{20}{100}) \times x \), and let \( z = x + y \). What is the value of \( z \)?
### Informal Answer
45


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


## Index 3261, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℕ)
(hx : Nat.Prime x)
(y : ℕ)
(hsum : x + y = 49)
(hy : Nat.Prime y)
: x = 2 ∧ y = 47 ∨ x = 47 ∧ y = 2
:= sorry
```
### Informal Problem
Let \( x \) and \( y \) be prime numbers such that \( x + y = 49 \). Find all possible values of \( x \) and \( y \).
### Informal Answer
x = 47 and y = 2


## Index 2551, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℕ → ℝ)
(h0 : a 0 = 1)
(h1 : a 1 = 10)
(hn : ∀ n : ℕ, n ≥ 2 → a n = ((n + 7) * a (n - 1) - 14 * a (n - 2)) / (n + 1))
: a 10 = 2021299 / 7425
:= sorry
```
### Informal Problem
Define a sequence \( a_0, a_1, a_2, \ldots \) of real numbers by:
- \( a_0 = 1 \),
- \( a_1 = 10 \),
- For every integer \( n \geq 2 \), \( a_n = \frac{(n + 7) \cdot a_{n-1} - 14 \cdot a_{n-2}}{n + 1} \).

Calculate the value of \( a_{10} \).
### Informal Answer
\frac{2021299}{7425}


## Index 1149, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(r : ℝ)
(s : ℝ)
(P : ℝ → ℝ)
(hp : ∀ x, P x = x ^ 2 + r * x + s)
(h1 : P r = 0)
(h2 : P s = 0)
(h3 : r ≠ s)
: (r, s) = (1, -2) ∨ (r, s) = (-1 / 2, -1 / 2)
:= sorry
```
### Informal Problem
Let \( r \) and \( s \) be real numbers. Define a quadratic polynomial \( P(x) = x^2 + r \cdot x + s \). Suppose that \( P(r) = 0 \), \( P(s) = 0 \), and \( r \ne s \). Determine all possible ordered pairs \( (r, s) \).
### Informal Answer
(1, -2) and (-\frac{1}{2}, -\frac{1}{2})


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


## Index 2520, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℕ → ℕ)
(h1 : f 1 = 1)
(h2 : ∀ k, f (2 * k) = k * f k)
(h3 : ∀ k, f (4 * k + 1) = (k + 1) * f (2 * k + 1))
(h4 : ∀ k, f (4 * k - 1) = (2 * k - 1) * f (2 * k - 1))
: f 5056 = 1264 * 2528 * f 1264
:= sorry
```
### Informal Problem
Let \( f \) be a function from natural numbers to natural numbers such that:
- \( f(1) = 1 \),
- For every \( k \), \( f(2k) = k \cdot f(k) \),
- For every \( k \), \( f(4k + 1) = (k + 1) \cdot f(2k + 1) \),
- For every \( k \), \( f(4k - 1) = (2k - 1) \cdot f(2k - 1) \).

Compute \( f(5056) \) in terms of \( f(1264) \).
### Informal Answer
1264 \times 2528 \times f(1264)


## Index 694, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example

: 1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19 = 100 ∧ 25 ^ 2 = 625
:= sorry
```
### Informal Problem
1. Compute the sum of the first 10 odd numbers: \(1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19\).
2. Compute \(25^2\).
### Informal Answer
625


## Index 2898, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x ∈ Set.Ioo (-Real.pi / 2) (Real.pi / 2))
(h : Real.tan x = 5 / 12)
: (Real.cos x ^ 2 - Real.sin x ^ 2) / (2 * Real.sin x * Real.cos x) = 119 / 120
:= sorry
```
### Informal Problem
Let \( x \) be a real number such that \( -\frac{\pi}{2} < x < \frac{\pi}{2} \). Suppose that \( \tan x = \frac{5}{12} \). Compute the value of the expression:
\[
\frac{\cos^2 x - \sin^2 x}{2 \sin x \cos x}.
\]
### Informal Answer
119/120


## Index 726, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(y : ℝ)
(h₀ : x - y = 1)
(h₁ : x ^ 2 + y ^ 2 = 7)
: x ^ 3 - y ^ 3 = 10
:= sorry
```
### Informal Problem
Let \( x \) and \( y \) be real numbers such that \( x - y = 1 \) and \( x^2 + y^2 = 7 \). Compute the value of \( x^3 - y^3 \).
### Informal Answer
10


## Index 2703, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(b : ℝ)
(c : ℝ)
(d : ℝ)
(h₀ : a - 7 = b + c + d)
(h₁ : b - 11 = a + c + d)
(h₂ : c + 5 = a + b + d)
(h₃ : d + 1 = a + b + c)
: a + b + c + d = -6
:= sorry
```
### Informal Problem
Let \( a, b, c, d \) be real numbers satisfying the following system of equations:
- \( a - 7 = b + c + d \)
- \( b - 11 = a + c + d \)
- \( c + 5 = a + b + d \)
- \( d + 1 = a + b + c \)

Find the value of \( a + b + c + d \).
### Informal Answer
-6


## Index 2574, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(b : ℝ)
(h : a + b = 1)
: a ^ 3 + b ^ 3 + 3 * a * b = 1
:= sorry
```
### Informal Problem
Let \( a \) and \( b \) be real numbers such that \( a + b = 1 \). Compute the value of \( a^3 + b^3 + 3ab \).
### Informal Answer
1


## Index 4661, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(b : ℝ)
(h₀ : a ≠ b)
(f : ℝ → ℝ)
(h₂ : f (2 * a) = 3 * a + b)
(h₃ : f (2 * b) = 3 * b + a)
(h₁ : ∀ x, f x = 4 * x - x ^ 2)
: a + b = 3 / 2
:= sorry
```
### Informal Problem
Let \( a \) and \( b \) be real numbers with \( a \ne b \). Suppose a function \( f : \mathbb{R} \to \mathbb{R} \) is defined by \( f(x) = 4x - x^2 \), and it satisfies:
- \( f(2a) = 3a + b \),
- \( f(2b) = 3b + a \).

Find the value of \( a + b \).
### Informal Answer
\frac{3}{2}


## Index 1953, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(ha : a ≤ 1)
(b : ℝ)
(hb : b ≤ 1)
(c : ℝ)
(hc : c ≤ 1)
(d : ℝ)
(h : a + b + c + d = 4)
(hd : d ≤ 1)
: a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1
:= sorry
```
### Informal Problem
Let \( a, b, c, d \) be real numbers such that:
- \( a \leq 1 \),
- \( b \leq 1 \),
- \( c \leq 1 \),
- \( d \leq 1 \),
- and \( a + b + c + d = 4 \).

Determine the values of \( a, b, c, \) and \( d \).
### Informal Answer
a = 1 and b = 1 and c = 1 and d = 1


## Index 350, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(t : ℝ)
(ht : t > 0)
(h : -6 * t ^ 2 - 10 * t + 56 = 0)
: t = 7 / 3
:= sorry
```
### Informal Problem
Given that \( t \) is a positive real number and satisfies the equation:
\[
-6t^2 - 10t + 56 = 0,
\]
find the value of \( t \).
### Informal Answer
\frac{7}{3}


## Index 3580, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℕ)
(h₁ : x / 6 = 15)
(y : ℕ)
(h₂ : y / 6 = 15)
(z : ℕ)
(h₃ : z / 6 = 15)
(w : ℕ)
(h₄ : w / 6 = 15)
(v : ℕ)
(h₅ : v / 6 = 15)
(u : ℕ)
(h₀ : x + y + z + w + v + u = 534)
: (x + y + z + w + v + u) / 6 = 89
:= sorry
```
### Informal Problem
Let \( x, y, z, w, v, u \) be natural numbers such that:
- \( x / 6 = 15 \),
- \( y / 6 = 15 \),
- \( z / 6 = 15 \),
- \( w / 6 = 15 \),
- \( v / 6 = 15 \),
- and \( x + y + z + w + v + u = 534 \).

Compute the value of \( (x + y + z + w + v + u) / 6 \).
### Informal Answer
89


## Index 4140, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(P : Polynomial ℝ)
(hP : P.degree = 4)
(hP2 : P.eval 2 = 10)
(hP3 : P.eval 3 = 23)
(hP4 : P.eval 4 = 40)
(hP5 : P.eval 5 = 63)
(hP6 : P.eval 6 = 94)
: Polynomial.eval 7 P = 135
:= sorry
```
### Informal Problem
Let \( P \) be a real-coefficient polynomial of degree 4. Given the following values:
- \( P(2) = 10 \)
- \( P(3) = 23 \)
- \( P(4) = 40 \)
- \( P(5) = 63 \)
- \( P(6) = 94 \)

Find the value of \( P(7) \).
### Informal Answer
135


## Index 948, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℝ → ℝ)
(hf : ∃ p q : Polynomial ℝ, f = fun x => p.eval x / q.eval x)
(h : ∀ x, x ≠ 0 → 3 * f (1 / x) + 2 * f x / x = x ^ 2)
: f (-2) = 67 / 20
:= sorry
```
### Informal Problem
Let \( f : \mathbb{R} \to \mathbb{R} \) be a rational function (i.e., \( f(x) = \frac{p(x)}{q(x)} \) for some polynomials \( p \) and \( q \)). Suppose that for all nonzero \( x \),  
\[
3 f\left(\frac{1}{x}\right) + \frac{2 f(x)}{x} = x^2.
\]  
Find the value of \( f(-2) \).
### Informal Answer
67/20


## Index 2888, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x > 0)
(y : ℝ)
(h : x * 9 = y * 10)
(h' : y = x - 4000)
: x = 40000
:= sorry
```
### Informal Problem
Let \( x \) be a positive real number and \( y \) be a real number such that:
- \( x \cdot 9 = y \cdot 10 \), and
- \( y = x - 4000 \).

Find the value of \( x \).
### Informal Answer
40000


## Index 616, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℤ)
(ha : a % 1995 = 2)
(b : ℤ)
(hb : b % 1995 = 4)
(c : ℤ)
(hc : c % 1995 = 6)
(d : ℤ)
(hd : d % 1995 = 8)
(e : ℤ)
(he : e % 1995 = 10)
(f : ℤ)
(hf : f % 1995 = 12)
: (a + b + c + d + e + f) % 1995 = 42
:= sorry
```
### Informal Problem
Let \( a, b, c, d, e, f \) be integers such that:
- \( a \mod 1995 = 2 \),
- \( b \mod 1995 = 4 \),
- \( c \mod 1995 = 6 \),
- \( d \mod 1995 = 8 \),
- \( e \mod 1995 = 10 \),
- \( f \mod 1995 = 12 \).

Find the value of \( (a + b + c + d + e + f) \mod 1995 \).
### Informal Answer
42


## Index 4262, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℕ)
(b : ℕ)
(c : ℕ)
(d : ℕ)
(h : a * b = c * d)
: ∃ (x : ℕ), x = a * (b + c) ∧ x = c * (a + d) ∧ ∃ (y : ℕ), y = a * (b + d) ∧ y = d * (a + c)
:= sorry
```
### Informal Problem
Let \( a, b, c, d \) be natural numbers such that \( a \cdot b = c \cdot d \). Find natural numbers \( x \) and \( y \) such that:
- \( x = a \cdot (b + c) \) and \( x = c \cdot (a + d) \),
- \( y = a \cdot (b + d) \) and \( y = d \cdot (a + c) \).
### Informal Answer
x = a(b + c) = c(a + d) and y = a(b + d) = d(a + c)


## Index 2159, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(y : ℝ)
(hxy : x ≠ y)
(h : x / y + x = y / x + y)
(hy : y ≠ 0)
(hx : x ≠ 0)
: 1 / x + 1 / y = -1
:= sorry
```
### Informal Problem
Let \( x \) and \( y \) be nonzero real numbers with \( x \ne y \). Suppose that  
\[
\frac{x}{y} + x = \frac{y}{x} + y.
\]  
Find the value of \( \frac{1}{x} + \frac{1}{y} \).
### Informal Answer
-1


## Index 4885, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℝ → ℝ)
(hf : f = λ x => sqrt (x - 2 * sqrt (x - 1)))
: ∀ (x : ℝ), f x = √(x - 2 * √(x - 1))
:= sorry
```
### Informal Problem
Let \( f : \mathbb{R} \to \mathbb{R} \) be defined by \( f(x) = \sqrt{x - 2 \cdot \sqrt{x - 1}} \). Express \( f(x) \) in terms of \( x \).
### Informal Answer
\sqrt{x - 2 \cdot \sqrt{x - 1}}


## Index 4809, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℕ)
(ha : a = 2^306 * 3^340)
(b : ℕ)
(hb : b = 2^342 * 3^308)
: a.gcd b = 2 ^ 306 * 3 ^ 308
:= sorry
```
### Informal Problem
Let \( a = 2^{306} \cdot 3^{340} \) and \( b = 2^{342} \cdot 3^{308} \). Compute the greatest common divisor (gcd) of \( a \) and \( b \).
### Informal Answer
2^{306} \cdot 3^{308}


## Index 2723, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℤ)
(d : ℤ)
(h₀ : a % d = 0)
(h₁ : (a + d) % d = 0)
(h₂ : (a + 2 * d) % d = 0)
(h₃ : (a + 3 * d) % d = 0)
(h₄ : (a + 4 * d) % d = 0)
(h₅ : (a + 5 * d) % d = 0)
(h₆ : d ≠ 0)
(n : ℕ)
: ∃ (x : ℤ), (a + (5 - ↑n) * d) % x = 0
:= sorry
```
### Informal Problem
Let \( a \) and \( d \) be integers with \( d \neq 0 \). Suppose that:
- \( a \mod d = 0 \),
- \( (a + d) \mod d = 0 \),
- \( (a + 2d) \mod d = 0 \),
- \( (a + 3d) \mod d = 0 \),
- \( (a + 4d) \mod d = 0 \),
- \( (a + 5d) \mod d = 0 \).

For any natural number \( n \), find an integer \( x \) such that \( (a + (5 - n)d) \mod x = 0 \).
### Informal Answer
1


## Index 4487, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℤ)
(hx : x ≥ 15)
(h : (x - 14) * (1 / 2 : ℚ) = (x + 14) * (1 / 3 : ℚ))
: x = 70
:= sorry
```
### Informal Problem
Let \( x \) be an integer such that \( x \geq 15 \). If  
\[
(x - 14) \cdot \frac{1}{2} = (x + 14) \cdot \frac{1}{3},
\]  
what is the value of \( x \)?
### Informal Answer
70


## Index 1399, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x > 0)
(h : 10 / (3 + x) = 4 / 3)
: x = 4.5
:= sorry
```
### Informal Problem
Let \( x \) be a positive real number. If
\[
\frac{10}{3 + x} = \frac{4}{3},
\]
what is the value of \( x \)?
### Informal Answer
4.5


## Index 551, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(ha : a ≠ 0)
(b : ℝ)
(hb : b ≠ 0)
(c : ℝ)
(hc : c ≠ 0)
(d : ℝ)
(hd : d ≠ 0)
: (a ^ 2 + b ^ 2) * (c ^ 2 + d ^ 2) - (a * c + b * d) ^ 2 = (a * d - b * c) ^ 2
:= sorry
```
### Informal Problem
Let \( a, b, c, d \) be nonzero real numbers. Compute the value of the expression:
\[
(a^2 + b^2)(c^2 + d^2) - (a \cdot c + b \cdot d)^2.
\]
### Informal Answer
(a \cdot d - b \cdot c)^2


## Index 361, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℝ → ℝ)
(hf : f = fun x => x ^ 2 + 12 * x + 5)
(g : ℝ → ℝ)
(hg : g = fun x => x ^ 2 + 4 * x - 12)
: {x : ℝ | f x - g x = 7} = {-5 / 4}
:= sorry
```
### Informal Problem
Let \( f \) and \( g \) be real-valued functions defined by  
\[
f(x) = x^2 + 12x + 5, \quad g(x) = x^2 + 4x - 12.
\]  
Find the set of all real numbers \( x \) such that \( f(x) - g(x) = 7 \).
### Informal Answer
\{-\frac{5}{4}\}


## Index 1659, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x = 14)
(y : ℝ)
(hy : y = -30)
(z : ℝ)
(hz : z = 9)
(ρ : ℝ)
(hρ : ρ > 0)
(θ : ℝ)
(hθ : θ ∈ Ioo (-π) π)
(φ : ℝ)
(h : x = ρ * sin φ * cos θ)
(h' : y = ρ * sin φ * sin θ)
(h'' : z = ρ * cos φ)
(hφ : φ ∈ Ioo 0 π)
: Real.cos θ ^ 2 = (14 / √(14 ^ 2 + (-30) ^ 2)) ^ 2 ∧ Real.sin θ ^ 2 = (-30 / √(14 ^ 2 + (-30) ^ 2)) ^ 2
:= sorry
```
### Informal Problem
Let \( x = 14 \), \( y = -30 \), and \( z = 9 \). Let \( \rho > 0 \), \( \theta \in (-\pi, \pi) \), and \( \varphi \in (0, \pi) \) be real numbers such that:
\[
x = \rho \sin \varphi \cos \theta, \quad y = \rho \sin \varphi \sin \theta, \quad z = \rho \cos \varphi.
\]
Compute \( \cos^2 \theta \) and \( \sin^2 \theta \).
### Informal Answer
\cos^2 \theta = ( \frac{14}{\sqrt{14^2 + (-30)^2}} )^2, \quad \sin^2 \theta = ( \frac{-30}{\sqrt{14^2 + (-30)^2}} )^2


## Index 2127, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℤ)
(hn : n > 0)
(h7 : 7 ∣ n)
(h8 : 8 ∣ 3 * n)
(h9 : 9 ∣ 5 * n)
: n ≥ 504
:= sorry
```
### Informal Problem
Let \( n \) be a positive integer such that:
- 7 divides \( n \),
- 8 divides \( 3n \), and
- 9 divides \( 5n \).

Find the smallest possible value of \( n \).
### Informal Answer
504


## Index 618, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(F : ℝ → ℝ)
(hF : F = fun x => x ^ 2)
(G : ℝ → ℝ)
(hG : G = fun x => 2 * x ^ 2)
: G (F 3) / F (G 3) = 1 / 2
:= sorry
```
### Informal Problem
Let \( F \) and \( G \) be functions from the real numbers to the real numbers defined by \( F(x) = x^2 \) and \( G(x) = 2x^2 \). Compute the value of \( \frac{G(F(3))}{F(G(3))} \).
### Informal Answer
\frac{1}{2}


## Index 767, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(h₀ : n ∈ Finset.Icc 10 99)
(h₁ : ∃ k, n = k * 11)
(h₂ : (Nat.digits 10 n).sum = 10)
: n = 55
:= sorry
```
### Informal Problem
Let \( n \) be a natural number between 10 and 99 inclusive. Suppose \( n \) is a multiple of 11, and the sum of its digits (in base 10) is 10. Find the value of \( n \).
### Informal Answer
55


## Index 3542, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(u : ℝ)
(v : ℝ)
(h₂ : u ^ 2 + v ^ 2 = 4)
(x : ℝ)
(y : ℝ)
(h₀ : u + v = x + y)
(h₁ : u * v = x * y)
(h₃ : x ^ 2 - y ^ 2 = 2 * Real.sqrt 3)
: x ^ 2 = 2 + √3 ∧ y ^ 2 = 2 - √3
:= sorry
```
### Informal Problem
Let \( u, v, x, y \) be real numbers such that:
- \( u^2 + v^2 = 4 \),
- \( u + v = x + y \),
- \( u \cdot v = x \cdot y \),
- \( x^2 - y^2 = 2 \sqrt{3} \).

Find the values of \( x^2 \) and \( y^2 \).
### Informal Answer
x^2 = 2 + \sqrt{3} and y^2 = 2 - \sqrt{3}


## Index 4930, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℂ)
(ha : a ≠ 0)
(b : ℂ)
(hb : b ≠ 0)
(c : ℂ)
(h1 : a + b + c = 0)
(h2 : a^2 + b^2 + c^2 = 1 / 2 + Complex.I)
(hc : c ≠ 0)
: a * b + b * c + c * a = -(1 / 4 + Complex.I / 2)
:= sorry
```
### Informal Problem
Let \( a, b, c \) be nonzero complex numbers such that:
- \( a + b + c = 0 \),
- \( a^2 + b^2 + c^2 = \frac{1}{2} + i \),
- \( c \neq 0 \).

Compute the value of \( a \cdot b + b \cdot c + c \cdot a \).
### Informal Answer
-( \frac{1}{4} + \frac{i}{2} )


## Index 2739, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(ha : a = 3 / 8)
(l : ℝ)
(hl : l = 3 / 10)
(t : ℝ)
(ht : t = 1 - a - l)
: t = 13 / 40
:= sorry
```
### Informal Problem
Let \( a = \frac{3}{8} \), \( l = \frac{3}{10} \), and \( t = 1 - a - l \). Compute the value of \( t \).
### Informal Answer
\frac{13}{40}


## Index 1191, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℕ)
(b : ℕ)
(h₀ : a + b = 83)
(c : ℕ)
(h₁ : b + c = 86)
(d : ℕ)
(h₂ : c + d = 88)
: a + d = 85
:= sorry
```
### Informal Problem
Let \( a, b, c, d \) be natural numbers such that:
- \( a + b = 83 \),
- \( b + c = 86 \),
- \( c + d = 88 \).

Find the value of \( a + d \).
### Informal Answer
85


## Index 2797, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(b : ℝ)
(h₀ : a / b = 3 / 5)
(c : ℝ)
(h₁ : b / c = 15 / 6)
(d : ℝ)
(h₂ : c / d = 6)
: a / d = 9
:= sorry
```
### Informal Problem
Let \( a, b, c, d \) be real numbers such that:
- \( \frac{a}{b} = \frac{3}{5} \),
- \( \frac{b}{c} = \frac{15}{6} \),
- \( \frac{c}{d} = 6 \).

Find the value of \( \frac{a}{d} \).
### Informal Answer
9


## Index 2678, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(l : ℝ)
(hl : l = 720)
(v1 : ℝ)
(hv1 : v1 = 18)
(v2 : ℝ)
(hv2 : v2 = 12)
(v3 : ℝ)
(hv3 : v3 = 15)
(t3 : ℝ)
(ht3 : t3 = l / v3)
(t1 : ℝ)
(ht1 : t1 = l / v1)
(t2 : ℝ)
(ht2 : t2 = l / v2)
: t1 = 40 ∧ t2 = 60 ∧ t3 = 48
:= sorry
```
### Informal Problem
Let \( l = 720 \), \( v_1 = 18 \), \( v_2 = 12 \), and \( v_3 = 15 \). Define \( t_1 = \frac{l}{v_1} \), \( t_2 = \frac{l}{v_2} \), and \( t_3 = \frac{l}{v_3} \). Find the values of \( t_1 \), \( t_2 \), and \( t_3 \).
### Informal Answer
t_1 = 40 and t_2 = 60 and t_3 = 48


## Index 1515, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℤ)
(b : ℤ)
(ha : a < b)
(h : a + b = 16)
(h1 : ∃ c, a + 1 = c ∧ c = b - 1)
: a * b = 63
:= sorry
```
### Informal Problem
Let \( a \) and \( b \) be integers such that \( a < b \) and \( a + b = 16 \). Suppose there exists an integer \( c \) such that \( a + 1 = c \) and \( c = b - 1 \). Compute the product \( a \times b \).
### Informal Answer
63


## Index 1966, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(α : ℝ)
(β : ℝ)
(h₁ : β - α = a)
(γ : ℝ)
(h₂ : γ - β = a)
(δ : ℝ)
(h₀ : α < β ∧ β < γ ∧ γ < δ)
(h₃ : δ - γ = a)
(h : (α + β) / 2 = 15 ∧ (β + γ) / 2 = 25 ∧ (γ + δ) / 2 = 35)
: (δ - α) / 3 = 10
:= sorry
```
### Informal Problem
Let \( \alpha, \beta, \gamma, \delta \) be real numbers such that:
- The differences \( \beta - \alpha \), \( \gamma - \beta \), and \( \delta - \gamma \) are all equal to some real number \( a \).
- The numbers satisfy \( \alpha < \beta < \gamma < \delta \).
- The averages \( \frac{\alpha + \beta}{2} = 15 \), \( \frac{\beta + \gamma}{2} = 25 \), and \( \frac{\gamma + \delta}{2} = 35 \).

Find the value of \( \frac{\delta - \alpha}{3} \).
### Informal Answer
10


## Index 563, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(u : ℤ)
(hu : u = 5)
(v : ℤ)
(hv : v = -3)
(w : ℤ)
(hw : w = -4)
(x : ℤ)
(hx : x = 6)
(y : ℤ)
(hy : y = -2)
(z : ℤ)
(hz : z = 7)
: (u * v * x * y * z - u * w * x * z - u * w * y * z - u * v * w * x + u * v * w * y + v * w * x * y + u * w * x -
      v * w) /
    (u * v * w * x) =
  97 / 42
:= sorry
```
### Informal Problem
Let \( u = 5 \), \( v = -3 \), \( w = -4 \), \( x = 6 \), \( y = -2 \), and \( z = 7 \) be integers. Compute the value of the expression:

\[
\frac{u \cdot v \cdot x \cdot y \cdot z - u \cdot w \cdot x \cdot z - u \cdot w \cdot y \cdot z - u \cdot v \cdot w \cdot x + u \cdot v \cdot w \cdot y + v \cdot w \cdot x \cdot y + u \cdot w \cdot x - v \cdot w}{u \cdot v \cdot w \cdot x}.
\]
### Informal Answer
\frac{97}{42}


## Index 4719, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(hx : x ≠ 0)
(h : x^3 + 1 / x^3 = 52)
: x + 1 / x = 4
:= sorry
```
### Informal Problem
Let \( x \) be a nonzero real number such that \( x^3 + \frac{1}{x^3} = 52 \). Find the value of \( x + \frac{1}{x} \).
### Informal Answer
4


## Index 4927, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(ha : a > 0)
(f : ℝ → ℝ)
(hf : ∀ x, f x = a * x ^ 2 - Real.sqrt 2)
(h : f (f 2) = -Real.sqrt 2)
: a = √2 / 4
:= sorry
```
### Informal Problem
Let \( a \) be a positive real number. Define a function \( f : \mathbb{R} \to \mathbb{R} \) by  
\[
f(x) = a \cdot x^2 - \sqrt{2}.
\]  
Given that \( f(f(2)) = -\sqrt{2} \), find the value of \( a \).
### Informal Answer
\frac{\sqrt{2}}{4}


## Index 1273, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(ha : a > 0)
(h : (a + 1) * 9 = 7 * (a + 5))
: a = 13
:= sorry
```
### Informal Problem
Let \( a \) be a positive real number. If \( (a + 1) \cdot 9 = 7 \cdot (a + 5) \), what is the value of \( a \)?
### Informal Answer
13


## Index 4130, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(r : ℝ → ℝ → ℝ)
(hr : r = fun ρ θ => ρ * Real.cos θ)
: r 2 (Real.pi / 3) = 1
:= sorry
```
### Informal Problem
Let \( r \) be a function from \( \mathbb{R} \times \mathbb{R} \) to \( \mathbb{R} \) defined by \( r(\rho, \theta) = \rho \cdot \cos(\theta) \).  
Compute the value of \( r(2, \pi/3) \).
### Informal Answer
1


## Index 1194, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(x : ℝ)
(hx : x > 0)
(h : 23 = x + a * x)
(ha : a = 1 + 20 / 100)
: x = 115 / 11
:= sorry
```
### Informal Problem
Let \( a \) and \( x \) be real numbers with \( x > 0 \). Suppose that \( 23 = x + a \cdot x \) and \( a = 1 + \frac{20}{100} \). Find the value of \( x \).
### Informal Answer
\frac{115}{11}


## Index 3616, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(p : Polynomial ℝ)
(h1 : ∀ x, (x^2 + 2 * x - 1) * p.eval x = (x^2 - 4 * x + 2) * p.eval (x - 1))
(h2 : p.eval (-2) = -64)
: Polynomial.eval 3 p = 32 / 7
:= sorry
```
### Informal Problem
Let \( p \) be a polynomial with real coefficients. Suppose that for every real number \( x \), the following identity holds:
\[
(x^2 + 2x - 1) \cdot p(x) = (x^2 - 4x + 2) \cdot p(x - 1),
\]
where \( p(x) \) denotes the evaluation of \( p \) at \( x \). Given that \( p(-2) = -64 \), find the value of \( p(3) \).
### Informal Answer
\frac{32}{7}


## Index 2896, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℝ → ℝ)
(hf : f = fun x => x ^ 2 + 2 * x - 5)
(A : ℝ × ℝ)
(hA : A.2 = f A.1)
(hA' : A.2 = -2)
(B : ℝ × ℝ)
(hB : B.2 = f B.1)
(hB' : B.2 = -2)
(hAB : A.1 < B.1)
: A.1 * B.1 = -3
:= sorry
```
### Informal Problem
Let \( f : \mathbb{R} \to \mathbb{R} \) be the function defined by \( f(x) = x^2 + 2x - 5 \). Consider two points \( A = (x_A, y_A) \) and \( B = (x_B, y_B) \) on the graph of \( f \), both with \( y \)-coordinate equal to \(-2\), and such that \( x_A < x_B \). Find the product \( x_A \cdot x_B \).
### Informal Answer
-3


## Index 1993, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(hn : n = 48)
(m : ℕ)
(hm : m = 16)
: n * m / 6 = 128
:= sorry
```
### Informal Problem
Let \( n = 48 \) and \( m = 16 \). Compute the value of \( (n \times m) \div 6 \).
### Informal Answer
128


## Index 4954, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(y : ℝ)
(x : ℝ)
(h3 : x * 5  =  1500)
(h4 : x * 4  =  1200 )
(h1 : y * 6 = x * 5)
: y = 250
:= sorry
```
### Informal Problem
Let \( x \) and \( y \) be real numbers such that:
- \( x \times 5 = 1500 \),
- \( x \times 4 = 1200 \), and
- \( y \times 6 = x \times 5 \).

Find the value of \( y \).
### Informal Answer
250


## Index 4289, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(α : ℕ)
(β : ℕ)
(γ : ℕ)
(δ : ℕ)
(k : ℕ)
(h₁ : (γ + δ) / 2 = k)
(h₂ : ∃ m, γ + δ = 2 * m ∧ k = m)
(h₀ : (α + β) / 2 = γ + δ)
: (α + β) / 2 = 2 * k
:= sorry
```
### Informal Problem
Let \( \alpha, \beta, \gamma, \delta, k \) be natural numbers such that:
- \( \frac{\gamma + \delta}{2} = k \),
- There exists a natural number \( m \) such that \( \gamma + \delta = 2m \) and \( k = m \),
- \( \frac{\alpha + \beta}{2} = \gamma + \delta \).

Find the value of \( \frac{\alpha + \beta}{2} \) in terms of \( k \).
### Informal Answer
2k


## Index 1605, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(ha : (2013 : ℝ) / a = -1)
(b : ℝ)
(hb : b / (2014 : ℝ) = -1)
: a * b = 2013 * 2014
:= sorry
```
### Informal Problem
Let \( a \) and \( b \) be real numbers such that:
- \( \frac{2013}{a} = -1 \), and
- \( \frac{b}{2014} = -1 \).

Compute the product \( a \times b \).
### Informal Answer
2013 \times 2014


## Index 2419, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(hn : n = 100)
(m : ℕ)
(hm : m = 20)
: n / m = 5
:= sorry
```
### Informal Problem
Let \( n = 100 \) and \( m = 20 \). Compute the integer division \( n / m \).
### Informal Answer
5


## Index 450, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(hn : n > 0)
(a : ℕ)
(ha : a ∈ Finset.Icc 1 n)
(h : ∑ i ∈ Finset.Icc 1 n, i = 1986 - a)
: a = 33
:= sorry
```
### Informal Problem
Let \( n \) be a positive natural number, and let \( a \) be a natural number such that \( 1 \leq a \leq n \). Suppose the sum of all natural numbers from 1 to \( n \) is equal to \( 1986 - a \). Determine the value of \( a \).
### Informal Answer
33


## Index 1668, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℤ)
(b : ℤ)
(c : ℤ)
(d : ℤ)
(P : ℤ → ℤ)
(hP : ∀ x, P x = a * x ^ 3 + b * x ^ 2 + c * x + d)
(hP' : ∀ x, |x| ≤ 2 → |P x| ≤ 100)
: |a| + |b| + |c| + |d| ≤ 334
:= sorry
```
### Informal Problem
Let \( a, b, c, d \) be integers. Define a cubic polynomial \( P(x) = a x^3 + b x^2 + c x + d \). Suppose that for all integers \( x \) with \( |x| \leq 2 \), we have \( |P(x)| \leq 100 \). Find an upper bound for \( |a| + |b| + |c| + |d| \).
### Informal Answer
334


## Index 2352, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(f : ℝ → ℝ)
(hf : f = λ x => 4.75 * x + 257)
(x : ℝ)
(hx : x = 28)
: f x = 390
:= sorry
```
### Informal Problem
Let \( f \) be a function from the real numbers to the real numbers defined by \( f(x) = 4.75x + 257 \). If \( x = 28 \), what is the value of \( f(x) \)?
### Informal Answer
390


## Index 3063, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(r : ℕ)
(y : ℕ)
(h₁ : r = y + 5)
(o : ℕ)
(h₀ : r + y + o = 35)
(h₂ : o = 2 * r)
: r = 10 ∧ y = 5 ∧ o = 20
:= sorry
```
### Informal Problem
Let \( r, y, o \) be natural numbers such that:
- \( r = y + 5 \),
- \( r + y + o = 35 \),
- \( o = 2r \).

Find the values of \( r \), \( y \), and \( o \).
### Informal Answer
r = 10 and y = 5 and o = 20


## Index 1973, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example

: IsGreatest {n : ℕ | 9 ^ 2 ≤ n ^ 2 ∧ n ^ 2 < 9 ^ 3} 26
:= sorry
```
### Informal Problem
Find the greatest natural number \( n \) such that \( 9^2 \leq n^2 < 9^3 \).
### Informal Answer
26


## Index 521, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(h₀ : n < 100)
(h₁ : n > 9)
(a : ℕ)
(h₂ : n % 10 = a)
(b : ℕ)
(h₃ : n / 10 = b)
(h₄ : a * b = 24)
: n = 28 ∨ n = 82 ∨ n = 38 ∨ n = 83 ∨ n = 46 ∨ n = 64 ∨ n = 18 ∨ n = 81
:= sorry
```
### Informal Problem
Let \( n \) be a natural number between 10 and 99 (inclusive). Let \( a \) be the units digit of \( n \) and \( b \) be the tens digit of \( n \). If the product \( a \times b = 24 \), what are all the possible values of \( n \)?
### Informal Answer
81


## Index 1524, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℤ)
(h : -5678901234567890 ≡ x [ZMOD 123456789])
: x % 123456789 = 71778799
:= sorry
```
### Informal Problem
Let \( x \) be an integer such that  
\[
-5678901234567890 \equiv x \pmod{123456789}.
\]  
Find the value of \( x \mod 123456789 \).
### Informal Answer
71778799


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


## Index 2346, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℤ)
(y : ℤ)
(h : x * y = x + y)
: x * y = 0 ∨ x * y = 4
:= sorry
```
### Informal Problem
Let \( x \) and \( y \) be integers such that \( x \cdot y = x + y \). Determine all possible values of \( x \cdot y \).
### Informal Answer
4


## Index 1529, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(p : ℕ)
(hp : Nat.Prime p)
(h : Nat.Prime (8 * p ^ 2 + 1))
: p = 3
:= sorry
```
### Informal Problem
Let \( p \) be a prime number such that \( 8p^2 + 1 \) is also prime. Determine the value of \( p \).
### Informal Answer
3


## Index 3917, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℝ)
(b : ℝ)
(x : ℝ)
(y : ℝ)
(h1 : a * x + b * y = 3)
(h2 : a * (3 * x) + b * (3 * y) = 9)
(h3 : a * (5 * x) + b * (5 * y) = 15)
: a * (11 * x) + b * (11 * y) = 33
:= sorry
```
### Informal Problem
Let \( a, b, x, y \) be real numbers such that:
- \( a \cdot x + b \cdot y = 3 \),
- \( a \cdot (3x) + b \cdot (3y) = 9 \),
- \( a \cdot (5x) + b \cdot (5y) = 15 \).

Compute the value of \( a \cdot (11x) + b \cdot (11y) \).
### Informal Answer
33


## Index 3020, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(p : ℕ)
(q : ℕ)
(r : ℕ)
(hpqr : p.Prime ∧ q.Prime ∧ r.Prime)
(h : 2 * p + 3 * q = 6 * r)
(n : ℕ)
(h'' : p + q + r = n)
: n = 7
:= sorry
```
### Informal Problem
Let \( p, q, r \) be prime numbers such that \( 2p + 3q = 6r \). If \( n = p + q + r \), what is the value of \( n \)?
### Informal Answer
7


## Index 4970, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(x : ℝ)
(y : ℝ)
(h : x ≠ y)
(h1 : x^2 + Real.sqrt 2 * y = Real.sqrt 3)
(h2 : y^2 + Real.sqrt 2 * x = Real.sqrt 3)
: x + y = √2
:= sorry
```
### Informal Problem
Let \( x \) and \( y \) be real numbers such that \( x \ne y \). Suppose:
\[
x^2 + \sqrt{2} \cdot y = \sqrt{3}, \quad y^2 + \sqrt{2} \cdot x = \sqrt{3}.
\]
Find the value of \( x + y \).
### Informal Answer
\sqrt{2}


## Index 3579, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(b : ℕ)
(g : ℕ)
(h : b + g = 40)
(h' : b * 5 = g * 3)
: b = 15
:= sorry
```
### Informal Problem
Let \( b \) and \( g \) be natural numbers such that:
- Their sum is 40: \( b + g = 40 \).
- Five times \( b \) equals three times \( g \): \( 5b = 3g \).

Find the value of \( b \).
### Informal Answer
15


## Index 1057, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(a : ℕ → ℝ)
(ha1 : a 1 = 1)
(han : ∀ n ≥ 2, a n = if Even n then a (n / 2) + 1 else 1 / a (n - 1))
: ∃ (n : ℕ), a n = 19 / 2
:= sorry
```
### Informal Problem
Define a sequence \( a_1, a_2, a_3, \ldots \) of real numbers by:
- \( a_1 = 1 \), and
- For every integer \( n \geq 2 \),
  \[
  a_n = \begin{cases}
  a_{n/2} + 1 & \text{if } n \text{ is even}, \\
  \frac{1}{a_{n-1}} & \text{if } n \text{ is odd}.
  \end{cases}
  \]
Find a natural number \( n \) such that \( a_n = \frac{19}{2} \).
### Informal Answer
1536


## Index 1954, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(P : ℝ)
(x : ℝ)
(hx : x ≠ 0)
(y : ℝ)
(hy : y ≠ 0)
(z : ℝ)
(hP : P = 1 / x + 1 / y + 1 / z)
(hz : z ≠ 0)
(a : ℝ)
(b : ℝ)
(c : ℝ)
(h : x / a + y / b + z / c = 1)
(ha : a ≠ 0)
(hb : b ≠ 0)
(hc : c ≠ 0)
(h' : a / x + b / y + c / z = 0)
: (a - 1) / x + (b - 1) / y + (c - 1) / z = a / x + b / y + c / z - P
:= sorry
```
### Informal Problem
Let \( x, y, z \) be nonzero real numbers, and define \( P = \frac{1}{x} + \frac{1}{y} + \frac{1}{z} \). Let \( a, b, c \) be nonzero real numbers such that:
\[
\frac{x}{a} + \frac{y}{b} + \frac{z}{c} = 1
\]
and
\[
\frac{a}{x} + \frac{b}{y} + \frac{c}{z} = 0.
\]
Simplify the expression:
\[
\frac{a - 1}{x} + \frac{b - 1}{y} + \frac{c - 1}{z}.
\]
### Informal Answer
\frac{a}{x} + \frac{b}{y} + \frac{c}{z} - P


## Index 1810, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(N : ℕ)
(hN : N ≠ 0)
(h : 76 * N / 100 + 24 * N / 100 = N)
: N ≥ 25
:= sorry
```
### Informal Problem
Let \( N \) be a nonzero natural number. Given that:
\[
\frac{76 \cdot N}{100} + \frac{24 \cdot N}{100} = N,
\]
determine the smallest possible value of \( N \).
### Informal Answer
25


## Index 4154, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(n : ℕ)
(hn : n = 6)
(m : ℕ)
(hm : m = 8)
: n * m / 2 * 2 = 48
:= sorry
```
### Informal Problem
Let \( n = 6 \) and \( m = 8 \). Compute the value of \( (n \times m) \div 2 \times 2 \).
### Informal Answer
48


## Index 1938, Src sft_ar_v3/Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch
### Formal Statement
```lean4
example
(w : ℝ × ℝ)
(hw : w = (3, 2))
(d : ℝ × ℝ)
(hd : d = (-1, 2))
(p : ℝ × ℝ)
(hp : p = 5 * d - 2 * w)
: p.1 = -11
:= sorry
```
### Informal Problem
Let \( w \) and \( d \) be vectors in \( \mathbb{R}^2 \) defined by \( w = (3, 2) \) and \( d = (-1, 2) \). Define a vector \( p \) by \( p = 5d - 2w \). Compute the first component \( p_1 \) of \( p \).
### Informal Answer
-11


