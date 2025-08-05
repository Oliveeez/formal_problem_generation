import Mathlib
import Aesop

set_option maxHeartbeats 0
set_option maxRecDepth 1000000

example
  -- $ABC$ is a triangle
  (A B C : ℝ)
  (h_triangle_sum : A + B + C = Real.pi)
  (h_triangle_A : 0 < A ∧ A < Real.pi)
  (h_triangle_B : 0 < B ∧ B < Real.pi)
  (h_triangle_C : 0 < C ∧ C < Real.pi)
  -- the sides opposite to angles $A$, $B$, $C$ are denoted as $a$, $b$, $c$
  (a b c : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_pos_c : c > 0)
  (h_cos_A : Real.cos A = (b^2+c^2-a^2)/(2*b*c))
  (h_cos_B : Real.cos B = (a^2+c^2-b^2)/(2*a*c))
  (h_cos_C : Real.cos C = (a^2+b^2-c^2)/(2*a*b))
  -- $\\\\overrightarrow{m}=(\\\\sqrt{3}b-c,\\\\cos C)$, $\\\\overrightarrow{n}=(a,\\\\cos A)$
  (m n : ℝ × ℝ)
  (hm : m = (Real.sqrt 3 * b - c, Real.cos C))
  (hn : n = (a, Real.cos A))
  -- $\\\\overrightarrow{m} \\\\parallel \\\\overrightarrow{n}$
  (h_parallel : m.1 * n.2 = m.2 * n.1)
  -- determine the value of $\\\\cos A$.
  (answer : ℝ)
  (h_answer : answer = Real.cos A)
  -- # Answer \\\\dfrac{\\\\sqrt{3}}{3}
: (answer = Real.sqrt 3 / 3)
:= by

  have h_triangle : (Real.sqrt 3 * b - c)*(Real.cos A) = (Real.cos C)*a := by
    rw [hm, hn] at h_parallel
    exact h_parallel

  -- deduce the condition h_triangle_simp by substituting h_cos_A and h_cos_B into h_triangle and simplifying
  have h_triangle_simp : (Real.sqrt 3 * b - c)*(b^2+c^2-a^2)=(a^2+b^2-c^2)*c := by
    rw [h_cos_A, h_cos_C] at h_triangle
    have heq1 : (√3 * b - c) * ((b ^ 2 + c ^ 2 - a ^ 2) / (2 * b * c)) = (a ^ 2 + b ^ 2 - c ^ 2) / (2 * a * b) * a := by
      linarith [h_triangle]
    have heq2 : (a ^ 2 + b ^ 2 - c ^ 2) / (2 * a * b) * a = (a ^ 2 + b ^ 2 - c ^ 2) / (2 * b) := by
      field_simp [show a ≠ 0 by linarith, show b ≠ 0 by linarith]
      all_goals nlinarith
    rw [heq2] at heq1
    have h1 : b ≠ 0 := by linarith
    have h2 : c ≠ 0 := by linarith
    have heq3 : (√3 * b - c) * ((b ^ 2 + c ^ 2 - a ^ 2) / (2 * b * c)) * (2 * b) = (a ^ 2 + b ^ 2 - c ^ 2) / (2 * b) * (2 * b) := by
      rw [heq1]
    have heq4 : (√3 * b - c) * ((b ^ 2 + c ^ 2 - a ^ 2) / (2 * b * c)) * (2 * b) = (√3 * b - c) * ((b ^ 2 + c ^ 2 - a ^ 2) / c) := by
      field_simp [show b ≠ 0 by linarith, show c ≠ 0 by linarith]
      all_goals nlinarith
    have heq5 : (a ^ 2 + b ^ 2 - c ^ 2) / (2 * b) * (2 * b) = a ^ 2 + b ^ 2 - c ^ 2 := by
      field_simp [show b ≠ 0 by linarith]
    rw [heq4, heq5] at heq3
    have heq6 : (√3 * b - c) * (b ^ 2 + c ^ 2 - a ^ 2) = (a ^ 2 + b ^ 2 - c ^ 2) * c := by
      have h3 : c ≠ 0 := by linarith
      field_simp [h3] at heq3
      rw [heq3]
    exact heq6

  -- deduce the condition h_a_square by further simplifying the expression h_triangle_simp and expressing a^2
  have h_a_square : a^2 = b^2 + c^2 - 2/Real.sqrt 3 * b * c := by
    have h1 : 0 < √3 := by positivity
    have h2 : √3 > 0 := by positivity
    have eq1 : (√3 * b - c) * (b ^ 2 + c ^ 2 - a ^ 2) = (a ^ 2 + b ^ 2 - c ^ 2) * c := by linarith [h_triangle_simp]
    have eq2 : √3 * b ^ 3 + √3 * b * c ^ 2 - √3 * a ^ 2 * b - c * b ^ 2 - c ^ 3 + a ^ 2 * c = a ^ 2 * c + b ^ 2 * c - c ^ 3 := by
      nlinarith [eq1, sq_pos_of_pos h_pos_b, sq_pos_of_pos h_pos_c, sq_pos_of_pos h_pos_a, Real.sqrt_pos.mpr (show (3 : ℝ) > 0 by norm_num), Real.sq_sqrt (show (0 : ℝ) ≤ (3 : ℝ) by norm_num)]
    have eq3 : √3 * b ^ 3 + √3 * b * c ^ 2 - √3 * a ^ 2 * b - c * b ^ 2 - b ^ 2 * c = 0 := by
      nlinarith [eq2, sq_pos_of_pos h_pos_b, sq_pos_of_pos h_pos_c, sq_pos_of_pos h_pos_a, Real.sqrt_pos.mpr (show (3 : ℝ) > 0 by norm_num), Real.sq_sqrt (show (0 : ℝ) ≤ (3 : ℝ) by norm_num)]
    have eq4 : b * (√3 * b ^ 2 + √3 * c ^ 2 - √3 * a ^ 2 - c * b - b * c) = 0 := by
      nlinarith [eq3, sq_pos_of_pos h_pos_b, sq_pos_of_pos h_pos_c, sq_pos_of_pos h_pos_a, Real.sqrt_pos.mpr (show (3 : ℝ) > 0 by norm_num), Real.sq_sqrt (show (0 : ℝ) ≤ (3 : ℝ) by norm_num)]
    cases' (mul_eq_zero.mp eq4) with hb h5
    · -- Case where b = 0, which contradicts h_pos_b
      exfalso
      linarith
    · -- Case where the second factor is zero
      have eq6 : √3 * b ^ 2 + √3 * c ^ 2 - √3 * a ^ 2 - c * b - b * c = 0 := by
        nlinarith
      have eq7 : √3 * (b ^ 2 + c ^ 2 - a ^ 2) = 2 * b * c := by
        nlinarith [eq6, sq_pos_of_pos h_pos_b, sq_pos_of_pos h_pos_c, sq_pos_of_pos h_pos_a, Real.sqrt_pos.mpr (show (3 : ℝ) > 0 by norm_num), Real.sq_sqrt (show (0 : ℝ) ≤ (3 : ℝ) by norm_num)]
      have eq8 : b ^ 2 + c ^ 2 - a ^ 2 = (2 * b * c) / √3 := by
        have h9 : √3 ≠ 0 := by positivity
        field_simp
        nlinarith [eq7, Real.sqrt_pos.mpr (show (3 : ℝ) > 0 by norm_num), Real.sq_sqrt (show (0 : ℝ) ≤ (3 : ℝ) by norm_num)]
      have eq9 : a ^ 2 = b ^ 2 + c ^ 2 - 2 / √3 * b * c := by
        have h10 : √3 > 0 := by positivity
        have eq10 : (2 * b * c) / √3 = (2 / √3) * b * c := by
          field_simp
          all_goals ring
        nlinarith [eq8, eq10, Real.sqrt_pos.mpr (show (3 : ℝ) > 0 by norm_num), Real.sq_sqrt (show (0 : ℝ) ≤ (3 : ℝ) by norm_num)]
      exact_mod_cast eq9

  -- deduce the answer that Real.cos A = Real.sqrt 3 / 3 by substituting h_a_square into h_cos_A
  have h_submission : Real.cos A = Real.sqrt 3 / 3 := by
    rw [h_cos_A, h_a_square]
    field_simp [show b ≠ 0 by linarith, show c ≠ 0 by linarith]
    have h1 : √3 ^ 2 = 3 := by
      rw [Real.sq_sqrt]
      norm_num
    calc
      2 * b * c * 3 = 6 * b * c := by ring
      _ = √3 * (√3 * (2 * b * c)) := by
        have h2 : √3 * (√3 * (2 * b * c)) = (√3 ^ 2) * (2 * b * c) := by
          ring
        rw [h2]
        rw [h1]
        ring

  rw [h_submission] at h_answer

  exact h_answer

example
  -- $m$ and $n$ are 2 integers
  (m n : ℤ)
  -- $m > n$
  (hmn : m > n)
  -- $m+n=20$ and $m^2+n^2=328$
  (h_sum : m+n=20)
  (h_square_sum : m^2+n^2=328)
  -- find the value of $m^2-n^2$
  (answer : ℤ)
  (h_answer : answer=m^2-n^2)
  -- # Answer 320
 : (answer = 320)
 := by

  -- deduce the condition m*n=36 from h_sum and h_square_sum
  have h_mul : m * n = 36 := by
    have : (m + n)^2 = m^2 + n^2 + 2 * m * n := by ring
    rw [h_sum, h_square_sum] at this
    linarith

  -- deduce the condition (m-n)^2 = 256 from h_square_sum and h_mul
  have h_diff_square : (m - n)^2 = 256 := by
    have : (m - n)^2 = m^2 + n^2 - 2 * (m * n) := by ring
    rw [h_square_sum, h_mul] at this
    linarith

  -- deduce the condition m-n=16 from h_diff_square and hmn
  have h_diff : m - n = 16 := by
    nlinarith

  -- deduce the answer that m^2-n^2 = 320 from h_sum and h_diff
  have h_submission : m^2 - n^2 = 320 := by
    nlinarith

  rw [h_submission] at h_answer

  exact h_answer

example
  -- $p,q,r,s,t$ are five primes
  (p q r s t : Nat)
  (hp : p.Prime)
  (hq : q.Prime)
  (hr : r.Prime)
  (hs : s.Prime)
  (ht : t.Prime)
  -- $p$ can be expressed as the difference of $q$ and $r$
  (heq1 : p = q - r)
  -- $p$ can be expressed as the sum of $s$ and $t$
  (heq2 : p = s + t)
  -- w.l.o.g. assume t > s to facilitate proving
  (hle1 : t > s)
  -- find the value of $p$
  (answer : Nat)
  (h_answer : answer = p)
  -- # Answer 5
: answer = 5
:= by

  -- introduce five primes $p, q, r, s, t$
  have hle2 : q > r := by
    by_contra h
    have : p = 0 := by
      rw [heq1]
      omega
    apply Nat.Prime.ne_zero hp this

  -- deduce the condition that p > 2
  have hge : p > 2 := by
    rw [heq2]
    have : s ≥ 2 := by exact Nat.Prime.two_le hs
    have : t ≥ 2 := by exact Nat.Prime.two_le ht
    linarith

  -- deduce the condition that p is odd
  have hoddp : Odd p := by
    by_contra h
    simp at h
    have : p = 2 := by exact (Nat.Prime.even_iff hp).mp h
    linarith

  -- deduce the condition that r = 2
  have heqr : r = 2 := by
    have : Even q ∨ Even r := by
      rw [heq1] at hoddp
      by_contra h
      simp at h
      have : Even (q-r) := Nat.Odd.sub_odd h.1 h.2
      have : ¬ Odd (q-r) := by exact Nat.not_odd_iff_even.mpr this
      contradiction
    rcases this with h1 | h2
    · have : q = 2 := by exact (Nat.Prime.even_iff hq).mp h1
      rw [this] at hle2
      have : r ≥ 2 := by exact Nat.Prime.two_le hr
      linarith
    · exact (Nat.Prime.even_iff hr).mp h2

  -- deduce the condition that either s or t is even
  have heven : Even s ∨ Even t := by
    rw [heq2] at hoddp
    by_contra h
    simp at h
    have : Even (s + t) := Odd.add_odd h.1 h.2
    have : ¬ Odd (s+t) := by exact Nat.not_odd_iff_even.mpr this
    contradiction

  -- deduce the condition that s is Even.
  have hevens : Even s := by
    rcases heven with h1 | h2
    · exact h1
    · have : t = 2 := by exact (Nat.Prime.even_iff ht).mp h2
      have : s ≥ 2 := by exact Nat.Prime.two_le hs
      linarith

  -- duduce the condition that s = 2
  have heqs : s = 2 := by exact (Nat.Prime.even_iff hs).mp hevens

  -- duduce the condition that t is 3 by considering the possible values of t modulo 3
  have heqt : t = 3 := by
    have : t % 3 = 0 ∨ t % 3 = 1 ∨ t % 3 = 2 := by
      by_contra h
      simp at h
      have : t % 3 ≥ 3 := by
        rcases h with ⟨h1, h2, h3⟩
        apply Nat.add_one_le_iff.mpr
        apply Nat.two_lt_of_ne
        · apply h1
        · apply h2
        · apply h3
      have : t % 3 < 3 := Nat.mod_lt t (by norm_num)
      linarith
    rcases this with h1 | h2 | h3
    · have : 3 ∣ t := by exact Nat.dvd_of_mod_eq_zero h1
      symm
      apply (Nat.dvd_prime_two_le ht (by linarith)).mp this
    · have : p % 3 = 0 := by
        rw [heq2, heqs, Nat.add_mod 2 t 3, h2]
      have : 3 ∣ p := by exact Nat.dvd_of_mod_eq_zero this
      have : p = 3 := by
        symm
        apply (Nat.dvd_prime_two_le hp (by linarith)).mp this
      have : t = 1 := by
        calc
          _ = p - s := by rw [heq2]; norm_num
          _ = 3 - 2 := by rw [this, heqs]
      rw [this] at heq2
      have := Nat.Prime.ne_one ht this
      exact False.elim this
    · have : q % 3 = 0 := by
        calc
          _ = (p + r) % 3 := by
            rw [heq1]
            rw [Nat.sub_add_cancel (by linarith)]
          _ = (s + t + r) % 3 := by rw [heq2]
          _ = _ := by
            rw [Nat.add_mod]
            rw [Nat.add_mod s t 3]
            rw [heqs, h3, heqr]
      have : 3 ∣ q := by exact Nat.dvd_of_mod_eq_zero this
      have : q = 3 := by
        symm
        apply (Nat.dvd_prime_two_le hq (by linarith)).mp this
      rw [this, heqr] at heq1
      simp at heq1
      rw [heq1] at hge
      linarith

  -- deduce the answer that p = 5
  have h_submission : p = 5 := by
    omega

  rw [h_submission] at h_answer

  exact h_answer

example
  -- $x, y$ are two real numbers
  (x y : Real)
  -- $x, y$ are both greater than 1
  (hx : x > 1)
  (hy : y > 1)
  -- $\\log_x (y^x) = \\log_y (x^(4y)) = 10$
  (hxy1 : Real.logb x (y^x) = 10)
  (hxy2 : Real.logb y (x^(4*y)) = 10)
  -- find the value of $xy$
  (answer : Real)
  (h_answer : answer = x*y)
  -- # Answer 25
: answer = 25
:= by

  -- deduce that x * Real.logb x y = 10
  have hxy3 : x * Real.logb x y = 10 := by
    rw [← hxy1]
    rw [Real.logb_rpow_eq_mul_logb_of_pos]
    linarith

  -- deduce that 4 * y * Real.logb y x = 10
  have hxy4 : 4 * y * Real.logb y x = 10 := by
    rw [← hxy2]
    rw [Real.logb_rpow_eq_mul_logb_of_pos]
    linarith

  -- deduce that 4 * x * y * Real.logb x y * Real.logb y x = 100 by multiplying the two equations
  have hxy5 : 4 * x * y * (Real.logb x y * Real.logb y x) = 100 := by
    nlinarith

  -- deduce that 4 * x * y = 100 according to the equation that Real.logb x y * Real.logb y x = 1
  have hxy6 : 4 * x * y = 100 := by
    rw [Real.mul_logb (by linarith) (by linarith) (by linarith)] at hxy5
    rw [Real.logb_self_eq_one (by linarith)] at hxy5
    nlinarith

  -- deduce the answer that x * y = 25
  have h_submission : x * y = 25 := by
    nlinarith

  rw [h_submission] at h_answer

  exact h_answer

example
  -- s is the index set {1,2,..., 999}
  (s : Finset ℤ)
  (hs : s = Finset.Icc 1 999)
  -- a and b are two sequence
  (a b : ℤ → ℤ)
  -- the set {1,2, ..., 1998} has been partitioned into disjoint pairs {a_i, b_i} (1 ≤ i ≤ 999)
  (ha : ∀ i ∈ s, a i ∈ Finset.Icc 1 1998)
  (hb : ∀ i ∈ s, b i ∈ Finset.Icc 1 1998)
  (hij : ∀ i ∈ s, ∀ j ∈ s, i ≠ j → a i ≠ a j ∧ b i ≠ b j)
  (hij' : ∀ i ∈ s, ∀ j ∈ s, a i ≠ b j)
  -- |a_i - b_i| equals 1 or 6
  (habs : ∀ i ∈ s, |a i - b i| = 1 ∨ |a i - b i| = 6)
  -- Find that the sum |a_1 - b_1| + |a_2 - b_2| + ... + |a_999 - b_999| ends in which digit
  (answer : ℤ)
  (h_answer : answer = (∑ i ∈ s, |a i - b i|) % 10)
  -- # Answer 9
  : answer = 9
:= by

  -- deduce that the equation |a i - b i| % 2 = (a i + b i) % 2
  have heq : ∀ i ∈ s, |a i - b i| % 2 = (a i + b i) % 2 := by
    intro x hx
    by_cases hge : a x ≥ b x
    · have : |a x - b x| = a x - b x := by
        rw [@abs_eq_self]
        linarith
      rw [this]
      apply Int.modEq_iff_dvd.mpr
      ring_nf
      exact Int.dvd_mul_left (b x) 2
    · have : |a x - b x| = - (a x - b x) := by
        rw [@abs_eq_neg_self]
        linarith
      rw [this]
      apply Int.modEq_iff_dvd.mpr
      ring_nf
      exact Int.dvd_mul_left (a x) 2

  -- deduce that the sum of differences mod 2 is equal to the sum of (a_i + b_i) mod 2
  have heqsum : (∑ i ∈ s, |a i - b i|) % 2 = (∑ i ∈ s, (a i + b i)) % 2 := by
    rw [Finset.sum_int_mod]
    nth_rw 2 [Finset.sum_int_mod]
    rw [Finset.sum_congr rfl heq]

  -- deduce that the sum of differences is odd according to the former equation
  have hmod2 : (∑ i ∈ s, |a i - b i|) % 2 = 1 := by
    have hainj : ∀ x ∈ s, ∀ y ∈ s, a x = a y → x = y := by
      intro x hx y hy hxy
      norm_cast at hx hy
      by_contra hneq
      exact (hij x hx y hy hneq).1 hxy
    have hbinj : ∀ x ∈ s, ∀ y ∈ s, b x = b y → x = y := by
      intro x hx y hy hxy
      norm_cast at hx hy
      by_contra hneq
      exact (hij x hx y hy hneq).2 hxy
    have hdisjoint : Disjoint (Finset.image a s) (Finset.image b s) := by
      unfold Disjoint
      intro x hxa hxb
      intro y hy
      have hya : y ∈ Finset.image a s := by exact hxa hy
      have hyb : y ∈ Finset.image b s := by exact hxb hy
      simp at hya hyb
      rcases hya with ⟨z1, hz1in, hz1eq⟩
      rcases hyb with ⟨z2, hz2in, hz2eq⟩
      have := hij' z1 hz1in z2 hz2in
      rw [hz1eq, hz2eq] at this
      norm_num at this
    have hunion : Finset.image a s ∪ Finset.image b s = Finset.Icc 1 1998 := by
      have hacard : Finset.card (Finset.image a s) = 999 := by
        rw [Finset.card_image_of_injOn hainj]
        rw [hs]
        simp
      have hbcard : Finset.card (Finset.image b s) = 999 := by
        rw [Finset.card_image_of_injOn hbinj]
        rw [hs]
        simp
      have hcard : Finset.card (Finset.image a s ∪ Finset.image b s) = 1998 := by
        rw [Finset.card_union_eq_card_add_card.mpr hdisjoint, hacard, hbcard]
      symm
      refine (Finset.eq_iff_card_ge_of_superset ?hst).mp ?_
      · intro x hx
        simp at hx
        rcases hx with hx1 | hx2
        · rcases hx1 with ⟨y, hyin, hyeq⟩
          have := ha y hyin
          rw [hyeq] at this
          exact this
        · rcases hx2 with ⟨y, hyin, hyeq⟩
          have := hb y hyin
          rw [hyeq] at this
          exact this
      simp
      rw [hcard]
    let f : ℤ → ℤ := fun x => x
    have haeq : ∑ i ∈ s, a i = ∑ j ∈ Finset.image a s, j := by
      have := @Finset.sum_image ℤ ℤ ℤ f _ _ s a hainj
      unfold f at this
      rw [this]
    have hbeq : ∑ i ∈ s, b i = ∑ j ∈ Finset.image b s, j := by
      have := @Finset.sum_image ℤ ℤ ℤ f _ _ s b hbinj
      unfold f at this
      rw [this]
    have heq : ∑ i ∈ s, (a i + b i) = (∑ j ∈ Finset.image a s, j) + (∑ j ∈ Finset.image b s, j) := by
      rw [Finset.sum_add_distrib, haeq, hbeq]
    have := @Finset.sum_disjUnion ℤ ℤ (Finset.image a s) (Finset.image b s) f _ hdisjoint
    simp at this
    rw [← this, hunion] at heq
    unfold f at heq
    rw [heqsum, heq]
    native_decide

  -- deduce that the sum of differences mod 5 is 4
  have hmod5 : (∑ i ∈ s, |a i - b i|) % 5 = 4 := by
    rw [Finset.sum_int_mod]
    have : ∀ i ∈ s, |a i - b i| % 5 = 1 := by
      intro i hi
      have := habs i hi
      rcases this with h1 | h2
      · rw [h1]
        norm_num
      · rw [h2]
        norm_num
    rw [Finset.sum_congr rfl this]
    rw [hs]
    simp

  -- deduce the answer is 9
  have hsum : (∑ i ∈ s, |a i - b i|) % 10 = 9 := by
    have : 10 ∣ ((∑ i ∈ s, |a i - b i|) - 9) := by
      have h2dvd : 2 ∣ ((∑ i ∈ s, |a i - b i|) - 9) := by
        exact Int.ModEq.dvd (id (Eq.symm hmod2))
      have h5dvd : 5 ∣ ((∑ i ∈ s, |a i - b i|) - 9) := by
        exact Int.ModEq.dvd (id (Eq.symm hmod5))
      apply IsCoprime.mul_dvd (by norm_num) h2dvd h5dvd
    exact id (Int.ModEq.symm (Int.modEq_of_dvd this))

  rw [hsum] at h_answer

  exact h_answer
