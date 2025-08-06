# Data Parsing
```shell
# Debug
ulimit -s unlimited;
python data_parse_cycle0123.py \
    --use_mp False \
    --reverse_order True

ulimit -s unlimited;
python data_parse_cycle0123.py \
    --use_mp False \
    --reverse_order True \
    --num_processes 
```

## Bug
```lean4
example
(answer : ℕ)
  (numbers : List ℕ)
  (h_numbers : numbers = List.filter (λ n : ℕ => n % 3 = 0 ∧ n % 4 ≠ 0) (List.range 61)) -- The numbers from 1, 2, 3, …, 60 that can be divided by 3 but not by 4
  (h_average : answer = (List.sum numbers) / (List.length numbers)) -- The average value of these numbers
: -- the answer is 30
answer = 30
:= sorry

have h_average : answer = numbers.sum / numbers.length := sorry
```
`h_average` depends on `numbers`!
