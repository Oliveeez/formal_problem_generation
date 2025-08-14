# Data Parsing
```shell
# Debug
ulimit -s unlimited;
python data_parse_cycle0123.py \
    --use_mp False \
    --reverse_order True

# Kangjie 8 GPUs
ulimit -s unlimited;
python data_parse_cycle0123.py \
    --use_mp True \
    --reverse_order True \
    --num_processes 24 \
    --reverse_order False

# Main 8 GPUs
ulimit -s unlimited;
python data_parse_cycle0123.py \
    --use_mp True \
    --reverse_order True \
    --num_processes 8 \
    --reverse_order Ture
```

# SFT
```shell
# xTuner 7B
export SFT_TASK_NAME=DeepSeek-Prover-V2-7B.Numina1.5_nonsynth.Cycle123.0808
echo $SFT_TASK_NAME
NPROC_PER_NODE=8 xtuner train ./train_recipes/${SFT_TASK_NAME}.py --deepspeed deepspeed_zero2;
xtuner convert pth_to_hf ./train_recipes/${SFT_TASK_NAME}.py ./work_dirs/${SFT_TASK_NAME}/epoch_1.pth /cache/ckpts/${SFT_TASK_NAME}/;
find ./work_dirs/${SFT_TASK_NAME} | grep state | xargs rm

# Llama-Factory 32B
accelerate launch \
--config_file /home/ma-user/workspace/formal_problem_generation/formal_problem_generation/train_recipes/fsdp.yaml \
src/train.py /home/ma-user/workspace/formal_problem_generation/formal_problem_generation/train_recipes/Goedel-Prover-V2-32B.Numina1.5_nonsynth.Cycle123.0808.yaml
```
# Inference
```shell
# Load LLM
export ASCEND_RT_VISIBLE_DEVICES=2;
python -m vllm.entrypoints.openai.api_server \
    --model /sfs/liuqi/ckpts/hf_ckpts/DeepSeek-Prover-V2-7B.Numina1.5_nonsynth.Cycle123.0808 \
    --port 3721${ASCEND_RT_VISIBLE_DEVICES} \
    --dtype bfloat16 \
    --api-key cycle0123_dspv2 \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-log-requests \
    --max-model-len 8192

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3;
python -m vllm.entrypoints.openai.api_server \
    --model /sfs/liuqi/ckpts/hf_ckpts/Goedel-Prover-V2-32B.cycle123_problem_generation_steps \
    --port 37210 \
    --dtype bfloat16 \
    --tensor_parallel_size 4 \
    --api-key cycle0123_goedel \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-log-requests \
    --max-model-len 8192

# Agent Run
ulimit -s unlimited;
python -m evaluator.fps_problem_generation \
    --log_root output/DeepSeek-Prover-V2-7B.Numina1.5_nonsynth.Cycle123.0808 \
    --agent_name sft_ar \
    --base_url http://0.0.0.0:37212/v1 \
    --api_key cycle0123_dspv2 \
    --model_name /sfs/liuqi/ckpts/hf_ckpts/DeepSeek-Prover-V2-7B.Numina1.5_nonsynth.Cycle123.0808 \
    --resume_from output/DeepSeek-Prover-V2-7B.Numina1.5_nonsynth.Cycle123.0808 \
    --num_concurrency 1

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

-- Solved

```python
({'tacticErrors': ['<Pantograph>:0:54: error: failed to synthesize\n  HMul ℕ ℤ ℤ\nAdditional diagnostic information may be available using the `set_option diagnostics true` command.\n']}, GoalState(state_id=50, goals=[Goal(variables=[Variable(t='ℤ', v=None, name='answer')], target='False', sibling_dep=[], name=None, is_conversion=False)], payload={'nextStateId': 50, 'goals': [{'vars': [{'userName': 'answer', 'type': {'pp': 'ℤ', 'dependentMVars': []}, 'name': '_uniq.25493', 'isInaccessible': False}], 'target': {'pp': 'False', 'dependentMVars': []}, 'name': '_uniq.25494', 'isConversion': False}]}, _sentinel=[2, 1, 4, 6, 7, 8, 10, 3, 12, 13, 14, 9, 15, 16, 5, 20, 17, 22, 23, 24, 25, 19, 26, 27, 18, 30, 29, 28, 11, 33, 35, 36, 37, 38, 39, 31, 40, 41, 32, 44, 43, 42, 21, 48, 49]), 'have h_sum_coeff : answer = ∑ k ∈ Finset.range 6, ↑(Nat.choose 5 k) * (-1) ^ k := sorry')

{'tacticErrors': ['<Pantograph>:0:41: error: linarith failed to find a contradiction\ncase h1.h\nanswer : ℚ\na b : ℤ\nh₁ : 9 / 10 + 5 / 6 = a / b\nh₂ : a.natAbs.gcd b.natAbs = 1\nh_answer : ↑a / ↑b = answer\nh_common_denominator : 30 = Nat.lcm 10 6\nh_frac1 : 9 / 10 = 27 / 30\nh_frac2 : 5 / 6 = 25 / 30\nh_sum : 27 / 30 + 25 / 30 = 52 / 30\nh_simplified : 52 / 30 = 26 / 15\na✝ : answer < 26 / 15\n⊢ False failed\n']}, GoalState(state_id=130, goals=[Goal(variables=[Variable(t='ℚ', v=None, name='answer'), Variable(t='ℤ', v=None, name='a'), Variable(t='ℤ', v=None, name='b'), Variable(t='9 / 10 + 5 / 6 = a / b', v=None, name='h₁'), Variable(t='a.natAbs.gcd b.natAbs = 1', v=None, name='h₂'), Variable(t='↑a / ↑b = answer', v=None, name='h_answer'), Variable(t='30 = Nat.lcm 10 6', v=None, name='h_common_denominator'), Variable(t='9 / 10 = 27 / 30', v=None, name='h_frac1'), Variable(t='5 / 6 = 25 / 30', v=None, name='h_frac2'), Variable(t='27 / 30 + 25 / 30 = 52 / 30', v=None, name='h_sum'), Variable(t='52 / 30 = 26 / 15', v=None, name='h_simplified')], target='False', sibling_dep=[], name=None, is_conversion=False)], payload={'nextStateId': 130, 'goals': [{'vars': [{'userName': 'answer', 'type': {'pp': 'ℚ', 'dependentMVars': []}, 'name': '_uniq.22784', 'isInaccessible': False}, {'userName': 'a', 'type': {'pp': 'ℤ', 'dependentMVars': []}, 'name': '_uniq.22790', 'isInaccessible': False}, {'userName': 'b', 'type': {'pp': 'ℤ', 'dependentMVars': []}, 'name': '_uniq.22796', 'isInaccessible': False}, {'userName': 'h₁', 'type': {'pp': '9 / 10 + 5 / 6 = a / b', 'dependentMVars': []}, 'name': '_uniq.23420', 'isInaccessible': False}, {'userName': 'h₂', 'type': {'pp': 'a.natAbs.gcd b.natAbs = 1', 'dependentMVars': []}, 'name': '_uniq.23460', 'isInaccessible': False}, {'userName': 'h_answer', 'type': {'pp': '↑a / ↑b = answer', 'dependentMVars': []}, 'name': '_uniq.23583', 'isInaccessible': False}, {'userName': 'h_common_denominator', 'type': {'pp': '30 = Nat.lcm 10 6', 'dependentMVars': []}, 'name': '_uniq.23630', 'isInaccessible': False}, {'userName': 'h_frac1', 'type': {'pp': '9 / 10 = 27 / 30', 'dependentMVars': []}, 'name': '_uniq.23816', 'isInaccessible': False}, {'userName': 'h_frac2', 'type': {'pp': '5 / 6 = 25 / 30', 'dependentMVars': []}, 'name': '_uniq.24108', 'isInaccessible': False}, {'userName': 'h_sum', 'type': {'pp': '27 / 30 + 25 / 30 = 52 / 30', 'dependentMVars': []}, 'name': '_uniq.24512', 'isInaccessible': False}, {'userName': 'h_simplified', 'type': {'pp': '52 / 30 = 26 / 15', 'dependentMVars': []}, 'name': '_uniq.24839', 'isInaccessible': False}], 'target': {'pp': 'False', 'dependentMVars': []}, 'name': '_uniq.24840', 'isConversion': False}]}, _sentinel=[2, 1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 3, 20, 21, 22, 23, 24, 25, 26, 17, 27, 28, 29, 30, 31, 32, 34, 5, 36, 37, 38, 39, 40, 41, 42, 43, 33, 44, 45, 46, 47, 48, 49, 50, 52, 19, 54, 55, 56, 57, 58, 59, 60, 61, 62, 51, 63, 64, 65, 66, 67, 68, 69, 70, 72, 35, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 71, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 53, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 93, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]), 'have h_answer : answer = 26 / 15 := by {\nlinarith [h₁, h₂, h_answer, h_common_denominator, h_frac1, h_frac2, h_sum, h_simplified]\n}'
```