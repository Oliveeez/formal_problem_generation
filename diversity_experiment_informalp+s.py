import json, random, re, os, glob, csv, hashlib
import os.path as osp
from typing import List, Tuple, Any, Dict, Optional
import multiprocessing as mp

from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np

_GLOBAL_TEXTS = None
_GLOBAL_METRIC = None
_GLOBAL_SCORER: rouge_scorer.RougeScorer = None

def _mp_init(metric: str, use_stemmer: bool, texts: list[str]):
    """Pool 初始化：在子进程里构造一次 RougeScorer，并存入全局。"""
    global _GLOBAL_TEXTS, _GLOBAL_METRIC, _GLOBAL_SCORER
    _GLOBAL_TEXTS = texts
    _GLOBAL_METRIC = metric
    _GLOBAL_SCORER = rouge_scorer.RougeScorer([metric], use_stemmer=use_stemmer)

def _score_pair_ij(pair: tuple[int, int]) -> float:
    """子进程：计算一对 (i, j) 的 ROUGE-L/ROUGE-Lsum F1"""
    i, j = pair
    s = _GLOBAL_SCORER.score(_GLOBAL_TEXTS[i], _GLOBAL_TEXTS[j])[_GLOBAL_METRIC]
    return s.fmeasure

def format_informalization_response(
    problem_type: str,
    informal_problem: str,
    informal_solution: str,
    informal_answer: Optional[str]
) -> str:
    # assert all(split not in field
    #     for split in [f'## Problem-Solving Question', '## Proof Question', '## Answer', '## Solution', '## Proof']
    #     for field in [problem_type, informal_problem, informal_solution, (informal_answer or '')]
    # )
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

def format_informal_code(d_orig: Dict) -> str:
    d = d_orig['informalization']
    return format_informalization_response(problem_type=d['problem_type'], informal_problem=(d.get('informal_problem') or d.get('informal_statement')), informal_solution=d.get('informal_solution') or d.get('informal_proof'), informal_answer=d.get('informal_answer')).strip()

def main(
    rounds: int = 10,
    sample_size: int = 400,
    use_stemmer: bool = True,       # 英文建议 True
    num_workers: int = 100,            # 0/1 = 单进程；>1 使用多进程
    chunksize: int = 400,
) -> None:
    data_root = '/sfs/liuqi/data/fpg_valid_fixed_evaluated/'

    print(f'''
rounds: int = {rounds},
sample_size: int = {sample_size},
use_stemmer: bool = {use_stemmer},
num_workers: int = {num_workers},
chunksize: int = {chunksize}
''')

    print('exp_name', 'mean', 'std', sep='\t')
    for exp_name in [
        'autoformalization_pg_kimina7b-PromptCoT-DS_kimina7b-valid_samples.jsonl',
        'autoformalization_pg_kimina7b-PromptCoT-QwQ_kimina7b-valid_samples.jsonl',
        'autoformalization_pg_kimina7b-ScaleQuest-Math_kimina7b-valid_samples.jsonl',
        'MUSTARDSauce_lean4_parsed-valid_samples.jsonl',
        'sft_ar_v3-Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch.0913.staged-valid_samples.jsonl',
        'sft_ar_v3-Goedel-Prover-V2-8B.Numina-Lean-reasseblmed.39509.problem_generator.nopack.3epoch-valid_samples.jsonl',
        'sft_wg_starified-Goedel-Prover-V2-8B.Numina-Lean.whole_statement_generatior.nopack-valid_samples.jsonl',
    ]:
        jsonl_path: str = osp.join(data_root, exp_name)
        # field="formal_statement"
        metric: str = "rougeL"          # "rougeL" 或 "rougeLsum"
        use_stemmer: bool = True       # 英文建议 True
        seed: int | None = 42
        ensure_unique_rounds: bool = False
        # mp_chunksize: int = 256          # map 的任务分块大小

        assert metric in ("rougeL", "rougeLsum")

        # 读取文本
        texts: List[str] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            data = [json.loads(l) for l in f.readlines()]
            texts = [format_informal_code(d) for d in data]

        n = len(texts)
        if n == 0:
            raise

        k = min(sample_size, n)
        rnd = random.Random(seed) if seed is not None else random.Random()
        # 单进程备用 scorer
        single_scorer = rouge_scorer.RougeScorer([metric], use_stemmer=use_stemmer)

        per_round: List[float] = []
        seen_rounds = set()

        for i_round in range(rounds):
            # print(f'Round {i_round}')
            idxs = rnd.sample(range(n), k)
            if ensure_unique_rounds:
                tries = 0
                key = tuple(sorted(idxs))
                while key in seen_rounds and tries < 5:
                    idxs = rnd.sample(range(n), k)
                    key = tuple(sorted(idxs)); tries += 1
                seen_rounds.add(key)

            # 🔁 本轮所有 (i, j)（i 在抽样集合，j 遍历全集且 j!=i）
            # ✅ 为了省内存，用生成器而不是把所有 pair 放进大列表
            pairs_iter = ((i, j) for i in idxs for j in idxs)

            if num_workers and num_workers > 1:
                # ✅ 多进程：每个进程里初始化一次 scorer & texts
                with mp.Pool(
                    processes=num_workers,
                    initializer=_mp_init,
                    initargs=(metric, use_stemmer, texts)
                ) as pool:
                    # 用 imap_unordered 拉流式取回结果，避免一次性 materialize
                    f1s_iter = pool.map(_score_pair_ij, pairs_iter, chunksize=chunksize)
                    pair_f1s = list(f1s_iter)
            else:
            # 单进程后备
                pair_f1s = []
                for (i, j) in pairs_iter:
                    s = single_scorer.score(texts[i], texts[j])[metric]
                    pair_f1s.append(s.fmeasure)

            per_round.append(sum(pair_f1s) / len(pair_f1s) if pair_f1s else float("nan"))
        # break

        # overall = sum(per_round)/len(per_round) if per_round else float("nan")
        print(exp_name, np.mean(per_round), np.std(per_round), sep='\t')

if __name__ == '__main__':
    import fire
    fire.Fire(main)
