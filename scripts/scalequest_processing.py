import os
import os.path as osp
import json
import pickle
import collections as C
import functools as F
import itertools as I
import multiprocessing as mp

import math_verify as M
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

def worker(args):
    i, d, load_path = args
    try:
        informal_answer = M.parse(d['response'])
        # assert len(informal_answer) == 2
        informal_answer = informal_answer[-1]
        assert isinstance(informal_answer, str)
        return {
            'informal_problem': d['query'],
            'informal_solution': d['response'],
            'informal_answer': informal_answer,
            'source_path': load_path,
            'source_idx': i
        }
    except Exception as e:
        return {
            'informal_problem': d['query'],
            'informal_solution': d['response'],
            'exception': str(repr(e)),
            'source_path': load_path,
            'source_idx': i
        }

if __name__ == '__main__':
    load_path = '/home/ma-user/local_cache/dyyyyyyyy/ScaleQuest-Math/train.json'
    with open(load_path, 'r') as f:
        data = [json.loads(l) for l in f.readlines()]
    print(len(data))

    args_list = [(i, d, load_path) for i, d in enumerate(data)]
    data_main_processed = process_map(worker, args_list, max_workers=16, chunksize=1000)
    
    print(len(data_main_processed))
    with open('/cache/data/fpg_informal_baselines/ScaleQuest-Math.processed.jsonl', 'w') as f:
        for d in data_main_processed:
            f.write(json.dumps(d)+'\n')
