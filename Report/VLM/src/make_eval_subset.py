#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 test.json 固定抽取评测子集（seed 可复现），写入 subset.json。"""
import json, argparse, random

ap = argparse.ArgumentParser()
ap.add_argument("--test_json", default="VLM/data/report/test.json")
ap.add_argument("--out", default="VLM/data/report/eval_subset.json")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

data = json.load(open(args.test_json, encoding="utf-8"))
random.seed(args.seed)
idx = sorted(random.sample(range(len(data)), min(args.n, len(data))))
sub = [data[i] for i in idx]
json.dump(sub, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"test 总数 {len(data)}，评测子集 {len(sub)} -> {args.out}  (seed={args.seed})")
