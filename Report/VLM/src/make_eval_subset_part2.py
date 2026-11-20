#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 test.json 抽取"第二批评测子集"：排除已用过的 id，再取 n 条（seed 固定）。"""
import json, argparse, random

ap = argparse.ArgumentParser()
ap.add_argument("--test_json", default="VLM/data/report/test.json")
ap.add_argument("--exclude_preds", default="out/report/ckpt-3500/preds.json",
                help="已完成推理的 preds.json（其中的 id 会被排除）")
ap.add_argument("--out", default="VLM/data/report/eval_subset_200.json")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--seed", type=int, default=2026)
args = ap.parse_args()

data = json.load(open(args.test_json, encoding="utf-8"))
used = set()
try:
    used = {p["id"] for p in json.load(open(args.exclude_preds, encoding="utf-8"))}
    print(f"排除已用: {len(used)} 条")
except Exception as e:
    print(f"[提示] 未读取到排除文件({e})，将不排除")

cand = [d for d in data if d["id"] not in used]
random.seed(args.seed)
idx = sorted(random.sample(range(len(cand)), min(args.n, len(cand))))
sub = [cand[i] for i in idx]
json.dump(sub, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
# 自检
assert not (set(x["id"] for x in sub) & used), "存在重复 id!"
print(f"可用 {len(cand)}，抽取 {len(sub)} -> {args.out} (seed={args.seed}, 与已用无重叠)")
