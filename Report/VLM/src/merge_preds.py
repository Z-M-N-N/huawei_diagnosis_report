#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并多个 preds.json 为一个（按 id 去重），保持顺序。"""
import json, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

seen, out = set(), []
for f in args.inputs:
    for p in json.load(open(f, encoding="utf-8")):
        if p["id"] in seen:
            continue
        seen.add(p["id"]); out.append(p)
json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"合并 {len(args.inputs)} 个文件 -> {args.out}：{len(out)} 条（去重后）")
