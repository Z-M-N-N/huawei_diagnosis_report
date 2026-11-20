#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按已评测的 id 生成"带输出模版"的评测子集（给未训练模型用）。"""
import json, argparse

BASE_TMPL = (
    "请查看视频，给出诊断发现报告。请严格按以下格式输出，不要输出其它多余内容：\n"
    "finding:<检查发现>\n"
    "diagnosticopinion:<诊断意见>\n"
    "measurements:{'主动脉根部内径': <数值>, '左房内径': <数值>, '左室舒张末内径': <数值>, "
    "'左室收缩末内径': <数值>, '室间隔厚度': <数值>, '左室后壁厚度': <数值>, '肺动脉收缩压': <数值>}"
)

ap = argparse.ArgumentParser()
ap.add_argument("--test_json", default="VLM/data/report/test.json")
ap.add_argument("--ids_from", required=True, help="含 id 的 preds.json（据此取同样的样本）")
ap.add_argument("--out", required=True)
args = ap.parse_args()

test = {d["id"]: d for d in json.load(open(args.test_json, encoding="utf-8"))}
ids = [p["id"] for p in json.load(open(args.ids_from, encoding="utf-8"))]

out = []
missing = 0
for i in ids:
    d = test.get(i)
    if d is None:
        missing += 1; continue
    nv = len(d["video"])
    out.append({
        "id": d["id"], "video": d["video"], "measurements": d.get("measurements"),
        "diag_item": d.get("diag_item", ""), "mark": d.get("mark", None),
        "conversations": [
            {"from": "human", "value": ("<video>\n" * nv) + BASE_TMPL},
            {"from": "gpt", "value": d["conversations"][1]["value"]},
        ],
    })
json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"生成 {len(out)} 条（缺失 {missing}）-> {args.out}")
