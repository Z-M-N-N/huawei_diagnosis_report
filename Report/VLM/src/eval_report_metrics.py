#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_report_metrics.py —— 由 preds.json 计算报告生成指标 + 输出耗时统计

指标:
  - 文本: ROUGE-1/2/L（字符级，整体 + finding + diagnosticopinion）、BLEU（jieba 分词）
  - 测量值: 解析成功率、7 项各自 MAE / 完全命中率
  - 耗时: 单条 min/median/mean/max/p95、超 300s 条数、总耗时及是否超 300s
输出: <dir>/metrics.json、<dir>/times.csv；并打印摘要
"""
import os, re, json, ast, argparse, statistics

import jieba
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

MEAS_KEYS = ["主动脉根部内径", "左房内径", "左室舒张末内径", "左室收缩末内径",
             "室间隔厚度", "左室后壁厚度", "肺动脉收缩压"]


class _CharTok:
    """字符级 tokenizer（中文按字切分），供 rouge_score 使用。"""
    def tokenize(self, text):
        return [c for c in text if not c.isspace()]


def parse_meas(text):
    m = re.search(r"measurements\s*[:：]\s*(\{.*?\})", text, re.S)
    if not m:
        return {}
    try:
        d = ast.literal_eval(m.group(1))
        if isinstance(d, dict):
            return {str(k).strip(): float(v) for k, v in d.items()
                    if isinstance(v, (int, float))}
    except Exception:
        pass
    return {}


def split_parts(t):
    finding = opinion = ""
    m = re.search(r"finding\s*[:：](.*?)(?=diagnosticopinion\s*[:：]|measurements\s*[:：]|$)", t, re.S)
    if m: finding = m.group(1).strip()
    m2 = re.search(r"diagnosticopinion\s*[:：](.*?)(?=measurements\s*[:：]|$)", t, re.S)
    if m2: opinion = m2.group(1).strip()
    return finding, opinion


def rouge_f1(refs, hyps):
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=_CharTok(), use_stemmer=False)
    acc = {"rouge1": [], "rouge2": [], "rougeL": []}
    for r, h in zip(refs, hyps):
        s = sc.score(r, h)
        for k in acc:
            acc[k].append(s[k].fmeasure)
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold", type=float, default=300.0)
    args = ap.parse_args()

    preds = json.load(open(args.preds, encoding="utf-8"))
    gts = [p["gt"] for p in preds]
    prd = [p["pred"] for p in preds]

    # ---------- 文本指标 ----------
    overall = rouge_f1(gts, prd)
    gt_f = [split_parts(g)[0] for g in gts]; pd_f = [split_parts(p)[0] for p in prd]
    gt_o = [split_parts(g)[1] for g in gts]; pd_o = [split_parts(p)[1] for p in prd]
    rouge_finding = rouge_f1(gt_f, pd_f)
    rouge_opinion = rouge_f1(gt_o, pd_o)

    sf = SmoothingFunction().method1
    bleu = corpus_bleu([[list(jieba.cut(g))] for g in gts],
                       [list(jieba.cut(p)) for p in prd], smoothing_function=sf)

    # ---------- 测量值指标 ----------
    gt_m = [parse_meas(g) for g in gts]
    pd_m = [parse_meas(p) for p in prd]
    parse_ok = sum(1 for m in pd_m if m)
    per_key = {}
    for k in MEAS_KEYS:
        diffs, hits, n = [], 0, 0
        for a, b in zip(gt_m, pd_m):
            if k in a and k in b:
                n += 1
                diffs.append(abs(a[k] - b[k]))
                if abs(a[k] - b[k]) < 1e-6:
                    hits += 1
        per_key[k] = {"n": n, "MAE": (sum(diffs) / n if n else None),
                      "exact": (hits / n if n else None)}
    all_diffs = [abs(a[k] - b[k]) for a, b in zip(gt_m, pd_m) for k in MEAS_KEYS if k in a and k in b]
    meas = {"parse_rate": parse_ok / len(preds) if preds else 0.0,
            "MAE_all": (sum(all_diffs) / len(all_diffs) if all_diffs else None),
            "n_parsed_pairs": len(all_diffs), "per_key": per_key}

    # ---------- 耗时 ----------
    times = [p.get("gen_seconds", 0.0) for p in preds]
    tot = sum(times)
    tstat = {
        "n": len(times),
        "min": min(times) if times else 0.0,
        "median": statistics.median(times) if times else 0.0,
        "mean": (tot / len(times)) if times else 0.0,
        "max": max(times) if times else 0.0,
        "p95": (sorted(times)[int(0.95 * (len(times) - 1))] if times else 0.0),
        "total_seconds": tot,
        "threshold": args.threshold,
        "n_over_threshold": sum(1 for t in times if t > args.threshold),
        "any_sample_over_threshold": any(t > args.threshold for t in times),
        "total_over_threshold": tot > args.threshold,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = {"n_samples": len(preds), "rouge_overall": overall,
               "rouge_finding": rouge_finding, "rouge_opinion": rouge_opinion,
               "bleu": bleu, "measurements": meas, "timing": tstat}
    json.dump(metrics, open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "times.csv"), "w", encoding="utf-8") as f:
        f.write("i,id,gen_seconds,over_300s\n")
        for i, p in enumerate(preds):
            t = p.get("gen_seconds", 0.0)
            f.write(f"{i},{p['id']},{t},{int(t > args.threshold)}\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
