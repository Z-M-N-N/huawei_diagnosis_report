#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""由 preds.json + metrics.json 生成 markdown 评测报告（含输入/输出/指标/耗时）。"""
import os, json, ast, argparse, datetime


def fmt(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--num_frames", default="")
    ap.add_argument("--n_show", type=int, default=5)
    args = ap.parse_args()

    preds = json.load(open(args.preds, encoding="utf-8"))
    M = json.load(open(args.metrics, encoding="utf-8"))
    t = M["timing"]; ro = M["rouge_overall"]; rf = M["rouge_finding"]; rp = M["rouge_opinion"]; ms = M["measurements"]

    L = []
    L.append(f"# 报告生成评测报告 — {args.tag}")
    L.append("")
    L.append(f"- 生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- LoRA checkpoint: `{args.checkpoint or args.tag}`")
    L.append(f"- 评测样本数: {M['n_samples']}（固定子集 eval_subset.json, seed=42）")
    L.append(f"- 每视频抽帧: {args.num_frames}")
    L.append("")
    L.append("## 一、评价指标")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("|---|---|")
    L.append(f"| ROUGE-1 (整体) | {fmt(ro['rouge1'])} |")
    L.append(f"| ROUGE-2 (整体) | {fmt(ro['rouge2'])} |")
    L.append(f"| ROUGE-L (整体) | {fmt(ro['rougeL'])} |")
    L.append(f"| ROUGE-L (finding) | {fmt(rf['rougeL'])} |")
    L.append(f"| ROUGE-L (diagnosticopinion) | {fmt(rp['rougeL'])} |")
    L.append(f"| BLEU (jieba) | {fmt(M['bleu'])} |")
    L.append(f"| 测量值解析成功率 | {fmt(ms['parse_rate'])} |")
    L.append(f"| 测量值 MAE(总体, 配对 {ms['n_parsed_pairs']}) | {fmt(ms['MAE_all'],3)} |")
    L.append("")
    L.append("### 各项测量值")
    L.append("")
    L.append("| 测量项 | MAE | 完全命中率 | 配对数 |")
    L.append("|---|---|---|---|")
    for k, v in ms["per_key"].items():
        L.append(f"| {k} | {fmt(v['MAE'],3)} | {fmt(v['exact'])} | {v['n']} |")
    L.append("")
    L.append("## 二、输出耗时")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append(f"| 单条 min | {fmt(t['min'],2)} s |")
    L.append(f"| 单条 median | {fmt(t['median'],2)} s |")
    L.append(f"| 单条 mean | {fmt(t['mean'],2)} s |")
    L.append(f"| 单条 max | {fmt(t['max'],2)} s |")
    L.append(f"| 单条 p95 | {fmt(t['p95'],2)} s |")
    L.append(f"| 总耗时 | {fmt(t['total_seconds'],1)} s |")
    L.append(f"| 阈值 | {int(t['threshold'])} s |")
    L.append(f"| 超阈值样本数 | {t['n_over_threshold']} / {t['n']} |")
    L.append(f"| **是否有单条输出超过 {int(t['threshold'])}s** | **{'是' if t['any_sample_over_threshold'] else '否'}** |")
    L.append(f"| **整体评测是否超过 {int(t['threshold'])}s** | **{'是' if t['total_over_threshold'] else '否'}** |")
    L.append("")
    L.append(f"## 三、样例（前 {args.n_show} 条：输入 / 模型输出 / 参考）")
    L.append("")
    for i, p in enumerate(preds[:args.n_show]):
        nv = len(p.get("video", []))
        L.append(f"### 样例 {i+1} — `{p['id'].split('/')[-1]}`（{nv} 段视频，耗时 {fmt(p.get('gen_seconds'),1)}s）")
        L.append("")
        L.append("**输入信息**")
        L.append("```")
        L.append(f"视频: " + " | ".join(os.path.basename(v) for v in p.get("video", [])))
        L.append(f"测量值(参考): {p.get('ref_measurements')}")
        L.append(f"提示词: {p['prompt'].replace('<video>','<video> ')}")
        L.append("```")
        L.append("**模型输出**")
        L.append("```")
        L.append(p["pred"])
        L.append("```")
        L.append("**参考输出**")
        L.append("```")
        L.append(p["gt"])
        L.append("```")
        L.append("")

    open(args.out_md, "w", encoding="utf-8").write("\n".join(L))
    print(f"报告: {args.out_md}")


if __name__ == "__main__":
    main()
