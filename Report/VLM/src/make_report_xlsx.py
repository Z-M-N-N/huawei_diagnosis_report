#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""由 preds.json + metrics.json 生成 xlsx 评测报告（概述 / 样本明细 / 测量值指标）。"""
import os, json, argparse, datetime
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out_xlsx", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--num_frames", default="")
    args = ap.parse_args()

    preds = json.load(open(args.preds, encoding="utf-8"))
    M = json.load(open(args.metrics, encoding="utf-8"))
    t = M["timing"]; ro = M["rouge_overall"]; rf = M["rouge_finding"]; rp = M["rouge_opinion"]; ms = M["measurements"]

    wb = Workbook()
    bold = Font(bold=True)
    hdr_fill = PatternFill("solid", fgColor="DDEBF7")
    wrap = Alignment(wrap_text=True, vertical="top")
    thin = Border(*[Side(style="thin", color="BBBBBB")] * 4)
    thr = int(t["threshold"])

    # ---------- Sheet1: 概述 ----------
    ws = wb.active; ws.title = "概述"
    rows = [
        ("项目", "报告生成 · 评测报告"),
        ("标签", args.tag),
        ("LoRA checkpoint", args.checkpoint),
        ("生成时间", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("评测样本数", M["n_samples"]),
        ("每视频抽帧", args.num_frames),
        ("", ""),
        ("【指标】", ""),
        ("ROUGE-1 (整体)", round(ro["rouge1"], 4)),
        ("ROUGE-2 (整体)", round(ro["rouge2"], 4)),
        ("ROUGE-L (整体)", round(ro["rougeL"], 4)),
        ("ROUGE-L (finding)", round(rf["rougeL"], 4)),
        ("ROUGE-L (diagnosticopinion)", round(rp["rougeL"], 4)),
        ("BLEU (jieba)", round(M["bleu"], 4)),
        ("测量值解析成功率", round(ms["parse_rate"], 4)),
        ("测量值 MAE(总体)", None if ms["MAE_all"] is None else round(ms["MAE_all"], 3)),
        ("测量值配对数", ms["n_parsed_pairs"]),
        ("", ""),
        ("【输出耗时】", ""),
        ("单条 min (s)", round(t["min"], 2)),
        ("单条 median (s)", round(t["median"], 2)),
        ("单条 mean (s)", round(t["mean"], 2)),
        ("单条 max (s)", round(t["max"], 2)),
        ("单条 p95 (s)", round(t["p95"], 2)),
        ("总耗时 (s)", round(t["total_seconds"], 1)),
        ("阈值 (s)", thr),
        ("超阈值样本数", f"{t['n_over_threshold']} / {t['n']}"),
        (f"是否有单条超过 {thr}s", "是" if t["any_sample_over_threshold"] else "否"),
        (f"整体是否超过 {thr}s", "是" if t["total_over_threshold"] else "否"),
    ]
    for r_i, (k, v) in enumerate(rows, start=1):
        ws.cell(r_i, 1, k).font = bold
        ws.cell(r_i, 2, v)
        if k.startswith("【"):
            ws.cell(r_i, 1).fill = hdr_fill; ws.cell(r_i, 2).fill = hdr_fill
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 60

    # ---------- Sheet2: 样本明细 ----------
    ws2 = wb.create_sheet("样本明细")
    headers = ["序号", "检查ID", "视频数", "视频(切面)", "提示词", "参考测量值",
               "模型输出", "参考输出", "耗时(s)", f"是否超{thr}s"]
    for c, h in enumerate(headers, start=1):
        cell = ws2.cell(1, c, h); cell.font = bold; cell.fill = hdr_fill; cell.border = thin
    for i, p in enumerate(preds, start=1):
        gen = p.get("gen_seconds", 0.0)
        vals = [i, p["id"], len(p.get("video", [])),
                " | ".join(os.path.basename(v) for v in p.get("video", [])),
                p["prompt"], json.dumps(p.get("ref_measurements"), ensure_ascii=False),
                p["pred"], p["gt"], round(gen, 2), "是" if gen > thr else "否"]
        for c, v in enumerate(vals, start=1):
            cell = ws2.cell(i + 1, c, v); cell.alignment = wrap; cell.border = thin
    widths = [6, 42, 7, 34, 22, 34, 70, 70, 9, 10]
    for c, w in enumerate(widths, start=1):
        ws2.column_dimensions[get_column_letter(c)].width = w
    ws2.freeze_panes = "A2"

    # ---------- Sheet3: 测量值指标 ----------
    ws3 = wb.create_sheet("测量值指标")
    for c, h in enumerate(["测量项", "MAE", "完全命中率", "配对数"], start=1):
        cell = ws3.cell(1, c, h); cell.font = bold; cell.fill = hdr_fill; cell.border = thin
    for r, (k, v) in enumerate(ms["per_key"].items(), start=2):
        ws3.cell(r, 1, k).border = thin
        ws3.cell(r, 2, None if v["MAE"] is None else round(v["MAE"], 3)).border = thin
        ws3.cell(r, 3, None if v["exact"] is None else round(v["exact"], 4)).border = thin
        ws3.cell(r, 4, v["n"]).border = thin
    for c, w in enumerate([20, 12, 14, 10], start=1):
        ws3.column_dimensions[get_column_letter(c)].width = w

    wb.save(args.out_xlsx)
    print(f"xlsx: {args.out_xlsx}")


if __name__ == "__main__":
    main()
