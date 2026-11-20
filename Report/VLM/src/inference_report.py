#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_report.py —— 报告生成推理（支持固定评测子集 / LoRA checkpoint）

输出 preds.json，每条含: id, video, prompt, ref_measurements, gt, pred
用法:
  python VLM/src/inference_report.py \
      --model_path /home/mzhao/华为项目/VLM/Model/good_pvd_merged \
      --subset_json VLM/data/report/eval_subset.json \
      --lora_path VLM/lora/report/checkpoint-1500 \
      --save_results out/report/ckpt-1500 \
      --num_frames 16 --max_new_tokens 768
"""
import os, sys, json, argparse, time
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.append(str(Path(__file__).resolve().parent))
from myqwen_vl_utils import process_vision_info  # noqa


def build_messages(human_value, video_paths, num_frames):
    text = human_value
    while text.startswith("<video>"):
        text = text[len("<video>"):].lstrip("\n")
    content = []
    for vp in video_paths:
        content.append({"type": "video", "video": vp, "nframes": num_frames,
                        "min_pixels": 224 * 224, "max_pixels": 224 * 224})
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lora_path", default="")
    ap.add_argument("--subset_json", default="VLM/data/report/eval_subset.json")
    ap.add_argument("--test_json", default="")
    ap.add_argument("--save_results", required=True)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--torch_dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=999999)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(args.torch_dtype, torch.bfloat16)
    os.makedirs(args.save_results, exist_ok=True)

    src = args.subset_json or args.test_json
    data = json.load(open(src, encoding="utf-8"))
    if args.limit < len(data):
        data = data[:args.limit]
    print(f"评测样本: {len(data)}  (来源 {src})")

    print(f"加载模型: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype,
        attn_implementation="flash_attention_2", device_map="cuda:0")
    if args.lora_path:
        from peft import PeftModel
        print(f"加载 LoRA: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    preds = []
    t_run0 = time.perf_counter()
    for i, rec in enumerate(data):
        human = rec["conversations"][0]["value"]
        gt = rec["conversations"][1]["value"]
        messages = build_messages(human, rec["video"], args.num_frames)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(text=[text], images=images, videos=videos, padding=True,
                           return_tensors="pt", **video_kwargs).to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        dt = time.perf_counter() - t0
        preds.append({
            "id": rec["id"],
            "video": rec["video"],
            "prompt": human,
            "ref_measurements": rec.get("measurements"),
            "gt": gt,
            "pred": gen,
            "gen_seconds": round(dt, 3),
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(data)}]")

    total = time.perf_counter() - t_run0
    out_p = os.path.join(args.save_results, "preds.json")
    json.dump(preds, open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"预测保存: {out_p}")
    n_over = sum(1 for p in preds if p["gen_seconds"] > 300)
    print(f"[耗时] 总 {total:.1f}s；单条 max={max(p['gen_seconds'] for p in preds):.1f}s；"
          f"超300s条数={n_over}；整体{'超过' if total>300 else '未超过'}300s")


if __name__ == "__main__":
    main()
