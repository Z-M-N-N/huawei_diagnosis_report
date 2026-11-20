import os
import argparse
import json
import math
import hashlib
from pathlib import Path
import sys
import ast
import random
import warnings

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

project_root = Path("qwenvl/train")
sys.path.append(str(project_root))
from myqwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

print("=" * 50)
print("GPU 信息:")
print("=" * 50)
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"当前设备: {torch.cuda.current_device()} - {torch.cuda.get_device_name(0)}")
    print(f"可见GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
        print(f"GPU {i}: {props.name} | 总显存: {props.total_memory / 1024 ** 3:.2f}GB "
              f"| 已分配: {allocated:.2f}GB | 保留: {reserved:.2f}GB")
else:
    print("⚠️ CUDA不可用，使用CPU推理")


# ============ 工具函数 ============

def convert_to_native(obj):
    """递归地将 NumPy 类型转换为 Python 原生类型"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(x) for x in obj]
    return obj


def get_video_frames(video_path, num_frames=128, cache_dir=".cache"):
    os.makedirs(cache_dir, exist_ok=True)
    video_hash = hashlib.md5(video_path.encode("utf-8")).hexdigest()
    video_file_path = video_path

    frames_cache_file = os.path.join(cache_dir, f"{video_hash}_{num_frames}_frames.npy")
    ts_cache_file = os.path.join(cache_dir, f"{video_hash}_{num_frames}_timestamps.npy")

    if os.path.exists(frames_cache_file) and os.path.exists(ts_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(ts_cache_file)
        return video_file_path, frames, timestamps

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(ts_cache_file, timestamps)
    return video_file_path, frames, timestamps


def create_image_grid(images, num_columns=8):
    pil_images = [Image.fromarray(im) for im in images]
    num_rows = math.ceil(len(images) / num_columns)
    w, h = pil_images[0].size
    grid = Image.new("RGB", (num_columns * w, num_rows * h))
    for idx, im in enumerate(pil_images):
        r, c = idx // num_columns, idx % num_columns
        grid.paste(im, (c * w, r * h))
    return grid


def apply_lora(model, lora_path):
    """加载 LoRA 权重到模型"""
    return PeftModel.from_pretrained(model, lora_path)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============ 推理 ============

def inference(model, processor, prompt, num_frames, video=None):
    """执行多模态推理，返回 (output_text, {'是': p_yes, '否': p_no})"""
    try:
        if video is None:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
        else:
            messages = [
                {"role": "user", "content": [
                    {"type": "video", "video": video,
                     "nframes": num_frames,
                     "min_pixels": 224 * 224,
                     "max_pixels": 224 * 224},
                    {"type": "text", "text": prompt},
                ]}
            ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = processor(
            text=text, images=images, videos=videos,
            padding=True, return_tensors="pt", **video_kwargs,
        )
        inputs = inputs.to(model.device)   # 注意：赋值回去

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,           # greedy，保证概率稳定
            )

            first_token_logits = outputs.scores[0]      # [B, V]
            probs = torch.softmax(first_token_logits, dim=-1)

            tokenizer = processor.tokenizer
            yes_id = tokenizer.encode("是", add_special_tokens=False)[0]
            no_id = tokenizer.encode("否", add_special_tokens=False)[0]

            yes_prob = probs[0, yes_id].item()
            no_prob = probs[0, no_id].item()
            total = yes_prob + no_prob
            if total > 0:
                yes_prob, no_prob = yes_prob / total, no_prob / total
            else:
                yes_prob, no_prob = 0.0, 0.0

            generated_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, outputs.sequences)
            ]
            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return output_text[0], {"是": yes_prob, "否": no_prob}

    except Exception as e:
        print(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        return "", {"是": 0.0, "否": 0.0}


def cmp_ref_response(result):
    """对比 gpt 参考答案与模型输出"""
    ref = result["msg"]["conversations"][-1]["value"]
    res = result["response"]
    return 1 if ref == res else 0


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL 视频理解推理脚本")
    parser.add_argument("--model_path", type=str, required=True, help="基础模型路径")
    parser.add_argument("--processor_path", type=str, default="", help="处理器路径")
    parser.add_argument("--diagnostic_mode", type=str, default="",
                        choices=["", "pvd", "avd", "cmd"], help="诊断模式")

    parser.add_argument("--lora_path_PVD", type=str, default="", help="PVD LoRA 权重路径")
    parser.add_argument("--lora_path_AVD", type=str, default="", help="AVD LoRA 权重路径")
    parser.add_argument("--lora_path_CMD", type=str, default="", help="CMD LoRA 权重路径")
    parser.add_argument("--diag_item_config", type=str,
                        default="../DATA_BUILDER/diag_info.json",
                        help="诊断项配置文件路径")
    parser.add_argument("--device_map", type=str, default="auto",
                        help="设备映射策略 (auto/cuda/cpu)")
    parser.add_argument("--torch_dtype", type=str, default="auto",
                        choices=["auto", "float16", "float32", "bfloat16"],
                        help="torch 数据类型")
    parser.add_argument("--cuda_num", type=int, default=2, help="指定使用的 GPU 编号")
    parser.add_argument("--expandable_segments", type=bool, default=True,
                        help="是否启用 PyTorch 动态显存扩展")

    # === 数据相关 ===
    parser.add_argument("--test_json", type=str, required=True,
                        help="归并后的测试集 JSON 文件路径（例如 .../test.json）")
    parser.add_argument("--data_root", type=str, default="",
                        help="视频数据根目录（当 video 为相对路径时使用）")
    parser.add_argument("--diag_item", type=str, default="all",
                        help="诊断项（'all' 表示全部）")
    parser.add_argument("--sample_size", type=int, default=200,
                        help="每个诊断项随机采样的样本数量（<=0 或 >=总数 表示全部）")

    # === 推理相关 ===
    parser.add_argument("--num_frames", type=int, default=16, help="提取的视频帧数")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="缓存目录")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="最大生成 token 数")
    parser.add_argument("--total_pixels", type=int, default=20480 * 32 * 32,
                        help="视频帧最大像素数")
    parser.add_argument("--min_pixels", type=int, default=64 * 32 * 32,
                        help="视频帧最小像素数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # === 输出相关 ===
    parser.add_argument("--save_results", type=str, default=None,
                        help="保存结果的目录（每个诊断项一个 json）")
    parser.add_argument("--print_result", action="store_true",
                        help="打印每条推理结果")

    args = parser.parse_args()

    args.processor_path = args.processor_path or args.model_path

    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # ============ 随机种子 ============
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ============ 加载模型 ============
    print("正在加载模型...")
    processor = AutoProcessor.from_pretrained(args.processor_path)

    if args.torch_dtype == "float16":
        torch_dtype = torch.float16
    elif args.torch_dtype == "float32":
        torch_dtype = torch.float32
    elif args.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = "auto"

    model, output_loading_info = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        output_loading_info=True,
        low_cpu_mem_usage=True,
    )
    print("加载信息:", output_loading_info)

    def load_and_merge_lora(model, lora_path):
        if not lora_path:
            raise ValueError("LoRA 路径为空，无法加载")
        print(f"加载 LoRA: {lora_path}")
        model = apply_lora(model, lora_path)
        print(f"合并 LoRA: {lora_path}")
        model = model.merge_and_unload()
        return model

    mode = args.diagnostic_mode
    lora_paths = []
    if mode == "pvd":
        lora_paths = [args.lora_path_PVD]
    elif mode == "avd":
        lora_paths = [args.lora_path_PVD, args.lora_path_AVD]
    elif mode == "cmd":
        lora_paths = [args.lora_path_PVD, args.lora_path_AVD, args.lora_path_CMD]
    elif mode == "":
        print("诊断模式为空，不加载 LoRA")
    else:
        print(f"未知诊断模式: {mode}，不加载 LoRA")

    for i, lora_path in enumerate(lora_paths):
        print(f"[{i + 1}/{len(lora_paths)}] 处理 LoRA")
        model = load_and_merge_lora(model, lora_path)

    model.eval()
    print("模型加载完成!")
    if hasattr(model, "device"):
        print(f"模型设备: {model.device}")

    if args.save_results:
        os.makedirs(args.save_results, exist_ok=True)

    # ============ 加载归并测试集 ============
    print(f"\n正在加载测试集: {args.test_json}")
    all_test_data = load_json(args.test_json)
    print(f"测试集总样本数: {len(all_test_data)}")

    by_diag = {}
    for item in all_test_data:
        d = item.get("diag_item", "__unknown__")
        by_diag.setdefault(d, []).append(item)

    print(f"测试集包含 {len(by_diag)} 个诊断项:")
    for d, lst in by_diag.items():
        print(f"  - {d}: {len(lst)} 条")

    # ============ 诊断项列表 ============
    try:
        diag_items_cfg = list(load_json(args.diag_item_config).keys())
    except Exception as e:
        print(f"无法加载 diag_item_config: {e}，使用 test.json 中出现的诊断项")
        diag_items_cfg = list(by_diag.keys())

    if args.diag_item != "all":
        diag_items = [args.diag_item]
    else:
        diag_items = diag_items_cfg

    # ============ 推理循环 ============
    for diag_item in diag_items:
        if diag_item not in by_diag:
            print(f"\n测试集中无此诊断项，跳过: {diag_item}")
            continue

        # 跳过已完成
        if args.save_results and os.path.exists(
            os.path.join(args.save_results, f"{diag_item}.json")
        ):
            print(f"\n已存在结果文件，跳过诊断项: {diag_item}")
            # continue

        sft_json = by_diag[diag_item]
        print(f"\n{'=' * 60}")
        print(f"诊断项: {diag_item} | 样本数: {len(sft_json)}")
        print(f"{'=' * 60}")

        # 采样
        if 0 < args.sample_size < len(sft_json):
            sampled_data = random.sample(sft_json, k=args.sample_size)
        else:
            sampled_data = sft_json
        print(f"采样样本数: {len(sampled_data)}")

        cmp_list = []
        results = []

        pbar = tqdm(sampled_data, desc=f"[{diag_item}]", unit="video")
        for msg in pbar:
            # video 路径：兼容绝对 / 相对
            try:
                v0 = msg["video"][0]
                if os.path.isabs(v0):
                    video_url = v0
                else:
                    video_url = os.path.join(args.data_root, v0)
            except Exception:
                video_url = None

            prompt = msg["conversations"][0]["value"].replace("<video>", "").strip()

            try:
                response, prob_y_n = inference(
                    model, processor, prompt, args.num_frames, video_url
                )
            except Exception as e:
                print(f"\n错误处理视频 {video_url}: {e}")
                continue

            result = {
                "msg": msg,
                "response": response,
                "prob_y": prob_y_n["是"],
                "prob_n": prob_y_n["否"],
            }
            is_correct = cmp_ref_response(result)
            result["cmp"] = "✅" if is_correct == 1 else "❌"
            cmp_list.append(is_correct)
            results.append(result)

            if len(cmp_list) > 0:
                current_acc = sum(cmp_list) / len(cmp_list)
                pbar.set_postfix({
                    "Acc": f"{current_acc:.4f}",
                    "Correct": f"{sum(cmp_list)}/{len(cmp_list)}",
                })

            if len(results) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if args.print_result:
                print(json.dumps(convert_to_native(result), ensure_ascii=False, indent=2))

        # ============ 指标 ============
        y_true, y_probs, y_pred = [], [], []
        for item in results:
            y_true.append(1 if item["msg"]["conversations"][-1]["value"] == "是" else 0)
            y_probs.append(item["prob_y"])
            y_pred.append(1 if item["response"] == "是" else 0)

        if len(y_true) == 0:
            print(f"[{diag_item}] 无有效样本，跳过指标计算")
            continue

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_probs)
        except ValueError:
            auc = 0.0

        try:
            cm = confusion_matrix(y_true, y_pred).tolist()
        except Exception:
            cm = None

        metrics_results = {
            "diag_item": diag_item,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc,
                "n_samples": len(y_true),
                "confusion_matrix": cm,
            },
            "results": results,
        }

        print(f"\n[{diag_item}] 准确率: {accuracy:.4f} | 精确率: {precision:.4f} | "
              f"召回率: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        if args.save_results:
            os.makedirs(args.save_results, exist_ok=True)
            out_path = os.path.join(args.save_results, f"{diag_item}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(convert_to_native(metrics_results), f,
                          ensure_ascii=False, indent=2)
            print(f"已保存: {out_path}")

    print("\n全部完成。")


if __name__ == "__main__":
    main()