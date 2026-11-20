# import json
# import os
# import pandas as pd
# from pathlib import Path

# from sklearn.metrics import confusion_matrix

# # 读取文件夹中所有json文件
# folder_path = "/home/mzhao/ECHO_VIEW/Qwen3-VL/results/video_and_mv"  # 替换为你的文件夹路径 # base | only_video | video_and_mv | video_to_mv | multi_video | all_video
# results = []

# for file_path in Path(folder_path).glob("*.json"):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         if "metrics" in data:
#             metrics = data["metrics"]
#             results.append({
#                 "file": file_path.name.replace("inference_","").replace(".json",""),
#                 "accuracy": metrics.get("accuracy", None),
#                 "precision": metrics.get("precision", None),
#                 "recall": metrics.get("recall", None),
#                 "f1": metrics.get("f1", None),
#                 "auc": metrics.get("auc", None)
#             })
        
        
#         if "results" in data:
#             y_true=[]
#             y_pred=[]
#             recodes=data["results"]
#             for item in recodes:
#                 y_true.append(1 if item["msg"]["conversations"][1]["value"] == '是' else 0)
#                 y_pred.append(1 if item["response"]=="是" else 0)

#             if y_true:
#                 tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#                 acc = (tp+tn)/(tp+tn+fp+fn)
#                 prec = tp/(tp+fp) if (tp+fp) else 0
#                 rec = tp/(tp+fn) if (tp+fn) else 0
#                 f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
                
#                 # print(f"{file_path.name.replace('inference_','').replace('.json','')}: 样本{len(y_true)}, 准确率{acc:.3f}, F1{f1:.3f}")
#                 print(f"{file_path.name.replace('inference_','').replace('.json',''):<30} 样本:{len(y_true):>6}  准确率:{acc:.3f}  F1:{f1:.3f}")
# # 转换为DataFrame
# df = pd.DataFrame(results)

# # 打印表格
# print(df.to_string(index=False))

# # 保存为CSV
# df.to_csv(f"{folder_path}/metrics_summary.csv", index=False)

# # 保存为Excel
# df.to_excel(f"{folder_path}/metrics_summary.xlsx", index=False)

# # 计算平均值
# print("\n平均值:")
# print(df[["accuracy", "precision", "recall", "f1", "auc"]].mean())
import json
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix

def main():
    parser = argparse.ArgumentParser(description='分析JSON推理结果文件')
    parser.add_argument(
        '--folder', '-f',
        type=str,
        required=True,
        help='JSON文件所在文件夹路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出文件目录（默认与输入相同）'
    )
    parser.add_argument(
        '--save_csv',
        action='store_true',
        default=True,
        help='是否保存CSV（默认True）'
    )
    parser.add_argument(
        '--save_excel',
        action='store_true',
        default=True,
        help='是否保存Excel（默认True）'
    )
    
    args = parser.parse_args()
    
    folder_path = args.folder
    output_dir = args.output or folder_path
    
    print(f"📁 处理文件夹: {folder_path}")
    
    results = []
    
    for file_path in Path(folder_path).glob("*.json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            file_name = file_path.name.replace("inference_", "").replace(".json", "")
            
            # 如果直接有metrics
            if "metrics" in data:
                metrics = data["metrics"]
                results.append({
                    "file": file_name,
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "auc": metrics.get("auc")
                })
            
            # 如果有results，重新计算
            if "results" in data:
                y_true, y_pred = [], []
                for item in data["results"]:
                    y_true.append(1 if item["msg"]["conversations"][1]["value"] == '是' else 0)
                    y_pred.append(1 if item["response"] == "是" else 0)
                
                if y_true:
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                    acc = (tp+tn)/(tp+tn+fp+fn)
                    prec = tp/(tp+fp) if (tp+fp) else 0
                    rec = tp/(tp+fn) if (tp+fn) else 0
                    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
                    
                    print(f"  {file_name:<35} 样本:{len(y_true):>5}  准确率:{acc:.3f}  F1:{f1:.3f}")
                    
                    # 避免重复
                    existing = next((r for r in results if r["file"] == file_name), None)
                    if existing:
                        existing.update({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
                    else:
                        results.append({
                            "file": file_name,
                            "accuracy": acc,
                            "precision": prec,
                            "recall": rec,
                            "f1": f1,
                            "auc": None
                        })
    
    if not results:
        print("⚠️ 未找到任何JSON文件或数据")
        return
    
    # 去重保留最后一个
    df = pd.DataFrame(results).drop_duplicates(subset=['file'], keep='last')
    
    # 打印汇总
    print("\n" + "="*80)
    print("📊 汇总结果:")
    print(df.to_string(index=False))
    
    # 保存文件
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.save_csv:
        csv_path = Path(output_dir) / "metrics_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ CSV已保存: {csv_path}")
    
    if args.save_excel:
        excel_path = Path(output_dir) / "metrics_summary.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"✅ Excel已保存: {excel_path}")
    
    # 平均值
    print("\n📈 平均值:")
    numeric_cols = ["accuracy", "precision", "recall", "f1", "auc"]
    print(df[numeric_cols].mean().round(4))

if __name__ == "__main__":
    main()