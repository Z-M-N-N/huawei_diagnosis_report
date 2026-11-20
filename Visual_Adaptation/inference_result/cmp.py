import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def load_data_from_json(json_data):
    """
    从JSON数据中提取prob_y和真实标签
    标签从 conversations 中 from="gpt" 的 value 字段提取
    """
    probs = []
    labels = []
    
    # 如果传入的是字符串，先解析为dict
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # 提取results中的每个样本
    for item in data.get('results', []):
        prob_y = item.get('prob_y', 0)
        
        # 从 conversations 中提取 gpt 的回复
        conversations = item.get('msg', {}).get('conversations', [])
        label_value = None
        
        for conv in conversations:
            if conv.get('from') == 'gpt':
                label_value = conv.get('value', '').strip()
                break
        
        # 将 '是' 转为 1，'否' 转为 0
        if label_value == '是':
            label = 1
        elif label_value == '否':
            label = 0
        else:
            # 如果没找到标签，跳过这个样本
            print(f"⚠️  警告: 样本 {item.get('msg', {}).get('id', 'unknown')} 未找到有效标签")
            continue
        
        probs.append(prob_y)
        labels.append(label)
    
    return np.array(probs), np.array(labels)


def find_optimal_threshold(probs, labels, metric='f1'):
    """
    寻找最优阈值
    
    参数:
        probs: 预测概率数组 (prob_y)
        labels: 真实标签数组 (1表示"是"，0表示"否")
        metric: 优化指标，可选 'accuracy', 'precision', 'recall', 'f1'
    
    返回:
        best_threshold: 最优阈值
        best_score: 最优分数
        all_thresholds: 所有候选阈值
        all_scores: 对应的分数
    """
    # 生成候选阈值：从所有唯一的概率值中选取
    unique_probs = np.unique(probs)
    
    # 添加边界值 0 和 1
    thresholds = np.sort(np.concatenate([[0], unique_probs, [1]]))
    
    best_threshold = 0.5
    best_score = 0
    
    all_scores = []
    
    for thresh in thresholds:
        # 预测：prob_y > thresh 为阳性(1)
        preds = (probs > thresh).astype(int)
        
        # 计算对应的指标
        if metric == 'accuracy':
            score = accuracy_score(labels, preds)
        elif metric == 'precision':
            # 处理分母为0的情况
            if np.sum(preds) == 0:
                score = 0
            else:
                score = precision_score(labels, preds, zero_division=0)
        elif metric == 'recall':
            score = recall_score(labels, preds, zero_division=0)
        elif metric == 'f1':
            if np.sum(preds) == 0 or np.sum(labels) == 0:
                score = 0
            else:
                score = f1_score(labels, preds, zero_division=0)
        else:
            raise ValueError(f"不支持的指标: {metric}")
        
        all_scores.append(score)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score, thresholds, all_scores


def evaluate_model(probs, labels, threshold):
    """
    在给定阈值下评估模型性能
    """
    preds = (probs > threshold).astype(int)
    
    # 混淆矩阵
    tp = np.sum((labels == 1) & (preds == 1))
    fp = np.sum((labels == 0) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fn = np.sum((labels == 1) & (preds == 0))
    
    # 各项指标
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    # AUC
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0
    
    return {
        'threshold': threshold,
        'confusion_matrix': {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn},
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': preds.tolist()
    }

def save_metrics_to_excel(results_dict, output_path=None):
    """
    将多个文件的评估指标保存到Excel文件（简化版）
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"evaluation_metrics_{timestamp}.xlsx"
    
    # 创建DataFrame
    data = []
    for file_name, metrics in results_dict.items():
        row = {
            '文件名称': file_name.replace("inference_",""),
            '最优阈值': metrics.get('best_threshold', 0.5),
            '准确率': metrics.get('accuracy', 0),
            '精确率': metrics.get('precision', 0),
            '召回率': metrics.get('recall', 0),
            'F1分数': metrics.get('f1', 0),
            'AUC': metrics.get('auc', 0),
            '总样本数': metrics.get('total_samples', 0),
            '阳性样本数': metrics.get('positive_samples', 0),
            '阴性样本数': metrics.get('negative_samples', 0),
            'TP': metrics.get('confusion_matrix', {}).get('TP', 0),
            'FP': metrics.get('confusion_matrix', {}).get('FP', 0),
            'TN': metrics.get('confusion_matrix', {}).get('TN', 0),
            'FN': metrics.get('confusion_matrix', {}).get('FN', 0),
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 添加汇总行
    if len(df) > 1:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary = {col: df[col].mean() for col in numeric_cols}
        summary['文件名称'] = '平均值'
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    
    # 直接保存，不调整列宽
    df.to_excel(output_path, sheet_name='评估指标', index=False)
    
    print(f"\n✅ 评估指标已保存到: {output_path}")
    return output_path

def plot_threshold_curve(thresholds, scores, metric, best_threshold, best_score, save_path=None):
    """
    绘制阈值-性能曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, scores, 'b-', linewidth=2, label=f'{metric} score')
    plt.axvline(x=best_threshold, color='r', linestyle='--', 
                label=f'最优阈值 = {best_threshold:.6f}')
    plt.axhline(y=best_score, color='g', linestyle=':', 
                label=f'最优{metric} = {best_score:.4f}')
    plt.xlabel('阈值 (Threshold)')
    plt.ylabel(f'{metric} 分数')
    plt.title(f'阈值 vs {metric} 曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"📊 曲线图已保存为: {save_path}")
    else:
        plt.savefig('threshold_analysis.png', dpi=300)
        print(f"📊 曲线图已保存为: threshold_analysis.png")
    
    plt.show()


def print_sample_details(probs, labels, threshold, json_data):
    """
    打印每个样本的详细信息
    """
    print("=" * 80)
    print("📝 每个样本的预测详情")
    print("=" * 80)
    print(f"{'样本':<6} {'真实标签':<10} {'prob_y':<15} {'预测结果':<10} {'匹配'}")
    print("-" * 80)
    
    preds = (probs > threshold).astype(int)
    
    # 重新获取数据以显示更多信息
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    for i, (prob, label, pred) in enumerate(zip(probs, labels, preds)):
        label_str = '是' if label == 1 else '否'
        pred_str = '是' if pred == 1 else '否'
        match = '✅' if label == pred else '❌'
        
        # 获取样本ID
        item = data.get('results', [])[i] if i < len(data.get('results', [])) else {}
        sample_id = item.get('msg', {}).get('id', 'unknown')
        
        print(f"{i+1:<6} {label_str:<10} {prob:<15.10f} {pred_str:<10} {match}  (ID: {sample_id})")
    
    print()


def main():
    """
    主函数 - 从JSON文件读取数据并分析
    """
    
    parser = argparse.ArgumentParser(description='分析JSON推理结果文件')
    parser.add_argument(
        '--json_floder', '-dn',
        type=str,
        required=True,
        help='JSON文件所在的文件夹路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出Excel文件路径（可选）'
    )

    args = parser.parse_args()

    # 存储所有文件的评估结果
    all_results = {}
    
    # 遍历文件夹中的所有JSON文件
    for file_path in Path(args.json_floder).glob("*.json"):
        print("\n" + "=" * 80)
        print(f"📁 处理文件: {file_path.name}")
        print("=" * 80)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        json_data = {"results": json_data["results"]}
        
        # 提取数据
        probs, labels = load_data_from_json(json_data)
        
        if len(probs) == 0:
            print(f"⚠️  文件 {file_path.name} 没有有效数据，跳过")
            continue
        
        print("=" * 80)
        print("📊 数据概况")
        print("=" * 80)
        print(f"总样本数: {len(probs)}")
        print(f"阳性样本 (是): {np.sum(labels)}")
        print(f"阴性样本 (否): {len(labels) - np.sum(labels)}")
        print(f"prob_y 范围: [{np.min(probs):.10f}, {np.max(probs):.10f}]")
        print(f"prob_y 均值: {np.mean(probs):.10f}")
        print()
        
        # 检查数据是否平衡
        print("📌 标签分布:")
        print(f"   '是' (阳性): {np.sum(labels)} 个")
        print(f"   '否' (阴性): {len(labels) - np.sum(labels)} 个")
        print()
        
        # ============ 寻找最优阈值 (按F1优化) ============
        print("=" * 80)
        print("🔍 寻找最优阈值 (优化指标: F1)")
        print("=" * 80)
        
        best_threshold, best_f1, thresholds, f1_scores = find_optimal_threshold(
            probs, labels, metric='f1'
        )
        
        # 使用最优阈值评估
        print(f"🏆 最优阈值 (F1最大化): {best_threshold:.10f}")
        print(f"   对应的F1分数: {best_f1:.6f}")
        print()
        
        # 使用最优阈值进行评估
        print("=" * 80)
        print(f"📋 使用最优阈值 (T={best_threshold:.6f}) 评估")
        print("=" * 80)
        
        results = evaluate_model(probs, labels, best_threshold)
        
        print(f"混淆矩阵:")
        print(f"   ┌─────────────┐")
        print(f"   │  TP: {results['confusion_matrix']['TP']:<2}   FP: {results['confusion_matrix']['FP']:<2} │")
        print(f"   │  FN: {results['confusion_matrix']['FN']:<2}   TN: {results['confusion_matrix']['TN']:<2} │")
        print(f"   └─────────────┘")
        print()
        print(f"📊 性能指标:")
        print(f"   ├─ 准确率 (Accuracy):  {results['accuracy']:.6f}  ({results['accuracy']*100:.2f}%)")
        print(f"   ├─ 精确率 (Precision): {results['precision']:.6f}  ({results['precision']*100:.2f}%)")
        print(f"   ├─ 召回率 (Recall):    {results['recall']:.6f}  ({results['recall']*100:.2f}%)")
        print(f"   ├─ F1分数:             {results['f1']:.6f}  ({results['f1']*100:.2f}%)")
        print(f"   └─ AUC:                {results['auc']:.6f}  ({results['auc']*100:.2f}%)")
        print()
        
        # 存储结果
        all_results[file_path.stem] = {
            'file_name': file_path.stem,
            'best_threshold': best_threshold,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'auc': results['auc'],
            'total_samples': len(probs),
            'positive_samples': np.sum(labels),
            'negative_samples': len(labels) - np.sum(labels),
            'confusion_matrix': results['confusion_matrix']
        }
    
    # ============ 保存所有结果到Excel ============
    if all_results:
        print("\n" + "=" * 80)
        print("💾 保存评估结果")
        print("=" * 80)
        
        output_path = save_metrics_to_excel(all_results, args.output)
        
        # 打印汇总表
        print("\n📊 汇总表:")
        print("=" * 80)
        df = pd.DataFrame([{
            '文件': v['file_name'],
            '阈值': v['best_threshold'],
            '准确率': v['accuracy'],
            '精确率': v['precision'],
            '召回率': v['recall'],
            'F1': v['f1'],
            'AUC': v['auc']
        } for v in all_results.values()])
        print(df.to_string(index=False))
        
        # 可选：为每个文件绘制曲线（如果样本数足够）
        # 如果需要，可以取消注释下面的代码
        # for file_path in Path(args.json_floder).glob("*.json"):
        #     with open(file_path, 'r', encoding='utf-8') as f:
        #         json_data = json.load(f)
        #     json_data = {"results": json_data["results"]}
        #     probs, labels = load_data_from_json(json_data)
        #     if len(probs) > 1:
        #         best_th, _, thresholds, f1_scores = find_optimal_threshold(probs, labels, metric='f1')
        #         plot_threshold_curve(
        #             thresholds, f1_scores, 'F1', 
        #             best_th, max(f1_scores),
        #             save_path=f"threshold_curve_{file_path.stem}.png"
        #         )
        
        print("\n✅ 处理完成！")
    else:
        print("\n⚠️  没有找到有效的JSON文件或数据")


if __name__ == "__main__":
    main()