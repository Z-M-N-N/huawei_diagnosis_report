import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_data_from_json(json_data):
    """从 JSON 中提取 prob_y 和真实标签（gpt 的 '是'/'否'）。"""
    probs, labels = [], []

    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data

    for item in data.get('results', []):
        prob_y = item.get('prob_y', 0)
        conversations = item.get('msg', {}).get('conversations', [])
        label_value = None
        for conv in conversations:
            if conv.get('from') == 'gpt':
                label_value = conv.get('value', '').strip()
                break

        if label_value == '是':
            label = 1
        elif label_value == '否':
            label = 0
        else:
            continue

        probs.append(prob_y)
        labels.append(label)

    return np.array(probs), np.array(labels)


def bootstrap_confidence_interval(probs, labels, threshold, n_bootstrap=2000, alpha=0.05, random_state=42):
    """百分位 Bootstrap 置信区间。"""
    data = list(zip(labels, probs))
    n = len(data)

    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    np.random.seed(random_state)

    for _ in range(n_bootstrap):
        boot = resample(data, n_samples=n, replace=True)
        yt = np.array([d[0] for d in boot])
        yp_proba = np.array([d[1] for d in boot])
        yp_class = (yp_proba > threshold).astype(int)

        if len(np.unique(yt)) < 2:
            continue

        try:
            metrics['accuracy'].append(accuracy_score(yt, yp_class))
            metrics['precision'].append(precision_score(yt, yp_class, zero_division=0))
            metrics['recall'].append(recall_score(yt, yp_class, zero_division=0))
            metrics['f1'].append(f1_score(yt, yp_class, zero_division=0))
            metrics['auc'].append(roc_auc_score(yt, yp_proba))
        except Exception:
            continue

    ci_results = {}
    for key, values in metrics.items():
        if len(values) == 0:
            ci_results[key] = {'mean': None, 'std': None, 'lower': None, 'upper': None, 'ci_95': None}
            continue
        mean_val = np.mean(values)
        std_val = np.std(values)
        lower = np.percentile(values, 100 * (alpha / 2))
        upper = np.percentile(values, 100 * (1 - alpha / 2))
        ci_results[key] = {
            'mean': mean_val,
            'std': std_val,
            'lower': lower,
            'upper': upper,
            'ci_95': f"[{lower:.4f}, {upper:.4f}]"
        }

    return ci_results


def find_optimal_threshold(probs, labels, metric='f1'):
    """在候选阈值上搜索使 metric 最大的阈值。"""
    unique_probs = np.unique(probs)
    thresholds = np.sort(np.concatenate([[0], unique_probs, [1]]))

    best_threshold = 0.5
    best_score = 0
    all_scores = []

    for thresh in thresholds:
        preds = (probs > thresh).astype(int)

        if metric == 'accuracy':
            score = accuracy_score(labels, preds)
        elif metric == 'precision':
            score = 0 if np.sum(preds) == 0 else precision_score(labels, preds, zero_division=0)
        elif metric == 'recall':
            score = recall_score(labels, preds, zero_division=0)
        elif metric == 'f1':
            score = 0 if (np.sum(preds) == 0 or np.sum(labels) == 0) else f1_score(labels, preds, zero_division=0)
        else:
            raise ValueError(f"不支持的指标: {metric}")

        all_scores.append(score)
        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score, thresholds, all_scores


def evaluate_model(probs, labels, threshold):
    """在给定阈值下评估。"""
    preds = (probs > threshold).astype(int)

    tp = int(np.sum((labels == 1) & (preds == 1)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fn = int(np.sum((labels == 1) & (preds == 0)))

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
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


def print_ci_report(ci_results, metric_names=None):
    """打印置信区间（精简，4 位小数）。"""
    if metric_names is None:
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']

    print("Bootstrap 95% CI:")
    for metric in metric_names:
        res = ci_results.get(metric, {})
        if res.get('mean') is not None:
            print(f"  {metric:<10} {res['mean']:.4f} ± {res['std']:.4f}  {res['ci_95']}")
        else:
            print(f"  {metric:<10} N/A")


def save_metrics_to_csv(results_dict, output_path=None):
    """保存指标到 CSV（含 CI 上下界，指标保留 4 位小数，计数保持整数）。"""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"evaluation_metrics_{timestamp}.csv"

    data = []
    for file_name, metrics in results_dict.items():
        row = {
            '文件名称': file_name.replace("inference_", ""),
            '最优阈值': metrics.get('best_threshold', 0.5),
            '准确率': metrics.get('accuracy', 0),
            '准确率_lower': metrics.get('ci', {}).get('accuracy', {}).get('lower', 0),
            '准确率_upper': metrics.get('ci', {}).get('accuracy', {}).get('upper', 0),
            '精确率': metrics.get('precision', 0),
            '精确率_lower': metrics.get('ci', {}).get('precision', {}).get('lower', 0),
            '精确率_upper': metrics.get('ci', {}).get('precision', {}).get('upper', 0),
            '召回率': metrics.get('recall', 0),
            '召回率_lower': metrics.get('ci', {}).get('recall', {}).get('lower', 0),
            '召回率_upper': metrics.get('ci', {}).get('recall', {}).get('upper', 0),
            'F1分数': metrics.get('f1', 0),
            'F1_lower': metrics.get('ci', {}).get('f1', {}).get('lower', 0),
            'F1_upper': metrics.get('ci', {}).get('f1', {}).get('upper', 0),
            'AUC': metrics.get('auc', 0),
            'AUC_lower': metrics.get('ci', {}).get('auc', {}).get('lower', 0),
            'AUC_upper': metrics.get('ci', {}).get('auc', {}).get('upper', 0),
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

    # 只对指标列保留 4 位小数，计数列保持整数
    metric_cols = [
        '最优阈值', '准确率', '准确率_lower', '准确率_upper',
        '精确率', '精确率_lower', '精确率_upper',
        '召回率', '召回率_lower', '召回率_upper',
        'F1分数', 'F1_lower', 'F1_upper',
        'AUC', 'AUC_lower', 'AUC_upper',
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    df[metric_cols] = df[metric_cols].round(4)

    # 添加平均值行
    if len(df) > 1:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary = {col: df[col].mean() for col in numeric_cols}
        summary['文件名称'] = '平均值'
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        df[metric_cols] = df[metric_cols].round(4)

    # 保存 CSV（utf-8-sig 让 Excel 打开不乱码）
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    return output_path


def plot_threshold_curve(thresholds, scores, metric, best_threshold, best_score, save_path=None):
    """绘制阈值-性能曲线（静默，不落盘）。"""
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, scores, 'b-', linewidth=2, label=f'{metric} score')
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'best={best_threshold:.4f}')
    plt.axhline(y=best_score, color='g', linestyle=':', label=f'best {metric}={best_score:.4f}')
    plt.xlabel('Threshold')
    plt.ylabel(f'{metric}')
    plt.title(f'Threshold vs {metric}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # if save_path:
    #     plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='分析 JSON 推理结果（含 Bootstrap CI）')
    parser.add_argument('--json_floder', '-dn', type=str, required=True,
                        help='JSON 文件所在文件夹')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出 CSV 路径（可选）')
    parser.add_argument('--bootstrap', '-b', type=int, default=200,
                        help='Bootstrap 重采样次数（默认 2000）')
    parser.add_argument('--no_bootstrap', action='store_true',
                        help='禁用 Bootstrap（加速）')
    args = parser.parse_args()

    all_results = {}

    for file_path in sorted(Path(args.json_floder).glob("*.json")):
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        json_data = {"results": json_data["results"]}

        probs, labels = load_data_from_json(json_data)
        if len(probs) == 0:
            print(f"跳过 {file_path.name}（无有效数据）")
            continue

        n_pos = int(np.sum(labels))
        n_neg = len(labels) - n_pos

        best_threshold, best_f1, thresholds, f1_scores = find_optimal_threshold(
            probs, labels, metric='f1'
        )
        results = evaluate_model(probs, labels, best_threshold)

        print(f"\n[{file_path.stem}]  n={len(probs)} (正{n_pos}/负{n_neg})  "
              f"T={best_threshold:.4f}  "
              f"Acc={results['accuracy']:.4f}  P={results['precision']:.4f}  "
              f"R={results['recall']:.4f}  F1={results['f1']:.4f}  AUC={results['auc']:.4f}")

        if not args.no_bootstrap:
            ci_results = bootstrap_confidence_interval(
                probs, labels, best_threshold, n_bootstrap=args.bootstrap
            )
            print_ci_report(ci_results)
            results['ci'] = ci_results
        else:
            results['ci'] = {}

        all_results[file_path.stem] = {
            'file_name': file_path.stem,
            'best_threshold': best_threshold,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'auc': results['auc'],
            'total_samples': len(probs),
            'positive_samples': n_pos,
            'negative_samples': n_neg,
            'confusion_matrix': results['confusion_matrix'],
            'ci': results.get('ci', {})
        }

        if len(thresholds) > 1:
            plot_threshold_curve(thresholds, f1_scores, 'F1', best_threshold, best_f1)

    if all_results:
        output_path = save_metrics_to_csv(all_results, args.output)

        print("\n汇总:")
        df = pd.DataFrame([{
            '文件': v['file_name'],
            '阈值': f"{v['best_threshold']:.4f}",
            'Acc': f"{v['accuracy']:.4f}",
            'P': f"{v['precision']:.4f}",
            'R': f"{v['recall']:.4f}",
            'F1': f"{v['f1']:.4f}",
            'AUC': f"{v['auc']:.4f}",
            'F1_CI': f"[{v.get('ci', {}).get('f1', {}).get('lower', 0):.4f}-{v.get('ci', {}).get('f1', {}).get('upper', 0):.4f}]",
        } for v in all_results.values()])

        # 追加终端平均值行
        if len(df) > 1:
            mean_threshold = np.mean([v['best_threshold'] for v in all_results.values()])
            mean_acc = np.mean([v['accuracy'] for v in all_results.values()])
            mean_p = np.mean([v['precision'] for v in all_results.values()])
            mean_r = np.mean([v['recall'] for v in all_results.values()])
            mean_f1 = np.mean([v['f1'] for v in all_results.values()])
            mean_auc = np.mean([v['auc'] for v in all_results.values()])

            mean_row = {
                '文件': '平均值',
                '阈值': f"{mean_threshold:.4f}",
                'Acc': f"{mean_acc:.4f}",
                'P': f"{mean_p:.4f}",
                'R': f"{mean_r:.4f}",
                'F1': f"{mean_f1:.4f}",
                'AUC': f"{mean_auc:.4f}",
                'F1_CI': '',
            }
            df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

        print(df.to_string(index=False))
        print(f"\n完成 -> {output_path}")
    else:
        print("\n没有有效的 JSON 数据")


if __name__ == "__main__":
    main()