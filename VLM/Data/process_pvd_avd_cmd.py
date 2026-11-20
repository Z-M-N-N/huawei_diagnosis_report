import os
import json
import argparse
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# 所有超声切面列表
ALL_VIEWS = [
    "A4C",
    "Parasternal Long",
    "Apical Doppler",
    "Parasternal Short",
    "A3C",
    "A2C",
    "A5C",
    "Subcostal",
    "Doppler Parasternal Long",
    "Doppler Parasternal Short",
    "SSN"
]


def load_json(file_path):
    """读取 JSON 文件并返回解析后的对象。"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


global_diag_info = load_json("Data/diag_info.json")


def save_json(data, file_path):
    """流式写入 JSON：逐条 dump，避免一次性大对象；带进度条；原子替换。"""
    tmp_path = file_path + ".tmp"
    n = len(data)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, item in enumerate(tqdm(data, desc=f"写入 {os.path.basename(file_path)}", unit="条")):
            json.dump(item, f, ensure_ascii=False, separators=(",", ":"))
            f.write(",\n" if i < n - 1 else "\n")
        f.write("]\n")
    os.replace(tmp_path, file_path)


def get_all_subfolders(root_dir, exclude_pattern="Train_Test"):
    """递归获取所有子文件夹，并过滤掉包含排除模式的文件夹"""
    subfolders = []
    try:
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path):
                if exclude_pattern not in item:
                    subfolders.append(item_path)
                    subfolders.extend(get_all_subfolders(item_path, exclude_pattern))
                else:
                    print(f"跳过: {item_path}")
    except PermissionError:
        print(f"权限不足，无法访问: {root_dir}")
    return subfolders


def get_subfolders_with_cache(root, cache_file, refresh=False):
    """获取所有包含 report.json 的子文件夹路径列表；复用缓存避免每次全量扫描。"""
    root = os.path.abspath(root)
    if not refresh and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
            folders = cache.get("folders")
            if isinstance(folders, list) and cache.get("root") == root:
                print(f"[缓存命中] 从 {cache_file} 读取 {len(folders)} 个文件夹（跳过扫描）")
                return folders
            print("缓存与当前数据目录不匹配，重新扫描...")
        except Exception as e:
            print(f"缓存读取失败({e})，重新扫描...")

    print("[扫描] 遍历数据目录，首次较慢（约10~20秒）...")
    all_subfolders = get_all_subfolders(root)
    test_folders = [f for f in all_subfolders
                    if os.path.exists(os.path.join(f, "report.json"))]

    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"root": root, "count": len(test_folders), "folders": test_folders},
                      f, ensure_ascii=False)
        print(f"[缓存已写入] {len(test_folders)} 个文件夹 -> {cache_file}")
    except Exception as e:
        print(f"写入缓存失败({e})，本次运行不缓存")
    return test_folders


# ----------------------------------------------------------------------
# 三种处理模式
# ----------------------------------------------------------------------

def deal_pvd(folder, diag_info):
    """处理单个文件夹（仅使用主要切面）"""
    if not isinstance(folder, str):
        print(f"错误: folder 类型为 {type(folder)}，期望 str")
        return None

    report_json_path = os.path.join(folder, "report.json")
    if not os.path.exists(report_json_path):
        return None

    try:
        data = load_json(report_json_path)
        labels = data.get("labels")
        mv = data.get("measurements")
        if labels is None:
            return None
    except Exception:
        return None

    diag_list = list(diag_info.keys())
    diag_convs = {}
    for diag_item in diag_list:
        if diag_item not in labels:
            continue

        primary_view = diag_info[diag_item]["primary_views"][0]
        prompt = diag_info[diag_item]["prompt"]
        # prompt = diag_info[diag_item]["prompt"].replace("请根据超声影像",f"请根据测量值和{mv}以及超声影像")
        value = labels[diag_item]["value"]
        mark = labels[diag_item]["mark"]

        if mark == 0:
            continue

        video_path = os.path.abspath(
            os.path.join(folder, f"{primary_view.replace(' ', '_')}.mp4")
        )
        if not os.path.exists(video_path):
            continue

        conv = {
            "id": folder,
            "video": [str(video_path)],
            "diag_item": diag_item,
            "mark": mark,
            "conversations": [
                {"from": "human", "value": "<video>\n" + prompt},
                {"from": "gpt", "value": "是" if value == 1 else "否"},
            ],
        }
        diag_convs[diag_item] = conv
    return diag_convs if diag_convs else None


def deal_avd(folder, diag_info):
    """处理单个文件夹（主要+辅助切面）"""
    if not isinstance(folder, str):
        print(f"错误: folder 类型为 {type(folder)}，期望 str")
        return None

    report_json_path = os.path.join(folder, "report.json")
    if not os.path.exists(report_json_path):
        return None

    try:
        data = load_json(report_json_path)
        labels = data.get("labels")
        mv = data.get("measurements")
        if labels is None:
            return None
    except Exception:
        return None

    diag_list = list(diag_info.keys())
    diag_convs = {}
    for diag_item in diag_list:
        if diag_item not in labels:
            continue

        prompt = diag_info[diag_item]["prompt"].replace(
            "请查看视频", f"请查看视频,并结合测量值数据：{mv}"
        )
        value = labels[diag_item]["value"]
        mark = labels[diag_item]["mark"]
        if mark == 0:
            continue

        views = (diag_info[diag_item]["primary_views"]
                 + diag_info[diag_item]["supplementary_views"])
        video_paths = []
        for view in views:
            video_path = os.path.abspath(
                os.path.join(folder, f"{view.replace(' ', '_')}.mp4")
            )
            if os.path.exists(video_path):
                video_paths.append(str(video_path))

        if not video_paths:
            continue

        conv = {
            "id": folder,
            "video": video_paths,
            "diag_item": diag_item,
            "mark": mark,
            "conversations": [
                {"from": "human",
                 "value": "<video>\n" * len(video_paths) + prompt},
                {"from": "gpt", "value": "是" if value == 1 else "否"},
            ],
        }
        diag_convs[diag_item] = conv
    return diag_convs if diag_convs else None


def deal_cmd(folder, diag_info):
    """处理单个文件夹（所有切面）"""
    if not isinstance(folder, str):
        print(f"错误: folder 类型为 {type(folder)}，期望 str")
        return None

    report_json_path = os.path.join(folder, "report.json")
    if not os.path.exists(report_json_path):
        return None

    try:
        data = load_json(report_json_path)
        labels = data.get("labels")
        mv = data.get("measurements")
        if labels is None:
            return None
    except Exception:
        return None

    diag_list = list(diag_info.keys())
    diag_convs = {}
    for diag_item in diag_list:
        if diag_item not in labels:
            continue

        prompt = diag_info[diag_item]["prompt"].replace(
            "请查看视频", f"请查看视频,并结合测量值数据：{mv}"
        )
        value = labels[diag_item]["value"]
        mark = labels[diag_item]["mark"]
        if mark == 0:
            continue

        video_paths = []
        for view in ALL_VIEWS:
            video_path = os.path.abspath(
                os.path.join(folder, f"{view.replace(' ', '_')}.mp4")
            )
            if os.path.exists(video_path):
                video_paths.append(str(video_path))

        if not video_paths:
            continue

        conv = {
            "id": folder,
            "video": video_paths,
            "diag_item": diag_item,
            "mark": mark,
            "conversations": [
                {"from": "human",
                 "value": "<video>\n" * len(video_paths) + prompt},
                {"from": "gpt", "value": "是" if value == 1 else "否"},
            ],
        }
        diag_convs[diag_item] = conv
    return diag_convs if diag_convs else None


# ----------------------------------------------------------------------
# 划分
# ----------------------------------------------------------------------

def split_by_folder(json_results, test_size=0.1, random_seed=42):
    """json_results: {diag_item: [conv, ...]}

    以 conv["id"]（=文件夹路径）为单位划分，保证同一患者不跨集。
    返回: (train_list, test_list)  两个扁平化列表。
    """
    all_ids = set()
    for conv_list in json_results.values():
        for c in conv_list:
            all_ids.add(c["id"])
    all_ids = sorted(all_ids)
    if not all_ids:
        return [], []

    train_ids, test_ids = train_test_split(
        all_ids, test_size=test_size, random_state=random_seed, shuffle=True
    )
    train_set = set(train_ids)
    test_set = set(test_ids)

    train_list, test_list = [], []
    for conv_list in json_results.values():
        for c in conv_list:
            if c["id"] in train_set:
                train_list.append(c)
            elif c["id"] in test_set:
                test_list.append(c)

    return train_list, test_list


# ----------------------------------------------------------------------
# 平衡（每个诊断项内部 正:负 = 1:1）
# ----------------------------------------------------------------------

def _get_gt(conv):
    """取 conv 的 gpt 回答，'是' / '否'"""
    try:
        return conv["conversations"][-1]["value"]
    except Exception:
        return None


def _print_dist(title, conv_list):
    """打印某列表按诊断项的 正/负 分布。"""
    tmp = {}
    for c in conv_list:
        tmp.setdefault(c.get("diag_item", "__unknown__"), {"是": 0, "否": 0})
        gt = _get_gt(c)
        if gt in ("是", "否"):
            tmp[c["diag_item"]][gt] += 1
    print(f"\n{title}")
    print(f"{'诊断项':<40} {'正(是)':<10} {'负(否)':<10} {'合计':<10}")
    print("-" * 70)
    for diag_item, cnt in tmp.items():
        total = cnt["是"] + cnt["否"]
        print(f"{diag_item:<40} {cnt['是']:<10} {cnt['否']:<10} {total:<10}")


def balance_by_diag(train_list, random_seed=42):
    """对 train_list 按 diag_item 分组，组内按'是'/'否'下采样到 1:1。"""
    rng = random.Random(random_seed)

    by_diag = {}
    for c in train_list:
        by_diag.setdefault(c.get("diag_item", "__unknown__"), []).append(c)

    balanced = []
    print("\n[下采样] 每个诊断项内部 正:负 = 1:1")
    print(f"{'诊断项':<40} {'原始(正/负)':<16} {'平衡后':<10}")
    print("-" * 70)

    for diag_item, group in by_diag.items():
        pos = [c for c in group if _get_gt(c) == "是"]
        neg = [c for c in group if _get_gt(c) == "否"]
        n = min(len(pos), len(neg))

        if n == 0:
            print(f"{diag_item:<40} {len(pos):>5}/{len(neg):<8} {'跳过(单类)':<10}")
            continue

        sampled_pos = rng.sample(pos, n)
        sampled_neg = rng.sample(neg, n)
        balanced.extend(sampled_pos)
        balanced.extend(sampled_neg)

        print(f"{diag_item:<40} {len(pos):>5}/{len(neg):<8} {2 * n:<10}")

    rng.shuffle(balanced)
    return balanced


def balance_by_diag_upsample(train_list, random_seed=42):
    """对 train_list 按 diag_item 分组，组内按'是'/'否'上采样到 1:1。

    多数类保留全部，少数类有放回采样（会重复）到同数量。
    注意：会导致数据量膨胀，谨慎使用。
    """
    rng = random.Random(random_seed)

    by_diag = {}
    for c in train_list:
        by_diag.setdefault(c.get("diag_item", "__unknown__"), []).append(c)

    balanced = []
    print("\n[上采样] 每个诊断项内部 正:负 = 1:1")
    print(f"{'诊断项':<40} {'原始(正/负)':<16} {'上采样后':<10}")
    print("-" * 70)

    for diag_item, group in by_diag.items():
        pos = [c for c in group if _get_gt(c) == "是"]
        neg = [c for c in group if _get_gt(c) == "否"]

        if len(pos) == 0 or len(neg) == 0:
            balanced.extend(group)
            print(f"{diag_item:<40} {len(pos):>5}/{len(neg):<8} "
                  f"{'保留原样(单类)':<10}")
            continue

        n = max(len(pos), len(neg))
        sampled_pos = list(pos)
        sampled_neg = list(neg)

        if len(pos) < n:
            sampled_pos = rng.choices(pos, k=n)
        if len(neg) < n:
            sampled_neg = rng.choices(neg, k=n)

        balanced.extend(sampled_pos)
        balanced.extend(sampled_neg)

        print(f"{diag_item:<40} {len(pos):>5}/{len(neg):<8} "
              f"{len(sampled_pos) + len(sampled_neg):<10}")

    rng.shuffle(balanced)
    return balanced


def main():
    parser = argparse.ArgumentParser(description="处理医学诊断视频数据集")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据根目录路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录，将在其中生成 train.json 和 test.json')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['pvd', 'avd', 'cmd'],
                        help='处理模式: pvd(主要切面), avd(主要+辅助切面), cmd(所有切面)')
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='测试集比例，默认 0.1（按文件夹划分）')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子，默认 42')
    parser.add_argument('--balance', action='store_true',
                        help='对训练集按诊断项做 正:负 = 1:1 平衡（默认关闭）')
    parser.add_argument('--balance_upsample', action='store_true',
                        help='平衡时用上采样（默认下采样；仅 --balance 生效时有效）')
    parser.add_argument('--cache_file', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "subfolders_cache.json"),
                        help='文件夹扫描结果缓存文件路径')
    parser.add_argument('--refresh_cache', action='store_true',
                        help='忽略缓存，强制重新扫描数据目录')
    args = parser.parse_args()

    root = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    mode = args.mode
    test_size = args.test_size
    random_seed = args.random_seed

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("开始处理诊断数据...")
    print("=" * 60)
    print(f"数据目录: {root}")
    print(f"输出目录: {output_dir}")
    print(f"处理模式: {mode}")
    print(f"测试集比例: {test_size}")
    print(f"随机种子: {random_seed}")
    print(f"训练集平衡: {'开启' if args.balance else '关闭'}"
          + (f"（{'上采样' if args.balance_upsample else '下采样'}）" if args.balance else ""))

    if mode == 'pvd':
        print("模式: PVD (Primary View Diagnosis) - 仅使用主要切面")
        deal_mode = deal_pvd
    elif mode == 'avd':
        print("模式: AVD (Associated Views Diagnosis) - 使用主要+辅助切面")
        deal_mode = deal_avd
    elif mode == 'cmd':
        print("模式: CMD (Comprehensive Multi-Views Diagnosis) - 使用所有切面")
        deal_mode = deal_cmd
    else:
        print(f"错误: 未知模式 '{mode}'")
        return

    # 第一步：扫描
    print("\n[1/3] 扫描并筛选包含 report.json 的文件夹...")
    folders = get_subfolders_with_cache(root, args.cache_file, args.refresh_cache)
    print(f"找到 {len(folders)} 个包含 report.json 的文件夹")

    if not global_diag_info:
        print("错误: 无法加载 diag_info.json")
        return
    print(f"加载了 {len(global_diag_info)} 个诊断项")

    # 第二步：处理
    print("\n[2/3] 处理诊断数据...")
    json_results = {}
    success_count = 0
    fail_count = 0

    for folder in tqdm(folders, desc="处理文件夹", unit="个"):
        result = deal_mode(folder, global_diag_info)
        if result is None:
            fail_count += 1
            continue
        success_count += 1
        for diag_item, conv in result.items():
            json_results.setdefault(diag_item, []).append(conv)

    total = sum(len(v) for v in json_results.values())
    print(f"✓ 成功: {success_count} 个文件夹")
    print(f"✗ 失败: {fail_count} 个文件夹")
    print(f"📁 诊断类型: {len(json_results)} 种")
    print(f"📄 总对话数: {total} 条")

    # ==== 新增：划分前先看分布 ====
    flat_all = [c for convs in json_results.values() for c in convs]
    _print_dist("[分布检查] 全量数据 各诊断项 正/负：", flat_all)

    # 第三步：划分 + 平衡 + 保存
    print("\n[3/3] 按文件夹（患者）级别划分 train/test...")
    train_list, test_list = split_by_folder(
        json_results, test_size=test_size, random_seed=random_seed
    )
    print(f"  划分后 -> 训练集: {len(train_list)} 条, 测试集: {len(test_list)} 条")

    if args.balance:
        if args.balance_upsample:
            train_list = balance_by_diag_upsample(train_list, random_seed=random_seed)
        else:
            train_list = balance_by_diag(train_list, random_seed=random_seed)
        print(f"  平衡后 -> 训练集: {len(train_list)} 条")

    train_path = os.path.join(output_dir, "train.json")
    test_path = os.path.join(output_dir, "test.json")
    save_json(train_list, train_path)
    save_json(test_list, test_path)

    # 统计
    print("\n" + "=" * 60)
    print("处理完成！统计信息：")
    print("=" * 60)
    print(f"✓ 成功处理: {success_count} 个文件夹")
    print(f"✗ 处理失败: {fail_count} 个文件夹")
    print(f"📁 诊断类型: {len(json_results)} 种")
    print(f"🚂 训练集: {len(train_list)} 条 -> {train_path}")
    print(f"🧪 测试集: {len(test_list)} 条 -> {test_path}")

    # 训练集最终分布
    _print_dist("训练集各诊断项 正/负 分布（最终）：", train_list)
    print("=" * 60)


if __name__ == "__main__":
    main()