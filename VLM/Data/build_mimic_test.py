# -*- coding: utf-8 -*-
"""
build_mimic_test.py
从 MIMIC-IV-ECHO 预处理好的逐切面 mp4 生成与 VLM/Data/pvd_no_mv/test.json
完全同格式的测试数据。

结构: {id, video[主切面mp4绝对路径], diag_item, mark, conversations[h:是/否]}
- 每个 study 对 28 个诊断项各生成一条记录，使用该诊断项的主切面视频。
- mark/gpt 标签: 优先用 exam.json 的 classification 映射(见 MAPPING)，
  能映射且标签非空 -> mark=1(阳)/-1(阴) 且 gpt=是/否; 无法映射 -> mark=0, gpt=否(占位).
"""
import os, json, argparse
from tqdm import tqdm

VLM_DIR = "/home/mzhao/华为项目/VLM"
DIAG_JSON = os.path.join(VLM_DIR, "Data/diag_info.json")
DEFAULT_ROOT = "/home/mzhao/超声诊断/Data/mimic-iv-echo/1.0.1/files_fixed"
DEFAULT_OUT = os.path.join(VLM_DIR, "Data/mimic_pvd")

# 诊断项 -> (MIMIC classification 键, 阳性规则)
#   'any'  : 存在且 >=1 视为阳性
#   'eq1'  : ==1 视为阳性
#   'both' : 组合(多键均 >=1)
MAPPING = {
    "心包积液":     ("pericardial_effusion", "eq1"),
    "三尖瓣反流":   ("tricuspid_regurg",     "any"),
    "二尖瓣反流":   ("mitral_regurg",        "any"),
    "主动脉瓣反流": ("aortic_regurg",        "any"),
    "右房增大":     ("ra_size",              "any"),
    "左房增大":     ("la_size",              "any"),
    "右室增大":     ("rv_chamber_size",      "any"),
    "左室增大":     ("lv_chamber_size",      "any"),
    "左室壁增厚":   ("lv_wall_thickness",    "any"),
    "收缩活动减弱": ("rv_function",          "any"),
    "二尖瓣脱垂":   ("mv_prolapse",          "eq1"),
    "瓣环钙化":     ("mac_severity",         "any"),
    "间隔缺损":     ("pfo_asd",              "eq1"),
    "双房增大":     (["la_size", "ra_size"], "both"),
}


def label_for(diag, cls):
    """返回 (mark, value)。class 为 None 或缺键 -> (0, '否') 占位。"""
    if diag not in MAPPING or not isinstance(cls, dict):
        return 0, "否"
    key, rule = MAPPING[diag]
    if rule == "both":
        vs = [cls.get(k) for k in key]
        if any(v is None for v in vs):
            return 0, "否"
        pos = all(v >= 1 for v in vs)
    else:
        v = cls.get(key)
        if v is None:
            return 0, "否"
        pos = (v >= 1) if rule == "any" else (v == 1)
    return (1, "是") if pos else (-1, "否")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 个 study(调试用)")
    args = ap.parse_args()

    diag_info = json.load(open(DIAG_JSON, encoding="utf-8"))
    diags = list(diag_info.keys())
    primary = {d: diag_info[d]["primary_views"][0].replace(" ", "_") + ".mp4" for d in diags}
    prompts = {d: diag_info[d]["prompt"] for d in diags}

    studies = []
    for p in sorted(os.listdir(args.root)):
        pdir = os.path.join(args.root, p)
        if not os.path.isdir(pdir):
            continue
        for s in sorted(os.listdir(pdir)):
            sdir = os.path.join(pdir, s)
            if os.path.isdir(sdir):
                studies.append(sdir)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "test.json")
    tmp = out_path + ".tmp"

    n_rec = 0
    skipped_no_exam = 0
    per_diag = {d: 0 for d in diags}
    per_diag_pos = {d: 0 for d in diags}
    per_diag_gt = {d: 0 for d in diags}  # 有真实标签的条数

    with open(tmp, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for i, sdir in enumerate(tqdm(studies, desc="生成记录", unit="study")):
            if args.limit and i >= args.limit:
                break
            exam = os.path.join(sdir, "exam.json")
            cls = None
            if os.path.exists(exam):
                try:
                    cls = json.load(open(exam, encoding="utf-8")).get("classification")
                except Exception:
                    cls = None
            else:
                skipped_no_exam += 1
            for d in diags:
                vp = os.path.join(sdir, primary[d])
                if not os.path.exists(vp):
                    continue
                mark, value = label_for(d, cls)
                rec = {
                    "id": sdir,
                    "video": [vp],
                    "diag_item": d,
                    "mark": mark,
                    "conversations": [
                        {"from": "human", "value": "<video>\n" + prompts[d]},
                        {"from": "gpt", "value": value},
                    ],
                }
                if not first:
                    f.write(",\n")
                json.dump(rec, f, ensure_ascii=False, separators=(",", ":"))
                first = False
                n_rec += 1
                per_diag[d] += 1
                if mark in (1, -1):
                    per_diag_gt[d] += 1
                if mark == 1:
                    per_diag_pos[d] += 1
        f.write("\n]\n")
    os.replace(tmp, out_path)

    print("=" * 60)
    print(f"studies: {len(studies)} (无 exam.json: {skipped_no_exam})")
    print(f"总记录: {n_rec} -> {out_path}")
    print(f"{'诊断项':<12}{'记录':>8}{'阳性':>8}{'有标签':>8}")
    for d in diags:
        print(f"{d:<12}{per_diag[d]:>8}{per_diag_pos[d]:>8}{per_diag_gt[d]:>8}")


if __name__ == "__main__":
    main()
