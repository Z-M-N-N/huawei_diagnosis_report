

import os
from pathlib import Path
import random
from tqdm import tqdm
from _tool import balance_data, load_json, save_json
from _tool import diag_info,train_reports,test_reports,video_path
def float_measurements(measurements):
    """
    对测量值在原始值上下浮动最多 ±3，但不超过原始值的 ±20%
    
    Args:
        measurements: dict, 原始测量值
    
    Returns:
        dict: 浮动后的测量值（整数）
    """
    result = {}
    for key, value in measurements.items():
        if isinstance(value, (int, float)):
            # 计算最大允许浮动值
            max_delta_by_percent = value * 0.20  # 20% 限制
            max_delta = min(3, max_delta_by_percent)  # 取 3 和 20% 中的较小值
            
            # 在 [-max_delta, max_delta] 范围内随机浮动
            new_value = value + random.uniform(-max_delta, max_delta)
            # 四舍五入为整数
            result[key] = round(new_value)
        else:
            result[key] = value
    return result
def video_and_mv(save_path,diag_item):
    """主视角 + 测量值"""
    if "train" in save_path:
        reports=train_reports
    if "test" in save_path:
        reports=test_reports
    os.makedirs(save_path, exist_ok=True)

    # reports = load_json("../DATA/reports.json")
    # video_path = "../DATA/dicom_videos"
    # diag_info=load_json("diag_info.json")
    prompt=diag_info[diag_item]["prompt"]
    P=diag_info[diag_item]["P"]
    N=diag_info[diag_item]["N"]

    split_key_file=f"diag_words/{diag_item}.json"
    if os.path.exists(split_key_file):
        print(f"分类文件存在: {split_key_file}")
        P=load_json(split_key_file)
        print("读取分类参考！")

    
    ALL_CONV = []
    CONV=[]
    for report in tqdm(reports, unit="report", leave=False):
        report=report["data"]
        path = report["path"]
        mp4_files = list((Path(video_path) / path).glob("*.mp4"))
        mv=report["measurements"]
        mv=float_measurements(mv)


        newprompt=prompt.replace("判断",f"以及给出的测量值{mv},请判断")
        try:
            report_txt=report["diagnosticopinion"]+report["findings_text"]
        except Exception as e:
            continue
        for mp4 in mp4_files:
            
            view = str(mp4).split("/")[-1].replace("_"," ").replace(".mp4","")
            if view not in diag_info[diag_item]["primary_views"]:
                continue


            value=None
            for p in P:
                if p in report_txt:
                    value="是"
             
            for n in N:
                if n in report_txt:
                    value="否"
      
            if value is None:
                continue
      
            conv = {
                "id": path,
                "video": [str(mp4)],
                "diag_item": diag_item,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<video>\n"+newprompt
                    },
                    {
                        "from": "gpt",
                        "value": value
                        # "value": f"""【诊断结果】：{value}
                        #         【诊断意见】：{ref_diag}"""
                    }
                ]
            }
            CONV.append(conv)
    CONV = balance_data(CONV)
    ALL_CONV.extend(CONV)
    print(len(ALL_CONV))
    save_json(ALL_CONV, os.path.join(save_path, f"{diag_item}.json"))