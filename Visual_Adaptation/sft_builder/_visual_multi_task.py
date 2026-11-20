

import json
import os
from pathlib import Path
from tqdm import tqdm
from _tool import balance_data, load_json, save_json
from _tool import diag_info,train_reports,test_reports,video_path

def video_to_multi_task(save_path,diag_items):
    """主视角 + 测量值"""
    if "train" in save_path:
        reports=train_reports
    if "test" in save_path:
        reports=test_reports

    os.makedirs(save_path, exist_ok=True)
    mv_name=['主动脉根部内径', '左房内径', '左室舒张末内径', '左室收缩末内径', '室间隔厚度', '左室后壁厚度', '肺动脉收缩压']

    ALL_CONV = []
    CONV=[]
    for report in tqdm(reports, unit="report", leave=False):
        report=report["data"]
        path = report["path"]
        mp4_files = list((Path(video_path) / path).glob("*.mp4"))
        mv=report["measurements"]
        try:
            report_txt=report["diagnosticopinion"]+report["findings_text"]
        except Exception as e:
            continue
        binary_labels=[]
        for idx,diag_item in enumerate(diag_items):
            P=diag_info[diag_item]["P"]
            split_key_file=f"diag_words/{diag_item}.json"
            if os.path.exists(split_key_file):
                # print(f"分类文件存在: {split_key_file}")
                P=load_json(split_key_file)
                # print("读取分类参考！")
            value=0
            for p in P:
                if p in report_txt:
                    value=1
                    # print(value)
                    break
     
            binary_labels.append(value)
 

        regression_labels=[mv[k] for k in mv_name]
            
        for mp4 in mp4_files:
            
            conv={
                "video": str(mp4),
                "binary_labels": binary_labels,
                "regression_labels": regression_labels
            }
            # conv = {
            #     "id": path,
            #     "video": str(mp4),

            #     "conversations": [
            #         {
            #             "from": "human",
            #             "value": "<video>\n"+newprompt
            #         },
            #         {
            #             "from": "gpt",
            #             "value": value
            #             # "value": f"""【诊断结果】：{value}
            #             #         【诊断意见】：{ref_diag}"""
            #         }
            #     ]
            # }
            # print(conv)
            CONV.append(conv)
    # CONV = balance_data(CONV)
    ALL_CONV.extend(CONV)
    print(len(ALL_CONV))
    save_json(ALL_CONV, os.path.join(save_path, f"visual_multi_task.json"))