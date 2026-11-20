import json
import random
import os
import json
import glob


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


reports = load_json("/home/mzhao/华为项目/EchoView/data_builders/video_data/reports.json")
train_reports=load_json("/home/mzhao/华为项目/EchoView/data_builders/video_data/train_data.json")
test_reports=load_json("/home/mzhao/华为项目/EchoView/data_builders/video_data/test_data.json")

video_path = "/home/mzhao/华为项目/EchoView/data_builders/video_data/dicom_videos"
diag_info=load_json("diag_info.json")

def balance_data(CONV,SAMPLE_SIZE=9000):
    # print(CONV[0])
    # P_Example=[item for item in CONV if f"诊断结果】：异常" in item["conversations"][-1]["value"]]
    # N_Example=[item for item in CONV if f"诊断结果】：正常" in item["conversations"][-1]["value"]]
    P_Example=[item for item in CONV if f"是" in item["conversations"][-1]["value"]]
    N_Example=[item for item in CONV if f"否" in item["conversations"][-1]["value"]]
    print("P:",len(P_Example),"N:",len(N_Example))
    # 各采样500个，不够则上采样
    # SAMPLE_SIZE = 500
    # if len(P_Example) == 0 or len(N_Example) == 0:
    #     print("Warning: One of the classes has no examples. Returning original data.")
    #     return CONV

    # if len(P_Example) >= SAMPLE_SIZE:
    #     P_Example = random.sample(P_Example, SAMPLE_SIZE)
    # else:
    #     P_Example = random.choices(P_Example, k=SAMPLE_SIZE)  # 有放回采样，可重复

    # if len(N_Example) >= SAMPLE_SIZE:
    #     N_Example = random.sample(N_Example, SAMPLE_SIZE)
    # else:
    #     N_Example = random.choices(N_Example, k=SAMPLE_SIZE)  # 有放回采样，可重复

    min_len = min(len(P_Example), len(N_Example), SAMPLE_SIZE)
    P_Example = random.sample(P_Example, min_len)
    N_Example = random.sample(N_Example, min_len)

    print("After balancing: P:", len(P_Example), "N:", len(N_Example))
    # 合并并打乱
    CONV = P_Example + N_Example
    random.shuffle(CONV)
    return CONV

def balance_data2(CONV,SAMPLE_SIZE=100):
    # print(CONV[0])
    # P_Example=[item for item in CONV if f"诊断结果】：异常" in item["conversations"][-1]["value"]]
    # N_Example=[item for item in CONV if f"诊断结果】：正常" in item["conversations"][-1]["value"]]
    P_Example=[item for item in CONV if f"是" in item["label"]]
    N_Example=[item for item in CONV if f"否" in item["label"]]
    print("P:",len(P_Example),"N:",len(N_Example))
    # 各采样500个，不够则上采样
    # SAMPLE_SIZE = 500
    # if len(P_Example) == 0 or len(N_Example) == 0:
    #     print("Warning: One of the classes has no examples. Returning original data.")
    #     return CONV

    # if len(P_Example) >= SAMPLE_SIZE:
    #     P_Example = random.sample(P_Example, SAMPLE_SIZE)
    # else:
    #     P_Example = random.choices(P_Example, k=SAMPLE_SIZE)  # 有放回采样，可重复

    # if len(N_Example) >= SAMPLE_SIZE:
    #     N_Example = random.sample(N_Example, SAMPLE_SIZE)
    # else:
    #     N_Example = random.choices(N_Example, k=SAMPLE_SIZE)  # 有放回采样，可重复

    min_len = min(len(P_Example), len(N_Example), SAMPLE_SIZE)
    P_Example = random.sample(P_Example, min_len)
    N_Example = random.sample(N_Example, min_len)

    print("After balancing: P:", len(P_Example), "N:", len(N_Example))
    # 合并并打乱
    CONV = P_Example + N_Example
    random.shuffle(CONV)
    return CONV



def merge_json_files_as_list(directory, output_file="merged.json"):
    """
    合并指定目录下的所有JSON文件为列表
    
    Args:
        directory: 包含JSON文件的目录路径
        output_file: 输出文件名
    """
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    if not json_files:
        print(f"在 {directory} 中没有找到JSON文件")
        return None
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    merged_data = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 如果是列表，扩展合并
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    # 如果是字典，添加文件名作为标识
                    file_name = os.path.splitext(os.path.basename(file_path))[0]
                    merged_data.append({
                        "source": file_name,
                        "data": data
                    })
                
                print(f"✓ 已合并: {os.path.basename(file_path)}")
                
        except Exception as e:
            print(f"✗ 读取失败 {os.path.basename(file_path)}: {e}")
    
    # 保存合并后的文件
    output_path = os.path.join(directory, output_file)
    random.shuffle(merged_data)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 合并完成！共 {len(merged_data)} 条记录保存到: {output_path}")
    return merged_data
