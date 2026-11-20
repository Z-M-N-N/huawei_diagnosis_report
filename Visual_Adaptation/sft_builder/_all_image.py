import os
from pathlib import Path
from tqdm import tqdm
from _tool import balance_data2, load_json, save_json
from _tool import diag_info, train_reports, test_reports, video_path

import cv2

def extract_frames(video_path, output_dir, num_frames=16):
    """
    从视频中均匀抽取指定数量的帧并保存为图像，返回保存的图像路径列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        #print(f"❌ 无法打开视频文件: {video_path}")
        return []
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    #print(f"📹 视频信息:")
    #print(f"  总帧数: {total_frames}")
    #print(f"  帧率: {fps:.2f} FPS")
    #print(f"  时长: {duration:.2f} 秒")
    #print(f"  抽取帧数: {num_frames}")
    
    if total_frames < num_frames:
        #print(f"⚠️ 视频总帧数 ({total_frames}) 少于请求抽取的帧数 ({num_frames})，将抽取所有帧")
        num_frames = total_frames
    
    # 计算均匀间隔
    if num_frames == 1:
        frame_indices = [0]
    else:
        frame_indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    
    #print(f"  抽取帧索引: {frame_indices[:5]}... (共{len(frame_indices)}帧)")
    
    # 提取并保存帧，同时收集路径
    saved_paths = []
    saved_count = 0
    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            #print(f"⚠️ 读取第 {frame_idx} 帧失败")
            continue
        
        # 保存图像
        filename = f"frame_{idx+1:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        saved_paths.append(filepath)
        saved_count += 1
        
        if (idx + 1) % 4 == 0 or idx == len(frame_indices) - 1:
            #print(f"  进度: {idx+1}/{len(frame_indices)} 帧已保存")
            pass
    
    cap.release()
    #print(f"✅ 完成! 共保存 {saved_count} 帧到目录: {output_dir}")
    return saved_paths


def only_images(save_path, diag_item):
    """主视角 + 默认提示词"""

    if "train" in save_path:
        reports = train_reports
    elif "test" in save_path:
        reports = test_reports
    else:
        #print(f"⚠️ 无法识别 save_path: {save_path}")
        return
    
    os.makedirs(save_path, exist_ok=True)

    prompt = diag_info[diag_item]["prompt"]
    P = diag_info[diag_item]["P"]
    N = diag_info[diag_item]["N"]

    split_key_file = f"diag_words/{diag_item}.json"
    if os.path.exists(split_key_file):
        #print(f"分类文件存在: {split_key_file}")
        P = load_json(split_key_file)
        #print("读取分类参考！")

    # 统一的图像输出目录
    base_image_dir = "../VIDEO_DATA/images/"
    
    # 第一步：收集所有符合条件的视频信息和标签
    video_items = []
    
    for report in tqdm(reports, unit="report", leave=False):
        report_data = report["data"]
        path = report_data["path"]  # 报告路径
        
        mp4_files = list((Path(video_path) / path).glob("*.mp4"))
        
        try:
            report_txt = report_data["diagnosticopinion"] + report_data["findings_text"]
        except Exception as e:
            continue
            
        for mp4 in mp4_files:
            view = str(mp4).split("/")[-1].replace("_", " ").replace(".mp4", "")
            if view not in diag_info[diag_item]["primary_views"]:
                continue
                
            # 判断标签
            value = None
            for p in P:
                if p in report_txt:
                    value = "是"
                    break  # 找到第一个匹配就跳出
            if value is None:
                for n in N:
                    if n in report_txt:
                        value = "否"
                        break
            
            if value is None:
                continue
            
            # 保存视频信息
            video_items.append({
                "id": path,
                "video_path": str(mp4),
                "diag_item": diag_item,
                "label": value,
                "report_path": path  # 保存报告路径用于创建子目录
            })
    
    #print(f"📊 共收集到 {len(video_items)} 个视频样本")
    
    # 第二步：平衡数据
    balanced_items = balance_data2(video_items)
    #print(f"📊 平衡后剩余 {len(balanced_items)} 个样本")
    
    # 第三步：处理每个视频，抽取帧并构建对话
    ALL_CONV = []
    
    for item in tqdm(balanced_items, unit="video", leave=False):
        mp4_path = item["video_path"]
        path = item["report_path"]
        
        # 创建输出目录：base_image_dir/diag_item/report_path/video_name/
        video_basename = Path(mp4_path).stem  # 去掉 .mp4 后缀
        output_dir = os.path.join(base_image_dir, path, video_basename)
        
        # 抽取帧，获取保存的图片路径列表
        frame_paths = extract_frames(mp4_path, output_dir, num_frames=16)
        
        # 构建 content 列表：所有图片 + 文本
        content_list = []
        for img_path in frame_paths:
            content_list.append({"type": "image", "image": img_path})
        content_list.append({"type": "text", "text": prompt})
        
        # 构建完整的对话数据
        conv = {
            "id": item["id"],
            "video": [mp4_path],
            "diag_item": item["diag_item"],
            "label": item["label"],
            "conversations": [
                {
                    "role": "user",
                    "content": content_list
                }
            ]
        }
        
        ALL_CONV.append(conv)
    
    # 保存结果
    #print(f"✅ 共生成 {len(ALL_CONV)} 条对话数据")
    save_json(ALL_CONV, os.path.join(save_path, f"{diag_item}.json"))
    #print(f"💾 已保存到: {os.path.join(save_path, f'{diag_item}.json')}")