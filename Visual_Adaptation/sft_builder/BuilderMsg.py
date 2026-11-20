import argparse
import os
import json
import random
import os



from _only_video import only_video
from _multi_video import multi_video
from _all_video import all_video
from _video_and_mv import video_and_mv
from _visual_multi_task import video_to_multi_task
from _tool import load_json,merge_json_files_as_list
from _all_image import only_images
# ==================== 主程序 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Qwen2.5-VL 视频理解推理脚本')
    parser.add_argument('--diag_item', type=str, default="二尖瓣反流", help='诊断项')
    parser.add_argument('--bulider_mode', type=str, default="构建模式", help='可选')
    args = parser.parse_args()

    builder_mode=args.bulider_mode
    diag_info=load_json("diag_info.json")
    diag_items=list(diag_info.keys())

    if builder_mode=="only_video":
        if args.diag_item=="all":
            for idx,diag_item in enumerate(diag_items):
                print("="*60)
                print("diag_item:",idx+1,diag_item)
                save_path = os.path.join("only_video", "train")
                only_video(save_path,diag_item)
                save_path = os.path.join("only_video", "test")
                only_video(save_path,diag_item)
                print("="*60)

            merge_json_files_as_list(
                directory=f"only_video/train",
                output_file="merged_train_list.json"
            )
            merge_json_files_as_list(
                directory=f"only_video/test",
                output_file="merged_test_list.json"
            )

    if builder_mode=="multi_video":
        if args.diag_item=="all":
            for idx,diag_item in enumerate(diag_items):
                print("="*60)
                print("diag_item:",idx+1,diag_item)
                save_path = os.path.join("multi_video", "train")
                multi_video(save_path,diag_item)
                save_path = os.path.join("multi_video", "test")
                multi_video(save_path,diag_item)
                print("="*60)
                
            merge_json_files_as_list(
                directory=f"multi_video/train",
                output_file="merged_train_list.json"
                )
            merge_json_files_as_list(
                directory=f"multi_video/test",
                output_file="merged_test_list.json"
                )

    if builder_mode=="all_video":
        if args.diag_item=="all":
            for idx,diag_item in enumerate(diag_items):
                print("="*60)
                print("diag_item:",idx+1,diag_item)
                save_path = os.path.join("all_video", "train")
                all_video(save_path,diag_item)
                save_path = os.path.join("all_video", "test")
                all_video(save_path,diag_item)
                print("="*60)
                
            merge_json_files_as_list(
                directory=f"all_video/train",
                output_file="merged_train_list.json"
                )
            merge_json_files_as_list(
                directory=f"all_video/test",
                output_file="merged_test_list.json"
                )

    if builder_mode=="video_and_mv":
        if args.diag_item=="all":
            for idx,diag_item in enumerate(diag_items):
                print("="*60)
                print("diag_item:",idx+1,diag_item)
                save_path = os.path.join("video_and_mv", "train")
                video_and_mv(save_path,diag_item)
                save_path = os.path.join("video_and_mv", "test")
                video_and_mv(save_path,diag_item)
                print("="*60)
                
            merge_json_files_as_list(
                directory=f"video_and_mv/train",
                output_file="merged_train_list.json"
                )
            merge_json_files_as_list(
                directory=f"video_and_mv/test",
                output_file="merged_test_list.json"
                )
    if builder_mode=="video_to_multi_task":
        print("="*60)
        save_path = os.path.join("video_to_multi_task", "train")
        video_to_multi_task(save_path,diag_items)
        save_path = os.path.join("video_to_multi_task", "test")
        video_to_multi_task(save_path,diag_items)
        print("="*60)

    if builder_mode=="only_images":
        if args.diag_item=="all":
            for idx,diag_item in enumerate(diag_items):
                save_path = os.path.join("only_images", "test")
                only_images(save_path,diag_item)