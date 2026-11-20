# Report —— 超声心动图诊断报告生成模型

基于 `/home/mzhao/华为项目/VLM` 的 Qwen2.5-VL LoRA SFT 框架，训练**检查级报告生成**模型：
输入一次超声检查的多段视频（+ 测量值），输出 `finding: ... diagnosticopinion: ...` 报告。

## 目录结构
```
Report/
├── run_train.sh                 # 训练入口（GPU_ID 需指定）
├── run_eval.sh                  # 推理/评估入口
├── EXPERIMENT_LOG.md            # 实验记录
└── VLM/                         # 框架代码（拷贝自 华为项目/VLM，并补回 data 模块）
    ├── train/                   # train_qwen.py 等（SFT 主程序）
    ├── src/                     # inference.py（分类）、inference_report.py（报告生成）
    ├── scripts/zero3.json       # DeepSpeed 配置
    └── data/
        ├── data_processor.py    # 数据加载（来自 EchoView-main/VLM/data，华为项目/VLM 中缺失）
        ├── build_report_data.py # 由 pvd 合并列表构建报告数据（含路径重写）
        └── report/              # 生成的 train.json / test.json
```

## 数据
- 来源：`/home/mzhao/华为项目/Data/full_dicom_videos_group/Train_Test_JSON_Report/pvd/{train,test}/merged_*_list.json`
- 路径重写：`/home/mzhao/ECHO_VIEW/Data/full_dicom_videos_group` → `/home/mzhao/华为项目/Data/full_dicom_videos_group`
- 去重：按 study id 合并（原始 567,032 → 训练 37,183 检查；36,838 → 测试 4,134 检查），视频取并集
- 提示词：`<video>*N\n请查看视频,给出诊断发现报告。`（**不含测量值**）
- 目标文本：`finding:... diagnosticopinion:... measurements:{...}`（**测量值并入生成文本末尾**）
- 格式：`{id, video:[...], measurements:{...}, diag_item, mark, conversations:[human, gpt]}`
- 重新生成：`python VLM/data/build_report_data.py`

## 训练
```bash
cd /home/mzhao/华为项目/Report
GPU_ID=5 bash run_train.sh          # 前台
# 后台： setsid nohup bash run_train.sh >/dev/null 2>&1 &
tail -f /tmp/echo_logs/logs_train_report.log
```
关键配置（在 `run_train.sh` 配置区修改）：
- 基座：`/home/mzhao/华为项目/VLM/Model/good_pvd_merged`（可换原始 Qwen2.5-VL-7B）
- batch=1，grad_accum=8，lr=1e-4，epoch=1，save_step=500，seed=42
- 输出：`VLM/lora/report/`

## 评估
```bash
GPU_ID=5 bash run_eval.sh           # 生成 out/report/preds.json，附带 ROUGE-1/L
```

## 备注
- 全部日志写入 `/tmp/echo_logs/`。
- 基座权重 16G，采用**绝对路径引用**，未拷贝，避免占用磁盘。
