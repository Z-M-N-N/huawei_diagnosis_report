# EXPERIMENT_LOG

## [2026-10-14 15:50] 创建项目 & 构建报告生成数据
- 目的：在 `/home/mzhao/华为项目/Report` 新建"超声报告生成模型"项目（基于
  `/home/mzhao/华为项目/VLM` 框架），并把 pvd 报告数据重写路径后转为训练/测试集。
- 执行命令：
  - 代码：拷贝 `VLM/{src,train,scripts}` + 补回缺失的 `data/` 模块（来自 `EchoView-main/VLM/data`）
  - 数据：`python VLM/data/build_report_data.py`
- 关键参数：
  - 路径重写：`/home/mzhao/ECHO_VIEW/Data/full_dicom_videos_group` → `/home/mzhao/华为项目/Data/full_dicom_videos_group`
  - 去重方式：按 study `id` 合并，视频取并集（保持出现顺序），`<video>` 占位符数 == 视频数
  - 基座模型（训练脚本）：`/home/mzhao/华为项目/VLM/Model/good_pvd_merged`
- 输入：
  - `.../Train_Test_JSON_Report/pvd/train/merged_train_list.json`（567,032 条）
  - `.../Train_Test_JSON_Report/pvd/test/merged_test_list.json`（36,838 条）
- 输出：
  - `VLM/data/report/train.json`（37,183 检查，69.2 MB）
  - `VLM/data/report/test.json`（4,134 检查，7.7 MB）
  - 脚本：`VLM/data/build_report_data.py`、`run_train.sh`、`run_eval.sh`、`VLM/src/inference_report.py`
- 结果/指标（数据统计）：
  - 每检查视频数分布：train {1:12901, 2:12889, 3:6897, 4:4496}；test {1:1407, 2:1444, 3:746, 4:537}
  - 文件缺失丢弃：0；占位符自检（前5000）：不匹配 0
  - train/test 无 id 重叠（划分干净）
- 结论与备注：
  - 数据格式与框架 `data_processor` 兼容，可直接训练。
  - 报告较长（中位 ~378 字），batch=1 / grad_accum=8 / max_len=40960。
  - **待用户确认**：训练显卡（当前 8 卡均被占用）；基座用 good_pvd_merged 还是原始 Qwen2.5-VL-7B。
  - 尚未启动训练（GPU 未指定）。
- 查看命令：`tail -n 60 /home/mzhao/华为项目/Report/EXPERIMENT_LOG.md`

## [2026-10-14 15:58] 调整数据：测量值移出提示词、并入生成文本
- 目的：按用户要求，提示词中不再给测量值，改为让模型在生成文本中输出测量值。
- 执行命令：`python VLM/data/build_report_data.py`（日志 /tmp/echo_logs/logs_build_report_data.log）
- 关键改动（`VLM/data/build_report_data.py`）：
  - 正则 `[,，]并结合测量值数据：({...})[,，]` 抽取并删除提示词中的测量值
  - human：`<video>*N\n请查看视频,给出诊断发现报告。`
  - gpt：原报告末尾追加 ` measurements:{...}`；新增 `measurements` 字段
- 输入：pvd merged train/test（同上次）
- 输出：`VLM/data/report/train.json`（37,183，76.1 MB）、`test.json`（4,134，8.5 MB）
- 结果/指标：含测量值 37183/37183、4134/4134；提示词残留"测量值" 0 条；gpt 缺 measurements 0 条；占位符自检 0 不匹配
- 结论与备注：数据已更新，训练脚本无需改动。仍未启动训练（GPU 未指定）。
- 查看命令：`tail -n 30 /home/mzhao/华为项目/Report/EXPERIMENT_LOG.md`

## [2026-10-14 16:27] 启动报告生成 LoRA 训练（GPU 2）
- 目的：用 pvd 报告数据训练超声报告生成模型。
- 执行命令：`GPU_ID=2 bash run_train.sh`（screen 会话 `report_train`）
- 关键参数：基座 `Model/good_pvd_merged`；LoRA r=64；batch=1 × grad_accum=8；lr=1e-4；epoch=1；save_step=500；seed=42；bf16；gradient checkpointing
- 输入：`VLM/data/report/train.json`（37,183 检查）；验证 `VLM/data/report/test.json`（5%）
- 输出：`VLM/lora/report/checkpoint-*`；日志 `/tmp/echo_logs/logs_train_report.log`
- 结果/指标：LoRA 可训练参数 40,370,176（占 0.48%）；总步数 4648；约 20 s/step，GPU2 显存 ~33 GB / 利用率 ~60%
- 结论与备注：已进入正常 step（>6）。预计 1 epoch ≈ 25 h。可回收/续跑。
- 查看命令：`tail -f /tmp/echo_logs/logs_train_report.log`

## [2026-10-14 16:39] 搭建训练里程碑自动评测（1/3、1/2、结束）+ 输出耗时统计
- 目的：在训练 1/3、1/2、结束三个节点各跑一次固定测试子集，生成含【输入/输出/指标/耗时】的报告。
- 关键文件：
  - `VLM/src/make_eval_subset.py` → 固定评测子集 `VLM/data/report/eval_subset.json`（300 条，seed=42）
  - `VLM/src/inference_report.py`（含逐条计时 `gen_seconds`）
  - `VLM/src/eval_report_metrics.py`（ROUGE-1/2/L 字符级、BLEU(jieba)、测量值 MAE/命中率、耗时统计）
  - `VLM/src/make_report_md.py`（生成 markdown 报告）
  - `run_eval_report.sh`（单次评测）、`eval_milestones.sh`（里程碑监控）
- 里程碑映射（总步数 4648）：1/3→`checkpoint-1500`；1/2→`checkpoint-2500`；结束→`VLM/lora/report/`
- 耗时阈值：300 s（记录单条 min/median/mean/max/p95、超阈值条数、整体是否超阈值）
- 输出：`out/report/<tag>/{preds.json,metrics.json,times.csv,report.md}`；日志 `/tmp/echo_logs/logs_eval_*.log`
- 执行命令：`EVAL_GPU=5 bash eval_milestones.sh`（screen 会话 `eval_milestones`）
- 结果/指标：冒烟测试（基座，2 条，GPU5）通过，~6 s/条；评测管线端到端可用。
- 结论与备注：评测用 GPU 5（评测时约 17 GB 空闲）；若被占满可能 OOM，可改 `EVAL_GPU` 重跑。
- 查看命令：`tail -f /tmp/echo_logs/logs_eval_milestones.log`

## [2026-10-15 11:00] 里程碑评测失败排查 & 补测
- 现象：`ckpt-1500`、`ckpt-2500` 评测在 GPU 5 上 **OOM 失败**，`out/report/ckpt-1500|ckpt-2500/` 为空。
  - 原因：GPU 5 被其他进程占用（32.7 GiB），本进程仅拿到 ~14 GiB，加载 7B 基座时 `torch.OutOfMemoryError`。
  - 同时 `run_eval_report.sh` 未检查子进程退出码，失败后仍打印"完成"，导致监控误判。
- 处置：
  1. 暂停训练：`kill -SIGINT <torchrun pid>`（优雅停止；checkpoint 500–3500 完好，可从 checkpoint-3500 续跑）
  2. 停止里程碑监控 screen
  3. 加固脚本：`run_eval_report.sh` 加 `set -o pipefail` + 校验 `preds.json`；`eval_milestones.sh` 失败自动重试 3 次、以 `metrics.json` 非空为完成标志
  4. 改用空闲 GPU 2 补测；评测协议统一为：**150→100 条固定子集(seed=42)、每视频 8 帧、输出上限 512 tokens**
- 执行命令：`EVAL_GPU=2 NUM_FRAMES=8 MAX_NEW_TOKENS=512 bash run_eval_report.sh <tag> <ckpt>`（screen `eval_catchup`）
- 结果/指标：补测进行中（ckpt-1500 → ckpt-2500），单条约 30 s，预计 ~1.5–2 h 完成。
- 结论与备注：**训练当前处于暂停状态**；三个里程碑协议已统一。待补测完成后可续训至结束，再做 `final` 评测。
- 查看命令：`tail -f /tmp/echo_logs/logs_eval_ckpt-1500.log`

## [2026-10-15 11:27] 评测 ckpt-3500（原配置）并导出 xlsx
- 目的：用最新 checkpoint-3500、原输入配置（16 帧/视频、输出上限 768、100 条固定子集）评测并导出 Excel。
- 执行命令：`EVAL_GPU=2 NUM_FRAMES=16 MAX_NEW_TOKENS=768 bash run_eval_report.sh ckpt-3500 VLM/lora/report/checkpoint-3500`（screen `eval_3500`）
- 新增：`VLM/src/make_report_xlsx.py` → 生成 `report.xlsx`（概述 / 样本明细 / 测量值指标 三个 sheet）
- 输出：`out/report/ckpt-3500/{preds.json,metrics.json,times.csv,report.md,report.xlsx}`
- 结果/指标：进行中，单样品预跑 ~22.7 s/条，预计 100 条约 50–70 min。
- 结论与备注：训练仍暂停；ckpt-1500/2500 补测未跑（等用户确认）。
- 查看命令：`tail -f /tmp/echo_logs/logs_eval_ckpt-3500.log`

## [2026-10-15 12:26] ckpt-3500 第二批 200 条推理（与前 100 合并为 300）
- 目的：在原 100 条基础上再推 200 条（不重叠），合并为 300 条统一评测。
- 前 100 条结果：`out/report/ckpt-3500/`（report.xlsx 已生成，平均 22.0 s/条）
- 第二批子集：`VLM/data/report/eval_subset_200.json`（排除已用 100 条；seed=2026）
- 执行命令：`CUDA_VISIBLE_DEVICES=2 python VLM/src/inference_report.py --lora_path VLM/lora/report/checkpoint-3500 --subset_json VLM/data/report/eval_subset_200.json --save_results out/report/ckpt-3500_p2 --num_frames 16 --max_new_tokens 768`（screen `eval_3500_p2`）
- 合并脚本：`VLM/src/merge_preds.py` + `merge_and_report.sh`
- 输出：合并后 `out/report/ckpt-3500_all/`（preds/metrics/md/xlsx，300 条）
- 结论与备注：预计 ~70 min；训练仍暂停。
- 查看命令：`tail -f /tmp/echo_logs/logs_eval_ckpt-3500_p2.log`

## [2026-10-15 14:36] 未训练底模对照（带输出模版提示词，300 条）
- 目的：用"未训练"的原始底模 `Qwen2.5-VL-7B-Instruct`（无 LoRA）对同样 300 条做生成，作为对照；提示词中加入输出模版。
- 提示词模版：`请查看视频，给出诊断发现报告。请严格按以下格式输出…` + `finding: / diagnosticopinion: / measurements:{7项}` 占位模板
- 子集：`VLM/data/report/eval_subset_300_tmpl.json`（= 已评测的同样 300 条 id，仅提示词换成带模版）
- 执行命令：`CUDA_VISIBLE_DEVICES=2 python VLM/src/inference_report.py --model_path /home/mzhao/公共大模型/Qwen2.5-VL-7B-Instruct --lora_path '' --subset_json ...eval_subset_300_tmpl.json --save_results out/report/base_untrained --num_frames 16 --max_new_tokens 768`（screen `eval_base`）
- 输出：`out/report/base_untrained/`（preds/metrics/md/xlsx，300 条）
- 结论与备注：预计 ~2 h；训练仍暂停。
- 查看命令：`tail -f /tmp/echo_logs/logs_eval_base_untrained.log`
