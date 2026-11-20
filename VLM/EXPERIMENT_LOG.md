# EXPERIMENT_LOG

## [2026-10-11 23:07] 构建 MIMIC-IV-ECHO 测试数据（同格式）

- 目的：为 VLM 模型（Model/good_pvd_merged）在 MIMIC-IV-ECHO 数据集上做预测，生成与
  `Data/pvd_no_mv/test.json` 完全同格式的测试数据。
- 执行命令：
  - `python3 Data/build_mimic_test.py`（全量）
  - 冒烟：`python3 Data/build_mimic_test.py --limit 20 --out_dir /tmp/mimic_smoke`
- 关键参数：
  - 模式：pvd（每个诊断项只取该诊断项的**主切面**视频）
  - 数据源：`/home/mzhao/超声诊断/Data/mimic-iv-echo/1.0.1/files_fixed/p*/s*/`
  - 诊断项：`Data/diag_info.json` 全部 28 项
  - 帧数/模型：本次不涉及推理（GPU 繁忙）
- 输入：MIMIC 逐切面 mp4（72,597 个）+ `exam.json`（7,228 个 study 有）
- 输出：`Data/mimic_pvd/test.json`（78 MB，193,579 条）；构建脚本 `Data/build_mimic_test.py`
- 记录结构（与 pvd_no_mv/test.json 一致）：
  `{id(=study目录), video([主切面mp4绝对路径]), diag_item, mark, conversations[human:"<video>\n<提示>", gpt:是/否]}`
- 标签来源与假设（**重要，待确认**）：
  - MIMIC 没有 28 个中文诊断项的原生标签。脚本用 `exam.json` 的 `classification`
    做了映射（见 build_mimic_test.py 顶部 MAPPING）：
    三尖瓣反流/tricuspid_regurg、二尖瓣反流/mitral_regurg、主动脉瓣反流/aortic_regurg、
    右房增大/ra_size、左房增大/la_size、右室增大/rv_chamber_size、左室增大/lv_chamber_size、
    左室壁增厚/lv_wall_thickness、收缩活动减弱/rv_function、二尖瓣脱垂/mv_prolapse、
    瓣环钙化/mac_severity、间隔缺损/pfo_asd、双房增大(la&ra)、心包积液/pericardial_effusion。
    规则：severity>=1 视为阳性(mark=1,gpt=是)，==0 为阴性(mark=-1,gpt=否)。
  - **无法映射**或标签缺失的诊断项：`mark=0, gpt="否"`（占位，仅用于出预测，不能算指标）。
- 结果/指标（生成统计）：
  - study 7243（15 个无 exam.json），总记录 193579。
  - 有真实标签(可评估)的诊断项 14 项；其余 14 项 mark=0（占位）。
  - 各诊断项条数/阳性数见下方运行输出（如 三尖瓣反流 7114/阳性3516，左室壁增厚 7092/阳性3108）。
  - 心包积液、主动脉瓣反流、主动脉瓣钙化、左室增大 等在 MIMIC classification 中多为 null，暂无标签。
- 结论与备注：
  - 数据格式已与项目既有测试集对齐，可直接用于 `src/inference.py` 推理。
  - **映射表为初步方案，需用户确认**；确认后 14 个无标签项可调整（更换映射或改为纯预测占位）。
  - 下一步：待 GPU 空闲后指定卡号，先小样本冒烟推理，再全量。
- 查看命令：`tail -n 60 /home/mzhao/华为项目/VLM/EXPERIMENT_LOG.md`

## [2026-10-15 16:25] VLM(pvd_no_mv) 全诊断项分类评测

- 目的：在 `Data/pvd_no_mv/test.json` 测试集上，对 `Model/good_pvd_merged`（Qwen2.5-VL，无 LoRA，MODE=""）做**逐诊断项**二分类评测（是否存在），并给出指标与置信区间。
- 执行命令：
  - 路径重映射（数据搬迁）：把 test.json 的视频绝对路径 `/home/mzhao/ECHO_VIEW/Data/` → `/home/mzhao/华为项目/Data/`，输出 `out/pvd_no_mv_2026-10-15/input/test_remap.json`（23123 条，0 缺失）。
  - 推理：`CUDA_VISIBLE_DEVICES=2 python -u src/inference.py --model_path Model/good_pvd_merged --test_json out/pvd_no_mv_2026-10-15/input/test_remap.json --diag_item_config Data/diag_info.json --diag_item all --sample_size 300 --num_frames 16 --torch_dtype bfloat16 --save_results out/pvd_no_mv_2026-10-15`
  - 置信区间：`python src/置信区间计算.py --json_floder out/pvd_no_mv_2026-10-15 --output out/pvd_no_mv_2026-10-15_CI.csv --bootstrap 2000`
  - 画图：`python out/pvd_no_mv_2026-10-15/make_figs.py`
- 关键参数：greedy 解码，max_new_tokens=1，num_frames=16，bf16；随机种子 42；**每个 diag_item 最多随机取 300 条**；28 个诊断项独立评测。
- 输入：`Data/pvd_no_mv/test.json`（重映射后）；视频根目录现位于 `/home/mzhao/华为项目/Data/dicom_videos_group/Dataset`。
- 输出：`out/pvd_no_mv_2026-10-15/<诊断项>.json` ×28；`out/pvd_no_mv_2026-10-15_CI.csv`；`figs/{classification_metrics_by_diag,roc_curves,confusion_matrices,label_distribution}.png`；`out/pvd_no_mv_2026-10-15/config_snapshot.txt`。
- 结果/指标（8400 条 = 28 项 × 300；阳性 2249）：
  - **最优阈值(F1)下的平均值：Acc 0.8973 / Precision 0.7859 / Recall 0.7893 / F1 0.7683 / AUC 0.9196**（二分类，正类=阳性；指标为各诊断项的算术平均）。
  - 强项(AUC≥0.96)：二尖瓣狭窄 0.996、二尖瓣反流 0.984、双房增大 0.979、右室增大 0.975、三尖瓣反流 0.967、右房增大 0.968、人工主动脉瓣 0.965、肥厚型心肌病 0.966、主动脉瓣狭窄 0.964、主动脉瓣反流 0.958。
  - 弱项(AUC<0.90)：**心包积液 0.711**、**左室壁增厚 0.767**、二叶式主动脉瓣 0.812、主动脉窦部增宽 0.890。
  - 低F1主要受**类别极不平衡 + 阈值 0.5**影响：心包积液(F1 0.21)、左室壁增厚(F1 0.30) 在随机 300 条中阳性仅各 18/23 例；用最优阈值后 F1 明显改善（见 CI.csv）。
  - 与 10-03 旧结果(1000/项，旧路径)对比一致：旧 Acc 0.8983 / F1 0.7576 / AUC 0.9216 相近，说明搬迁前后结果可复现。
- 结论与备注：
  - 数据搬迁仅需重映射路径，模型/指标与旧结果一致；本次按"每项 300 条"完成全 28 项独立评测。
  - 心包积液、左室壁增厚 阳性样本过少，指标方差大，建议提高采样上限或做类别平衡采样后复评。
  - 查看命令：`tail -n 60 /home/mzhao/华为项目/VLM/EXPERIMENT_LOG.md`

## [2026-10-15 17:24] VLM(pvd_no_mv) 正负样本平衡评测

- 目的：针对上一轮随机采样导致的**类别高度不平衡**（如心包积液阳性率 6%、室壁瘤阳性仅 33 例），改用**每诊断项正负 1:1 平衡采样**重新评测，考察不平衡对指标的影响。
- 执行命令：
  - 构建平衡集：每项正/负各取 `k=min(阳性数, 阴性数, 150)`，输出 `out/pvd_no_mv_2026-10-15_balanced/input/test_balanced.json`（6346 条，seed=42；**每项 ≤300**）。
  - 推理：`CUDA_VISIBLE_DEVICES=2 python -u src/inference.py --model_path Model/good_pvd_merged --test_json out/pvd_no_mv_2026-10-15_balanced/input/test_balanced.json --diag_item_config Data/diag_info.json --diag_item all --sample_size 0 --num_frames 16 --torch_dtype bfloat16 --save_results out/pvd_no_mv_2026-10-15_balanced`
  - CI：`python src/置信区间计算.py --json_floder out/pvd_no_mv_2026-10-15_balanced --output out/pvd_no_mv_2026-10-15_balanced_CI.csv --bootstrap 2000`
  - 图表：`python out/pvd_no_mv_2026-10-15_balanced/make_figs.py`
- 关键参数：同上一轮（good_pvd_merged、无 LoRA、greedy、bf16、seed=42）；**每项总测评数 ≤300，正负各 ≤150**。
- 输入：`out/pvd_no_mv_2026-10-15_balanced/input/test_balanced.json`（正负平衡）。
- 输出：`out/pvd_no_mv_2026-10-15_balanced/<项>.json` ×28；`out/pvd_no_mv_2026-10-15_balanced_CI.csv` / `.xlsx`；`out/pvd_no_mv_2026-10-15_balanced_逐条预测.xlsx`（28 sheet+汇总）；`figs_balanced/*.png`。
- 结果/指标（最优阈值下，平均值）：
  - **平衡：Acc 0.8665 / Precision 0.8580 / Recall 0.8920 / F1 0.8719 / AUC 0.9237**
  - 对比不平衡（随机 300/项）：Acc 0.8973 / P 0.7859 / R 0.7893 / F1 0.7683 / AUC 0.9196
  - **F1 均值 +0.104（0.768→0.872）**，P/R 更均衡（不平衡时 P<R，平衡后近似相等），AUC 基本持平（0.9196→0.9237）。
  - F1 提升最大：心包积液 0.212→0.774、左室壁增厚 0.298→0.742、室壁瘤 0.621→0.901、二叶式主动脉瓣 0.592→0.800、心尖部增厚 0.667→0.872、人工支架 0.667→0.866。
  - 说明：不平衡随机采样下"准确率高"是被阴性主导的假象（模型偏向多数类），平衡后准确率下降属正常，F1/P/R 才反映真实判别力。
- 结论与备注：
  - **平衡采样显著改善 F1 与 P/R 平衡**，弱项（心包积液、左室壁增厚）从"几乎不可用"回升到 F1 0.74–0.77；AUC 排序基本稳定。
  - 心包积液 AUC 仅 0.785、左室壁增厚 0.745，仍是全项目最弱项，即使平衡后判别力仍有限（阳性固有难分）。
  - 查看命令：`tail -n 60 /home/mzhao/华为项目/VLM/EXPERIMENT_LOG.md`

## [2026-10-15 19:31] VLM(pvd_no_mv) 选择性预测（全量非平衡 test）

- 目的：在**完整非平衡测试集**上评估"选择性预测（带弃权）"：仅对高置信样本给出判断，如实报告覆盖率与准确率。**明确声明：这是选择性预测，不是模型整体性能，禁止只报准确率而不报覆盖率。**
- 说明（诚信）：曾评估"把平均准确率做到95%"的其他路径——非平衡真实标签下逐项最优阈值上限仅 87.0%（28项中仅1项≥95%），加大阴性比例反而更低（0.5阈值下）。唯一能达到≥95%的方式是按模型是否答对挑样本（cherry-picking＝伪造指标），已拒绝；改用正当的选择性预测。
- 执行命令：
  - 全量推理：`CUDA_VISIBLE_DEVICES=2 python -u src/inference.py --model_path Model/good_pvd_merged --test_json out/pvd_no_mv_2026-10-15/input/test_remap.json --diag_item_config Data/diag_info.json --diag_item all --sample_size 0 --num_frames 16 --torch_dtype bfloat16 --save_results out/pvd_no_mv_2026-10-15_full`
  - 选择性分析：`python out/pvd_no_mv_2026-10-15_full/make_selective.py`
- 关键参数：同前（good_pvd_merged、无 LoRA、greedy、bf16、seed=42）；置信度=首token 是/否 softmax；置信样本定义 `prob_y>=τ 或 prob_y<=1-τ`；预测按 0.5。
- 输入：完整 test（23123 条，自然非平衡）。
- 输出：`out/pvd_no_mv_2026-10-15_full/<项>.json` ×28；`selective_prediction.xlsx`（覆盖率-准确率曲线 sheet + 各诊断项@τ=0.95 sheet）；`input/selective_subset_tau0.950.json`（高置信子集1561条）；`figs_selective/selective_coverage_accuracy.png`。
- 结果/指标：
  - 全量基线(τ=0，0.5阈值)：宏平均 Acc 0.7737 / 池化 0.7403（=非平衡真实分布下的真实水平）。
  - 选定 τ*=0.95（满足宏平均≥95%的最大覆盖率）：**覆盖率 6.75%**（1561/23123），**宏平均 Acc 0.9544**，**池化 Acc 0.9058**。
  - 覆盖率-准确率曲线：τ=0.80→42.96%(0.878) / 0.85→27.60%(0.912) / 0.90→16.37%(0.934) / 0.925→12.19%(0.944) / 0.95→6.75%(0.954) / 0.975→3.35%(0.954) / 0.99→1.63%(1.000)。
  - 重要保留：τ=0.95 时 心包积液/心尖部增厚/起搏器 **无高置信样本**（被排除在宏平均外，宏平均实际基于25项）；宏平均受极小样本项(人工支架n=1、左室增大n=2、间隔缺损n=3)拉高；左室壁增厚高置信准确率仅0.192（自信出错）。
- 结论与备注：
  - **可作为"高置信子集上的≥95%"汇报，但必须同时给出覆盖率6.75%及"3项无样本"的说明**；池化准确率仅90.6%才是这批高置信样本的真实总体水平。
  - 模型置信度标定较差（约6%样本 prob 恰为0.5），且部分项"自信地错"，选择性预测在这套数据上覆盖代价很大。
  - 查看命令：`tail -n 60 /home/mzhao/华为项目/VLM/EXPERIMENT_LOG.md`
