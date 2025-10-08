# GLoCIM‑simple

> 以最少依赖、清晰结构复现 **GLoCIM**（Global‑view Long‑Chain Interest Modeling）在 MIND 系列数据集上的训练/评测流程，支持 Docker 或本地 Conda 环境，一键脚本与可复现配置。

---

## 目录

* [简介](#简介)
* [仓库结构](#仓库结构)
* [环境与依赖](#环境与依赖)

  * [方式 A：Conda 本地环境](#方式-a-conda-本地环境)
  * [方式 B：Docker 环境](#方式-b-docker-环境)
* [数据准备](#数据准备)
* [训练与评测](#训练与评测)

  * [训练](#训练)
  * [验证 / 测试评测](#验证--测试评测)
  * [预测与导出](#预测与导出)
* [可复现实验与期望指标](#可复现实验与期望指标)
* [日志、断点与输出](#日志断点与输出)
* [常见问题（FAQ/故障排查）](#常见问题faq故障排查)
* [引用 / Citation](#引用--citation)
* [许可证](#许可证)

---

## 简介

**GLoCIM** 通过在全局点击图上选择并编码“长链兴趣”（long‑chain interests），并与邻居兴趣进行门控融合，提升新闻推荐排序表现。相较仅建模局部子图的方法，GLoCIM 更好地捕捉远距离关联，同时维持可控的计算与显存开销。

> 论文要点（简述）：
>
> * **长链选择算法**：结合热度（累积点击频次）、语义与类别，动态为每条已点新闻选择价值最高的长链；
> * **长链兴趣编码器**：分别在新闻级、用户级进行注意力聚合得到全局视角的长链兴趣；
> * **协同兴趣融合**：通过门控单元将长链兴趣与邻居兴趣进行协同建模；
> * **实验**：在 MIND‑small / MIND‑large 上多个指标取得领先。

> **温馨提示**：本仓库为“simple”版本，注重流程清晰与复现友好；如需更完整的工程化/打包分发方案，可结合 Docker 镜像与自解压包的工作流使用。

---

## 仓库结构

```
GLoCIM_simple/
├─ src/                     # 模型、数据管道、图构建与长链选择等核心代码
├─ configs/                 # Hydra/YAML 配置（如 small.yaml 等）
├─ scripts/                 # 一键脚本（训练/评测/数据获取）
│  ├─ train.sh
│  ├─ eval_test.sh
│  └─ fetch_release_data.sh # 可选：从 Release 或本地源拉取数据/词向量
├─ data/                    # 默认数据根目录（MINDsmall, MINDsmall_bots_10pct 等）
├─ checkpoint/              # 训练断点与已发布 ckpt 存放处
├─ logs/                    # 训练/评测日志
├─ pred/                    # 预测输出（提交或分析用）
├─ requirements.txt         # Python 依赖
├─ Dockerfile               # （可选）CUDA + PyTorch 运行镜像
└─ README.md                # 当前文档
```

---

## 环境与依赖

### 方式 A：Conda 本地环境

```bash
# 1) 创建环境
conda create -n glocim python=3.10 -y
conda activate glocim

# 2) 安装依赖
pip install -r requirements.txt

# 3)（可选）验证 PyTorch/CUDA
python - <<'PY'
import torch
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
PY
```

### 方式 B：Docker 环境

> 要求：主机已安装 **Docker** 与 **NVIDIA Container Toolkit**，`nvidia-smi` 可用。

**构建镜像**（示例标签自定）：

```bash
docker build -t glocim:cu121-v1 .
```

**运行容器**（挂载数据、日志、断点等目录）：

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=16g \
  -e WANDB_MODE=offline -e HYDRA_FULL_ERROR=1 \
  -e PROJECT_ROOT=/opt/app -e PYTHONPATH=/opt/app/src \
  -v $PWD/data:/opt/app/data \
  -v $PWD/checkpoint:/opt/app/checkpoint \
  -v $PWD/logs:/opt/app/logs \
  -v $PWD/pred:/opt/app/pred \
  glocim:cu121-v1 bash
```

进入容器后在 `/opt/app` 内执行与本地一致的训练/评测命令即可。

---

## 数据准备

> 默认路径约定：`data/` 作为数据根；不同数据集分别存放在 `data/<DATASET_NAME>` 下。

### 选项 1：使用脚本（如已提供）

```bash
# 仅示例：按需修改脚本中的下载/解压来源
bash scripts/fetch_release_data.sh MINDsmall
bash scripts/fetch_release_data.sh MINDsmall_bots_10pct  # 如需
```

### 选项 2：手动放置

将 MIND‑small（或自定义增强版，如 `MINDsmall_bots_10pct`）的 `news.tsv / behaviors.tsv / entity_embed.vec` 等文件整理为：

```
data/
├─ MINDsmall/
│  ├─ train/behaviors.tsv
│  ├─ train/news.tsv
│  ├─ val/behaviors.tsv
│  ├─ val/news.tsv
│  ├─ test/behaviors.tsv
│  ├─ test/news.tsv
│  └─ entity_embed.vec  # 若使用实体向量
└─ MINDsmall_bots_10pct/  # 可选
   └─ ...
```

> 词向量：若使用 GloVe/TransE，可放入 `data/embeddings/` 或脚本默认位置（详见配置）。

---

## 训练与评测

在仓库根目录：

```bash
export PROJECT_ROOT=$PWD
export PYTHONPATH=$PWD/src
export WANDB_MODE=offline   # 无外网时建议
```

### 训练

**示例：在 MINDsmall 上训练（small 配置）**

```bash
python -u scripts/train.py \
  --config-name small \
  dataset.dataset_name=MINDsmall \
  dataset.dataset_dir=$PWD/data/MINDsmall \
  train.seed=2025 \
  train.num_workers=4
```

常用开关（按需在 CLI 覆盖或在 configs 内设置）：

* `val_embed_mode=recompute`：验证阶段重编码新闻向量（更一致的评估）
* `train.amp=true/false`：是否启用混合精度
* `model.long_chain.len=8`、`model.long_chain.branches=3`：长链长度与每跳分支数

### 验证 / 测试评测

如需加载指定断点进行 **测试集评测**：

```bash
export EVAL_CKPT=checkpoint/MINDsmall/best_auc.pth   # 或实际 ckpt 路径
python -u scripts/eval_test.py \
  --config-name small \
  dataset.dataset_name=MINDsmall \
  dataset.dataset_dir=$PWD/data/MINDsmall \
  val_embed_mode=recompute val_sample_ratio=1.0 val_amp=false
```

> 说明：若提供 `EVAL_CKPT` / `CKPT_PATH` 环境变量，脚本将优先使用该断点。

### 预测与导出

```bash
python -u scripts/predict.py \
  --config-name small \
  dataset.dataset_name=MINDsmall \
  dataset.dataset_dir=$PWD/data/MINDsmall \
  output_dir=$PWD/pred/MINDsmall
```

输出文件默认位于 `pred/<DATASET_NAME>/`，便于提交或下游分析。

---

## 可复现实验与期望指标

> 以下数值给出“量级参考”，不同硬件/随机种子/实现细节可能有±微小差异。

**MIND‑small（GLoCIM，full method）**：

* AUC ≈ **0.682**
* MRR ≈ **0.330**
* nDCG@5 ≈ **0.367**
* nDCG@10 ≈ **0.428**

**建议**：

* 验证使用 `val_embed_mode=recompute`，避免旧向量导致评估偏差；
* 长链长度 `len≈8` 常较优，过短难以提取远距离兴趣、过长易引入噪声；
* 训练期间按若干步周期**重新选择长链**，使选择与表示学习协同演化。

---

## 日志、断点与输出

* **logs/**：`*.log` 训练/评测日志（可配合 `tee` 保存）；
* **checkpoint/**：按数据集与配置分目录保存 `*.pth`；
* **pred/**：推理/提交结果；
* **wandb/**（可选）：如连网可开启在线追踪；默认 `WANDB_MODE=offline`。

---

## 常见问题（FAQ/故障排查）

**Q1：Docker 报错 `nvidia-container-cli: initialization error: load library failed: libnvidia-...`**

* 确认主机 `nvidia-smi` 正常；
* 已安装并配置 **NVIDIA Container Toolkit**；
* 使用 `--gpus all`（或 `--device` 指定 GPU）与 `--ipc=host --shm-size=16g`；
* 驱动/运行时版本不匹配也会导致该错误，必要时升级驱动或重新安装 nvidia‑ctk。

**Q2：显存不足 / 速度慢**

* 适当降低 batch size、长链长度 `len` 或分支 `branches`；
* 关闭部分特征（如 `val_amp=false` 改为 true，或在训练启用 `amp`）。

**Q3：验证指标与理论不符**

* 确保验证阶段重编码（`val_embed_mode=recompute`）；
* 检查数据划分路径、候选采样与负采样设置是否与训练一致。

---

## 引用 / Citation

若本项目对您的研究/产品有帮助，请引用：

```
@inproceedings{GLoCIM2025,
  title     = {GLoCIM: Global-view Long Chain Interest Modeling for News Recommendation},
  author    = {Yang, Zhen and Wang, Wenhui and Qi, Tao and Zhang, Peng and Zhang, Tianyun and Ru, Zhang and Liu, Jianyi and Huang, Yongfeng},
  booktitle = {Proceedings of the 31st International Conference on Computational Linguistics},
  year      = {2025}
}
```

---

## 许可证

本仓库（simple 版本）默认采用 MIT 许可证（如需变更请在根目录 `LICENSE` 中注明）。

---

### 联系方式 / 贡献

* 欢迎通过 Issue/PR 反馈问题与提交改进建议；
* 如需集成到自解压包或更复杂的分发方案，请在 Issue 说明您的目标环境（操作系统、驱动版本、是否离线等）。
