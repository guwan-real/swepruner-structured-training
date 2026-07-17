# SWE-Pruner Structured Training

本项目用于为 SWE-Pruner 压缩模型构造新的结构化训练数据，并在同一个官方初始化 checkpoint 上运行两组可公平比较的训练实验。

## 1. 项目目标

- 汇总 `SWE-Pruner official`、`SWE-Smith`、`SWE-Gym` 三类真实任务。
- 构造行级监督、代码块结构关系和候选排序数据。
- M1 只使用扩充后的行级数据，验证数据扩充本身的收益。
- M2 使用与 M1 相同的主数据和初始化，并额外加入结构关系与排序辅助损失。
- 固定数据划分、初始化、优化器和主任务设置，使 M1/M2 的差异集中在结构监督策略。

仓库内置数据集统计：

| 内容 | 数量 |
| --- | ---: |
| 主样本 | 2,001 |
| 训练集 | 1,566 |
| 验证集 | 207 |
| 测试集 | 228 |
| 结构关系 | 66,565 |
| 排序样本 | 728 |
| 结构关系类别 | 10 |

## 2. M1 与 M2

| 实验 | 主行级数据 | 结构关系损失 | 排序损失 | 用途 |
| --- | --- | --- | --- | --- |
| M1 | 使用 | 不使用 | 不使用 | 数据扩充基线 |
| M2 | 使用 | 使用 | 使用 | 结构监督方法 |

M1 和 M2 使用相同的数据划分、官方 SWE-Pruner 初始化、tokenizer、backbone 配置及主任务超参数。配置公平性筛查覆盖 24 个字段。

## 3. 项目结构

```text
swepruner-structured-training/
├── README.md
├── pyproject.toml
├── config/                         # 数据构建配置
├── scripts/                        # 数据构建与样本检查脚本
├── src/swepruner_dataset_builder/  # 数据集构建实现
├── tests/                          # 数据构建测试
├── training/
│   ├── assets/
│   │   └── swepruner_real_dataset_2k_seed42.tar.gz
│   ├── configs/
│   │   ├── m1_data_only.json
│   │   └── m2_structural.json
│   ├── scripts/
│   │   ├── create_server_conda.sh
│   │   ├── download_assets.sh
│   │   ├── download_wheelhouse.sh
│   │   ├── install_offline_conda.sh
│   │   ├── train_m1.sh
│   │   ├── train_m2.sh
│   │   └── unpack_dataset.sh
│   ├── checkpoint.py               # 官方 checkpoint/HF 权重加载
│   ├── data.py                     # 主任务、关系和排序数据读取
│   ├── losses.py                   # M1/M2 损失
│   ├── model.py                    # SWE-Pruner 兼容模型
│   ├── screen.py                   # 数据与 M1/M2 公平性筛查
│   └── train.py                    # torchrun 训练入口
└── artifacts/                      # 本地产物，默认不提交
```

## 4. 服务器目录约定

服务器基础目录固定为：

```text
/home/yuantao/futao
```

新建独立工作目录并 clone 私有仓库：

```bash
export BASE_DIR=/home/yuantao/futao
export WORK_DIR=$BASE_DIR/swepruner_workspace
export PROJECT_DIR=$WORK_DIR/swepruner-structured-training

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

git clone git@github.com:guwan-real/swepruner-structured-training.git
cd "$PROJECT_DIR"

mkdir -p "$WORK_DIR/logs" "$WORK_DIR/runs"
```

如果服务器没有配置 GitHub SSH key，可以使用已登录的 GitHub CLI：

```bash
cd /home/yuantao/futao/swepruner_workspace
gh repo clone guwan-real/swepruner-structured-training
cd swepruner-structured-training
```

## 5. 创建 conda 环境并在线安装依赖

当前服务器能够联网，只是不能从本机直接上传文件。因此推荐全部资源由服务器通过 GitHub、PyPI、PyTorch 和 Hugging Face 直接下载，不需要准备 wheelhouse。

B200 建议使用 Python 3.11 和 PyTorch CUDA 12.8 wheel：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -n swepruner-train python=3.11 pip
conda activate swepruner-train

python -m pip install --upgrade pip
python -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r training/requirements-training.txt
```

检查 B200 是否可用：

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))

assert torch.cuda.is_available(), "CUDA is unavailable"
PY
```

## 6. 模型资源目录

训练脚本默认使用以下离线目录：

```text
training/offline_assets/
├── code-pruner/                    # 官方 SWE-Pruner checkpoint 与 tokenizer
└── qwen3-reranker-config/          # backbone 配置
```

依赖安装完成后，直接在服务器下载模型资源：

```bash
export PROJECT_DIR=/home/yuantao/futao/swepruner_workspace/swepruner-structured-training

cd "$PROJECT_DIR"
conda activate swepruner-train

bash training/scripts/download_assets.sh
```

下载完成后，`train_m1.sh` 和 `train_m2.sh` 会自动使用上述默认目录，不需要从本机上传 checkpoint 或 tokenizer。

## 7. 解压和筛查数据

仓库已经包含 2K 数据包，clone 后不需要再次下载数据集：

```bash
export PROJECT_DIR=/home/yuantao/futao/swepruner_workspace/swepruner-structured-training

cd "$PROJECT_DIR"
conda activate swepruner-train

bash training/scripts/unpack_dataset.sh

python -m training.screen \
  --data-root training/data/upload_bundle_2k
```

预期核心结果：

```json
{
  "valid": true,
  "main_rows": 2001,
  "auxiliary_counts": {
    "relation": 66565,
    "ranking": 728
  },
  "m1_m2_parity_fields": 24
}
```

数据包 SHA-256：

```text
25b83b5bab239599aa8b49021260d24e4e11becacd10e6759f3ea25da60d26bf
```

## 8. 运行 M1

第一个参数是物理 GPU 编号列表。下面使用 B200 `0,1,2,3`：

```bash
export WORK_DIR=/home/yuantao/futao/swepruner_workspace
export PROJECT_DIR=$WORK_DIR/swepruner-structured-training

cd "$PROJECT_DIR"
conda activate swepruner-train

bash training/scripts/train_m1.sh 0,1,2,3 \
  --epochs 3 \
  --output-dir "$WORK_DIR/runs/m1_data_only"
```

## 9. 运行 M2

下面使用 B200 `4,5,6,7`：

```bash
export WORK_DIR=/home/yuantao/futao/swepruner_workspace
export PROJECT_DIR=$WORK_DIR/swepruner-structured-training

cd "$PROJECT_DIR"
conda activate swepruner-train

bash training/scripts/train_m2.sh 4,5,6,7 \
  --epochs 3 \
  --output-dir "$WORK_DIR/runs/m2_structural"
```

脚本会自动设置 `CUDA_VISIBLE_DEVICES`，并根据 GPU 列表长度设置 `torchrun --nproc_per_node`。

## 10. 在 8 张 GPU 上同时运行 M1 和 M2

只有显存、CPU、存储带宽足够时才建议并行运行：

```bash
export WORK_DIR=/home/yuantao/futao/swepruner_workspace
export PROJECT_DIR=$WORK_DIR/swepruner-structured-training

cd "$PROJECT_DIR"
conda activate swepruner-train

mkdir -p "$WORK_DIR/logs" "$WORK_DIR/runs"

nohup bash training/scripts/train_m1.sh 0,1,2,3 \
  --epochs 3 \
  --output-dir "$WORK_DIR/runs/m1_data_only" \
  > "$WORK_DIR/logs/m1_data_only.log" 2>&1 &
M1_PID=$!

nohup bash training/scripts/train_m2.sh 4,5,6,7 \
  --epochs 3 \
  --output-dir "$WORK_DIR/runs/m2_structural" \
  > "$WORK_DIR/logs/m2_structural.log" 2>&1 &
M2_PID=$!

echo "M1 PID=$M1_PID"
echo "M2 PID=$M2_PID"
```

查看日志：

```bash
tail -f /home/yuantao/futao/swepruner_workspace/logs/m1_data_only.log
tail -f /home/yuantao/futao/swepruner_workspace/logs/m2_structural.log
```

查看进程和 GPU：

```bash
ps -fp "$M1_PID" "$M2_PID"
watch -n 2 nvidia-smi
```

## 11. 恢复和自定义训练

额外训练参数可以直接放在 GPU 列表后面：

```bash
bash training/scripts/train_m2.sh 0,2,4,6 \
  --epochs 5 \
  --learning-rate 1e-5 \
  --output-dir /home/yuantao/futao/swepruner_workspace/runs/m2_custom
```

查看训练入口支持的完整参数：

```bash
python -m training.train --help
```

## 12. 运行前检查清单

- `nvidia-smi` 能看到计划使用的 B200。
- `torch.cuda.is_available()` 返回 `True`。
- `training/data/upload_bundle_2k` 已生成。
- `training/offline_assets/code-pruner` 已存在。
- `training/offline_assets/qwen3-reranker-config` 已存在。
- `python -m training.screen` 返回 `valid: true`。
- M1 和 M2 使用不同输出目录。
- 并行运行时 M1/M2 使用不重叠的物理 GPU。

## 13. 安全说明

仓库不包含 API Key。数据构建阶段如需调用 LLM API，请通过环境变量或服务器密钥管理系统注入，不要把密钥写入配置文件或提交到 Git。
