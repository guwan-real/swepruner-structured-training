# M1/M2 offline training

This directory implements the two controlled experiments described in the experiment plan.

| Strategy | Initialization | Main loss | Structural heads | Auxiliary files |
|---|---|---|---|---|
| M1 `data_only` | official SWE-Pruner | `L_keep + 0.05 L_document` | none | none |
| M2 `structural_pruner` | same official SWE-Pruner | `L_keep + 0.25 L_role + 0.10 L_relation + 0.10 L_rank + 0.05 L_document` | role + relation | relation + ranking |

Both configs use seed 42, the same repository-aware splits, 1,566 main samples per epoch, 80% new-data sampling, 20% official replay, three epochs, the same maximum length and the same number of optimizer steps. Official replay rows never receive fabricated role or relation labels.

The model core follows the public SWE-Pruner `TokenScorer`: Qwen3-Reranker-0.6B backbone, early/middle/final hidden-state concatenation, one fusion-attention layer, CRF keep/prune head and yes/no document scorer. The loader accepts the official `best_model.pt` or Hugging Face `model.safetensors`; M2 initializes only its new role/relation heads randomly.

## 1. Dataset included in Git

The validated 2,001-row dataset archive is stored at:

```text
training/assets/swepruner_real_dataset_2k_seed42.tar.gz
```

After cloning:

```bash
bash training/scripts/unpack_dataset.sh
python -m training.screen --data-root training/data/upload_bundle_2k
```

## 2. Prepare packages for an offline conda server

The B200 server cannot download from the public internet. Run the following on an internet-connected **Linux x86_64 machine with the same Python version**. If the server has an internal mirror, set `PIP_INDEX_URL`, `TORCH_INDEX_URL` and `HF_ENDPOINT` and run there directly.

```bash
conda activate YOUR_ENV
bash training/scripts/download_wheelhouse.sh training/wheelhouse
bash training/scripts/download_assets.sh training/offline_assets
tar -czf swepruner_offline_assets.tar.gz training/wheelhouse training/offline_assets
```

Transfer/extract that archive beside this repository on the server, then install without accessing an index:

```bash
bash training/scripts/install_offline_conda.sh YOUR_ENV training/wheelhouse
```

`download_assets.sh` downloads only one full model, `ayanami-kitasan/code-pruner`, plus the small Qwen3-Reranker backbone `config.json`. The official checkpoint already contains the backbone weights, so a second 0.6B weight copy is unnecessary.

## 3. B200 commands

Two GPUs, recommended first run:

```bash
conda activate YOUR_ENV
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
bash training/scripts/train_m1.sh 2
bash training/scripts/train_m2.sh 2
```

Four GPUs:

```bash
bash training/scripts/train_m1.sh 4
bash training/scripts/train_m2.sh 4
```

Override paths when assets live elsewhere:

```bash
export DATA_ROOT=/data/upload_bundle_2k
export TOKENIZER_PATH=/models/code-pruner
export INIT_CHECKPOINT=/models/code-pruner
export BACKBONE_PATH=/models/qwen3-reranker-config
export OUTPUT_DIR=/checkpoints/m2_structural
bash training/scripts/train_m2.sh 2
```

Pure-new-data ablation:

```bash
bash training/scripts/train_m2.sh 2 --set replay_ratio=0.0
```

The default uses PyTorch SDPA, avoiding a server-side `flash-attn` build. B200 runs in BF16. Start with per-device batch size 2; if memory is comfortable, use the same override for both strategies:

```bash
bash training/scripts/train_m1.sh 2 --set per_device_batch_size=4 --set gradient_accumulation_steps=2
bash training/scripts/train_m2.sh 2 --set per_device_batch_size=4 --set gradient_accumulation_steps=2
```

Outputs include `best_model.pt`, `last_model.pt`, `run_manifest.json`, `metrics.jsonl`, `best_metrics.json` and a threshold curve for comparing keep ratio at matched CORE recall.

## Offline guarantees

- All model/tokenizer loads set `local_files_only=True`.
- Training never calls an API or Hugging Face Hub.
- The install script uses `pip --no-index`.
- The dataset archive has a fixed SHA256 check before extraction.
- The screen command checks source counts, splits, label alignment, auxiliary references, API-key patterns and M1/M2 parity.

Architecture compatibility was checked against official SWE-Pruner commit `96171b5f3ecaf89745cbeb436c8893b57f3400bd` (MIT license).
## 指定物理 GPU

两个启动脚本的第一个参数都是逗号分隔的物理 GPU 编号。脚本会自动设置
`CUDA_VISIBLE_DEVICES`，并按 GPU 数量设置 `torchrun --nproc_per_node`：

```bash
# 使用物理 GPU 0 和 1
bash training/scripts/train_m1.sh 0,1

# 使用物理 GPU 2、3、6、7
bash training/scripts/train_m2.sh 2,3,6,7

# 也可以通过环境变量指定，未指定时默认使用 0,1
GPU_IDS=4,5 bash training/scripts/train_m1.sh
```

GPU 参数后仍可追加训练参数，例如：

```bash
bash training/scripts/train_m2.sh 0,2,4,6 --epochs 3 --output-dir training_outputs/m2-b200
```

## 服务器新建 conda 环境

服务器能访问已配置的 conda channel 时，可以直接创建环境，再从本地 wheelhouse
离线安装 Python 包：

```bash
bash training/scripts/create_server_conda.sh \
  swepruner-train \
  "$PWD/training/wheelhouse"
```

服务器完全离线时，推荐克隆服务器上已有的 Python/CUDA 环境，第三个参数是源环境名：

```bash
bash training/scripts/create_server_conda.sh \
  swepruner-train \
  "$PWD/training/wheelhouse" \
  base
```

如果 conda 的 Python 和 pip 包已经在本机缓存，也可以强制离线创建：

```bash
CONDA_OFFLINE=1 bash training/scripts/create_server_conda.sh \
  swepruner-train \
  "$PWD/training/wheelhouse"
```

创建完成后执行：

```bash
conda activate swepruner-train
bash training/scripts/unpack_dataset.sh
python -m training.screen --data-root training/data/swepruner_real_dataset_2k_seed42
```
