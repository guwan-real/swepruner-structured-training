# M1/M2 offline training

This directory implements the two controlled experiments described in the experiment plan.

| Strategy | Initialization | Main loss | Structural heads | Auxiliary files |
|---|---|---|---|---|
| M1 `data_only` | official SWE-Pruner | `L_keep + 0.05 L_document` | none | none |
| M2 `structural_pruner` | same official SWE-Pruner | `L_keep + 0.25 L_role + 0.10 L_relation + 0.10 L_rank + 0.05 L_document` | role + relation | relation + ranking |

Both configs use seed 42, the same repository-aware splits, 1,566 main samples per epoch, 80% new-data sampling, 20% official replay, three epochs, the same maximum length and the same number of optimizer steps. Official replay rows never receive fabricated role or relation labels.

## Objective ablations

The implementation distinguishes independent parameter heads from training objectives:

- `keep` uses the CRF compression head and is always enabled.
- `role` has an independent role classification head.
- `relation` has an independent relation head and uses line plus auxiliary block-relation supervision.
- `rank` is a margin objective over the shared yes/no document score; it has no independent rank head.
- `document` is BCE over the same yes/no document score; it has no independent document head.

Disabled role/relation heads are not created. Disabled relation/rank objectives do not load their auxiliary batches or run their extra forward passes.

Recommended four-run screen on physical GPUs 6 and 7:

```bash
export WORK_DIR=/home/yuantao/futao/swepruner_workspace

for PRESET in b0 b1 b2 b3; do
  bash training/scripts/train_ablation.sh "$PRESET" 6,7 \
    --set epochs=5 \
    --output-dir "$WORK_DIR/runs/ablation_${PRESET}"
done
```

| Preset | Active objectives |
|---|---|
| `b0` | keep |
| `b1` | keep + relation |
| `b2` | keep + relation + role |
| `b3` | keep + relation + role + rank + document |

Additional single-objective and leave-one-out presets are available through:

```bash
bash training/scripts/train_ablation.sh --help
```

Every run records `active_objectives`, raw losses, parameter-group gradient norms, and the existing threshold curve. It also reports nearest-grid `core_recall`, actual retention and threshold at target keep ratios 50% and 55%. Rank/document do not have separate parameter groups, so their gradients appear in the shared backbone/fusion norms rather than a fictitious head norm.

## Full-backbone continued-training comparison

The normal M1/M2 and ablation launchers keep `backbone_training_mode=last_n`, which freezes the backbone except for the final two transformer layers. A separate launcher tests full-parameter continued training without changing those existing commands:

```bash
export WORK_DIR=/home/yuantao/futao/swepruner_workspace

for PRESET in b0 b1 b2 b3; do
  bash training/scripts/train_full_backbone_ablation.sh "$PRESET" 4,5 \
    --set epochs=5 \
    --set per_device_batch_size=1 \
    --set gradient_accumulation_steps=2 \
    --output-dir "$WORK_DIR/runs/full_backbone_${PRESET}"
done
```

Both paths initialize from the same official SWE-Pruner checkpoint. The controlled difference is:

| Path | Backbone parameters trained | Effective batch in the two-GPU examples |
|---|---|---:|
| `train_ablation.sh` | final 2 transformer layers | `2 GPUs x batch 2 x accumulation 1 = 4` |
| `train_full_backbone_ablation.sh` | entire backbone | `2 GPUs x batch 1 x accumulation 2 = 4` |

The full-backbone launcher uses a separate default output root, `training_outputs/full_backbone_ablations/`. On B200 it disables gradient checkpointing by default for higher throughput. If explicitly enabled for a smaller GPU, the implementation uses non-reentrant checkpointing so relation/rank auxiliary forwards remain compatible with DDP. It is continued training from the official model, not random initialization from scratch. With only 2K examples, random initialization would test data insufficiency more than the value of the structured objectives.

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
bash training/scripts/train_m1.sh 0,1
bash training/scripts/train_m2.sh 0,1
```

Four GPUs:

```bash
bash training/scripts/train_m1.sh 0,1,2,3
bash training/scripts/train_m2.sh 0,1,2,3
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
bash training/scripts/train_m2.sh 0,2,4,6 --set epochs=3 --output-dir training_outputs/m2-b200
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
