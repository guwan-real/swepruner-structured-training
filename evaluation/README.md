# 将训练模型迁移到官方 SWE-Pruner 服务

本目录只解决一个问题：把训练生成的 `best_model.pt` 转成官方 SWE-Pruner 可加载的模型目录。现有 mini-swe-agent 不需要重写。

## 1. 为什么需要导出

我们的模型核心输入输出与官方 SWE-Pruner 一致，但 checkpoint 包装不同：

- 训练产物是包含 metadata 的 `best_model.pt`。
- 官方服务需要 `config.json + model.safetensors + tokenizer`。
- 我们的核心权重名是 `backbone.*`，官方文件中是 `model.backbone.*`。
- M2 的 `role_head` 和 `relation_head` 只用于训练，推理时应移除。

导出后仍由官方 `swe-pruner` 代码完成 tokenizer、分块、token 打分和行级剪枝。

## 2. 更新服务器仓库

\`\`\`bash
export WORK_DIR=/home/yuantao/futao/swepruner_workspace
export PROJECT_DIR=$WORK_DIR/swepruner-structured-training

cd "$PROJECT_DIR"
git pull origin main
\`\`\`

## 3. 导出一个可启动模型

先在已有训练环境中导出。正式模型使用验证集效果最好的 `best_model.pt`：

\`\`\`bash
cd "$PROJECT_DIR"
conda activate swepruner-train

mkdir -p "$WORK_DIR/eval_models"

python -m evaluation.export_official \\
  --checkpoint "$WORK_DIR/runs/m1_data_only_e5/best_model.pt" \\
  --tokenizer-dir training/offline_assets/code-pruner \\
  --backbone-config-dir training/offline_assets/qwen3-reranker-config \\
  --output-dir "$WORK_DIR/eval_models/m1_best" \\
  --label m1_best
\`\`\`

`--output-dir` 必须不存在或为空。成功后目录至少包含：

\`\`\`text
m1_best/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── backbone_config/
└── export_manifest.json
\`\`\`

M2 的命令完全相同，只需要替换 checkpoint、输出目录和 label。

## 4. 准备官方服务环境

官方 SWE-Pruner 要求 Python 3.12，建议与 Python 3.11 训练环境隔离。B200/CUDA 13.0：

\`\`\`bash
cd "$PROJECT_DIR"

MAX_JOBS=8 bash evaluation/scripts/create_eval_env.sh swepruner-eval
conda activate swepruner-eval

bash evaluation/scripts/prepare_official_tools.sh \\
  /home/yuantao/futao/swepruner_workspace/official_eval
\`\`\`

准备脚本固定官方 SWE-Pruner revision：

\`\`\`text
96171b5f3ecaf89745cbeb436c8893b57f3400bd
\`\`\`

官方模型使用 FlashAttention 2，环境脚本默认安装 `flash-attn`。

## 5. 启动并验证

下面在物理 GPU 6、端口 8000 启动刚导出的 M1：

\`\`\`bash
cd "$PROJECT_DIR"
conda activate swepruner-eval

bash evaluation/scripts/start_pruner.sh \\
  6 "$WORK_DIR/eval_models/m1_best" 8000 m1_best
\`\`\`

执行真实剪枝请求：

\`\`\`bash
python -m evaluation.prune_smoke \\
  --url http://127.0.0.1:8000/prune \\
  --threshold 0.5
\`\`\`

如果输出包含 `origin_token_cnt`、`left_token_cnt` 和 `pruned_code`，说明官方 loader、权重映射和输入输出链路均已跑通。

停止服务：

\`\`\`bash
bash evaluation/scripts/stop_pruner.sh m1_best
\`\`\`

服务日志位于：

\`\`\`text
evaluation/runtime/m1_best/service.log
\`\`\`

## 6. 接入现有 mini-swe-agent

不修改 mini-swe-agent 源码，只修改它现有配置中的 pruner 段：

\`\`\`yaml
agent:
  pruner:
    url: http://127.0.0.1:8000/prune
    threshold: 0.5
    timeout: 120
    retries: 3
    min_chars: 500
    chunk_overlap_tokens: 50
\`\`\`

A0、M1、M2 比较时分别启动对应模型目录，mini-swe-agent 使用同一份其余配置即可。

## 7. 当前验证边界

本地可以检查导出代码和脚本语法，但真实 `best_model.pt` 位于服务器，因此最终成功标准是在服务器上完成第 5 节的官方 loader 和 `/prune` smoke。只看到导出文件生成，不代表迁移已经验证完成。
