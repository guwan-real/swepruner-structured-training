# SWE-Pruner Structured Dataset Builder

这是一个面向 Python 3.11+ 的本地数据集构建工具。它把 SWE-smith、SWE-Gym 和可选的 SWE-Pruner 原始数据规范化为代码行级 keep/prune 数据，同时保留 `CORE`、`SUPPORT`、`DROP` 三级角色以及 block relation、ranking、置信度和来源证据。

## 数据来源与边界

- `swe_pruner_original` 保留官方自然语言 query、代码和原始标签。缺失时直接跳过，绝不使用 API 仿造。
- `swe_smith` 使用 buggy/base 仓库、mutation 或 patch、失败测试和 traceback 构造结构化标签。
- `swe_gym` 使用真实 issue 和 gold patch 校准合成 mutation 的分布偏差。
- Patch 只用于定位旧代码和生成标签。模型输入始终来自修复前仓库，不包含修复后完整代码或 patch 新增实现。
- 本项目不下载数据、不训练模型、不实现 GRPO，也不会自动安装依赖或运行目标仓库测试。

推荐训练权重为 `0.30 / 0.55 / 0.15`。构建器分别输出三个来源，不复制样本凑比例；服务器 DataLoader 决定实际采样权重。

## 标签与静态分析

`CORE` 是 patch 删除或修改的旧行、mutation 原始行、纯新增 hunk 的 buggy 上下文，以及最后一个项目 traceback frame 所在的最小完整 AST block。`SUPPORT` 是函数签名、局部 def-use、控制条件、异常配对、import、类型、decorator 和默认一跳 caller/callee。其余为 `DROP`。

二分类映射固定为 `CORE/SUPPORT -> keep=1`，`DROP -> keep=0`。冲突优先级为 `CORE > SUPPORT > DROP`，API 不能修改高置信度 CORE。

分析器基于标准库 `ast`，输出 module、class、function、if/else、loop、try/except/finally、with、match/case、return、raise 和连续语句 block。Python 动态 dispatch、反射和 monkey patch 只会产生低置信度近似证据，不会伪装成精确结果。

困难负样本使用 identifier/token Jaccard、相似函数名、同文件其他路径、同类其他方法和 issue 关键词相似度筛选。训练窗口来自同一个文件，边界与完整顶层 AST 单元对齐；跨文件关系只进入 relation/ranking metadata。

## 输入

```text
data_sources/
  swe_pruner/train.jsonl          # 可选
  swe_smith/tasks.jsonl
  swe_smith/repos/
  swe_gym/tasks.jsonl
  swe_gym/repos/
```

SWE-smith 和 SWE-Gym 最低需要 `task_id`、`repo_path`，以及 `patch` 或 `mutation_patch`。相对 `repo_path` 按 `tasks.jsonl` 所在目录解析。`inspect-input` 会在正式构建前列出无效任务。

## 快速开始

仓库根目录包含开发树 import shim，因此不安装也可直接运行：

以下示例中的 `python` 指 Python 3.11+。如果本机命令名不同，可使用 `python3.11`，或者运行脚本时设置 `PYTHON_BIN=/absolute/path/to/python3`；脚本会拒绝 Python 3.10 及以下版本。

```bash
python -m swepruner_dataset_builder init-config
python -m swepruner_dataset_builder inspect-input \
  --source swe_smith \
  --tasks data_sources/swe_smith/tasks.jsonl
```

真实第一阶段数据可由官方发布源直接准备并构建：

```bash
bash scripts/build_real_first_phase.sh
```

该脚本下载 SWE-Pruner 作者发布的 61k JSONL，并规范化前 100 条；从官方 `SWE-bench/SWE-smith-py` 按固定 seed、跨多个仓库抽取最多 100 个 Python 任务；从官方 `SWE-Gym/SWE-Gym` 抽取最多 20 个真实任务。SWE-smith checkout 会应用官方 bug-creation patch 得到 buggy worktree，再生成反向 diff 用于标注；SWE-Gym checkout 保持官方 `base_commit` 的 buggy 状态。原始候选、下载 SHA-256、checkout 失败和来源信息保存在 `data_sources/preparation_report.json`。

小规模离线构建：

```bash
python -m swepruner_dataset_builder build \
  --source swe_smith \
  --tasks data_sources/swe_smith/tasks.jsonl \
  --output artifacts/swe_smith \
  --config config/default.toml \
  --seed 42 \
  --task-limit 100 \
  --num-workers 4 \
  --offline \
  --resume
```

SWE-Gym 使用同一命令并把 source 改为 `swe_gym`。SWE-Pruner 原始数据使用 `swe_pruner_original`，只做规范化，不重新生成标签。

## API 配置与双 key 切换

API 是 OpenAI-compatible `/chat/completions` 接口，只审核低置信度 SUPPORT/DROP、静态伪相关、困难负样本和缺失 issue。所有请求写入 `api_cache.sqlite`，失败时保留静态标签并降低置信度。

```bash
export LLM_BASE_URL='https://api.llm.ustc.edu.cn/v1'
export LLM_API_KEYS='first-key,second-key'
export LLM_MODEL_PRIMARY='qwen-chat'
export LLM_MODEL_REVIEWER='deepseek-chat'
```

也支持单 key 的 `LLM_API_KEY`。遇到 `401`、`403` 或 `429` 时自动切换到下一个 key。Key 不会写入代码、配置、日志、SQLite 或 JSONL。启用 API：

```bash
python -m swepruner_dataset_builder build \
  --source swe_smith \
  --tasks data_sources/swe_smith/tasks.jsonl \
  --output artifacts/swe_smith \
  --config config/default.toml \
  --use-api \
  --resume
```

没有 tokenizer 时使用保守字符近似。已有离线 tokenizer 可通过 `--tokenizer-path /absolute/local/path` 指定；不会联网下载。

## 验证、报告和导出

```bash
python -m swepruner_dataset_builder validate \
  --dataset artifacts/swe_smith

python -m swepruner_dataset_builder report \
  --artifact-dir artifacts/swe_smith

python -m swepruner_dataset_builder export-swepruner \
  --input artifacts/swe_smith/pruning_sft.jsonl \
  --mapping config/swepruner_mapping.json \
  --output artifacts/swe_smith/swe_pruner_compatible.jsonl

python -m swepruner_dataset_builder export-swepruner-official \
  --input artifacts/swe_smith/pruning_sft.jsonl \
  --output artifacts/swe_smith/swe_pruner_official_format.jsonl

python -m swepruner_dataset_builder create-manifest \
  --artifacts-root artifacts \
  --output artifacts/combined_manifest.json
```

字段 mapping 只重命名核心数据字段。拿到服务器 SWE-Pruner loader 后，只需调整 `config/swepruner_mapping.json`，无需重建标签。

## 断点续跑与产物

每个任务的确定性输出保存在来源目录的 `.state/`，状态 key 包含配置 fingerprint、builder version、seed 和 tokenizer。`--resume` 复用相同版本结果；API 请求另由 SQLite 去重。单任务失败写入 `failed_tasks.jsonl`，不会终止其他任务。

`create-manifest` 按 repository 切分，并用去注释 token hash、标识符规范化 hash 和 AST 结构 hash 合并跨仓库近重复组。它同时生成固定 seed 的 20 条人工抽查样本：

```text
artifacts/samples_for_review/review_samples.jsonl
artifacts/samples_for_review/review_samples.md
```

Markdown 只展示 buggy 代码和 patch 旧位置摘要，不展示正确新增代码。

## 测试与 fixture 闭环

测试不会访问真实网络，API 场景全部使用 `FakeProvider`：

```bash
python -m unittest discover -s tests -v
bash scripts/build_small.sh
```

当前仓库没有真实数据时，`build_small.sh` 把测试产物写到 `artifacts/fixture_demo/`，避免误当成训练数据。它覆盖普通函数、类方法、纯新增 patch、控制和异常关系、caller/callee、def-use、相似名称 hard negative、traceback、issue 生成、双模型仲裁、缓存、repo split、近重复、泄漏检查和 SWE-Pruner 导出。

## 上传训练服务器

上传真实构建产生的各来源 `pruning_sft.jsonl`、可选的 `block_relation.jsonl` 和 `block_ranking.jsonl`、`combined_manifest.json`、`splits/`、`report.json`、字段 mapping 及 builder 配置。`api_cache.sqlite` 和 `.state/` 只在需要继续构建或审计 API 时上传，不是模型训练必需文件。

服务器端仍需确认 SWE-Pruner loader 的 query、document、line labels、document label 字段名，行分隔方式，以及 Qwen3-Reranker tokenizer 对特殊 token 的包装规则。
