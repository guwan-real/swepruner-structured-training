from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from .io_utils import stable_hash


DEFAULT_TOML = '''builder_version = "0.1.0"
confidence_file = "confidence.toml"

[analysis]
max_call_hops = 1
max_repo_files = 2000
include_tests = true
dynamic_call_confidence = 0.45

[budget]
max_lines = 240
max_chars = 30000
max_input_tokens = 7500
min_core_lines = 1
max_support_ratio = 0.65
max_drop_ratio = 0.85
hard_negative_count = 4

[sampling]
max_samples_per_task = 8
document_negative_ratio = 0.10
easy_drop_count = 2
review_sample_count = 20

[api]
base_url = "https://api.llm.ustc.edu.cn/v1"
model_primary = "qwen-chat"
model_reviewer = "deepseek-chat"
prompt_version = "v1"
temperature = 0.0
max_tokens = 500
timeout_seconds = 60
retry_count = 2
local_skip_threshold = 0.90
primary_accept_threshold = 0.85
review_lower_threshold = 0.45
max_candidates_per_task = 8

[split]
train_ratio = 0.80
validation_ratio = 0.10
test_ratio = 0.10
'''

CONFIDENCE_TOML = '''[confidence]
patch_deleted_or_modified_line = 1.00
mutation_original_line = 1.00
traceback_final_project_frame = 0.95
patch_addition_nearest_context = 0.90
direct_control_dependency = 0.88
direct_local_def_use = 0.88
exception_pair = 0.85
function_signature = 0.85
dynamic_traceback_neighbor = 0.85
import_dependency = 0.78
type_dependency = 0.78
attribute_dependency = 0.75
one_hop_direct_call = 0.72
one_hop_caller = 0.68
inheritance_or_override = 0.65
lexical_hard_negative = 0.75
easy_negative = 0.70
api_single_model = 0.60
api_two_model_agreement = 0.82
'''

MAPPING = {
    "query": "query",
    "code": "document",
    "line_keep_labels": "labels",
    "document_label": "document_label",
}


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    config = tomllib.loads(source.read_text(encoding="utf-8"))
    confidence_path = source.parent / str(config.get("confidence_file", "confidence.toml"))
    confidence = tomllib.loads(confidence_path.read_text(encoding="utf-8"))
    config["confidence"] = confidence.get("confidence", {})
    config["config_path"] = str(source)
    config["fingerprint"] = stable_hash(config)
    return config


def write_default_config(directory: str | Path) -> list[str]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    files = {
        "default.toml": DEFAULT_TOML,
        "confidence.toml": CONFIDENCE_TOML,
        "swepruner_mapping.json": json.dumps(MAPPING, indent=2) + "\n",
        "eval_repo_blacklist.txt": "# One repository name per line.\n",
    }
    written: list[str] = []
    for name, content in files.items():
        path = target / name
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            written.append(str(path))
    return written


def load_blacklist(path: str | Path | None) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    return {
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

