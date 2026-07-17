from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, write_default_config
from .exporters import export_swepruner, export_swepruner_official
from .pipeline import build_dataset, create_manifest
from .real_data import prepare_real_data
from .reporting import build_report
from .task_adapters import inspect_tasks
from .validation import validate_artifact_dir, validate_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swepruner_dataset_builder")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-config")
    init.add_argument("--directory", default="config")
    inspect = sub.add_parser("inspect-input")
    inspect.add_argument("--source", required=True, choices=["swe_smith", "swe_gym"])
    inspect.add_argument("--tasks", required=True)
    build = sub.add_parser("build")
    build.add_argument("--source", required=True, choices=["swe_smith", "swe_gym", "swe_pruner_original"])
    build.add_argument("--tasks", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--config", default="config/default.toml")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--task-limit", type=int)
    build.add_argument("--num-workers", type=int, default=1)
    build.add_argument("--resume", action="store_true")
    build.add_argument("--use-api", action="store_true")
    build.add_argument("--offline", action="store_true")
    build.add_argument("--tokenizer-path")
    validate = sub.add_parser("validate")
    validate.add_argument("--dataset", required=True)
    report = sub.add_parser("report")
    report.add_argument("--artifact-dir", required=True)
    export = sub.add_parser("export-swepruner")
    export.add_argument("--input", required=True)
    export.add_argument("--mapping", required=True)
    export.add_argument("--output", required=True)
    official_export = sub.add_parser("export-swepruner-official")
    official_export.add_argument("--input", required=True)
    official_export.add_argument("--output", required=True)
    manifest = sub.add_parser("create-manifest")
    manifest.add_argument("--artifacts-root", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--config", default="config/default.toml")
    manifest.add_argument("--seed", type=int, default=42)
    prepare = sub.add_parser("prepare-real-data")
    prepare.add_argument("--root", default="data_sources")
    prepare.add_argument("--smith-limit", type=int, default=100)
    prepare.add_argument("--gym-limit", type=int, default=20)
    prepare.add_argument("--pruner-limit", type=int, default=100)
    prepare.add_argument("--max-repos-per-source", type=int, default=5)
    prepare.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init-config":
        result = {"written": write_default_config(args.directory)}
    elif args.command == "inspect-input":
        result = inspect_tasks(args.source, args.tasks)
    elif args.command == "build":
        if args.use_api and args.offline:
            raise SystemExit("--use-api and --offline cannot be used together")
        result = build_dataset(args.source, args.tasks, args.output, load_config(args.config), args.seed,
                               args.task_limit, args.num_workers, args.resume, args.use_api, args.offline,
                               args.tokenizer_path)
    elif args.command == "validate":
        target = Path(args.dataset)
        result = validate_artifact_dir(target) if target.is_dir() else validate_dataset(target)
    elif args.command == "report":
        result = build_report(args.artifact_dir)
    elif args.command == "export-swepruner":
        result = export_swepruner(args.input, args.mapping, args.output)
    elif args.command == "export-swepruner-official":
        result = export_swepruner_official(args.input, args.output)
    elif args.command == "create-manifest":
        result = create_manifest(args.artifacts_root, args.output, load_config(args.config), args.seed)
    elif args.command == "prepare-real-data":
        result = prepare_real_data(args.root, args.smith_limit, args.gym_limit, args.pruner_limit,
                                   args.seed, args.max_repos_per_source)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("valid", True) else 1
