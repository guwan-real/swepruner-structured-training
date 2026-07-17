from __future__ import annotations

import argparse
import json
from pathlib import Path

from .http_client import prune, request_json


DEFAULT_CODE = """def normalize(value):
    if value is None:
        return 0
    return int(value)

def unrelated_banner():
    return 'hello'
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test an official SWE-Pruner HTTP service")
    parser.add_argument("--url", default="http://127.0.0.1:8000/prune")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--query", default="Handle a missing value while normalizing input")
    parser.add_argument("--code-file")
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    health_url = args.url.rsplit("/", 1)[0] + "/health"
    health = request_json(health_url, timeout=args.timeout)
    code = Path(args.code_file).read_text(encoding="utf-8") if args.code_file else DEFAULT_CODE
    response = prune(args.url, args.query, code, args.threshold, args.timeout)
    required = {"pruned_code", "origin_token_cnt", "left_token_cnt", "model_input_token_cnt"}
    missing = sorted(required - set(response))
    if missing:
        raise RuntimeError(f"invalid prune response; missing fields: {missing}")
    origin = int(response["origin_token_cnt"])
    left = int(response["left_token_cnt"])
    if origin < 0 or left < 0 or left > origin:
        raise RuntimeError(f"invalid token counts: origin={origin}, left={left}")
    summary = {
        "health": health,
        "threshold": args.threshold,
        "origin_token_cnt": origin,
        "left_token_cnt": left,
        "keep_ratio": left / origin if origin else None,
        "kept_frags": response.get("kept_frags", []),
        "pruned_code": response["pruned_code"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
