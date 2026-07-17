from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 120.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"cannot reach {url}: {error.reason}") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"expected a JSON object from {url}")
    return result


def prune(
    url: str,
    query: str,
    code: str,
    threshold: float,
    timeout: float,
    chunk_overlap_tokens: int = 50,
) -> dict[str, Any]:
    return request_json(
        url,
        {
            "query": query,
            "code": code,
            "threshold": threshold,
            "always_keep_first_frags": False,
            "chunk_overlap_tokens": chunk_overlap_tokens,
        },
        timeout,
    )
