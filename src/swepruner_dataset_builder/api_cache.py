from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class ApiCache:
    def __init__(self, path: str | Path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(destination, check_same_thread=False)
        self.lock = threading.Lock()
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS api_cache (
                request_hash TEXT PRIMARY KEY,
                base_url TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                prompt TEXT NOT NULL,
                raw_response TEXT,
                parsed_response TEXT,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL,
                error TEXT
            )"""
        )
        self.connection.commit()
        self.hits = 0
        self.misses = 0

    def get(self, request_hash: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT parsed_response, raw_response, input_tokens, output_tokens, status, error FROM api_cache WHERE request_hash = ?",
                (request_hash,),
            ).fetchone()
        if not row:
            self.misses += 1
            return None
        self.hits += 1
        return {
            "parsed": json.loads(row[0]) if row[0] else None,
            "raw": row[1] or "",
            "input_tokens": row[2] or 0,
            "output_tokens": row[3] or 0,
            "status": row[4],
            "error": row[5] or "",
            "cached": True,
        }

    def put(
        self,
        request_hash: str,
        base_url: str,
        model: str,
        prompt_version: str,
        prompt: str,
        raw_response: str,
        parsed_response: dict[str, Any] | None,
        input_tokens: int,
        output_tokens: int,
        status: str,
        error: str = "",
    ) -> None:
        with self.lock:
            self.connection.execute(
                """INSERT OR REPLACE INTO api_cache
                (request_hash, base_url, model, prompt_version, prompt, raw_response, parsed_response,
                 input_tokens, output_tokens, status, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (request_hash, base_url, model, prompt_version, prompt, raw_response,
                 json.dumps(parsed_response, ensure_ascii=False) if parsed_response is not None else None,
                 input_tokens, output_tokens, status, error[:1000]),
            )
            self.connection.commit()

    def stats(self) -> dict[str, int | float]:
        with self.lock:
            total, success, failed, input_tokens, output_tokens = self.connection.execute(
                "SELECT COUNT(*), SUM(status='success'), SUM(status!='success'), SUM(input_tokens), SUM(output_tokens) FROM api_cache"
            ).fetchone()
        lookups = self.hits + self.misses
        return {
            "requests": int(total or 0), "success": int(success or 0), "failed": int(failed or 0),
            "input_tokens": int(input_tokens or 0), "output_tokens": int(output_tokens or 0),
            "cache_hits": self.hits, "cache_misses": self.misses,
            "cache_hit_rate": self.hits / lookups if lookups else 0.0,
        }

    def close(self) -> None:
        self.connection.close()

