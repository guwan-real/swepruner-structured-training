from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .api_cache import ApiCache
from .io_utils import canonical_json, redact_secret, stable_hash


@dataclass(slots=True)
class ProviderResponse:
    raw: str
    parsed: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_keys: list[str], timeout: int = 60, retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.api_keys = [key for key in api_keys if key]
        self.timeout = timeout
        self.retries = retries

    @classmethod
    def from_env(cls, config: dict) -> "OpenAICompatibleProvider":
        api = config["api"]
        keys: list[str] = []
        for value in (os.environ.get("LLM_API_KEYS", ""), os.environ.get("LLM_API_KEY", "")):
            keys.extend(part.strip() for part in value.split(",") if part.strip())
        keys = list(dict.fromkeys(keys))
        if not keys:
            raise ValueError("LLM_API_KEY or LLM_API_KEYS is required when --use-api is enabled")
        return cls(
            os.environ.get("LLM_BASE_URL", str(api["base_url"])), keys,
            int(api["timeout_seconds"]), int(api["retry_count"]),
        )

    def complete(self, model: str, messages: list[dict[str, str]], options: dict[str, Any]) -> ProviderResponse:
        payload = {"model": model, "messages": messages, **options}
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error = ""
        for key in self.api_keys:
            for attempt in range(self.retries + 1):
                request = urllib.request.Request(
                    f"{self.base_url}/chat/completions", data=encoded, method="POST",
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                )
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        raw = response.read().decode("utf-8")
                    envelope = json.loads(raw)
                    content = envelope["choices"][0]["message"]["content"]
                    parsed = _parse_json_content(content)
                    usage = envelope.get("usage", {})
                    return ProviderResponse(raw, parsed, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
                except urllib.error.HTTPError as exc:
                    last_error = f"HTTP {exc.code}"
                    if exc.code in {401, 403, 429}:
                        break
                    if attempt < self.retries:
                        time.sleep(min(2 ** attempt, 4))
                except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
                    last_error = redact_secret(str(exc))
                    if attempt < self.retries:
                        time.sleep(min(2 ** attempt, 4))
        raise RuntimeError(f"API request failed after key failover: {last_error}")


class FakeProvider:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls = 0
        self.base_url = "fake://provider"

    def complete(self, model: str, messages: list[dict[str, str]], options: dict[str, Any]) -> ProviderResponse:
        self.calls += 1
        if not self.responses:
            raise RuntimeError("FakeProvider has no queued response")
        parsed = self.responses.pop(0)
        return ProviderResponse(json.dumps(parsed), parsed, 10, 5)


def _parse_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    text = str(content).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("model response must be a JSON object")
    return parsed


class CachedLLMClient:
    def __init__(self, provider: Any, cache: ApiCache, config: dict):
        self.provider = provider
        self.cache = cache
        self.config = config

    def request(self, model: str, prompt_version: str, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = canonical_json(payload)
        base_url = getattr(self.provider, "base_url", "unknown")
        request_hash = stable_hash({
            "base_url": base_url, "model": model, "prompt_version": prompt_version,
            "issue_hash": stable_hash(payload.get("issue_text", ""), 64),
            "candidate_code_hash": stable_hash(payload.get("candidate_code", ""), 64),
            "relation_metadata_hash": stable_hash(payload.get("relation_metadata", {}), 64),
            "payload": payload,
        }, 64)
        cached = self.cache.get(request_hash)
        if cached and cached["status"] == "success":
            return cached["parsed"]
        api = self.config["api"]
        options = {
            "temperature": float(api["temperature"]),
            "max_tokens": int(api["max_tokens"]),
            "response_format": {"type": "json_object"},
        }
        try:
            response = self.provider.complete(model, [{"role": "user", "content": prompt}], options)
            self.cache.put(request_hash, base_url, model, prompt_version, prompt, response.raw,
                           response.parsed, response.input_tokens, response.output_tokens, "success")
            return response.parsed
        except Exception as exc:
            error = redact_secret(str(exc))
            self.cache.put(request_hash, base_url, model, prompt_version, prompt, "", None, 0, 0, "error", error)
            raise RuntimeError(error) from exc


def validate_block_decision(value: dict[str, Any], block_id: str) -> dict[str, Any]:
    role = str(value.get("role", ""))
    relation = str(value.get("relation", "NONE"))
    confidence = float(value.get("confidence", -1))
    allowed_relations = {"CALL", "DEF_USE", "CONTROL", "TYPE", "IMPORT", "ATTRIBUTE", "INHERITANCE",
                         "OVERRIDE", "EXCEPTION", "TRACEBACK", "DECORATOR", "NONE"}
    if str(value.get("block_id")) != block_id or role not in {"SUPPORT", "DROP"} or relation not in allowed_relations or not 0 <= confidence <= 1:
        raise ValueError("invalid API block decision schema")
    return {"block_id": block_id, "role": role, "relation": relation, "confidence": confidence,
            "reason": str(value.get("reason", ""))[:500]}


def review_candidate(client: CachedLLMClient, payload: dict[str, Any], local_role: str, local_confidence: float,
                     primary_model: str, reviewer_model: str, config: dict) -> dict[str, Any]:
    api = config["api"]
    block_id = payload["block_id"]
    primary = validate_block_decision(client.request(primary_model, api["prompt_version"] + ":primary", payload), block_id)
    accepted = {**primary, "agreement": None, "models": [primary_model]}
    if primary["confidence"] >= float(api["primary_accept_threshold"]):
        return accepted
    if primary["confidence"] < float(api["review_lower_threshold"]):
        return {"block_id": block_id, "role": local_role, "relation": payload["relation_metadata"].get("relation", "NONE"),
                "confidence": max(0.0, local_confidence - 0.10), "reason": "primary confidence too low; kept static label",
                "agreement": False, "models": [primary_model]}
    reviewer = validate_block_decision(client.request(reviewer_model, api["prompt_version"] + ":reviewer", payload), block_id)
    if reviewer["role"] == primary["role"]:
        return {**primary, "confidence": float(config["confidence"].get("api_two_model_agreement", 0.82)),
                "agreement": True, "models": [primary_model, reviewer_model]}
    return {"block_id": block_id, "role": local_role, "relation": payload["relation_metadata"].get("relation", "NONE"),
            "confidence": max(0.0, local_confidence - 0.15), "reason": "model disagreement; kept static label",
            "agreement": False, "models": [primary_model, reviewer_model]}

