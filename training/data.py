from __future__ import annotations

import bisect
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
from torch.utils.data import Dataset, Sampler

OFFICIAL_SOURCE = "swe_pruner_original"
ROLE_NAMES = ("DROP", "SUPPORT", "CORE")
PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    '<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc


def load_split_ids(data_root: str | Path, split: str) -> set[str]:
    path = Path(data_root) / "splits" / f"{split}_manifest.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    values = row.get("sample_ids")
    if not isinstance(values, list):
        raise ValueError(f"sample_ids missing from {path}")
    return {str(value) for value in values}


class MainDataset(Dataset):
    def __init__(self, path: str | Path, sample_ids: set[str]):
        self.rows = [row for row in read_jsonl(path) if row.get("sample_id") in sample_ids]
        if len(self.rows) != len(sample_ids):
            found = {row.get("sample_id") for row in self.rows}
            raise ValueError(f"main dataset is missing {len(sample_ids - found)} requested sample_ids")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class ReplayDistributedSampler(Sampler[int]):
    """Deterministic global 80/20 schedule, then shard it across DDP ranks."""

    def __init__(
        self,
        dataset: MainDataset,
        samples_per_epoch: int,
        replay_ratio: float,
        seed: int,
        rank: int,
        world_size: int,
    ):
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.replay_ratio = replay_ratio
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self.replay = [i for i, row in enumerate(dataset.rows) if row["dataset_source"] == OFFICIAL_SOURCE]
        self.new = [i for i, row in enumerate(dataset.rows) if row["dataset_source"] != OFFICIAL_SOURCE]
        if not self.new:
            raise ValueError("train split has no new structured samples")
        if replay_ratio and not self.replay:
            raise ValueError("replay_ratio is nonzero but train split has no official replay rows")
        self.total_size = int(math.ceil(samples_per_epoch / world_size) * world_size)
        self.num_samples = self.total_size // world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @staticmethod
    def _draw(pool: Sequence[int], count: int, rng: random.Random) -> list[int]:
        if count <= len(pool):
            return rng.sample(list(pool), count)
        return [pool[rng.randrange(len(pool))] for _ in range(count)]

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + 1009 * self.epoch)
        replay_count = int(round(self.samples_per_epoch * self.replay_ratio))
        new_count = self.samples_per_epoch - replay_count
        schedule = self._draw(self.new, new_count, rng)
        schedule.extend(self._draw(self.replay, replay_count, rng) if replay_count else [])
        rng.shuffle(schedule)
        if len(schedule) < self.total_size:
            schedule.extend(schedule[: self.total_size - len(schedule)])
        return iter(schedule[self.rank : self.total_size : self.world_size])

    def __len__(self) -> int:
        return self.num_samples


class ShardedSequentialSampler(Sampler[int]):
    def __init__(self, size: int, rank: int, world_size: int):
        self.indices = list(range(rank, size, world_size))

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class OffsetJsonlDataset(Dataset):
    """Index large JSONL files by byte offset without retaining parsed rows."""

    def __init__(self, paths: Iterable[str | Path], sample_ids: set[str], label_field: str):
        self.entries: list[tuple[str, int]] = []
        self.label_values: set[str] = set()
        self._handles: dict[str, Any] = {}
        for raw_path in paths:
            path = str(Path(raw_path))
            with open(path, "rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("sample_id") not in sample_ids:
                        continue
                    self.entries.append((path, offset))
                    if label_field in row:
                        self.label_values.add(str(row[label_field]))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, offset = self.entries[index]
        handle = self._handles.get(path)
        if handle is None:
            handle = open(path, "rb")
            self._handles[path] = handle
        handle.seek(offset)
        return json.loads(handle.readline())

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass


def format_instruction(instruction: str, query: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: "


class BatchEncoder:
    def __init__(
        self,
        tokenizer: Any,
        max_length: int,
        aux_max_length: int,
        instruction: str,
        role_to_id: dict[str, int],
        relation_to_id: dict[str, int],
    ):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("a fast tokenizer is required for offset-based line alignment")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.aux_max_length = aux_max_length
        self.instruction = instruction
        self.role_to_id = role_to_id
        self.relation_to_id = relation_to_id
        self.prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
            tokenizer.pad_token = tokenizer.eos_token

    @staticmethod
    def _line_for_offset(line_ends: list[int], start: int) -> int:
        return min(bisect.bisect_right(line_ends, start), len(line_ends) - 1)

    def _query_and_budget(self, query: str, limit: int) -> tuple[list[int], int]:
        query_ids = self.tokenizer.encode(
            format_instruction(self.instruction, query), add_special_tokens=False
        )
        available = limit - len(self.prefix_ids) - len(self.suffix_ids)
        if available < 2:
            raise ValueError("max_length is too small for the official SWE-Pruner prompt")
        query_ids = query_ids[: available - 1]
        return query_ids, available - len(query_ids)

    def encode_main(self, row: dict[str, Any]) -> dict[str, Any]:
        code = str(row["code"])
        lines = code.splitlines(keepends=True)
        plain_lines = code.splitlines()
        fields = ("line_keep_labels", "line_roles", "line_relation_types", "line_confidences", "line_numbers")
        if not plain_lines or any(len(row.get(field, [])) != len(plain_lines) for field in fields):
            raise ValueError(f"line arrays do not align for {row.get('sample_id')}")
        encoded = self.tokenizer(
            code,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        code_ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        query_ids, code_budget = self._query_and_budget(str(row["query"]), self.max_length)
        code_ids = code_ids[:code_budget]
        offsets = offsets[:code_budget]
        if not code_ids:
            raise ValueError(f"tokenized code is empty for {row.get('sample_id')}")
        cumulative = 0
        line_ends: list[int] = []
        for line in lines:
            cumulative += len(line)
            line_ends.append(cumulative)
        structural = row.get("dataset_source") != OFFICIAL_SOURCE
        hard_id = row.get("metadata", {}).get("hard_negative", {}).get("block_id")
        hard_spans = [
            (int(span["start_line"]), int(span["end_line"]))
            for span in row.get("metadata", {}).get("block_spans", [])
            if hard_id and span.get("block_id") == hard_id
        ]
        keep: list[int] = []
        roles: list[int] = []
        relations: list[int] = []
        confidences: list[float] = []
        line_indices: list[int] = []
        hard_negative: list[bool] = []
        for start, _ in offsets:
            line_index = self._line_for_offset(line_ends, int(start))
            line_number = int(row["line_numbers"][line_index])
            keep.append(int(row["line_keep_labels"][line_index]))
            roles.append(self.role_to_id.get(str(row["line_roles"][line_index]), -100) if structural else -100)
            relations.append(
                self.relation_to_id.get(str(row["line_relation_types"][line_index]), -100)
                if structural
                else -100
            )
            confidences.append(float(row["line_confidences"][line_index]))
            line_indices.append(line_index)
            hard_negative.append(any(start_line <= line_number <= end_line for start_line, end_line in hard_spans))
        doc_start = len(self.prefix_ids) + len(query_ids)
        input_ids = self.prefix_ids + query_ids + code_ids + self.suffix_ids
        length = len(input_ids)

        def place(values: list[Any], fill: Any) -> list[Any]:
            result = [fill] * length
            result[doc_start : doc_start + len(values)] = values
            return result

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * length,
            "code_mask": place([True] * len(code_ids), False),
            "keep_labels": place(keep, -100),
            "role_labels": place(roles, -100),
            "relation_labels": place(relations, -100),
            "confidence_weights": place(confidences, 0.0),
            "line_indices": place(line_indices, -1),
            "hard_negative_mask": place(hard_negative, False),
            "document_labels": float(row["document_label"]),
            "sample_id": str(row["sample_id"]),
            "dataset_source": str(row["dataset_source"]),
        }

    def encode_aux_document(self, query: str, document: str) -> tuple[list[int], list[int]]:
        query_ids, document_budget = self._query_and_budget(query, self.aux_max_length)
        document_ids = self.tokenizer.encode(document, add_special_tokens=False)[:document_budget]
        input_ids = self.prefix_ids + query_ids + document_ids + self.suffix_ids
        return input_ids, [1] * len(input_ids)

    def collate_main(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = [self.encode_main(row) for row in rows]
        return self._pad(encoded)

    def collate_relation(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        items = []
        for row in rows:
            document = (
                "[ANCHOR]\n" + str(row["anchor_block"]["code"]) +
                "\n[CANDIDATE]\n" + str(row["candidate_block"]["code"])
            )
            ids, mask = self.encode_aux_document(str(row["query"]), document)
            items.append({
                "input_ids": ids,
                "attention_mask": mask,
                "labels": self.relation_to_id[str(row["relation"])],
                "weights": float(row.get("confidence", 1.0)),
            })
        return self._pad_aux(items)

    def collate_ranking(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        positive, negative = [], []
        weights = []
        for row in rows:
            anchor = "[ANCHOR]\n" + str(row["anchor_block"]["code"])
            pos_doc = anchor + "\n[CANDIDATE]\n" + str(row["positive_block"]["code"])
            neg_doc = anchor + "\n[CANDIDATE]\n" + str(row["hard_negative_block"]["code"])
            positive.append(self.encode_aux_document(str(row["query"]), pos_doc))
            negative.append(self.encode_aux_document(str(row["query"]), neg_doc))
            weights.append(min(float(row.get("positive_confidence", 1.0)), float(row.get("negative_confidence", 1.0))))
        pos = self._pad_pairs(positive)
        neg = self._pad_pairs(negative)
        return {
            "positive_input_ids": pos[0], "positive_attention_mask": pos[1],
            "negative_input_ids": neg[0], "negative_attention_mask": neg[1],
            "weights": torch.tensor(weights, dtype=torch.float32),
        }

    def _pad(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(row["input_ids"]) for row in rows)
        pad_id = int(self.tokenizer.pad_token_id)
        specs = {
            "input_ids": (pad_id, torch.long), "attention_mask": (0, torch.long),
            "code_mask": (False, torch.bool), "keep_labels": (-100, torch.long),
            "role_labels": (-100, torch.long), "relation_labels": (-100, torch.long),
            "confidence_weights": (0.0, torch.float32), "line_indices": (-1, torch.long),
            "hard_negative_mask": (False, torch.bool),
        }
        batch: dict[str, Any] = {}
        for field, (fill, dtype) in specs.items():
            batch[field] = torch.tensor(
                [row[field] + [fill] * (max_len - len(row[field])) for row in rows], dtype=dtype
            )
        batch["document_labels"] = torch.tensor([row["document_labels"] for row in rows], dtype=torch.float32)
        batch["sample_ids"] = [row["sample_id"] for row in rows]
        batch["dataset_sources"] = [row["dataset_source"] for row in rows]
        return batch

    def _pad_aux(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pairs = [(row["input_ids"], row["attention_mask"]) for row in rows]
        input_ids, attention_mask = self._pad_pairs(pairs)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor([row["labels"] for row in rows], dtype=torch.long),
            "weights": torch.tensor([row["weights"] for row in rows], dtype=torch.float32),
        }

    def _pad_pairs(self, pairs: list[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(ids) for ids, _ in pairs)
        pad_id = int(self.tokenizer.pad_token_id)
        ids = [value + [pad_id] * (max_len - len(value)) for value, _ in pairs]
        masks = [value + [0] * (max_len - len(value)) for _, value in pairs]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.long)

