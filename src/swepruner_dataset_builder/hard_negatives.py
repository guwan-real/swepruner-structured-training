from __future__ import annotations

import io
import keyword
import re
import token
import tokenize

from .python_parser import RepositoryIndex
from .schemas import BlockLabel, HardNegative, Role


def identifier_tokens(text: str) -> set[str]:
    values: set[str] = set()
    try:
        for item in tokenize.generate_tokens(io.StringIO(text).readline):
            if item.type == token.NAME and not keyword.iskeyword(item.string):
                values.add(item.string.lower())
    except (IndentationError, tokenize.TokenError):
        values.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower()))
    return values


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def mine_hard_negatives(
    index: RepositoryIndex,
    anchor_symbol_id: str,
    labels: dict[str, BlockLabel],
    issue_text: str,
) -> list[HardNegative]:
    anchor = index.symbol(anchor_symbol_id)
    if not anchor:
        return []
    anchor_block = index.block(anchor.block_id)
    if not anchor_block:
        return []
    anchor_tokens = identifier_tokens(anchor.qualname + "\n" + anchor_block.source_text)
    issue_tokens = identifier_tokens(issue_text)
    related = {
        relation.target_block_id for relation in index.relations if relation.source_block_id in {anchor.block_id, anchor.signature_block_id}
    } | {
        relation.source_block_id for relation in index.relations if relation.target_block_id in {anchor.block_id, anchor.signature_block_id}
    }
    candidates: list[HardNegative] = []
    for symbol in index.symbols.values():
        if symbol.symbol_id == anchor_symbol_id or symbol.kind == "ClassDef":
            continue
        block = index.block(symbol.block_id)
        if not block or block.block_id in related:
            continue
        descendants = [
            candidate for candidate in index.blocks.values()
            if candidate.relative_file_path == block.relative_file_path
            and block.start_line <= candidate.start_line <= candidate.end_line <= block.end_line
        ]
        if any(labels.get(item.block_id, BlockLabel()).role != Role.DROP for item in descendants):
            continue
        lexical = jaccard(anchor_tokens, identifier_tokens(symbol.qualname + "\n" + block.source_text))
        issue_similarity = jaccard(issue_tokens, identifier_tokens(block.source_text))
        same_file = symbol.relative_file_path == anchor.relative_file_path
        same_class = bool(anchor.class_qualname and anchor.class_qualname == symbol.class_qualname)
        name_left = set(re.split(r"[_\W]+", anchor.name.lower()))
        name_right = set(re.split(r"[_\W]+", symbol.name.lower()))
        name_similarity = jaccard(name_left, name_right)
        if same_class:
            negative_type = "same_class_other_method"
        elif name_similarity >= 0.5 or lexical >= 0.30:
            negative_type = "similar_name_no_relation"
        elif issue_similarity >= 0.20:
            negative_type = "issue_keyword_match_no_relation"
        elif same_file:
            negative_type = "same_file_other_branch"
        else:
            continue
        candidates.append(HardNegative(block.block_id, negative_type, round(max(lexical, name_similarity, issue_similarity), 4), 99, same_file, same_class))
    return sorted(candidates, key=lambda item: (-item.same_file, -item.same_class, -item.lexical_similarity, item.block_id))

