from __future__ import annotations

import ast
import io
import keyword
import token
import tokenize

from .io_utils import stable_hash


def _tokens(code: str, normalize_identifiers: bool) -> list[str]:
    output: list[str] = []
    try:
        for item in tokenize.generate_tokens(io.StringIO(code).readline):
            if item.type in {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT}:
                continue
            value = item.string
            if normalize_identifiers and item.type == token.NAME and not keyword.iskeyword(value):
                value = "IDENT"
            output.append(value)
    except (IndentationError, tokenize.TokenError):
        output = code.split()
    return output


def code_fingerprints(code: str) -> dict[str, str]:
    token_hash = stable_hash(_tokens(code, False), 64)
    identifier_hash = stable_hash(_tokens(code, True), 64)
    try:
        structure = ast.dump(ast.parse(code), annotate_fields=False, include_attributes=False)
    except SyntaxError:
        structure = "syntax-error:" + " ".join(_tokens(code, True))
    return {
        "token_hash": token_hash,
        "identifier_normalized_hash": identifier_hash,
        "ast_structure_hash": stable_hash(structure, 64),
    }


def annotate_samples(samples: list[dict]) -> None:
    for sample in samples:
        sample.setdefault("metadata", {})["dedup_hashes"] = code_fingerprints(sample.get("code", ""))

