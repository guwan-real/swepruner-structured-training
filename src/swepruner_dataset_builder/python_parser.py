from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .schemas import Block, ParsedFile, Relation, RelationType, Symbol


CONTAINER_TYPES = {
    "Module", "ClassDef", "FunctionDef", "AsyncFunctionDef", "If", "For", "AsyncFor",
    "While", "Try", "TryStar", "With", "AsyncWith", "Match", "Else", "Except",
    "Finally", "MatchCase", "StatementGroup",
}


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _facts(nodes: ast.AST | Iterable[ast.AST]) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    roots = [nodes] if isinstance(nodes, ast.AST) else list(nodes)
    defined: set[str] = set()
    used: set[str] = set()
    calls: set[str] = set()
    attributes: set[str] = set()
    annotations: set[str] = set()
    for root in roots:
        for node in ast.walk(root):
            if isinstance(node, ast.Name):
                (defined if isinstance(node.ctx, (ast.Store, ast.Del)) else used).add(node.id)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
                if node.annotation:
                    annotations.update(item.id for item in ast.walk(node.annotation) if isinstance(item, ast.Name))
            elif isinstance(node, ast.Call):
                name = _dotted(node.func)
                if name:
                    calls.add(name)
            elif isinstance(node, ast.Attribute):
                attributes.add(_dotted(node))
            elif isinstance(node, ast.AnnAssign):
                annotations.update(item.id for item in ast.walk(node.annotation) if isinstance(item, ast.Name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns:
                annotations.update(item.id for item in ast.walk(node.returns) if isinstance(item, ast.Name))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    defined.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    defined.add(alias.asname or alias.name)
    return tuple(sorted(values) for values in (defined, used, calls, attributes, annotations))


class _FileBuilder:
    def __init__(self, relative_path: str, source: str):
        self.relative_path = relative_path
        self.source = source
        self.lines = source.splitlines()
        self.blocks: list[Block] = []
        self.symbols: list[Symbol] = []
        self._symbol_qualnames: dict[str, str] = {"module": "<module>"}

    def text(self, start: int, end: int) -> str:
        return "\n".join(self.lines[max(0, start - 1):max(start, end)])

    def add_block(
        self,
        block_type: str,
        start: int,
        end: int,
        parent: str | None,
        symbol_id: str,
        node: ast.AST | Iterable[ast.AST] | None = None,
        suffix: str = "",
    ) -> Block:
        start = max(1, start)
        end = max(start, end)
        qualname = self._symbol_qualnames.get(symbol_id, "<module>")
        discriminator = f"::{suffix}" if suffix else ""
        block_id = f"{self.relative_path}::{qualname}::{block_type}::{start}::{end}{discriminator}"
        facts = _facts(node) if node is not None else ([], [], [], [], [])
        block = Block(
            block_id, block_type, self.relative_path, start, end, parent, symbol_id,
            self.text(start, end), qualname, *facts,
        )
        self.blocks.append(block)
        return block

    def build(self, tree: ast.Module) -> tuple[list[Block], list[Symbol]]:
        module_end = max(1, len(self.lines))
        module = self.add_block("Module", 1, module_end, None, "module")
        self._walk_body(tree.body, module.block_id, "module", "", "")
        return self.blocks, self.symbols

    def _groups(self, body: list[ast.stmt], parent: str, symbol_id: str) -> None:
        simple = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Return, ast.Raise, ast.Assert,
                  ast.Import, ast.ImportFrom, ast.Delete, ast.Pass, ast.Break, ast.Continue)
        group: list[ast.stmt] = []
        for stmt in body + [ast.Pass()]:
            if isinstance(stmt, simple) and hasattr(stmt, "lineno"):
                group.append(stmt)
            else:
                if len(group) >= 2:
                    self.add_block(
                        "StatementGroup", group[0].lineno, group[-1].end_lineno or group[-1].lineno,
                        parent, symbol_id, group, suffix=str(group[0].lineno),
                    )
                group = []

    def _walk_body(
        self,
        body: list[ast.stmt],
        parent: str,
        symbol_id: str,
        parent_qualname: str,
        class_qualname: str,
    ) -> None:
        self._groups(body, parent, symbol_id)
        for stmt in body:
            self._walk_stmt(stmt, parent, symbol_id, parent_qualname, class_qualname)

    def _header_end(self, stmt: ast.AST, body: list[ast.stmt]) -> int:
        if body:
            return max(stmt.lineno, body[0].lineno - 1)
        return stmt.lineno

    def _walk_stmt(
        self,
        stmt: ast.stmt,
        parent: str,
        symbol_id: str,
        parent_qualname: str,
        class_qualname: str,
    ) -> None:
        end = stmt.end_lineno or stmt.lineno
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = type(stmt).__name__
            start = min([stmt.lineno] + [item.lineno for item in stmt.decorator_list])
            qualname = f"{parent_qualname}.{stmt.name}" if parent_qualname else stmt.name
            sid = f"{self.relative_path}::{qualname}::{kind}::{start}::{end}"
            self._symbol_qualnames[sid] = qualname
            container = self.add_block(kind, start, end, parent, sid)
            signature = self.add_block(
                f"{kind}Signature", start, self._header_end(stmt, stmt.body), container.block_id,
                sid, [stmt.args, *stmt.decorator_list] + ([stmt.returns] if stmt.returns else []),
            )
            self.symbols.append(
                Symbol(sid, stmt.name, qualname, kind, self.relative_path, start, end, symbol_id,
                       container.block_id, signature.block_id, class_qualname)
            )
            for index, decorator in enumerate(stmt.decorator_list):
                self.add_block("Decorator", decorator.lineno, decorator.end_lineno or decorator.lineno,
                               container.block_id, sid, decorator, suffix=str(index))
            self._walk_body(stmt.body, container.block_id, sid, qualname, class_qualname)
            return
        if isinstance(stmt, ast.ClassDef):
            start = min([stmt.lineno] + [item.lineno for item in stmt.decorator_list])
            qualname = f"{parent_qualname}.{stmt.name}" if parent_qualname else stmt.name
            sid = f"{self.relative_path}::{qualname}::ClassDef::{start}::{end}"
            self._symbol_qualnames[sid] = qualname
            container = self.add_block("ClassDef", start, end, parent, sid)
            signature = self.add_block(
                "ClassDefSignature", start, self._header_end(stmt, stmt.body), container.block_id,
                sid, [*stmt.bases, *stmt.keywords, *stmt.decorator_list],
            )
            self.symbols.append(
                Symbol(sid, stmt.name, qualname, "ClassDef", self.relative_path, start, end, symbol_id,
                       container.block_id, signature.block_id, qualname)
            )
            for index, decorator in enumerate(stmt.decorator_list):
                self.add_block("Decorator", decorator.lineno, decorator.end_lineno or decorator.lineno,
                               container.block_id, sid, decorator, suffix=str(index))
            self._walk_body(stmt.body, container.block_id, sid, qualname, qualname)
            return
        control = (
            (ast.If, "If", "IfHeader"), (ast.For, "For", "ForHeader"),
            (ast.AsyncFor, "AsyncFor", "ForHeader"), (ast.While, "While", "WhileHeader"),
            (ast.With, "With", "WithHeader"), (ast.AsyncWith, "AsyncWith", "WithHeader"),
        )
        for node_type, block_type, header_type in control:
            if isinstance(stmt, node_type):
                container = self.add_block(block_type, stmt.lineno, end, parent, symbol_id)
                header_node = getattr(stmt, "test", None) or getattr(stmt, "iter", None) or getattr(stmt, "items", None)
                self.add_block(header_type, stmt.lineno, self._header_end(stmt, stmt.body),
                               container.block_id, symbol_id, header_node)
                self._walk_body(stmt.body, container.block_id, symbol_id, parent_qualname, class_qualname)
                orelse = getattr(stmt, "orelse", [])
                if orelse:
                    else_start = max(stmt.lineno, orelse[0].lineno - 1)
                    else_block = self.add_block("Else", else_start, orelse[-1].end_lineno or orelse[-1].lineno,
                                                container.block_id, symbol_id, suffix=str(else_start))
                    self.add_block("ElseHeader", else_start, else_start, else_block.block_id, symbol_id)
                    self._walk_body(orelse, else_block.block_id, symbol_id, parent_qualname, class_qualname)
                return
        if isinstance(stmt, (ast.Try, ast.TryStar)):
            container = self.add_block(type(stmt).__name__, stmt.lineno, end, parent, symbol_id)
            self.add_block("TryHeader", stmt.lineno, stmt.lineno, container.block_id, symbol_id)
            self._walk_body(stmt.body, container.block_id, symbol_id, parent_qualname, class_qualname)
            for index, handler in enumerate(stmt.handlers):
                h_end = handler.end_lineno or handler.lineno
                h_block = self.add_block("Except", handler.lineno, h_end, container.block_id, symbol_id,
                                         suffix=str(index))
                self.add_block("ExceptHeader", handler.lineno,
                               self._header_end(handler, handler.body), h_block.block_id, symbol_id,
                               handler.type, suffix=str(index))
                self._walk_body(handler.body, h_block.block_id, symbol_id, parent_qualname, class_qualname)
            if stmt.orelse:
                start = max(stmt.lineno, stmt.orelse[0].lineno - 1)
                block = self.add_block("Else", start, stmt.orelse[-1].end_lineno or stmt.orelse[-1].lineno,
                                       container.block_id, symbol_id, suffix=f"try-{start}")
                self.add_block("ElseHeader", start, start, block.block_id, symbol_id, suffix="try")
                self._walk_body(stmt.orelse, block.block_id, symbol_id, parent_qualname, class_qualname)
            if stmt.finalbody:
                start = max(stmt.lineno, stmt.finalbody[0].lineno - 1)
                block = self.add_block("Finally", start, stmt.finalbody[-1].end_lineno or stmt.finalbody[-1].lineno,
                                       container.block_id, symbol_id)
                self.add_block("FinallyHeader", start, start, block.block_id, symbol_id)
                self._walk_body(stmt.finalbody, block.block_id, symbol_id, parent_qualname, class_qualname)
            return
        if isinstance(stmt, ast.Match):
            container = self.add_block("Match", stmt.lineno, end, parent, symbol_id)
            first_case = min((case.pattern.lineno for case in stmt.cases), default=stmt.lineno + 1)
            self.add_block("MatchHeader", stmt.lineno, max(stmt.lineno, first_case - 1),
                           container.block_id, symbol_id, stmt.subject)
            for index, case in enumerate(stmt.cases):
                start = case.pattern.lineno
                case_end = max((item.end_lineno or item.lineno for item in case.body), default=start)
                block = self.add_block("MatchCase", start, case_end, container.block_id, symbol_id,
                                       [case.pattern] + ([case.guard] if case.guard else []), suffix=str(index))
                self._walk_body(case.body, block.block_id, symbol_id, parent_qualname, class_qualname)
            return
        block = self.add_block(type(stmt).__name__, stmt.lineno, end, parent, symbol_id, stmt)
        for child_name in ("body", "orelse", "finalbody"):
            child = getattr(stmt, child_name, None)
            if isinstance(child, list) and child and all(isinstance(item, ast.stmt) for item in child):
                self._walk_body(child, block.block_id, symbol_id, parent_qualname, class_qualname)


def parse_python_file(repo_path: str | Path, path: str | Path) -> ParsedFile:
    repo = Path(repo_path).resolve()
    absolute = Path(path).resolve()
    relative = absolute.relative_to(repo).as_posix()
    source = absolute.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=relative, type_comments=True)
    except SyntaxError as exc:
        return ParsedFile(relative, str(absolute), source, source.splitlines(), [], [], str(exc))
    builder = _FileBuilder(relative, source)
    blocks, symbols = builder.build(tree)
    return ParsedFile(relative, str(absolute), source, source.splitlines(), blocks, symbols)


@dataclass
class RepositoryIndex:
    repo_path: str
    files: dict[str, ParsedFile] = field(default_factory=dict)
    blocks: dict[str, Block] = field(default_factory=dict)
    symbols: dict[str, Symbol] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    ast_failures: list[str] = field(default_factory=list)

    def block(self, block_id: str) -> Block | None:
        return self.blocks.get(block_id)

    def symbol(self, symbol_id: str) -> Symbol | None:
        return self.symbols.get(symbol_id)

    def blocks_for_symbol(self, symbol_id: str) -> list[Block]:
        return [block for block in self.blocks.values() if block.enclosing_symbol_id == symbol_id]


def _is_excluded(path: Path) -> bool:
    return any(part in {".git", ".venv", "venv", "node_modules", "build", "dist", "__pycache__"} for part in path.parts)


def build_repository_index(
    repo_path: str | Path,
    max_files: int = 2000,
    priority_paths: Iterable[str] | None = None,
) -> RepositoryIndex:
    repo = Path(repo_path).resolve()
    index = RepositoryIndex(str(repo))
    prioritized: list[Path] = []
    seen: set[Path] = set()
    for relative_path in priority_paths or []:
        candidate = (repo / relative_path).resolve()
        try:
            relative = candidate.relative_to(repo)
        except ValueError:
            continue
        if candidate.is_file() and candidate.suffix == ".py" and not _is_excluded(relative) and candidate not in seen:
            prioritized.append(candidate)
            seen.add(candidate)
    remaining = [
        path for path in sorted(repo.rglob("*.py"))
        if path not in seen and not _is_excluded(path.relative_to(repo))
    ]
    paths = prioritized + remaining[:max(0, max_files - len(prioritized))]
    for path in paths:
        parsed = parse_python_file(repo, path)
        index.files[parsed.relative_path] = parsed
        if parsed.syntax_error:
            index.ast_failures.append(parsed.relative_path)
            continue
        index.blocks.update({block.block_id: block for block in parsed.blocks})
        index.symbols.update({symbol.symbol_id: symbol for symbol in parsed.symbols})
    _build_relations(index)
    return index


def _build_relations(index: RepositoryIndex) -> None:
    relations: list[Relation] = []
    signatures = {symbol.symbol_id: index.blocks.get(symbol.signature_block_id) for symbol in index.symbols.values()}
    symbols_by_name: dict[str, list[Symbol]] = defaultdict(list)
    for symbol in index.symbols.values():
        symbols_by_name[symbol.name].append(symbol)
    parent_to_headers: dict[str, list[Block]] = defaultdict(list)
    for block in index.blocks.values():
        if block.parent_block_id and (block.block_type.endswith("Header") or block.block_type.endswith("Signature")):
            parent_to_headers[block.parent_block_id].append(block)
    for symbol_id in {block.enclosing_symbol_id for block in index.blocks.values()}:
        blocks = sorted(
            [block for block in index.blocks.values() if block.enclosing_symbol_id == symbol_id and block.block_type not in CONTAINER_TYPES],
            key=lambda item: (item.start_line, item.end_line - item.start_line),
        )
        definitions: dict[str, Block] = {}
        for block in blocks:
            for name in block.used_names:
                target = definitions.get(name)
                if target and target.block_id != block.block_id:
                    relations.append(Relation(block.block_id, target.block_id, RelationType.DEF_USE, 0.88, ["local_def_use"]))
            for name in block.defined_names:
                definitions[name] = block
    for block in index.blocks.values():
        parent_id = block.parent_block_id
        visited: set[str] = set()
        while parent_id and parent_id not in visited:
            visited.add(parent_id)
            parent = index.blocks.get(parent_id)
            if not parent:
                break
            headers = parent_to_headers.get(parent.block_id, [])
            if headers and block.block_id not in {item.block_id for item in headers}:
                relation_type = RelationType.EXCEPTION if parent.block_type in {"Except", "Finally"} else RelationType.CONTROL
                confidence = 0.85 if relation_type == RelationType.EXCEPTION else 0.88
                relations.append(Relation(block.block_id, headers[0].block_id, relation_type, confidence,
                                          ["exception_pair" if relation_type == RelationType.EXCEPTION else "direct_control_dependency"]))
                break
            parent_id = parent.parent_block_id
        for call in block.calls:
            candidates = symbols_by_name.get(call.split(".")[-1], [])
            if len(candidates) == 1 and signatures.get(candidates[0].symbol_id):
                target = signatures[candidates[0].symbol_id]
                relations.append(Relation(block.block_id, target.block_id, RelationType.CALL, 0.72, ["one_hop_direct_call"]))
                caller = index.symbol(block.enclosing_symbol_id)
                if caller and signatures.get(caller.symbol_id):
                    relations.append(Relation(target.block_id, signatures[caller.symbol_id].block_id,
                                              RelationType.CALLED_BY, 0.68, ["one_hop_caller"]))
    for parsed in index.files.values():
        imports = [block for block in parsed.blocks if block.block_type in {"Import", "ImportFrom"}]
        for block in parsed.blocks:
            for imported in imports:
                if set(block.used_names) & set(imported.defined_names):
                    relations.append(Relation(block.block_id, imported.block_id, RelationType.IMPORT, 0.78, ["import_dependency"]))
        for symbol in parsed.symbols:
            signature = index.blocks.get(symbol.signature_block_id)
            if not signature:
                continue
            for decorator in parsed.blocks:
                if decorator.parent_block_id == symbol.block_id and decorator.block_type == "Decorator":
                    relations.append(Relation(signature.block_id, decorator.block_id, RelationType.DECORATOR, 0.78, ["decorator_dependency"]))
            for annotation in signature.annotations:
                candidates = symbols_by_name.get(annotation, [])
                if len(candidates) == 1:
                    target = index.blocks.get(candidates[0].signature_block_id)
                    if target:
                        relations.append(Relation(signature.block_id, target.block_id, RelationType.TYPE, 0.78, ["type_dependency"]))
    seen: set[tuple[str, str, str]] = set()
    index.relations = []
    for relation in relations:
        key = (relation.source_block_id, relation.target_block_id, str(relation.relation))
        if relation.source_block_id != relation.target_block_id and key not in seen:
            seen.add(key)
            index.relations.append(relation)


def smallest_block_for_line(index: RepositoryIndex, file_path: str, line: int) -> Block | None:
    parsed = index.files.get(file_path)
    if not parsed:
        return None
    candidates = [
        block for block in parsed.blocks
        if block.start_line <= line <= block.end_line and block.block_type not in CONTAINER_TYPES
    ]
    if not candidates:
        candidates = [block for block in parsed.blocks if block.start_line <= line <= block.end_line]
    return min(candidates, key=lambda item: (item.end_line - item.start_line, -item.start_line), default=None)


def top_level_block(index: RepositoryIndex, block: Block) -> Block:
    current = block
    parent = index.block(current.parent_block_id) if current.parent_block_id else None
    while parent and parent.block_type != "Module":
        current = parent
        parent = index.block(current.parent_block_id) if current.parent_block_id else None
    return current
