from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


FRAME_RE = re.compile(r'^\s*File "([^"]+)", line (\d+), in ([^\n]+)$', re.MULTILINE)


@dataclass(slots=True)
class TraceFrame:
    file_path: str
    line: int
    symbol: str


def project_frames(traceback_text: str, repo_path: str | Path, known_files: set[str]) -> list[TraceFrame]:
    repo = Path(repo_path).resolve()
    frames: list[TraceFrame] = []
    for raw_path, line, symbol in FRAME_RE.findall(traceback_text or ""):
        path = Path(raw_path)
        relative = ""
        try:
            if path.is_absolute():
                relative = path.resolve().relative_to(repo).as_posix()
            else:
                candidate = raw_path.replace("\\", "/")
                matches = [known for known in known_files if known == candidate or known.endswith("/" + candidate) or candidate.endswith("/" + known)]
                if len(matches) == 1:
                    relative = matches[0]
        except (OSError, ValueError):
            relative = ""
        if relative in known_files:
            frames.append(TraceFrame(relative, int(line), symbol.strip()))
    return frames

