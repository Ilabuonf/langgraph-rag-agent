"""Load documents from disk and split them into overlapping chunks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**data)


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def load_documents(root: str | Path) -> list[tuple[str, str]]:
    """Return (source, text) pairs for every supported file under `root` (or `root` itself)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    docs = []
    for path in files:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = read_file(path).strip()
        if text:
            source = path.name if root.is_file() else str(path.relative_to(root))
            docs.append((source, text))
    return docs


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into chunks of at most `chunk_size` characters with `overlap` characters shared.

    Cuts prefer paragraph breaks, then whitespace, so chunks rarely split a sentence mid-word.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            floor = start + chunk_size // 2
            paragraph = text.rfind("\n\n", floor, end)
            space = text.rfind(" ", floor, end)
            if paragraph != -1:
                end = paragraph
            elif space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        # Start the next chunk at a word boundary.
        boundary = text.find(" ", next_start, end)
        start = boundary + 1 if overlap and boundary != -1 else next_start
    return chunks


def build_chunks(
    docs: list[tuple[str, str]], chunk_size: int = 800, overlap: int = 150
) -> list[Chunk]:
    chunks = []
    for source, text in docs:
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(Chunk(id=f"{source}#{i}", text=piece, source=source, chunk_index=i))
    return chunks
