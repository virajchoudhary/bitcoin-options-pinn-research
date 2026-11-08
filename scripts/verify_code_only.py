"""Check the explicit public file allowlist without importing research code."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".venv", ".pytest_cache", "__pycache__"}
FORBIDDEN = {".csv", ".json", ".jsonl", ".gz", ".zip", ".npz", ".npy", ".pt", ".pth",
             ".safetensors", ".pdf", ".docx", ".png", ".jpg", ".parquet", ".pkl", ".tex"}


def verify(root=ROOT):
    expected = set((root / "PUBLIC_FILES.txt").read_text().splitlines())
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and not IGNORED.intersection(path.relative_to(root).parts)}
    assert actual == expected, f"Public allowlist differs: extra={actual-expected}, missing={expected-actual}"
    for name in sorted(actual):
        path = root / name
        assert path.suffix.lower() not in FORBIDDEN, f"Excluded asset format: {name}"
        content = path.read_bytes()
        assert len(content) <= 256_000 and b"\0" not in content, f"Unexpected binary or size: {name}"
        text = content.decode("utf-8")
        assert not re.search(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----", text), name
        assert not re.search(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|\bgh[pousr]_[A-Za-z0-9]{30,}\b", text), name
    return len(actual)


if __name__ == "__main__":
    print(f"CODE-ONLY ALLOWLIST PASS: {verify()} files; no bundled data or model assets")
