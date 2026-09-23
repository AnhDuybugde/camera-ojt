"""Check publishable files without printing matched secrets or private data."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKED = re.compile(r"(^|/)(\.env$|node_modules/|\.venv/|var/|data/.*\.(?:jpg|jpeg|png|npy|db)$)|\.(?:engine|pt|onnx|db|sqlite|npy)$", re.I)
SECRETS = re.compile(r"(?:AIza[0-9A-Za-z_-]{30,}|tvly-[0-9A-Za-z_-]{20,}|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")


def main():
    names = subprocess.check_output(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"], cwd=ROOT).decode().split("\0")
    failures = []
    for name in filter(None, names):
        if BLOCKED.search(name):
            failures.append((name, "private/runtime artifact"))
            continue
        blob = subprocess.check_output(["git", "show", ":" + name], cwd=ROOT)
        if len(blob) > 20_000_000:
            failures.append((name, "large binary"))
        if SECRETS.search(blob.decode("utf-8", errors="ignore")):
            failures.append((name, "credential pattern"))
    for name, reason in failures:
        print(name + ": " + reason)
    print(f"Publish check: {len(failures)} blocked files")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
