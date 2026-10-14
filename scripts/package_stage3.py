"""Make a source ZIP for Colab, excluding datasets, weights and local environments.

The archive includes a SHA-256 manifest of every bundled file and the upstream
Git revision. Colab verifies the file hashes before running this local snapshot.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(ROOT.parent / "stage3_colab/Visual_retrieval_system_stage3.zip"))
    args = ap.parse_args()
    paths = [ROOT / name for name in ("README.md", "requirements.txt", "requirements-stage3.txt", ".gitignore")]
    for folder in ("src", "scripts", "configs", "tests", "notebooks", "docs", "results/first_models"):
        paths += [p for p in (ROOT / folder).rglob("*") if p.is_file()
                  and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".log")]
    paths = sorted(set(paths))
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    metadata = {"purpose": "Stage 3 source snapshot; local edits included",
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "split_manifest_sha256": hashes["configs/split_manifest.csv"], "files": hashes}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, "Visual_retrieval_system/" + p.relative_to(ROOT).as_posix())
        z.writestr("Visual_retrieval_system/bundle_manifest.json", json.dumps(metadata, indent=2) + "\n")
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        for rel, digest in hashes.items():
            assert hashlib.sha256(z.read("Visual_retrieval_system/" + rel)).hexdigest() == digest
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".sha256").write_text(f"{digest}  {output.name}\n")
    print(f"{output}\nfiles: {len(paths)}; bytes: {output.stat().st_size}; SHA-256: {digest}")


if __name__ == "__main__":
    main()
