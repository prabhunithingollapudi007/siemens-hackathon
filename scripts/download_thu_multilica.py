"""Download and prepare the preprocessed THU-MultiLiCa cerebral MRA dataset.

The source provides preprocessed MRA volumes and vascular annotations. The
dataset is for non-commercial research/education; keep its license and cite
the dataset paper when publishing.

Usage:
    python scripts/download_thu_multilica.py
    python scripts/download_thu_multilica.py --output data/external/thu_multilica
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from urllib.request import urlopen


URL = "https://ivg.au.tsinghua.edu.cn/dataset/Cerebral-MRA/Cerebral_files/annotation-aftpreprocess.zip"


def download(url: str, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with urlopen(url, timeout=60) as response:
        with target.open("wb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if chunk:
                    stream.write(chunk)
                    digest.update(chunk)
                else:
                    break
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("data/external/thu_multilica"))
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()
    root = args.output
    archive = root / "annotation-aftpreprocess.zip"
    extracted = root / "dataset"
    print(f"[download] {args.url}")
    sha256 = download(args.url, archive)
    print(f"[download] wrote {archive} sha256={sha256}")
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"corrupt archive member: {bad}")
        zf.extractall(extracted)
        files = [str(p.relative_to(extracted)) for p in extracted.rglob("*") if p.is_file()]
    manifest = {
        "dataset": "THU-MultiLiCa Cerebral MRA",
        "source_url": args.url,
        "archive_sha256": sha256,
        "modality": "MRA",
        "units": "voxel coordinates; inspect source metadata before physical planning",
        "purpose": "vascular landmark and semantic segmentation annotations",
        "centerlines": "not guaranteed; derive only from supplied annotations or VMTK",
        "files": sorted(files),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[prepare] extracted {len(files)} files to {extracted}")
    print(f"[prepare] manifest: {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
