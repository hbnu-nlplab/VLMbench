#!/usr/bin/env python
"""Download LACE-Bench captions + Visual Genome images and lay them out on disk.

Run from the project root:
    python scripts/download_data.py
    python scripts/download_data.py --skip-images   # captions only
    python scripts/download_data.py --skip-captions # images only

Produces, under dataset/:
    visual_genome/VG_100K_all/<image_id>.jpg   # merged VG v1 + v2
    lacebench/train/<image_id>.json            # per-image annotation
    lacebench/test/<image_id>.json
    lacebench/lace_test.json                   # consolidated test dict (for --qual-anal)
    lacebench/train_keyword_dict.json          # synset/candidate dict for KE eval
    lacebench/test_keyword_dict.json
"""

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

# Ensure the project root is importable when run as `python scripts/download_data.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from lacebench import CAPTION_DIR, IMG_DIR  # noqa: E402

HF_REPO = "lacebench/LACE-Bench"
VG_URLS = [
    "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
    "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
]


# ---------- Visual Genome ----------

def _download_with_progress(url, dest):
    if dest.exists():
        print(f"[VG] zip already present, skipping download: {dest}")
        return
    print(f"[VG] downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0)) or None
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while True:
                chunk = resp.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def _extract_flat(zip_path, target_dir):
    """Extract image files from `zip_path` directly into `target_dir`, dropping any internal folders."""
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/") and Path(m).name]
        skipped = 0
        for member in tqdm(members, desc=f"extract {zip_path.name}"):
            dest = target_dir / Path(member).name
            if dest.exists():
                skipped += 1
                continue
            with zf.open(member) as src, open(dest, "wb") as out:
                out.write(src.read())
        if skipped:
            print(f"[VG] {zip_path.name}: {skipped} files already existed, skipped")


def download_visual_genome():
    cache_dir = IMG_DIR.parent  # dataset/visual_genome/
    cache_dir.mkdir(parents=True, exist_ok=True)
    for url in VG_URLS:
        zip_path = cache_dir / Path(url).name
        _download_with_progress(url, zip_path)
        _extract_flat(zip_path, IMG_DIR)
    n_imgs = sum(1 for _ in IMG_DIR.glob("*.jpg"))
    print(f"[VG] {n_imgs} images now under {IMG_DIR}")


# ---------- LACE captions ----------

def _record_payload(record):
    """Strip a HF record down to the keys the rest of the codebase expects."""
    return {
        "regions": record["regions"],
        "relation_centric_regions": record["relation_centric_regions"],
    }


def _build_keyword_dict(records):
    """Build {keyword_name: [counterfactual_candidates...]} from the `keywords` column."""
    out = {}
    for r in records:
        for kw in r.get("keywords") or []:
            cf = kw.get("counterfactual") if isinstance(kw, dict) else None
            if not cf:
                continue
            name = cf.get("human_annotation")
            cands = cf.get("candidate") or []
            if name:
                out.setdefault(name, []).extend(cands)
    return {k: sorted(set(v)) for k, v in out.items()}


def download_captions():
    CAPTION_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[captions] loading {HF_REPO} from HuggingFace ...")
    ds = load_dataset(HF_REPO)

    for split in ("train", "test"):
        split_dir = CAPTION_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        records = list(ds[split])
        print(f"[captions] writing {len(records)} {split} records to {split_dir} ...")

        consolidated = {}
        for record in tqdm(records, desc=f"write {split}"):
            image_id = str(record["image_id"])
            payload = _record_payload(record)
            with open(split_dir / f"{image_id}.json", "w") as f:
                json.dump({image_id: payload}, f)
            consolidated[image_id] = payload

        if split == "test":
            lace_test_path = CAPTION_DIR / "lace_test.json"
            with open(lace_test_path, "w") as f:
                json.dump(consolidated, f)
            print(f"[captions] wrote consolidated test set to {lace_test_path}")

        kw_dict = _build_keyword_dict(records)
        kw_path = CAPTION_DIR / f"{split}_keyword_dict.json"
        with open(kw_path, "w") as f:
            json.dump(kw_dict, f)
        print(f"[captions] wrote {len(kw_dict)} keyword entries to {kw_path}")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-captions", action="store_true",
                        help="Skip the LACE caption download.")
    parser.add_argument("--skip-images", action="store_true",
                        help="Skip the Visual Genome image download.")
    args = parser.parse_args()

    if not args.skip_images:
        download_visual_genome()
    if not args.skip_captions:
        download_captions()
    print("Done.")


if __name__ == "__main__":
    main()
