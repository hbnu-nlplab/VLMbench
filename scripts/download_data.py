#!/usr/bin/env python
"""Download LACE-Bench captions from HuggingFace and convert to the expected on-disk layout.

Run from the project root:
    python scripts/download_data.py

Produces, under dataset/lacebench/:
    train/<image_id>.json        # per-image annotation, used by training/eval globs
    test/<image_id>.json
    lace_test.json               # consolidated test dict, used by --qual-anal
    train_keyword_dict.json      # synset/candidate dict for KE eval
    test_keyword_dict.json

Visual Genome images must be downloaded separately (see README).
"""

import json
import sys
from pathlib import Path

# Ensure the project root is importable when run as `python scripts/download_data.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402

from lacebench import CAPTION_DIR  # noqa: E402

HF_REPO = "lacebench/LACE-Bench"


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


def main():
    CAPTION_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {HF_REPO} from HuggingFace ...")
    ds = load_dataset(HF_REPO)

    for split in ("train", "test"):
        split_dir = CAPTION_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        records = list(ds[split])
        print(f"Writing {len(records)} {split} records to {split_dir} ...")

        consolidated = {}
        for record in records:
            image_id = str(record["image_id"])
            payload = _record_payload(record)
            with open(split_dir / f"{image_id}.json", "w") as f:
                json.dump({image_id: payload}, f)
            consolidated[image_id] = payload

        if split == "test":
            lace_test_path = CAPTION_DIR / "lace_test.json"
            with open(lace_test_path, "w") as f:
                json.dump(consolidated, f)
            print(f"Wrote consolidated test set to {lace_test_path}")

        kw_dict = _build_keyword_dict(records)
        kw_path = CAPTION_DIR / f"{split}_keyword_dict.json"
        with open(kw_path, "w") as f:
            json.dump(kw_dict, f)
        print(f"Wrote {len(kw_dict)} keyword entries to {kw_path}")

    print("Done.")


if __name__ == "__main__":
    main()
