from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
IMG_DIR = DATASET_DIR / "visual_genome" / "VG_100K_all"
CAPTION_DIR = DATASET_DIR / "lacebench"
