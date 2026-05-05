# LACE-Bench

## Description
LACE-Bench is a benchmarking tool designed to evaluate the performance of various algorithms and models in a standardized manner. It provides a framework for running experiments and collecting metrics to facilitate comparisons and improvements.

## Installation
To install the necessary dependencies for LACE-Bench, please follow these steps:

1. Clone the repository and navigate into it.
2. Navigate to the project directory:
   ```
   cd LACE-Bench
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Dataset
All data lives under `dataset/` in the project root. A single command pulls and prepares everything:

```bash
python scripts/download_data.py
```

This downloads:
- **Visual Genome v1 + v2** images from `cs.stanford.edu/people/rak248/VG_100K_2/`,
  merged flat into `dataset/visual_genome/VG_100K_all/<image_id>.jpg`.
- **LACE-Bench captions** from the
  [HuggingFace dataset](https://huggingface.co/datasets/lacebench/LACE-Bench),
  converted into `dataset/lacebench/{train,test}/<image_id>.json`,
  `lace_test.json`, and the keyword dictionaries used for knowledge-editing eval.

Pass `--skip-images` or `--skip-captions` to download only one half. Existing zips and
extracted files are skipped on re-run.

## Usage
Train and evaluation share a single entry point with subcommands:

```bash
# Train
bash scripts/train.sh
# or directly:
python run_qwen2_vl.py train --output-dir ./outputs/run1

# Evaluate
bash scripts/eval.sh
# with adapter:
ADAPTER=./outputs/run1/checkpoint-50000 bash scripts/eval.sh
```

See `python run_qwen2_vl.py train --help` and `eval --help` for all flags.
