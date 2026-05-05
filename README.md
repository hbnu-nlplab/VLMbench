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
All data lives under `dataset/` in the project root.

### Visual Genome
```bash
BASE="dataset/visual_genome"
TARGET="$BASE/VG_100K_all"
mkdir -p "$TARGET"

wget -P "$BASE" https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget -P "$BASE" https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

unzip -q "$BASE/images.zip"  -d "$TARGET"/
unzip -q "$BASE/images2.zip" -d "$TARGET"/

ls "$TARGET" | head
```

### LACE data
Download the captions from the [LACE-Bench HuggingFace dataset](https://huggingface.co/datasets/lacebench/LACE-Bench)
and convert them to the layout the loaders expect:
```bash
python scripts/download_data.py
```
This populates `dataset/lacebench/` with per-image JSON files, `lace_test.json`, and the
keyword dictionaries used for knowledge-editing evaluation.

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
