#!/usr/bin/env python3
"""
Cross-Attention Analysis (Image-Token -> LM Decoder)
- Extract cross-attention weights for decoder output tokens.
- Reconstruct 2D patch-grid heatmaps.
- Visualize per-token and aggregated heatmaps over the image.
- Quantitatively compare attention mass inside a local bbox vs global context.

Usage (example):
    python scripts/cross_attention_analysis.py \
      --image /path/to/img.jpg \
      --prompt "Describe the image." \
      --bbox "x1,y1,x2,y2" \
      --model "Qwen/Qwen2-VL-2B-Instruct" \
      --grid 14 \
      --out out_dir

Notes:
- This script tries to work with HuggingFace vision->seq / encoder-decoder models.
- If the model places both text+visual tokens in encoder seq, the script will
  attempt to infer which encoder positions correspond to image patches. You can
  override grid size with --grid if auto-detection is inaccurate.
"""

import argparse, os, math, json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from peft import PeftModel

from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM

# small helpers
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def parse_bbox(s):
    # accepts "x1,y1,x2,y2" or "x,y,w,h"
    parts = [float(x) for x in s.split(",")]
    if len(parts) == 4:
        x1, y1, x2, y2 = parts
        # detect if given as x,w style (if x2<=1 or x2<w) can't be perfect; assume x2>x1 -> x2 is x2
        if x2 > x1 and y2 > y1:
            return [x1, y1, x2, y2]
    raise ValueError("bbox must be 'x1,y1,x2,y2' in pixel coords")

def to_device(obj, device):
    if isinstance(obj, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k,v in obj.items()}
    return obj.to(device)

def aggregate_attentions(cross_attentions, layer_agg="mean", head_agg="sum"):
    # cross_attentions: tuple/list of layers, each (batch, n_heads, tgt_len, src_len)
    # returns (tgt_len, src_len) aggregated across layers & heads
    arrs = [ca.detach().cpu().numpy() for ca in cross_attentions]
    # stack -> (n_layers, batch, heads, tgt, src)
    stacked = np.stack(arrs, axis=0)
    # assume batch=1
    if stacked.shape[1] != 1:
        stacked = stacked[:,0]  # (n_layers, heads, tgt, src)
    else:
        stacked = stacked[:,0]
    # layer reduction
    if layer_agg == "mean":
        L = stacked.mean(axis=0)  # (heads, tgt, src)
    elif layer_agg == "sum":
        L = stacked.sum(axis=0)
    else:
        L = stacked[-1]  # last layer
    # head reduction
    if head_agg == "sum":
        H = L.sum(axis=0)  # (tgt, src)
    else:
        H = L.mean(axis=0)
    return H  # (tgt_len, src_len)

def detect_image_token_span(src_len, encoder_input_ids=None, processor=None):
    # Best-effort: if tokenizer/processor provided and encoder_input_ids exists,
    # try to detect special image tokens by checking for special token ids.
    # Fallback: assume all src are image tokens.
    if encoder_input_ids is None:
        return 0, src_len
    # If encoder_input_ids is tensor shape (batch, seq)
    arr = encoder_input_ids.cpu().numpy().reshape(-1)
    # heuristic: image placeholder might be a single sentinel token repeated -> find long run of same id
    # fallback: treat entire encoder as image tokens if there are no text tokens (e.g., all -100)
    # This is heuristic and may not be correct for all models.
    unique, counts = np.unique(arr, return_counts=True)
    # if there's single repeated token dominating -> treat as image region
    if len(unique) == 1:
        return 0, src_len
    # else fallback: assume last N tokens are image tokens where N is perfect square ~ src_len
    return 0, src_len

def map_bbox_to_grid(bbox, img_w, img_h, grid_size):
    # bbox: [x1,y1,x2,y2] in pixel coords
    x1,y1,x2,y2 = bbox
    # clamp
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    # normalized coords
    nx1, ny1 = x1 / img_w, y1 / img_h
    nx2, ny2 = x2 / img_w, y2 / img_h
    gx1, gy1 = int(math.floor(nx1 * grid_size)), int(math.floor(ny1 * grid_size))
    gx2, gy2 = int(math.ceil(nx2 * grid_size)), int(math.ceil(ny2 * grid_size))
    gx1 = max(0, min(grid_size-1, gx1))
    gy1 = max(0, min(grid_size-1, gy1))
    gx2 = max(0, min(grid_size, gx2))
    gy2 = max(0, min(grid_size, gy2))
    return gx1, gy1, gx2, gy2  # note gx2,gy2 are exclusive indices if used accordingly

def heatmap_from_attention(att_vec, grid_size, image_size):
    # att_vec: (src_len,) attention over image patches arranged row-major
    # grid_size: n x n
    # image_size: (W,H)
    grid = att_vec.reshape(grid_size, grid_size)
    # normalize for visualization
    norm = (grid - grid.min()) / (grid.max() - grid.min() + 1e-12)
    import cv2
    heat = cv2.resize(norm.astype("float32"), image_size, interpolation=cv2.INTER_LINEAR)
    return heat  # HxW float

def save_overlay(image_pil, heatmap, out_path, cmap="jet", alpha=0.5):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(image_pil)
    ax.imshow(heatmap, cmap=cmap, alpha=alpha, extent=(0,image_pil.width, image_pil.height, 0))
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def analyze(model, processor, image_path, prompt, bbox=None, grid_size=None, max_new_tokens=32, device="cuda", out_dir="out"):
    ensure_dir(out_dir)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    # prepare inputs
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    inputs = to_device(inputs, device)

    # generate tokens (capture generation outputs if possible)
    model = model.to(device)
    model.eval()
    gen_ids = None
    cross_attentions = None
    try:
        # use HF generate with attentions if supported
        gen_out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 output_attentions=True, return_dict_in_generate=True)
        # gen_out may be GeneratedSequence or GenerationOutput having 'sequences' and 'attentions'
        # get generated token ids
        generated_ids = getattr(gen_out, "sequences", None)
        if generated_ids is None:
            generated_ids = gen_out  # fallback (tensor)
        gen_ids = generated_ids
        # try to extract cross attentions from generation output
        if hasattr(gen_out, "cross_attentions") and gen_out.cross_attentions is not None:
            cross_attentions = gen_out.cross_attentions
    except Exception:
        # generation with attention flags may not be supported; fallback to greedy generate then forward
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # ensure gen_ids is tensor [batch, seq]
    if isinstance(gen_ids, list):
        gen_ids = torch.stack(gen_ids).to(device)
    if isinstance(gen_ids, torch.Tensor) and gen_ids.dim()==1:
        gen_ids = gen_ids.unsqueeze(0)

    # if we didn't receive cross_attentions, re-run forward with labels to get decoder cross-attentions
    if cross_attentions is None:
        # prepare labels from generated ids (shifted inside model)
        labels = gen_ids.to(device)
        with torch.no_grad():
            # many models return dict with decoder_cross_attentions at outputs.cross_attentions or decoder.cross_attentions
            outputs = model(**{**inputs, "labels": labels}, output_attentions=True, return_dict=True)
        # try multiple attribute names
        cross_attentions = getattr(outputs, "cross_attentions", None)
        if cross_attentions is None:
            # some models store decoder cross attentions in decoder_cross_attentions
            cross_attentions = getattr(outputs, "decoder_cross_attentions", None)
        if cross_attentions is None:
            raise RuntimeError("Could not obtain cross-attention tensors from the model outputs.")

    # cross_attentions: tuple(layers) each (batch, heads, tgt_len, src_len)
    agg = aggregate_attentions(cross_attentions, layer_agg="mean", head_agg="sum")  # (tgt_len, src_len)
    tgt_len, src_len = agg.shape
    print(f"Attention shape (tgt_len, src_len): {agg.shape}")

    # detect image token span in encoder src positions
    enc_input_ids = inputs.get("input_ids", None)
    img_start, img_end = detect_image_token_span(src_len, encoder_input_ids=enc_input_ids, processor=processor)
    img_token_len = img_end - img_start
    # if token length is non-square try to infer a square grid
    infer_grid = grid_size
    if infer_grid is None:
        g = int(round(math.sqrt(img_token_len)))
        if g*g != img_token_len:
            # fallback: assume entire src are image patches
            g = int(round(math.sqrt(src_len)))
            if g*g != src_len:
                # as last resort, set grid to nearest integer sqrt(src_len)
                g = max(1, int(math.sqrt(src_len)))
        infer_grid = g

    # extract only image token positions (assume contiguous)
    # if img_token_len matches infer_grid^2 use that span, otherwise use last infer_grid^2 tokens
    desired = infer_grid*infer_grid
    if img_token_len == desired:
        img_indices = np.arange(img_start, img_end)
    else:
        # take last desired tokens
        img_indices = np.arange(max(0, src_len - desired), src_len)

    # compute per-target-token heatmaps and aggregate
    results = []
    agg_all = None
    for t in range(tgt_len):
        att_src = agg[t]  # (src_len,)
        img_att = att_src[img_indices]  # (desired,)
        # normalize image-att
        img_att_sum = img_att.sum()
        if img_att_sum > 0:
            img_att = img_att / img_att_sum
        if agg_all is None:
            agg_all = img_att.copy()
        else:
            agg_all += img_att
        # create heatmap resized to image pixels
        heat = heatmap_from_attention(img_att, infer_grid, (img_w, img_h))
        results.append({"token_index": t, "att_patch": img_att, "heatmap": heat})

    # normalize aggregated
    if agg_all is not None:
        agg_all = agg_all / (agg_all.sum() + 1e-12)
        agg_heat = heatmap_from_attention(agg_all, infer_grid, (img_w, img_h))
        save_overlay(image, agg_heat, os.path.join(out_dir, "attention_aggregated.png"))
    else:
        agg_heat = None

    # per-token saves (limited to first 10 tokens)
    for r in results[: min(10, len(results))]:
        idx = r["token_index"]
        token_heat_path = os.path.join(out_dir, f"token_{idx}_heat.png")
        save_overlay(image, r["heatmap"], token_heat_path)

    # quantitative comparison for bbox
    metrics = {}
    if bbox is not None:
        gx1, gy1, gx2, gy2 = map_bbox_to_grid(bbox, img_w, img_h, infer_grid)
        # compute mask over patches
        mask = np.zeros((infer_grid, infer_grid), dtype=np.float32)
        mask[gy1:gy2, gx1:gx2] = 1.0
        mask = mask.reshape(-1)
        in_mass_per_token = []
        out_mass_per_token = []
        for r in results:
            att_patch = r["att_patch"]  # normalized per token
            in_mass = float((att_patch * mask).sum())
            out_mass = float(((1.0-mask) * att_patch).sum())
            in_mass_per_token.append(in_mass)
            out_mass_per_token.append(out_mass)
        # aggregated
        in_agg = float((agg_all * mask).sum()) if agg_all is not None else None
        out_agg = float((agg_all * (1-mask)).sum()) if agg_all is not None else None

        metrics = {
            "grid_size": infer_grid,
            "bbox_grid": [gx1,gy1,gx2,gy2],
            "per_token_in_mean": float(np.mean(in_mass_per_token)),
            "per_token_out_mean": float(np.mean(out_mass_per_token)),
            "per_token_in_std": float(np.std(in_mass_per_token)),
            "aggregated_in": in_agg,
            "aggregated_out": out_agg,
        }
        with open(os.path.join(out_dir,"attention_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    print("Saved results in", out_dir)
    return {"per_token": results, "aggregated_heat": agg_heat, "metrics": metrics}


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", type=str, default="Describe the image in one sentence.")
    parser.add_argument("--bbox", type=str, default=None, help="x1,y1,x2,y2 (pixels)")
    parser.add_argument("--grid", type=int, default=None, help="patch grid size (n for n x n)")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--lora_path", type=str, default=None, help="Path or HF id to LoRA/adapter to load with PeftModel")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Loading model:", args.model)
    try:
        model = AutoModelForVision2Seq.from_pretrained(args.model, device_map="auto", trust_remote_code=True)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model)

    # If a LoRA / PEFT adapter path/id is provided, try to wrap the model with PeftModel
    if args.lora_path:
        try:
            # prefer loading adapter with device_map to place weights properly
            model = PeftModel.from_pretrained(model, args.lora_path, device_map="auto", torch_dtype=torch.float16)
            print(f"Loaded LoRA adapter from {args.lora_path}")
        except Exception as e:
            # fallback: try loading base model first then attach adapter
            try:
                print("PeftModel.from_pretrained failed, retrying by loading base model then adapter...", e)
                base_model = AutoModelForVision2Seq.from_pretrained(args.model, trust_remote_code=True)
                model = PeftModel.from_pretrained(base_model, args.lora_path)
                print(f"Loaded LoRA adapter (fallback) from {args.lora_path}")
            except Exception as e2:
                raise RuntimeError(f"Failed to load LoRA adapter '{args.lora_path}': {e2}")

    bbox = parse_bbox(args.bbox) if args.bbox else None

    analyze(model, processor, args.image, args.prompt, bbox=bbox, grid_size=args.grid,
            max_new_tokens=args.max_new_tokens, device=device, out_dir=args.out)

if __name__ == "__main__":
    ''' Example usage:
        python scripts/cross_attention_analysis.py \
            --image /path/to/img.jpg \
            --prompt "A person riding a bike." \
            --bbox "100,50,300,250" \
            --model "Qwen/Qwen2-VL-2B-Instruct" \
            --grid 14 \
            --out results_example \
            --lora_path ./outputs/my_lora_adapter
    '''
    main_cli()