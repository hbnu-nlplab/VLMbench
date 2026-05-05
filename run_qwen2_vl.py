#!/usr/bin/env python
# Adapted from HuggingFace Transformers examples (Apache-2.0).
"""Train or evaluate Qwen2-VL on LACE-Bench.

Usage:
    python run_qwen2_vl.py train --output-dir ./outputs/foo [flags]
    python run_qwen2_vl.py eval  [--adapter-path PATH] [flags]
"""

import argparse
import gc
import glob
import json
import logging
import os
import sys
import time

os.environ.setdefault("WANDB_DISABLED", "true")

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version
from trl import SFTConfig, SFTTrainer

from lacebench import CAPTION_DIR, IMG_DIR
from lacebench.chat import (
    CAP_PROMPT, EDIT_PROMPT, PAR_PROMPT, PAR_PROMPT_EACH_BBOX,
    format_data, generate_text_from_sample,
)
from lacebench.data import get_captions, get_each_json, get_edit_examples
from lacebench.eval import (
    QUAL_ANAL_IMAGE_IDS, apply_task_b, build_eval_inputs,
    filter_null_candidates, load_synsets, make_image_ids, make_image_transform,
)
from lacebench.image import blur_except_boxes, draw_bounding_boxes
from lacebench.metric import compute_acc, compute_clipscore, compute_metrics_custom

logger = logging.getLogger(__name__)


def _setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------- TRAIN ----------------------

def _build_labels(input_ids, processor):
    """Replace pad and image tokens with -100 so they're ignored in the LM loss."""
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    if isinstance(processor, Qwen2VLProcessor):
        image_tokens = [151652, 151653, 151655]
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for tok_id in image_tokens:
        labels[labels == tok_id] = -100
    return labels


def _merge_task_b(train_data, train_p_data, eval_data, eval_p_data, replace=False):
    """Append (or replace) task-B paragraph data into task-A dicts in place."""
    op = (lambda a, b: b) if replace else (lambda a, b: a + b)
    for src, dst in [(train_p_data, train_data), (eval_p_data, eval_data)]:
        dst['prompt'] = op(dst['prompt'], src['prompt'])
        dst['image_path'] = op(dst['image_path'], src['image_path'])
        dst['caption'] = op(dst['caption'], src['paragraph'])
        dst['bounding_box'] = op(dst['bounding_box'], src['sub_region_boxes'])


def _eager_caption_records(annotation, eval=False):
    """Eager (non-efficient) caption loader. Loads all images upfront."""
    vis_root = str(IMG_DIR)
    data = []
    for record in tqdm(annotation, total=len(annotation)):
        image_id = next(iter(record.keys()))
        image = Image.open(os.path.join(vis_root, image_id + ".jpg")).convert("RGB")
        for region in record[image_id]['regions']:
            bbox = (region['x'], region['y'],
                    region['x'] + region['width'], region['y'] + region['height'])
            blured = blur_except_boxes(image, [bbox])
            for caption in (region['captions'][:1] if eval else region['captions']):
                data.append({'image': blured, 'bounding_box': bbox, 'caption': caption['caption']})
    return data


def _clear_memory():
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    print(f"GPU allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"GPU reserved:  {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")


def train(args):
    require_version("datasets>=1.8.0",
                    "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt")

    par_prompt_active = PAR_PROMPT_EACH_BBOX if args.each_bbox else PAR_PROMPT

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        logging_steps=10,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=args.save_steps,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        push_to_hub=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    training_args.remove_unused_columns = False
    training_args.save_safetensors = False
    training_args.dataloader_num_workers = args.num_workers

    _setup_logging()
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, "
        f"16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if (os.path.isdir(training_args.output_dir) and training_args.do_train
            and not training_args.overwrite_output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    train_data = get_each_json(sorted(glob.glob(str(CAPTION_DIR / 'train/*json'))))
    split = len(train_data) - len(train_data) // 10
    eval_data = train_data[split:]
    train_data = train_data[:split]

    if not args.efficient_memory:
        train_data = [format_data(d['caption'], d['image']) for d in _eager_caption_records(train_data)]
        eval_data = [format_data(d['caption'], d['image']) for d in _eager_caption_records(eval_data, eval=True)]
        column_names = None
    else:
        train_data, train_p_data = get_captions(
            train_data, eval=False, prompt_c=CAP_PROMPT, prompt_b=par_prompt_active,
            include_bbox=args.include_bbox, counterfactual=args.use_cf,
        )
        eval_data, eval_p_data = get_captions(
            eval_data, eval=True, prompt_c=CAP_PROMPT, prompt_b=par_prompt_active,
            include_bbox=args.include_bbox, counterfactual=args.use_cf,
        )
        del train_data['candidates']
        del eval_data['candidates']

        n_train, n_eval = len(train_data.get('caption', [])), len(eval_data.get('caption', []))
        print(f"Training samples: {n_train}, Eval samples: {n_eval}")
        for label, n in [("training", n_train), ("eval", n_eval)]:
            if n == 0:
                raise ValueError(f"No {label} samples found! Check your data path and JSON files.")

        if args.task_b or args.only_task_b:
            _merge_task_b(train_data, train_p_data, eval_data, eval_p_data, replace=args.only_task_b)

        train_data = Dataset.from_dict(train_data)
        eval_data = Dataset.from_dict(eval_data)
        column_names = train_data.column_names

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    peft_config = LoraConfig(
        lora_alpha=16, lora_dropout=0.05, r=8, bias="none",
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )

    resume_path = False
    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint
        model = PeftModel.from_pretrained(model, resume_path)
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
    else:
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    set_seed(training_args.seed)

    PROMPT_COL, CAP_COL, IMG_COL, BBOX_COL = "prompt", "caption", "image_path", "bounding_box"

    if args.efficient_memory:
        def tok_cap_func(examples):
            convs = [
                format_data(cap, img, prpt) for cap, img, prpt in zip(
                    examples[CAP_COL], examples[IMG_COL], examples[PROMPT_COL],
                )
            ]
            return {"texts": [processor.apply_chat_template(c, tokenize=False) for c in convs]}

        def transform_images_map(examples):
            convs = [format_data(cap, img) for cap, img in zip(examples[CAP_COL], examples[IMG_COL])]
            for conv, bbox in zip(convs, examples[BBOX_COL]):
                image = Image.open(conv[1]["content"][0]["image"]).convert("RGB")
                ret = blur_except_boxes(image, bbox)
                ret = draw_bounding_boxes(ret, bbox, each_bbox=args.each_bbox)
                conv[1]["content"][0]["image"] = ret
            return {
                "texts": examples["texts"],
                "image_inputs": [process_vision_info(c)[0] for c in convs],
            }

        keep_cols = [CAP_COL, IMG_COL, BBOX_COL]
        train_data = train_data.map(
            tok_cap_func, batched=True,
            remove_columns=[c for c in column_names if c not in keep_cols],
            num_proc=args.preprocessing_num_workers, load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
        )
        train_data.set_transform(transform_images_map)
        eval_data = eval_data.map(
            tok_cap_func, batched=True,
            remove_columns=[c for c in column_names if c not in keep_cols],
            num_proc=args.preprocessing_num_workers, load_from_cache_file=False,
            desc="Running tokenizer on eval dataset",
        )
        eval_data.set_transform(transform_images_map)

        def collate_fn(examples):
            batch = processor(
                text=[ex['texts'] for ex in examples],
                images=[ex['image_inputs'] for ex in examples],
                return_tensors="pt", padding=True,
            )
            batch["labels"] = _build_labels(batch["input_ids"], processor)
            return batch
    else:
        def collate_fn(examples):
            batch = processor(
                text=[processor.apply_chat_template(ex, tokenize=False) for ex in examples],
                images=[process_vision_info(ex)[0] for ex in examples],
                return_tensors="pt", padding=True,
            )
            batch["labels"] = _build_labels(batch["input_ids"], processor)
            return batch

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=train_data, eval_dataset=eval_data,
        data_collator=collate_fn, peft_config=peft_config,
        tokenizer=processor.tokenizer,
    )

    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model(training_args.output_dir)
    _clear_memory()


# ---------------------- EVAL ----------------------

def _max_new_tokens(knowledge_edit, task_b):
    return {
        (False, False): 64,
        (False, True): 512,
        (True, False): 3,
        (True, True): 5,
    }[knowledge_edit, task_b]


def _result_filename(args):
    parts = [args.model_name_or_path.split('/')[-1]]
    if args.knowledge_edit: parts.append("KE")
    if args.include_bbox:   parts.append("bbox")
    if args.use_cf_ke:      parts.append("CF")
    parts.append("taskB" if args.task_b else "taskC")
    return "_".join(parts) + ".txt"


def evaluate(args):
    _setup_logging()
    logger.info("Load data")

    if args.qual_anal:
        with open(str(CAPTION_DIR / "lace_test.json"), "r") as f:
            raw = json.load(f)
        eval_records = [{img_id: raw[img_id]} for img_id in QUAL_ANAL_IMAGE_IDS]
    else:
        eval_records = get_each_json(sorted(glob.glob(str(CAPTION_DIR / 'test/*json'))))

    objs = None
    te_synset = syn_lst = None

    if args.knowledge_edit:
        te_synset, syn_lst = load_synsets(CAPTION_DIR)
        eval_data, eval_p_data = get_edit_examples(
            eval_records, eval=True, prompt=EDIT_PROMPT, use_cf=args.use_cf_ke,
        )
        objs = eval_data['objs']
    else:
        eval_data, eval_p_data = get_captions(
            eval_records, eval=True, prompt_c=CAP_PROMPT, prompt_b=PAR_PROMPT,
            include_bbox=args.include_bbox,
        )

    prompt_active = CAP_PROMPT
    if args.task_b:
        objs_b = apply_task_b(eval_data, eval_p_data, knowledge_edit=args.knowledge_edit)
        prompt_active = PAR_PROMPT
        if args.knowledge_edit:
            objs = objs_b

    model_params = {"max_new_tokens": _max_new_tokens(args.knowledge_edit, args.task_b)}

    candidates = eval_data["candidates"]
    transform_fn = make_image_transform(args.model_name_or_path, each_bbox=args.each_bbox)
    vl_lst, crop_img_lst = build_eval_inputs(eval_data, transform_fn)
    img_ids = make_image_ids(eval_data['image_path'])

    if args.qual_anal:
        os.makedirs(args.qual_anal_dir, exist_ok=True)
        for record, img_id in zip(vl_lst, img_ids):
            record['image'].save(os.path.join(args.qual_anal_dir, img_id))

    formatted = [
        format_data(d['caption'], d['image'], d['prompt'] if args.knowledge_edit else prompt_active)
        for d in vl_lst
    ]

    llm_cls = Qwen2_5_VLForConditionalGeneration if "2.5" in args.model_name_or_path else Qwen2VLForConditionalGeneration

    model = llm_cls.from_pretrained(
        args.model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16,
    )
    processor = Qwen2VLProcessor.from_pretrained(args.model_name_or_path)

    if args.adapter_path:
        print(args.adapter_path)
        model = PeftModel.from_pretrained(model, args.adapter_path)

    formatted, candidates, crop_img_lst = filter_null_candidates(formatted, candidates, crop_img_lst)

    batched = [formatted[i:i + args.batch_size] for i in range(0, len(formatted), args.batch_size)]
    gen_batch = []
    print("do model")
    for batch in tqdm(batched, total=len(batched)):
        gen_batch.extend(generate_text_from_sample(model, processor, batch, **model_params))
    print("done")

    if args.knowledge_edit:
        acc_sum, gen_seq = compute_acc(gen_batch, objs, syn_lst, te_synset, processor)
        print(gen_seq)
        ke_acc = acc_sum / len(gen_seq)
        print(f"KE ACC: {ke_acc}")
        metric_ret = {"KE_ACC": ke_acc, "CLIPScore": compute_clipscore(gen_seq, crop_img_lst)}
    else:
        print("do metrics")
        metric_ret, gen_seq = compute_metrics_custom(gen_batch, candidates, crop_img_lst, processor)

    print(metric_ret)
    print(gen_seq)
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, _result_filename(args)), "w") as wf:
        wf.write(str(metric_ret) + "\n")
        for img_id, seq, cand in zip(img_ids, gen_seq, candidates):
            wf.write(f"IMG_ID: {img_id}\tGEN: {seq}\tGOLD: {cand[0]}\n")


# ---------------------- CLI ----------------------

def make_parser():
    parser = argparse.ArgumentParser(description="Train or evaluate Qwen2-VL on LACE-Bench.")
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-name-or-path", "-m", default="Qwen/Qwen2-VL-2B-Instruct")
    common.add_argument("--include-bbox", action=argparse.BooleanOptionalAction, default=True)
    common.add_argument("--each-bbox", action=argparse.BooleanOptionalAction, default=False)
    common.add_argument("--task-b", action=argparse.BooleanOptionalAction, default=True)
    common.add_argument("--use-cf", action=argparse.BooleanOptionalAction, default=False,
                        help="(train) Use counterfactual captions instead of original.")

    train_p = sub.add_parser("train", parents=[common], help="Train Qwen2-VL with LoRA.")
    train_p.add_argument("--output-dir", "-o", required=True)
    train_p.add_argument("--num-epochs", type=int, default=3)
    train_p.add_argument("--batch-size", type=int, default=1)
    train_p.add_argument("--grad-accum", type=int, default=16)
    train_p.add_argument("--lr", type=float, default=2e-4)
    train_p.add_argument("--save-steps", type=int, default=10000)
    train_p.add_argument("--eval-steps", type=int, default=10000)
    train_p.add_argument("--num-workers", type=int, default=16)
    train_p.add_argument("--preprocessing-num-workers", type=int, default=None)
    train_p.add_argument("--resume-from-checkpoint", default=None)
    train_p.add_argument("--efficient-memory", action=argparse.BooleanOptionalAction, default=True)
    train_p.add_argument("--only-task-b", action="store_true",
                         help="Replace task-A data with task-B (instead of concatenating).")

    eval_p = sub.add_parser("eval", parents=[common], help="Evaluate Qwen2-VL on LACE-Bench test set.")
    eval_p.add_argument("--adapter-path", default=None,
                        help="Path to LoRA adapter to load before eval (default: no adapter).")
    eval_p.add_argument("--batch-size", type=int, default=1)
    eval_p.add_argument("--knowledge-edit", action="store_true",
                        help="Run knowledge-editing evaluation instead of caption metrics.")
    eval_p.add_argument("--use-cf-ke", action="store_true",
                        help="Use counterfactual references for knowledge-editing eval.")
    eval_p.add_argument("--qual-anal", action="store_true",
                        help="Run on hand-picked qualitative-analysis subset and save images.")
    eval_p.add_argument("--qual-anal-dir", default="qual_anal_results")
    eval_p.add_argument("--results-dir", default="results")

    return parser


def main():
    args = make_parser().parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
