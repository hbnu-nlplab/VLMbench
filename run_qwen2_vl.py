#!/usr/bin/env python
# Adapted from HuggingFace Transformers examples (Apache-2.0).

import gc
import glob
import logging
import os
import sys
import time

os.environ["WANDB_DISABLED"] = "true"

import torch
from PIL import Image
from tqdm import tqdm

import transformers
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Qwen2VLProcessor,
    BitsAndBytesConfig,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from lacebench import CAPTION_DIR, IMG_DIR
from lacebench.chat import (
    CAP_PROMPT, PAR_PROMPT, PAR_PROMPT_EACH_BBOX, format_data,
)
from lacebench.data import get_captions, get_each_json
from lacebench.image import blur_except_boxes, draw_bounding_boxes

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0",
                "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt")

# ---------------------- CONFIG ----------------------
INCLUDE_BBOX = True
EACH_BBOX = False
efficient_memory = True
LOAD_MODEL = None  # None or checkpoint number, e.g. "60000"
USE_CF = False
use_task_b = True
only_task_b = False

PAR_PROMPT_ACTIVE = PAR_PROMPT_EACH_BBOX if EACH_BBOX else PAR_PROMPT

PROMPT_COLUMN = "prompt"
CAPTION_COLUMN = "caption"
IMAGE_COLUMN = "image_path"
BBOX_COLUMN = "bounding_box"


def get_captions_qwen(annotation, eval=False):
    """Eager (non-efficient) caption loader. Used when efficient_memory=False."""
    vis_root = str(IMG_DIR)
    data = []
    for record in tqdm(annotation, total=len(annotation)):
        image_id = next(iter(record.keys()))
        image_path = os.path.join(vis_root, image_id + ".jpg")
        image = Image.open(image_path).convert("RGB")

        for region in record[image_id]['regions']:
            bbox = (region['x'], region['y'],
                    region['x'] + region['width'], region['y'] + region['height'])
            blured_image = blur_except_boxes(image, [bbox])

            captions = region['captions'][:1] if eval else region['captions']
            for caption in captions:
                data.append({
                    'image': blured_image,
                    'bounding_box': bbox,
                    'caption': caption['caption'],
                })
    return data


def transform_images(examples):
    captions = list(examples[CAPTION_COLUMN])
    images = list(examples[IMAGE_COLUMN])
    convs = [format_data(cap, img) for cap, img in zip(captions, images)]

    for conv, bbox in zip(convs, examples[BBOX_COLUMN]):
        image_file = conv[1]["content"][0]["image"]
        image = Image.open(image_file).convert("RGB")
        ret_image = blur_except_boxes(image, bbox)
        ret_image = draw_bounding_boxes(ret_image, bbox, each_bbox=EACH_BBOX)
        conv[1]["content"][0]["image"] = ret_image

    from qwen_vl_utils import process_vision_info
    image_inputs = [process_vision_info(conv)[0] for conv in convs]
    return {"texts": examples["texts"], "image_inputs": image_inputs}


def clear_memory():
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    print(f"GPU allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"GPU reserved:  {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")


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
    """Append (or replace, if `replace=True`) task-B paragraph data into task-A dicts."""
    op = (lambda a, b: b) if replace else (lambda a, b: a + b)
    for src, dst in [(train_p_data, train_data), (eval_p_data, eval_data)]:
        dst['prompt'] = op(dst['prompt'], src['prompt'])
        dst['image_path'] = op(dst['image_path'], src['image_path'])
        dst['caption'] = op(dst['caption'], src['paragraph'])
        dst['bounding_box'] = op(dst['bounding_box'], src['sub_region_boxes'])


def main():
    model_name_or_path = "Qwen/Qwen2-VL-2B-Instruct"

    training_args = SFTConfig(
        output_dir=f"./outputs/{model_name_or_path}_GR_bbox_2e-4",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        logging_steps=10,
        eval_steps=10000,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=10000,
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
    training_args.dataloader_num_workers = 16
    preprocessing_num_workers = None

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
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

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Load dataset
    train_data = get_each_json(sorted(glob.glob(str(CAPTION_DIR / 'train/*json'))))
    eval_data = train_data[len(train_data) - len(train_data) // 10:]
    train_data = train_data[:len(train_data) - len(train_data) // 10]

    if not efficient_memory:
        train_data = get_captions_qwen(train_data)
        eval_data = get_captions_qwen(eval_data, eval=True)
        train_data = [format_data(d['caption'], d['image']) for d in train_data]
        eval_data = [format_data(d['caption'], d['image']) for d in eval_data]
    else:
        train_data, train_p_data = get_captions(
            train_data, eval=False, prompt_c=CAP_PROMPT, prompt_b=PAR_PROMPT_ACTIVE,
            include_bbox=INCLUDE_BBOX, counterfactual=USE_CF,
        )
        eval_data, eval_p_data = get_captions(
            eval_data, eval=True, prompt_c=CAP_PROMPT, prompt_b=PAR_PROMPT_ACTIVE,
            include_bbox=INCLUDE_BBOX, counterfactual=USE_CF,
        )

        del train_data['candidates']
        del eval_data['candidates']

        n_train, n_eval = len(train_data.get('caption', [])), len(eval_data.get('caption', []))
        print(f"Training samples: {n_train}, Eval samples: {n_eval}")
        for label, n in [("training", n_train), ("eval", n_eval)]:
            if n == 0:
                raise ValueError(f"No {label} samples found! Check your data path and JSON files.")

        if use_task_b or only_task_b:
            _merge_task_b(train_data, train_p_data, eval_data, eval_p_data, replace=only_task_b)

        train_data = Dataset.from_dict(train_data)
        eval_data = Dataset.from_dict(eval_data)
        column_names = train_data.column_names

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    processor = AutoProcessor.from_pretrained(model_name_or_path)

    peft_config = LoraConfig(
        lora_alpha=16, lora_dropout=0.05, r=8, bias="none",
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )

    adapter_path = False
    if LOAD_MODEL:
        adapter_path = f"./outputs/{model_name_or_path}_LRGR_bbox/checkpoint-{LOAD_MODEL}"
        model = PeftModel.from_pretrained(model, adapter_path)
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
    else:
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    set_seed(training_args.seed)

    if efficient_memory:
        def tok_cap_func(examples):
            captions = list(examples[CAPTION_COLUMN])
            images = list(examples[IMAGE_COLUMN])
            prompts = list(examples[PROMPT_COLUMN])
            convs = [format_data(cap, img, prpt) for cap, img, prpt in zip(captions, images, prompts)]
            texts = [processor.apply_chat_template(conv, tokenize=False) for conv in convs]
            return {"texts": texts}

        keep_cols = [CAPTION_COLUMN, IMAGE_COLUMN, BBOX_COLUMN]
        train_data = train_data.map(
            function=tok_cap_func, batched=True,
            remove_columns=[c for c in column_names if c not in keep_cols],
            num_proc=preprocessing_num_workers, load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
        )
        train_data.set_transform(transform_images)

        eval_data = eval_data.map(
            function=tok_cap_func, batched=True,
            remove_columns=[c for c in column_names if c not in keep_cols],
            num_proc=preprocessing_num_workers, load_from_cache_file=False,
            desc="Running tokenizer on eval dataset",
        )
        eval_data.set_transform(transform_images)

        def collate_fn(examples):
            texts = [ex['texts'] for ex in examples]
            image_inputs = [ex['image_inputs'] for ex in examples]
            batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
            batch["labels"] = _build_labels(batch["input_ids"], processor)
            return batch
    else:
        from qwen_vl_utils import process_vision_info

        def collate_fn(examples):
            texts = [processor.apply_chat_template(ex, tokenize=False) for ex in examples]
            image_inputs = [process_vision_info(ex)[0] for ex in examples]
            batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
            batch["labels"] = _build_labels(batch["input_ids"], processor)
            return batch

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collate_fn,
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
    )

    trainer.train(resume_from_checkpoint=adapter_path)
    trainer.save_model(training_args.output_dir)

    clear_memory()


if __name__ == "__main__":
    main()
