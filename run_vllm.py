import glob
import json
import logging
import os
import sys

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor

from lacebench import CAPTION_DIR
from lacebench.chat import (
    CAP_PROMPT, EDIT_PROMPT, PAR_PROMPT, SYSTEM_MESSAGE,
    generate_text_from_sample,
)
from lacebench.data import get_captions, get_each_json, get_edit_examples
from lacebench.eval import (
    QUAL_ANAL_IMAGE_IDS, apply_task_b, build_eval_inputs,
    filter_null_candidates, load_synsets, make_image_ids, make_image_transform,
)
from lacebench.metric import compute_acc, compute_metrics_custom

logger = logging.getLogger(__name__)

# ---------------------- CONFIG ----------------------
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")
PROMPT = "Describe the given image in one sentence."

LOAD_ADAPTER = False
ADAPTER_PATH = f"./outputs/{MODEL_NAME}_LRGR_bbox/checkpoint-candi-1/"
USE_VLLM = False
BATCH_SIZE = 1

KCC = False
INCLUDE_BBOX = True
KNOWLEDGE_EDIT = False
USE_CF_KE = False
QUAL_ANAL = False
EACH_BBOX = False
TASK_B = False

if KCC:
    INCLUDE_BBOX = False
    KNOWLEDGE_EDIT = True
    EACH_BBOX = False
    TASK_B = False
# ----------------------------------------------------


def format_data(data, prompt=PROMPT, processor=None, model_name=MODEL_NAME):
    caption, image, image_path = data['caption'], data['image'], data['image_path']

    if hasattr(processor, "apply_chat_template"):
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": caption}]},
        ]
    if "deepseek" in model_name.lower():
        return [
            {"role": "<|User|>",
             "content": f"<image>\n{prompt}",
             "images": [image_path] if isinstance(image_path, str) else [None]},
            {"role": "<|Assistant|>", "content": caption},
        ]
    return {"text": prompt, "image": image}


def load_model(model_name: str, use_vllm=False):
    if use_vllm:
        from vllm import LLM
        return LLM(model=model_name, tensor_parallel_size=1), None

    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def generate_text(model, processor, inputs, params):
    # InternVL-style chat
    if hasattr(model, "chat"):
        return [
            model.chat(d["content"][1]["image"], d["content"][1]["text"])
            for d in inputs
        ]

    # HuggingFace Vision2Seq with chat template (Qwen, LLaVA, ...)
    if hasattr(processor, "apply_chat_template"):
        return generate_text_from_sample(model, processor, inputs, **params)

    # Idefics / DeepSeek / CogVLM-style
    texts = [d["content"][1]["text"] for d in inputs]
    images = [d["content"][1]["image"] for d in inputs]
    model_inputs = processor(text=texts, images=images, return_tensors="pt").to(model.device)
    generated = model.generate(**model_inputs, max_new_tokens=params.get("max_new_tokens", 128))
    return processor.batch_decode(generated, skip_special_tokens=True)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(f"Loading model: {MODEL_NAME}")
    model, processor = load_model(MODEL_NAME, use_vllm=USE_VLLM)
    if LOAD_ADAPTER and not USE_VLLM:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    logger.info("Load data")
    if QUAL_ANAL:
        with open(str(CAPTION_DIR / "lace_test.json"), "r") as f:
            eval_data_raw = json.load(f)
        eval_data = [{img_id: eval_data_raw[img_id]} for img_id in QUAL_ANAL_IMAGE_IDS]
    elif KCC:
        eval_data = get_each_json(sorted(glob.glob(str(CAPTION_DIR / 'kcc/raw/*json'))))
    else:
        eval_data = get_each_json(sorted(glob.glob(str(CAPTION_DIR / 'test/*json'))))

    model_params = {}
    objs = None
    syn_lst = te_synset = None

    if KNOWLEDGE_EDIT:
        te_synset, syn_lst = load_synsets(CAPTION_DIR)
        eval_data, eval_p_data = get_edit_examples(
            eval_data, eval=True, prompt=EDIT_PROMPT, use_cf=USE_CF_KE,
        )
        model_params["max_new_tokens"] = 3
        objs = eval_data['objs']
    else:
        eval_data, eval_p_data = get_captions(
            eval_data, eval=True, prompt_c=CAP_PROMPT, prompt_b=PAR_PROMPT,
            include_bbox=INCLUDE_BBOX,
        )
        model_params["max_new_tokens"] = 64

    prompt_active = CAP_PROMPT
    if TASK_B:
        objs_b = apply_task_b(eval_data, eval_p_data, knowledge_edit=KNOWLEDGE_EDIT)
        prompt_active = PAR_PROMPT
        if KNOWLEDGE_EDIT:
            model_params["max_new_tokens"] = 5
            objs = objs_b
        else:
            model_params["max_new_tokens"] = 512

    candidates = eval_data["candidates"]
    transform_fn = make_image_transform(MODEL_NAME, each_bbox=EACH_BBOX)
    vl_lst, crop_img_lst = build_eval_inputs(eval_data, transform_fn)

    if QUAL_ANAL:
        for record, img_id in zip(vl_lst, make_image_ids(eval_data['image_path'])):
            record['image'].save(f"qual_anal_results/{img_id}")

    prompt_per_record = [d['prompt'] if KNOWLEDGE_EDIT else prompt_active for d in vl_lst]
    formatted = [
        format_data(d, p, processor, MODEL_NAME)
        for d, p in zip(vl_lst, prompt_per_record)
    ]

    formatted, candidates, crop_img_lst = filter_null_candidates(formatted, candidates, crop_img_lst)

    batches = [formatted[i:i + BATCH_SIZE] for i in range(0, len(formatted), BATCH_SIZE)]
    all_gen = []
    for batch in tqdm(batches, total=len(batches), desc="Run model"):
        all_gen.extend(generate_text(model, processor, batch, model_params))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if KNOWLEDGE_EDIT:
        acc_sum, _ = compute_acc(all_gen, objs, syn_lst, te_synset, processor)
        print(f"Knowledge Editing Accuracy: {acc_sum:.4f}")
    else:
        metric_ret, _ = compute_metrics_custom(all_gen, candidates, crop_img_lst, processor)
        print(metric_ret)


if __name__ == "__main__":
    main()
