import glob
import json
import logging
import sys

import torch
from tqdm import tqdm
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)

from lacebench import CAPTION_DIR
from lacebench.chat import (
    CAP_PROMPT, EDIT_PROMPT, PAR_PROMPT,
    format_data, generate_text_from_sample,
)
from lacebench.data import get_captions, get_each_json, get_edit_examples
from lacebench.eval import (
    QUAL_ANAL_IMAGE_IDS, apply_task_b, build_eval_inputs,
    filter_null_candidates, load_synsets, make_image_ids, make_image_transform,
)
from lacebench.metric import compute_acc, compute_clipscore, compute_metrics_custom

logger = logging.getLogger(__name__)

# ---------------------- CONFIG ----------------------
MODEL_NAME_OR_PATH = "Qwen/Qwen2-VL-2B-Instruct"

KCC = False
LOAD_MODEL = True
INCLUDE_BBOX = True
KNOWLEDGE_EDIT = True
USE_CF_KE = True
QUAL_ANAL = False
EACH_BBOX = False
TASK_B = True

if KCC:
    LOAD_MODEL = False
    INCLUDE_BBOX = False
    KNOWLEDGE_EDIT = True
    EACH_BBOX = False
    TASK_B = False


def adapter_path_for(model_name_or_path, use_cf_ke):
    suffix = "-LRGR_bbox_CF/checkpoint-120000" if use_cf_ke else "_LRGR_bbox/checkpoint-candi-1"
    return f"./outputs/{model_name_or_path}{suffix}/"


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

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
    te_synset = syn_lst = None

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
    transform_fn = make_image_transform(MODEL_NAME_OR_PATH, each_bbox=EACH_BBOX)
    vl_lst, crop_img_lst = build_eval_inputs(eval_data, transform_fn)
    img_ids = make_image_ids(eval_data['image_path'])

    if QUAL_ANAL:
        for record, img_id in zip(vl_lst, img_ids):
            record['image'].save(f"qual_anal_results/{img_id}")

    prompt_per_record = [d['prompt'] if KNOWLEDGE_EDIT else prompt_active for d in vl_lst]
    formatted = [
        format_data(d['caption'], d['image'], p)
        for d, p in zip(vl_lst, prompt_per_record)
    ]

    if "2.5" in MODEL_NAME_OR_PATH:
        from transformers import Qwen2_5_VLForConditionalGeneration
        llm_model = Qwen2_5_VLForConditionalGeneration
    else:
        llm_model = Qwen2VLForConditionalGeneration

    model = llm_model.from_pretrained(
        MODEL_NAME_OR_PATH, device_map="auto", torch_dtype=torch.bfloat16,
    )
    processor = Qwen2VLProcessor.from_pretrained(MODEL_NAME_OR_PATH)

    if LOAD_MODEL:
        adapter_path = adapter_path_for(MODEL_NAME_OR_PATH, USE_CF_KE)
        print(adapter_path)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    formatted, candidates, crop_img_lst = filter_null_candidates(formatted, candidates, crop_img_lst)

    BATCH_SIZE = 1
    batched = [formatted[i:i + BATCH_SIZE] for i in range(0, len(formatted), BATCH_SIZE)]
    gen_batch = []
    print("do model")
    for batch in tqdm(batched, total=len(batched)):
        gen_batch.extend(generate_text_from_sample(model, processor, batch, **model_params))
    print("done")

    if KNOWLEDGE_EDIT:
        acc_sum, gen_seq = compute_acc(gen_batch, objs, syn_lst, te_synset, processor)
        print(gen_seq)
        ke_acc = acc_sum / len(gen_seq)
        print(f"KE ACC: {ke_acc}")
        metric_ret = {"KE_ACC": ke_acc, "CLIPScore": compute_clipscore(gen_seq, crop_img_lst)}
    else:
        print("do metrics")
        metric_ret, gen_seq = compute_metrics_custom(gen_batch, candidates, crop_img_lst, processor)

    parts = [MODEL_NAME_OR_PATH.split('/')[-1]]
    if KNOWLEDGE_EDIT: parts.append("KE")
    if INCLUDE_BBOX:   parts.append("bbox")
    if USE_CF_KE:      parts.append("CF")
    parts.append("taskB" if TASK_B else "taskC")
    result_file_name = "_".join(parts) + ".txt"

    print(metric_ret)
    print(gen_seq)
    with open(f"results/{result_file_name}", "w") as wf:
        wf.write(str(metric_ret) + "\n")
        for img_id, seq, cand in zip(img_ids, gen_seq, candidates):
            wf.write(f"IMG_ID: {img_id}\tGEN: {seq}\tGOLD: {cand[0]}\n")


if __name__ == "__main__":
    main()
