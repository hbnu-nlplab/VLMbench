"""Shared evaluation helpers used by test/run scripts."""

import json

from .image import transform_images as _transform_images


# Subset of test images selected for qualitative analysis.
QUAL_ANAL_IMAGE_IDS = [
    '2363033', '2362969', '2362742', '2362682', '2341610', '2341436', '2341272',
    '2341163', '2341046', '2340962', '2340908', '2340907', '2340806', '2340762',
    '2340743', '2340711', '2340660', '2340484', '2340480', '2340476', '2340367',
    '2340303', '2340293', '2340230', '2340111', '2340104', '2340017', '2339863',
    '2339852', '2339759', '2339725', '2339715', '2339606', '2339537', '2339515',
    '2339510',
]


def load_synsets(caption_dir):
    """Load train/test keyword dictionaries used for KE accuracy scoring."""
    with open(str(caption_dir / "train_keyword_dict.json"), "r") as f:
        tr_synset = {k.split('.')[0]: v for k, v in json.load(f).items()}
    with open(str(caption_dir / "test_keyword_dict.json"), "r") as f:
        te_synset = {k.split('.')[0]: v for k, v in json.load(f).items()}

    syn_lst = list({
        v.lower()
        for synset in (tr_synset, te_synset)
        for vs in synset.values()
        for v in vs
    })
    return te_synset, syn_lst


def apply_task_b(eval_data, eval_p_data, knowledge_edit=False):
    """Swap eval_data fields with paragraph-level (task B) versions in place."""
    eval_data['image_path'] = eval_p_data['image_path']
    eval_data['caption'] = eval_p_data['paragraph']
    eval_data['bounding_box'] = eval_p_data['sub_region_boxes']
    eval_data['candidates'] = [[p] for p in eval_p_data['paragraph']]
    return eval_p_data['objs_in_paragraph'] if knowledge_edit else None


def make_image_ids(image_paths):
    """Generate '<img_id>-<region_idx>.jpg' for each image path."""
    ids = []
    prev = ''
    counter = 0
    for path in image_paths:
        img_id = path.split('/')[-1].split('.')[0]
        counter = 0 if img_id != prev else counter + 1
        ids.append(f"{img_id}-{counter}.jpg")
        prev = img_id
    return ids


def build_eval_inputs(eval_data, transform_fn):
    """Apply blur+crop transforms to each eval record."""
    vl_records, crop_images = [], []
    for c, ip, bb, pp in zip(
        eval_data["caption"], eval_data["image_path"],
        eval_data["bounding_box"], eval_data["prompt"],
    ):
        vl_records.append({
            "image": transform_fn(ip, bb, "blur"),
            "caption": c, "prompt": pp, "image_path": ip,
        })
        crop_images.append(transform_fn(ip, bb, "crop"))
    return vl_records, crop_images


def filter_null_candidates(formatted, candidates, crop_images):
    """Drop entries with empty candidate lists."""
    new_f, new_c, new_i = [], [], []
    for cand, ed, cimg in zip(candidates, formatted, crop_images):
        if cand:
            new_f.append(ed)
            new_c.append(cand)
            new_i.append(cimg)
    return new_f, new_c, new_i


def make_image_transform(model_name_or_path, each_bbox=False):
    """Return a transform_images callable that downscales for 7B-class models."""
    max_dim = 900 if "7B" in model_name_or_path else None

    def transform(image_file, bbox_lst, type="blur"):
        return _transform_images(
            image_file, bbox_lst, type=type, each_bbox=each_bbox, max_dim=max_dim,
        )

    return transform
