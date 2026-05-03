import os
import json

from tqdm import tqdm

from . import IMG_DIR


def get_each_json(ann_paths):
    annotation = []
    for ann_path in ann_paths:
        data = json.load(open(ann_path, "r"))
        if isinstance(data, dict):
            annotation.append(data)
    return annotation


def get_captions(annotation, eval=False, prompt_c="", prompt_b="", include_bbox=False, counterfactual=False):
    vis_root = str(IMG_DIR)
    capkey = "counterfactual_caption" if counterfactual else "caption"

    data = {'image_path': [], 'caption': [], 'prompt': [], 'bounding_box': [], 'candidates': []}
    p_data = {'image_path': [], 'paragraph': [], 'sub_regions': [], 'sub_region_boxes': [],
              'p_bounding_box': [], 'prompt': []}

    default_c = "Describe a sentence for the given image."
    default_b = ("The given image defines several objects as a group and creates a bounding box. "
                 "Describe this bounding box at paragraph level. Paragraphs must consist of at least three sentences.")

    for record in tqdm(annotation, total=len(annotation)):
        image_id = next(iter(record.keys()))
        image_path = os.path.join(vis_root, image_id + ".jpg")

        for region in record[image_id]['regions']:
            bbox = (region['x'], region['y'],
                    region['x'] + region['width'], region['y'] + region['height'])
            captions = region['captions'][:1] if eval else region['captions']

            for cap_dict in captions:
                prompt = prompt_c or default_c
                if include_bbox:
                    prompt += f" Refer to the position of the bounding box, which is {bbox}."

                data['prompt'].append(prompt)
                data['image_path'].append(image_path)
                data['caption'].append(cap_dict[capkey])
                data['bounding_box'].append([bbox])

            data['candidates'].append([cap_dict[capkey] for cap_dict in region['captions']])

        for rr in record[image_id]["relation_centric_regions"]:
            x, y, w, h = 999, 999, 0, 0
            err_flag = False
            tmp_region_boxes = []
            if len(rr['region_ids']) == 0:
                print(image_id)
                continue
            for rid in rr['region_ids']:
                try:
                    rid = int(rid.split('_')[-1].replace(',', ''))
                except Exception:
                    print(rid)
                    err_flag = True
                    continue

                try:
                    region = record[image_id]['regions'][rid]
                    b_x = region['x']
                    b_w = region['width'] + region['x']
                    b_y = region['y']
                    b_h = region['height'] + region['y']

                    if b_x < x: x = b_x
                    if b_y < y: y = b_y
                    if b_w > w: w = b_w
                    if b_h > h: h = b_h

                    tmp_region_boxes.append([b_x, b_y, b_w, b_h])
                except Exception:
                    print(rid)
                    err_flag = True
                    continue

            if not err_flag:
                p_bbox = (x, y, x + w, y + h)
                p_data['sub_region_boxes'].append(tmp_region_boxes)
                p_data['image_path'].append(image_path)
                p_data['paragraph'].append(rr['human_annotation'])
                p_data['sub_regions'].append(rr['region_ids'])
                p_data['p_bounding_box'].append(p_bbox)

                prompt = prompt_b or default_b
                if include_bbox:
                    prompt += f" Refer to the position of the bounding box, which is {p_bbox}."
                p_data['prompt'].append(prompt)

    return data, p_data


def get_objs(caption, cf_caption):
    """Extract the differing object span between a caption and its counterfactual."""
    tokens_caption = caption.split()
    tokens_cf = cf_caption.split()

    prefix_len = 0
    for w1, w2 in zip(tokens_caption, tokens_cf):
        if w1 == w2:
            prefix_len += 1
        else:
            break

    suffix_len = 0
    while (suffix_len < len(tokens_caption) - prefix_len and
           suffix_len < len(tokens_cf) - prefix_len and
           tokens_caption[-(suffix_len + 1)] == tokens_cf[-(suffix_len + 1)]):
        suffix_len += 1

    if suffix_len > 0:
        original_tokens = tokens_caption[prefix_len:-suffix_len]
        masked_caption = tokens_caption[:prefix_len] + [" {object} "] + tokens_caption[-suffix_len:]
        cf_tokens = tokens_cf[prefix_len:-suffix_len]
    else:
        original_tokens = tokens_caption[prefix_len:]
        masked_caption = tokens_caption[:prefix_len] + [" {object} "]
        cf_tokens = tokens_cf[prefix_len:]

    return " ".join(original_tokens), " ".join(cf_tokens), " ".join(masked_caption)


def get_edit_examples(annotation, eval=False, prompt="", use_cf=False):
    vis_root = str(IMG_DIR)
    data = {'image_path': [], 'caption': [], 'prompt': [], 'bounding_box': [],
            'candidates': [], 'objs': [], 'counterfactual_objs': []}
    p_data = {'image_path': [], 'paragraph': [], 'sub_regions': [], 'sub_region_boxes': [],
              'p_bounding_box': [], 'prompt': [], 'objs_in_paragraph': []}

    default_b = ("The given image defines several objects as a group and creates a bounding box. "
                 "Describe this bounding box at paragraph level. Paragraphs must consist of at least three sentences.")

    for record in tqdm(annotation, total=len(annotation)):
        image_id = next(iter(record.keys()))
        image_path = os.path.join(vis_root, image_id + ".jpg")

        obj_lst = []
        for region in record[image_id]['regions']:
            bbox = (region['x'], region['y'],
                    region['x'] + region['width'], region['y'] + region['height'])
            captions = region['captions'][:1] if eval else region['captions']

            for cap_dict in captions:
                data['image_path'].append(image_path)
                caption = cap_dict['caption']
                counterfactual_caption = cap_dict['counterfactual_caption']
                original_obj, counterfact_obj, masked_caption = get_objs(caption, counterfactual_caption)

                if use_cf:
                    obj_lst.append(counterfact_obj)
                    data['objs'].append(counterfact_obj)
                    data['candidates'].append([counterfactual_caption])
                else:
                    obj_lst.append(original_obj)
                    data['objs'].append(original_obj)
                    data['candidates'].append([caption])
                data['counterfactual_objs'].append(counterfact_obj)

                data['prompt'].append(prompt + '\n[CAPTION] ' + masked_caption)
                data['caption'].append(masked_caption)
                data['bounding_box'].append([bbox])

        for rr in record[image_id]["relation_centric_regions"]:
            x, y, w, h = 999, 999, 0, 0
            err_flag = False
            tmp_region_boxes = []
            for rid in rr['region_ids']:
                try:
                    rid = int(rid.split('_')[-1].replace(',', ''))
                except Exception:
                    print(rid)
                    err_flag = True
                    continue

                try:
                    region = record[image_id]['regions'][rid]
                    b_x = region['x']
                    b_w = region['width'] + region['x']
                    b_y = region['y']
                    b_h = region['height'] + region['y']

                    if b_x < x: x = b_x
                    if b_y < y: y = b_y
                    if b_w > w: w = b_w
                    if b_h > h: h = b_h

                    tmp_region_boxes.append([b_x, b_y, b_w, b_h])
                except Exception:
                    print(rid)
                    err_flag = True
                    continue

            if not err_flag:
                paragraph = rr['human_annotation']
                for obj in obj_lst:
                    obj_b_idx = paragraph.find(obj)
                    if obj_b_idx != -1:
                        objs_in_paragraph = paragraph[obj_b_idx:obj_b_idx + len(obj)]
                        new_paragraph = paragraph[:obj_b_idx] + "{object}" + paragraph[obj_b_idx + len(obj):]

                        p_data['sub_region_boxes'].append(tmp_region_boxes)
                        p_data['image_path'].append(image_path)
                        p_data['paragraph'].append(new_paragraph)
                        p_data['objs_in_paragraph'].append(objs_in_paragraph)
                        p_data['sub_regions'].append(rr['region_ids'])
                        p_data['p_bounding_box'].append((x, y, x + w, y + h))
                        p_data['prompt'].append(prompt or default_b)

    return data, p_data
