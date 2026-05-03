"""Qwen-VL chat-template helpers shared by training and evaluation scripts."""

from qwen_vl_utils import process_vision_info


SYSTEM_MESSAGE = (
    "You are a Vision Language Model specialized in image captioning.\n"
    "Your task is to analyze the provided image and generate an appropriate caption.\n"
    "The image is cleared only in the region corresponding to the caption, and all other parts are blurred.\n"
    "Focus on delivering accurate, succinct captions based on the visual information. "
    "Avoid additional explanation unless absolutely necessary."
)

DEFAULT_PROMPT = "Describe the given image."
CAP_PROMPT = "Describe a sentence for the given image."
PAR_PROMPT = (
    "The given image defines several objects as a group and creates a bounding box. "
    "Describe this bounding box at paragraph level. "
    "Paragraphs must consist of at least three sentences."
)
PAR_PROMPT_EACH_BBOX = (
    "The given image defines several colored-bounding-boxed objects as a group and creates a big region. "
    "Describe this big region at paragraph level. "
    "Paragraphs must consist of at least three sentences."
)
EDIT_PROMPT = (
    "In the given [caption], make sure to generate only the words without article that "
    "correspond to the {object}"
)


def format_data(caption, image, prompt=DEFAULT_PROMPT, system_message=SYSTEM_MESSAGE):
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": caption}]},
    ]


def generate_text_from_sample(model, processor, samples, max_new_tokens=1024, device="cuda"):
    texts = [
        processor.apply_chat_template(sample[1:2], tokenize=False, add_generation_prompt=True)
        for sample in samples
    ]
    image_inputs, _ = process_vision_info(samples)

    model_inputs = processor(
        text=texts, images=image_inputs, padding=True, return_tensors="pt"
    ).to(device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens).to("cpu")
    return [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
