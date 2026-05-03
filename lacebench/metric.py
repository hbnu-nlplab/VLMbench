import evaluate
import numpy as np
import torch
import torchvision
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from statistics import mean
from torchmetrics.multimodal.clip_score import CLIPScore

print("LOAD METRIC ...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metric = evaluate.combine(["rouge", "meteor"])
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
pil_to_tensor = torchvision.transforms.PILToTensor()
print("done.")

ROUGE_METEOR_KEYS = ["rouge1", "rouge2", "rougeL", "meteor"]


def compute_clipscore(decoded_preds, images):
    """Average CLIPScore between predicted captions and their corresponding images."""
    try:
        with torch.no_grad():
            scores = [clip_metric(pil_to_tensor(img), p) for img, p in zip(images, decoded_preds)]
        return float(torch.stack(scores).mean().item()) / 100
    except Exception as e:
        print(f"[WARN] CLIPScore computation failed: {e}")
        return float("nan")


def calc_bertscore(decoded_preds, candidates_n, lang, device):
    scores = []
    for pred, refs in zip(decoded_preds, candidates_n):
        bs = bertscore.compute(
            predictions=[pred] * len(refs),
            references=refs,
            lang=lang,
            device=device,
        )
        scores.append(mean(bs["f1"]))
    return scores


def _compute_text_metrics(decoded_preds, candidates, lang):
    """Compute ROUGE / METEOR / BLEU-{1..4} / BERTScore for the given references."""
    metric.add_batch(predictions=decoded_preds, references=candidates)
    out = metric.compute()
    out.update(bleu.compute(predictions=decoded_preds, references=candidates))
    for bleu_n in [1, 2, 3]:
        out[f"bleu-{bleu_n}"] = bleu.compute(
            predictions=decoded_preds, references=candidates, max_order=bleu_n,
        )["bleu"]
    out.update({k: float(out[k]) for k in ROUGE_METEOR_KEYS})
    out["bertscore_f1"] = float(mean(calc_bertscore(decoded_preds, candidates, lang, device=device)))
    return out


def _clean_predictions(preds, tokenizer):
    if hasattr(preds, "device") or hasattr(preds, "cpu"):
        preds = preds.cpu().numpy().tolist()
    decoded = tokenizer.batch_decode(
        preds, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    ) if tokenizer is not None else preds
    decoded = [
        p.replace("The image features ", "")
         .replace("The image shows ", "")
         .replace("\n", " ")
         .strip()
        for p in decoded
    ]
    return [decoded] if isinstance(decoded, str) else decoded


def compute_metrics_custom(preds, candidates, images, tokenizer, lang="en", n=5):
    decoded_preds = _clean_predictions(preds, tokenizer)

    candidates_n = [cands[:n] for cands in candidates]
    ret = {}
    for label, refs in [(f"n={n}", candidates_n), ("n=all", candidates)]:
        eval_metric = _compute_text_metrics(decoded_preds, refs, lang)
        if label == "n=all":
            eval_metric["clipscore"] = compute_clipscore(decoded_preds, images)
        ret[label] = {
            k: float(v) for k, v in eval_metric.items()
            if isinstance(v, (int, float, np.floating))
        }
    return ret, decoded_preds


def _extract_nouns(text):
    return ' '.join(word for word, pos in pos_tag(word_tokenize(text)) if pos.startswith('N'))


def compute_acc(gen_batch, objs, syn_lst, te_synset, processor):
    """Returns (acc_sum, noun_sequences). Caller divides acc_sum by len(noun_sequences)."""
    decoded_preds = processor.batch_decode(
        gen_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    gen_seq = [_extract_nouns(p) for p in decoded_preds]

    acc = 0
    for pred, gold in zip(gen_seq, objs):
        if pred == '':
            continue
        if pred == gold:
            acc += 1
        elif pred in syn_lst and gold in te_synset and pred in te_synset[gold]:
            acc += 0.8
        else:
            bs = bertscore.compute(predictions=[pred], references=[gold], lang="en", device=device)
            if bs['f1'][0] < 0.85:
                acc += bs['f1'][0]
    return acc, gen_seq
