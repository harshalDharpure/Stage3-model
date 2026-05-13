"""
Evaluation metrics for generation task.
Implements BLEU, ROUGE, METEOR, BERTScore, and NLI entailment consistency.
"""

import json
from typing import List, Dict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    from bert_score import score as bert_score

    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

try:
    from nltk.translate.meteor_score import meteor_score

    METEOR_AVAILABLE = True
except ImportError:
    METEOR_AVAILABLE = False


def calculate_bleu(reference: str, candidate: str) -> Dict[str, float]:
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    smoothing = SmoothingFunction().method1
    bleu_1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu_2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu_3 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu_4 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    return {"bleu_1": float(bleu_1), "bleu_2": float(bleu_2), "bleu_3": float(bleu_3), "bleu_4": float(bleu_4)}


def calculate_rouge(reference: str, candidate: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "rouge_1_f1": scores["rouge1"].fmeasure,
        "rouge_2_f1": scores["rouge2"].fmeasure,
        "rouge_l_f1": scores["rougeL"].fmeasure,
    }


def calculate_meteor(reference: str, candidate: str) -> float:
    if not METEOR_AVAILABLE:
        return 0.0
    try:
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        return float(meteor_score([ref_tokens], cand_tokens))
    except Exception:
        return 0.0


def calculate_bertscore(references: List[str], candidates: List[str], lang: str = "en") -> Dict[str, float]:
    if not BERTSCORE_AVAILABLE:
        return {"bertscore_f1": 0.0}
    try:
        _, _, F1 = bert_score(candidates, references, lang=lang, verbose=False)
        return {"bertscore_f1": float(F1.mean().item())}
    except Exception:
        return {"bertscore_f1": 0.0}


def calculate_all_metrics(reference: str, candidate: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics.update(calculate_bleu(reference, candidate))
    metrics.update(calculate_rouge(reference, candidate))
    metrics["meteor"] = calculate_meteor(reference, candidate)
    return metrics


def calculate_batch_metrics(
    references: List[str],
    candidates: List[str],
    lang: str = "en",
    *,
    include_bertscore: bool = True,
) -> Dict[str, float]:
    all_metrics = {
        "bleu_1": [],
        "bleu_2": [],
        "bleu_3": [],
        "bleu_4": [],
        "rouge_1_f1": [],
        "rouge_2_f1": [],
        "rouge_l_f1": [],
        "meteor": [],
    }
    for ref, cand in zip(references, candidates):
        m = calculate_all_metrics(ref, cand)
        for k in all_metrics:
            all_metrics[k].append(m[k])
    avg_metrics = {k: (sum(v) / len(v) if v else 0.0) for k, v in all_metrics.items()}
    if include_bertscore:
        avg_metrics.update(calculate_bertscore(references, candidates, lang=lang))
    else:
        avg_metrics["bertscore_f1"] = 0.0
    return avg_metrics


def calculate_response_length_stats(references: List[str], candidates: List[str]) -> Dict[str, float]:
    ref_lengths = [len(ref.split()) for ref in references]
    cand_lengths = [len(cand.split()) for cand in candidates]
    avg_ref = sum(ref_lengths) / len(ref_lengths) if ref_lengths else 0.0
    avg_cand = sum(cand_lengths) / len(cand_lengths) if cand_lengths else 0.0
    return {
        "avg_reference_length": avg_ref,
        "avg_candidate_length": avg_cand,
        "length_ratio": (avg_cand / avg_ref) if avg_ref > 0 else 0.0,
        "length_difference": avg_cand - avg_ref,
    }


# ---------------------------------------------------------------------------
# NLI (Natural Language Inference) score: entailment consistency
# Reference = premise, Candidate = hypothesis; score = mean P(entailment).
# ---------------------------------------------------------------------------
NLI_AVAILABLE = False
try:
    import torch as _torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _nli_model = None
    _nli_tokenizer = None
    _NLI_LABEL_ENTAILMENT = 2  # MNLI: 0=contradiction, 1=neutral, 2=entailment
    NLI_AVAILABLE = True
except ImportError:
    pass


def _get_nli_model():
    global _nli_model, _nli_tokenizer
    if _nli_model is None and NLI_AVAILABLE:
        try:
            _nli_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base-mnli")
            _nli_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base-mnli")
            _nli_model.eval()
            if _torch.cuda.is_available():
                _nli_model = _nli_model.cuda()
        except Exception:
            return None, None
    return _nli_model, _nli_tokenizer


def calculate_nli_score(
    references: List[str],
    candidates: List[str],
    max_length: int = 256,
    batch_size: int = 8,
) -> Dict[str, float]:
    if not references or not candidates or len(references) != len(candidates):
        return {"nli_score": 0.0}
    model, tokenizer = _get_nli_model()
    if model is None or tokenizer is None:
        return {"nli_score": 0.0}
    device = next(model.parameters()).device
    entailment_probs = []
    for i in range(0, len(references), batch_size):
        batch_ref = references[i : i + batch_size]
        batch_cand = candidates[i : i + batch_size]
        premises = [(" ".join(r.split()[:max_length])) for r in batch_ref]
        hypotheses = [(" ".join(c.split()[:max_length])) for c in batch_cand]
        try:
            inputs = tokenizer(premises, hypotheses, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with _torch.no_grad():
                logits = model(**inputs).logits
            probs = _torch.softmax(logits, dim=1)
            ent = probs[:, _NLI_LABEL_ENTAILMENT].cpu().tolist()
            entailment_probs.extend(ent)
        except Exception:
            entailment_probs.extend([0.0] * len(premises))
    return {"nli_score": float(sum(entailment_probs) / len(entailment_probs)) if entailment_probs else 0.0}

