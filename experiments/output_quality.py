from rouge_score import rouge_scorer
from collections import Counter
import itertools
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from bert_score import score as bertscore
from sentence_transformers import SentenceTransformer, util


def experiment_rouge_l(prompt, generated_answer, human_answer):
    """
        the greater the better
    """

    if human_answer is None:
        raise ValueError("human answer needed for rouge score")

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(human_answer, generated_answer)['rougeL'].fmeasure
    return score

def _distinct_n(tokens, n):
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)

def experiment_distinct(prompt, generated_answer, tokenizer, human_answer=None):
    """
        the greater the better
    """
    tokens = tokenizer.encode(generated_answer)
    d1 = _distinct_n(tokens, 1)
    d2 = _distinct_n(tokens, 2)
    return 0.5 * d1 + 0.5 * d2

def experiment_self_bleu(prompt, generated_answer):
    """
        need generated_answer to be a list of strings
        the smaller the better
    """
    smoothie = SmoothingFunction().method1
    scores = []
    for i, hypo in enumerate(generated_answer):
        references = [sent.split() for j, sent in enumerate(generated_answer) if j != i]
        score = sentence_bleu(references, hypo.split(),
                              smoothing_function=smoothie, weights=(0.25,0.25,0.25,0.25))
        scores.append(score)
    return sum(scores) / len(scores)

def experiment_bleu(prompt, generated_answer, human_answer):
    """
        the greater the bettert
    """
    smoothie = SmoothingFunction().method1
    return sentence_bleu(
        [human_answer.split()],
        generated_answer.split(),
        smoothing_function=smoothie,
        weights=(0.25, 0.25, 0.25, 0.25)
    )

def experiment_meteor(prompt, generated_answer, human_answer):
    """
        the greater the better
    """
    return single_meteor_score(human_answer, generated_answer)

def experiment_token_f1(prompt, generated_answer, tokenizer, human_answer):
    """
        the greater the better
    """
    pred_tokens  = set(tokenizer.encode(generated_answer))
    gold_tokens  = set(tokenizer.encode(human_answer))
    if not gold_tokens:
        return 0.0
    tp = len(pred_tokens & gold_tokens)
    precision = tp / len(pred_tokens) if pred_tokens else 0
    recall    = tp / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def experiment_bertscore(prompt, generated_answer, human_answer):
    """
        the greater the better
    """
    P, R, F = bertscore(
        [generated_answer], [human_answer],
        lang= "en",
        rescale_with_baseline=True
    )
    return F.item()


_model_cache = {}

def _get_sbert(model_name):
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]

def experiment_sbert_cosine(prompt, generated_answer, tokenizer, human_answer,
                            model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
        the greater the better
    """
    sbert = _get_sbert(model_name)
    emb_gen  = sbert.encode(generated_answer, convert_to_tensor=True)
    emb_gold = sbert.encode(human_answer,     convert_to_tensor=True)
    return float(util.cos_sim(emb_gen, emb_gold)[0][0])
