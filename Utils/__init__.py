from .Utils import Tokenizer
from collections import defaultdict
import json
from typing import List


def load_stop_words():
    f = open("sample_data/hindi_stop_words.json", 'r', encoding="utf-8")
    words = json.load(f)["stop_words"]
    return frozenset(words)


def calculate_rouge(pred_sum: str, ref_sum: List[str], n: int):
    pred_ngrams = create_ngrams(pred_sum.split(" "), n)
    intersection_ngrams_count = 0
    total_ngrams_count = 0
    for s in ref_sum:
        r_ngrams = create_ngrams(s.split(" "), n)

        for ngram in r_ngrams.keys():
            intersection_ngrams_count += min(r_ngrams[ngram],
                                             pred_ngrams[ngram])
        total_ngrams_count += sum(r_ngrams.values())

    rouge = intersection_ngrams_count / total_ngrams_count
    return rouge


def create_ngrams(tokens, n):
    ngrams = defaultdict(int)
    for ngram in (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)):
        ngrams[ngram] += 1
    return ngrams
