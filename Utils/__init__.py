from .Utils import Tokenizer
from collections import defaultdict
import json
import numpy as np
from typing import List
from sumy.parsers.plaintext import PlaintextParser


def load_document(file_name):
    t = Utils.Tokenizer()
    parser = PlaintextParser.from_file(file_name, t)
    document = []
    for s in parser.document.sentences:
        words = s.words
        if len(words) != 1:
            document.append(words)

    return document


def process_document(document):

    for i, s in enumerate(document):
        document[i] = list(map(Tokenizer.clean_word, s))

    return document


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


def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z


def get_random_nos(n):
    return np.random.rand(n)
