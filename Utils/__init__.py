from .Utils import Tokenizer
from collections import defaultdict
import json
import random
import numpy as np
from typing import List
from sumy.parsers.plaintext import PlaintextParser
import os


randBinList = lambda n: [random.randint(0, 1) for _ in range(1, n + 1)]


def join_sentences(doc):
    return " ".join([" ".join(_) for _ in doc])


def join_docs(docs):
    return [join_sentences(d) for d in docs]


def generate_summary(doc):
    return "।\n".join([" ".join(_) for _ in doc])


def load_document(doc_fname, ref_fname):
    t = Utils.Tokenizer()
    parser = PlaintextParser.from_file(doc_fname, t)
    document = []
    for s in parser.document.sentences:
        words = s.words
        if len(words) != 1:
            document.append(words)

    r_parser = PlaintextParser.from_file(ref_fname, t)
    reference = []
    for s in r_parser.document.sentences:
        words = s.words
        if len(words) != 1:
            reference.append(words)

    return document, reference


def load_documents(file_name, ref_dir):
    t = Utils.Tokenizer()
    parser = PlaintextParser.from_file(file_name, t)
    document = []
    for s in parser.document.sentences:
        words = s.words
        if len(words) != 1:
            document.append(words)

    ref = []
    for r_fn in os.listdir(ref_dir):
        parser = PlaintextParser.from_file(ref_dir + "/" + r_fn, t)
        doc = []
        for s in parser.document.sentences:
            words = s.words
            if len(words) != 1:
                doc.append(words)

        ref.append(doc)

    return document, ref


def process_document(document):
    stop_words = load_stop_words()
    p_doc = []
    for i, s in enumerate(document):
        q = []
        for word in s:
            if word not in stop_words:
                q.append(word)

        p_doc.append(list(map(process_word, q)))

    return p_doc


def process_documents(docs):
    p_docs = []
    for d in docs:
        p_docs.append(process_document(d))
    return p_docs


def process_word(w):
    word = Tokenizer.clean_word(w)
    word = Tokenizer.stem_word(word)
    return word


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
