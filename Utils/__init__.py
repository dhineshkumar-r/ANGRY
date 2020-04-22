from .Utils import Tokenizer
from collections import defaultdict
import json
import random
import numpy as np
from typing import List, DefaultDict
from sumy.parsers.plaintext import PlaintextParser
import os

random.seed(0)
np.random.seed(0)

randBinList = lambda n: [random.randint(0, 1) for _ in range(1, n + 1)]


def remove_headings(doc) -> List[List[str]]:
    n_doc = []
    for s in doc:
        if s[0] != '@':
            n_doc.append(s)
    return n_doc


def join_sentences(doc) -> str:
    return " ".join([" ".join(_) for _ in doc])


def join_docs(docs) -> List[str]:
    return [join_sentences(d) for d in docs]


def generate_summary(doc) -> str:
    return "।\n".join([" ".join(_) for _ in doc])


def load_document(doc_fname, ref_fname) -> (List[str], List[str]):
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


def process_document(document: List[List[str]], use_stop_words=True, use_lemmatizer=True) -> List[List[str]]:
    stop_words = set()
    if use_stop_words is True:
        stop_words = load_stop_words()
    p_doc = []
    for i, s in enumerate(document):
        q = []
        for word in s:
            if word not in stop_words:
                q.append(word)
        if use_lemmatizer:
            p_doc.append(list(map(process_word, q)))
        else:
            p_doc.append(list(map(process_only_clean_word, q)))

    return p_doc


def process_documents(docs: List[str]) -> List[List[List[str]]]:
    p_docs = []
    for d in docs:
        p_docs.append(process_document(d))
    return p_docs


def process_word(w: str) -> str:
    word = Tokenizer.clean_word(w)
    word = Tokenizer.stem_word(word)
    return word


def process_only_clean_word(w: str) -> str:
    word = Tokenizer.clean_word(w)
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


def find_overlap(f1: str, f2: str) -> int:
    f1_ngrams = create_ngrams(f1.split(" "), 1)
    f2_ngrams = create_ngrams(f2.split(" "), 1)
    intersection_ngrams_count = 0
    for ngram in f1_ngrams.keys():
        intersection_ngrams_count += min(f1_ngrams[ngram], f2_ngrams[ngram])

    return intersection_ngrams_count


def content_overlap_metric():
    ref_sum, sum_75, = load_document("test/references/s_chap_2.txt", "PSO_sum_1.txt")
    sum_25, sum_50 = load_document("PSO_sum_1_25.txt", "PSO_sum_1_50.txt")
    p_ref_sum = join_sentences(process_document(ref_sum, use_stop_words=True, use_lemmatizer=False))
    p_sum_75 = join_sentences(process_document(sum_75, use_stop_words=True, use_lemmatizer=False))
    p_sum_25 = join_sentences(process_document(sum_25, use_stop_words=True, use_lemmatizer=False))
    p_sum_50 = join_sentences(process_document(sum_50, use_stop_words=True, use_lemmatizer=False))
    o_75 = find_overlap(p_ref_sum, p_sum_75)
    o_50 = find_overlap(p_ref_sum, p_sum_50)
    o_25 = find_overlap(p_ref_sum, p_sum_25)
    print(o_75, o_50, o_25)
    return


def create_ngrams(tokens, n) -> DefaultDict:
    ngrams = defaultdict(int)
    for ngram in (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)):
        ngrams[ngram] += 1
    return ngrams


def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z


def get_random_nos(n):
    return np.random.rand(n)
