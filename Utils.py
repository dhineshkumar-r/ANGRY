# -*- coding: utf-8 -*-
from typing import List
import json


def load_stop_words():
    f = open("sample_data/hindi_stop_words.json", 'r', encoding="utf-8")
    words = json.load(f)["stop_words"]
    return frozenset(words)


class Tokenizer:

    def __init__(self):
        pass

    @staticmethod
    def to_sentences(text: str) -> List[str]:
        return [s.strip() for s in text.split(u"।")]

    @staticmethod
    def to_words(sentence: str) -> List[str]:
        return [w.strip() for w in sentence.split(" ")]


# TODO: Stemmer
# TODO: Lemmatizer

if __name__ == "__main__":
    # t = Tokenizer()
    # sent = t.to_words("इस पूरे काल के दौरान क्रमश: जंगलों की कटाई हो रही थी और खेती का इलाका बढ़ता जा रहा था")
    # print(len(sent))
    # print(sent)
    # print(load_stop_words())
    pass