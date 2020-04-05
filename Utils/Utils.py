# -*- coding: utf-8 -*-
from typing import List

suffixes = {
    1: ["ो", "े", "ू", "ु", "ी", "ि", "ा"],
    2: ["तृ", "ान", "ैत", "ने", "ाऊ", "ाव", "कर", "ाओ", "िए", "ाई", "ाए", "नी", "ना", "ते", "ीं", "ती",
        "ता", "ाँ", "ां", "ों", "ें", "ीय", "ति", "या", "पन", "पा", "ित", "ीन", "लु", "यत", "वट", "लू"],
    3: ["ेरा", "त्व", "नीय", "ौनी", "ौवल", "ौती", "ौता", "ापा", "वास", "हास", "काल", "पान", "न्त", "ौना", "सार",
        "पोश", "नाक",
        "ियल", "ैया", "ौटी", "ावा", "ाहट", "िया", "हार", "ाकर", "ाइए", "ाईं", "ाया", "ेगी", "वान", "बीन",
        "ेगा", "ोगी", "ोगे", "ाने", "ाना", "ाते", "ाती", "ाता", "तीं", "ाओं", "ाएं", "ुओं", "ुएं", "ुआं", "कला",
        "िमा", "कार",
        "गार", "दान", "खोर"],
    4: ["ावास", "कलाप", "हारा", "तव्य", "वैया", "वाला", "ाएगी", "ाएगा", "ाओगी", "ाओगे",
        "एंगी", "ेंगी", "एंगे", "ेंगे", "ूंगी", "ूंगा", "ातीं", "नाओं", "नाएं", "ताओं", "ताएं", "ियाँ", "ियों",
        "ियां",
        "त्वा", "तव्य", "कल्प", "िष्ठ", "जादा", "क्कड़"],
    5: ["ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां", "अक्कड़", "तव्य:", "निष्ठ"]}


class Tokenizer:

    def __init__(self):
        pass

    @staticmethod
    def clean_word(w):
        return w.strip().replace(":", "").replace("'", "").replace("\"", ""). \
            replace("(", "").replace(")", "").replace(",", "").replace("[", "").replace("]", "")

    @staticmethod
    def stem_word(w):
        len_suffix = len(suffixes)
        done = False
        for i in range(len_suffix, 0, -1):
            if len(w) > i + 1:
                for q in suffixes[i]:
                    if w.endswith(q):
                        w = w[:-i]
                        done = True
                        break

                if done is True:
                    break
        return w

    @staticmethod
    def to_sentences(text: str) -> List[str]:
        return [s.strip() for s in text.replace("?", u"।").split(u"।")]

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
