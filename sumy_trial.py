import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from Utils import *

parser = PlaintextParser.from_file("sample_data/sample_hindi_text.txt", Tokenizer())

summarizer = TextRankSummarizer()
summarizer.stop_words = load_stop_words()
summary = summarizer(parser.document, 2)

for sentence in summary:
    print(sentence)
