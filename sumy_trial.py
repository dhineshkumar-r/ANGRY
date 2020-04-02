from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
import Utils

parser = PlaintextParser.from_file("sample_data/sample_hindi_text.txt", Utils.Tokenizer())

summarizer = TextRankSummarizer()
summarizer.stop_words = Utils.load_stop_words()
summary = summarizer(parser.document, 2)

for sentence in summary:
    print(sentence)
