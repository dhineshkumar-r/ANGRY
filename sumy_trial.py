from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
import PSO
import Utils

fn = "history_ncert_class10/chap_3.txt" # sys.argv[1]
ref_dir_n = "history_ncert_class10/annotations/chapter3" # sys.argv[2]
summary_length = 75

parser = PlaintextParser.from_file(fn, Utils.Tokenizer())

# load file
document, ref_sum = Utils.load_documents(fn, ref_dir_n)

processed_ref_sum = Utils.process_documents(ref_sum)
ref_sum = PSO.join_docs(processed_ref_sum)

summarizer = TextRankSummarizer()
summarizer.stop_words = Utils.load_stop_words()
summary = summarizer(parser.document, summary_length)

p_sum1 = ""
for sentence in summary:
    p_sum1 += str(sentence) + " "

print("Final Rouge Score: ", Utils.calculate_rouge(p_sum1, ref_sum, 1))

f = open("predicted_summary_textrank.txt", 'w', encoding='utf-8')
for sentence in summary:
    f.write(str(sentence) + '\n')
f.close()
