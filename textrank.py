from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer

import os
import PSO
import Utils


def textrank_test(doc_dir, ref_dir, summary_length=75, stopwords=True):
    """
    Args:
        doc_dir (str): Input chapters directory
        ref_dir (str): Reference summaries directory
        summary_length (int) : Length of summary
    """
    docs = sorted(os.listdir(doc_dir))
    refs = sorted(os.listdir(ref_dir))

    documents = []
    references = []
    for d, r in zip(docs, refs):
        doc, ref = Utils.load_document(doc_dir + "/" + d, ref_dir + "/" + r)
        p_doc = Utils.process_document(doc)
        p_ref = Utils.process_document(ref)
        documents.append(p_doc)
        references.append(p_ref)

    references = Utils.join_docs(references)

    rouge_scores = [0.0] * len(documents)
    rogue_index = 0
    for d, r in zip(docs, refs):
        # Perform Textrank
        parser = PlaintextParser.from_file(doc_dir + "/" + d, Utils.Tokenizer())
        summarizer = TextRankSummarizer()
        if stopwords:
            summarizer.stop_words = Utils.load_stop_words()
        summary = summarizer(parser.document, summary_length)
        p_sum = ""
        for sentence in summary:
            p_sum += str(sentence) + " "

        rouge_scores[rogue_index] = Utils.calculate_rouge(p_sum, references[rogue_index], 1)
        rogue_index += 1

    print(rouge_scores)


if __name__ == "__main__":
    textrank_test('test/documents', 'test/references')
