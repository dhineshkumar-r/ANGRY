import os
import Utils
import PSO
import numpy as np


def pso_train(doc_dir, ref_dir):
    docs = sorted(os.listdir(doc_dir))
    refs = sorted(os.listdir(ref_dir))

    documents = []
    references = []
    features = []
    for d, r in zip(docs, refs):
        doc, ref = Utils.load_document(doc_dir + "/" + d, ref_dir + "/" + r)
        p_doc = Utils.process_document(doc)
        p_ref = Utils.process_document(ref)
        features.append(PSO.extract_features(p_doc))
        documents.append(p_doc)
        references.append(p_ref)

    references = Utils.join_docs(references)

    # Initialize a Binary PSO model.
    model = PSO.Swarm2(documents, references)

    # Train the model with extracted features.
    weights = model.train(features)

    # Generate summary with weights.
    rouge_scores = [0.0] * len(documents)
    for i, feature in enumerate(features):
        p_sum_idx = np.argsort(np.dot(feature, weights))[-PSO.SUMMARY_SIZE:]
        p_sum = Utils.join_sentences([documents[i][idx] for idx in p_sum_idx])
        rouge_scores[i] = Utils.calculate_rouge(p_sum, references[i], 1)

    print(rouge_scores)
    print(weights)
    return weights


def pso_test(doc_dir, ref_dir, weights):
    docs = sorted(os.listdir(doc_dir))
    refs = sorted(os.listdir(ref_dir))

    documents = []
    references = []
    features = []
    for d, r in zip(docs, refs):
        doc, ref = Utils.load_document(doc_dir + "/" + d, ref_dir + "/" + r)
        p_doc = Utils.process_document(doc)
        p_ref = Utils.process_document(ref)
        features.append(PSO.extract_features(p_doc))
        documents.append(p_doc)
        references.append(p_ref)

    references = Utils.join_docs(references)

    weights = np.array(weights)

    # Generate summary with weights.
    rouge_scores = [0.0] * len(documents)
    for i, feature in enumerate(features):
        p_sum_idx = np.argsort(np.dot(feature, weights))[-PSO.SUMMARY_SIZE:]
        p_sum = Utils.join_sentences([documents[i][idx] for idx in p_sum_idx])
        rouge_scores[i] = Utils.calculate_rouge(p_sum, references[i], 1)

    print(rouge_scores)
