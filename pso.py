import os
import Utils
import PSO
import numpy as np
import argparse


def pso_train(doc_dir, ref_dir,config):
    docs = sorted(os.listdir(doc_dir))
    refs = sorted(os.listdir(ref_dir))

    documents = []
    references = []
    features = []
    for d, r in zip(docs, refs):
        doc, ref = Utils.load_document(doc_dir + "/" + d, ref_dir + "/" + r)
        p_doc = Utils.process_document(doc,config.use_stopwords,config.use_lemmatizer)
        p_ref = Utils.process_document(ref,config.use_stopwords,config.use_lemmatizer)
        features.append(PSO.extract_features(p_doc,config))
        documents.append(p_doc)
        references.append(p_ref)

    references = Utils.join_docs(references)

    # Initialize a Binary PSO model.
    # python main.py  -mode train  -w_max 0.9  -w_min 0.4  -v_max 4  -v_min -4  -c1 1  -c2 1  -num_particles 2  -num_iterations 20
    # -num_features 3  -summary_size 75  -similarity_score 0.12  -n_grams 1  -freq_thresh 0.4  -max_sent_thresh 0.8  -min_sent_thresh 0.2  -use_stopwords True  -use_lemmatizer False  -file None  -index 25
    # CHECK w   is what
    model = PSO.Swarm2(documents, references,n_features=config.num_features, n_particles=config.num_particles,
                 n_iterations=config.num_iterations, w=0.9, c1=config.c1, c2=config.c2, sum_size=config.summary_size,config=config)

    # Train the model with extracted features.
    weights = model.train(features)

    # Generate summary with weights.
    rouge_scores = [0.0] * len(documents)
    for i, feature in enumerate(features):
        p_sum_idx = np.argsort(np.dot(feature, weights))[-config.summary_size:]
        p_sum = Utils.join_sentences([documents[i][idx] for idx in p_sum_idx])
        rouge_scores[i] = Utils.calculate_rouge(p_sum, references[i], 1)

    print(rouge_scores)
    print(weights)
    return weights


def pso_test(doc_dir, ref_dir, weights,config):
    docs = sorted(os.listdir(doc_dir))
    refs = sorted(os.listdir(ref_dir))

    documents = []
    references = []
    features = []
    for d, r in zip(docs, refs):
        doc, ref = Utils.load_document(doc_dir + "/" + d, ref_dir + "/" + r)

        p_doc = Utils.process_document(doc, config.use_stopwords, config.use_lemmatizer)
        p_ref = Utils.process_document(ref, config.use_stopwords, config.use_lemmatizer)
        features.append(PSO.extract_features(p_doc,config))
        documents.append(p_doc)
        references.append(p_ref)

    references = Utils.join_docs(references)

    weights = np.array(weights)

    # Generate summary with weights.
    rouge_scores = [0.0] * len(documents)
    for i, feature in enumerate(features):
        p_sum_idx = np.argsort(np.dot(feature, weights))[-config.summary_size:]
        p_sum = Utils.join_sentences([documents[i][idx] for idx in p_sum_idx])
        rouge_scores[i] = Utils.calculate_rouge(p_sum, references[i], 1)

    print(rouge_scores)
