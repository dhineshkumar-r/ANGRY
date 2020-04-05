import sys
import Utils
import PSO
import numpy as np

fn = "history_ncert_class10/chap_3.txt" # sys.argv[1]
ref_dir_n = "history_ncert_class10/annotations/chapter3" # sys.argv[2]

# load file
document, ref_sum = Utils.load_documents(fn, ref_dir_n)

# Pre-process with Stemmer and/or Lemmatizer.
processed_doc = Utils.process_document(document)
processed_ref_sum = Utils.process_documents(ref_sum)

# Extract features
features = PSO.extract_features(processed_doc)

# Initialize a Binary PSO model.
model = PSO.Swarm(processed_doc, processed_ref_sum)

# Train the model with extracted features.
weights = model.train(features)

# Generate summary with weights.
p_sum_idx = np.argsort(np.dot(features, weights))[-75:]
p_sum = PSO.generate_summary([document[idx] for idx in p_sum_idx])
p_sum1 = PSO.join_sentences([processed_doc[idx] for idx in p_sum_idx])
ref_sum = PSO.join_docs(processed_ref_sum)
print("Final Rouge Score: ", Utils.calculate_rouge(p_sum1, ref_sum, 1))

f = open("predicted_summary.txt", 'w', encoding='utf-8')
f.write(p_sum)
f.close()
