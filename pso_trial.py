import sys
import Utils
import PSO

fn = "sample_data/sample_hindi_text.txt" # sys.argv[1]
ref_dir_n = "history_ncert_class10/annotations/chapter3" # sys.argv[2]

# load file
document, ref_sum = Utils.load_documents(fn, ref_dir_n)

# Pre-process with Stemmer and/or Lemmatizer.
processed_doc = Utils.process_document(document)
processed_ref_sum = Utils.process_documents(ref_sum)

# Extract features
features = extract_features(processed_doc)

# Initialize a Binary PSO model.
model = PSO.Swarm(processed_doc, processed_ref_sum)

# Train the model with extracted features.
weights = model.train(features)

# Generate summary with weights.
summary = generate_summary(document, weights)

