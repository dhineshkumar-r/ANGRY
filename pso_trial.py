import sys
import Utils
import PSO

fn = "sample_data/sample_hindi_text.txt"
# ref_sum = sys.argv[2]

# load file
document = Utils.load_document(fn)

# Pre-process with Stemmer and/or Lemmatizer.
processed_doc = Utils.process_document(document)

# # Extract features
# features = extract_features(processed_doc)
#
# # Initialize a Binary PSO model.
# model = PSO.Swarm()
#
# # Train the model with extracted features.
# weights = model.train(features)
#
# # Generate summary with weights.
# summary = generate_summary(document, weights)

