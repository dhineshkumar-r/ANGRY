from .particle import Particle
from .swarm import Swarm, Swarm2
from .FeatureGen import FeatureGen
from .constants import *


def extract_features(tokenized_doc,config):
    """
    Args:
        tokenized_doc (List[List[str]]): Tokenized document

    Returns:
        features (Numpy.Array): N*D dimension Array of features
    """
    # python main.py  -w_max 0.9  -w_min 0.4  -v_max 4  -v_min -4  -c1 1  -c2 1
    # -num_particles 2  -num_iterations 20  -num_features 3  -summary_size 75
    # -similarity_score 0.12  -n_grams 1  -freq_thresh 0.3  -max_sent_thresh 0.8  -min_sent_thresh 0.2  -use_stopwords True  -use_lemmatizer False  -index 1
    features_object = FeatureGen(tokenized_doc,n_grams=config.n_grams, summary_len=config.summary_size, sim_th=config.similarity_score, max_th=config.max_sent_thresh, min_th=config.min_sent_thresh)
    feature_vector = features_object.getfeaturevec(tokenized_doc)

    return feature_vector
