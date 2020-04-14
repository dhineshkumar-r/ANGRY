from .particle import Particle
from .swarm import Swarm
from .FeatureGen import FeatureGen
from .constants import *


def extract_features(tokenized_doc):
    """
    Args:
        tokenized_doc (List[List[str]]): Tokenized document

    Returns:
        features (Numpy.Array): N*D dimension Array of features
    """
    features_object = FeatureGen(tokenized_doc)
    feature_vector = features_object.getfeaturevec(tokenized_doc)

    return feature_vector
