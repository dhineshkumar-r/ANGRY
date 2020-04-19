import sys
import argparse
import PSO
from pso import pso_train, pso_test


# python main.py  -mode train  -w_max 0.9  -w_min 0.4  -v_max 4  -v_min -4  -c1 1  -c2 1  -num_particles 2  -num_iterations 20
# -num_features 3  -summary_size 75  -similarity_score 0.12  -n_grams 1  -freq_thresh 0.4  -max_sent_thresh 0.8  -min_sent_thresh 0.2  -use_stopwords True  -use_lemmatizer False  -file None  -index 25

def parse_arguments(args):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-mode', '--mode', type=str, default='train')
    arg_parser.add_argument('-w_max', '--w_max', type=float, default=0.9)
    arg_parser.add_argument('-w_min', '--w_min', type=float, default=0.4)
    arg_parser.add_argument('-v_max', '--v_max', type=float, default=4)
    arg_parser.add_argument('-v_min', '--v_min', type=float, default=-4)
    arg_parser.add_argument('-c1', '--c1', type=float, default=1)
    arg_parser.add_argument('-c2', '--c2', type=float, default=1)
    arg_parser.add_argument('-num_particles', '--num_particles', type=int, default=3)
    arg_parser.add_argument('-num_iterations', '--num_iterations', type=int, default=20)
    arg_parser.add_argument('-num_features', '--num_features', type=int, default=3)
    arg_parser.add_argument('-summary_size', '--summary_size', type=int, default=75)
    arg_parser.add_argument('-similarity_score', '--similarity_score', type=float, default=0.12)
    arg_parser.add_argument('-n_grams', '--n_grams', type=int, default=3)
    arg_parser.add_argument('-freq_thresh', '--freq_thresh', type=float, default=0.5)
    arg_parser.add_argument('-max_sent_thresh', '--max_sent_thresh', type=float, default=1.0)
    arg_parser.add_argument('-min_sent_thresh', '--min_sent_thresh', type=float, default=0.5)
    arg_parser.add_argument('-use_stopwords', '--use_stopwords', type=bool, default=True)
    arg_parser.add_argument('-use_lemmatizer', '--use_lemmatizer', type=bool, default=True)
    arg_parser.add_argument('-file', '--file', type=str, help="Model to be loaded", default=None)
    arg_parser.add_argument('-index', '--index', type=int)

    # TODO Accept other arguments for the experiment

    return arg_parser.parse_args(args)


config = parse_arguments(sys.argv[1:])

# TODO Update exp-tag with updated exp parameters
exp_tag = '-{index}'
exp_tag = exp_tag.format(**vars(config))

if config.mode == "train":
    m_weights = pso_train("train/documents", "train/references", config)
    f = open("models/model" + exp_tag + ".txt", 'w')
    f.write(",".join([str(v) for v in m_weights]))
    f.close()
elif config.mode == "test":
    if config.file is not None:
        f = open(config.file, 'r')
    # Use index not config.file
    else:
        f = open("models/model" + exp_tag + ".txt", 'r')
    w = [float(v) for v in f.readline().split(",")]
    f.close()
    test_summaries = pso_test("test/documents", "test/references", w, config)
