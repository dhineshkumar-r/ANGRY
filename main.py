import sys
import argparse
import PSO
from pso import pso_train, pso_test


def parse_arguments(args):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-m', '--mode', type=str, help="Enter the mode.")
    arg_parser.add_argument('-c1', '--c1', type=float, help="Enter Inertia.", default=PSO.C1)
    arg_parser.add_argument('-c2', '--c2', type=float, help="Enter Inertia.", default=PSO.C2)
    arg_parser.add_argument('-f', '--file', type=str, help="Model to be loaded")
    # TODO Accept other arguments for the experiment

    return arg_parser.parse_args(args)


config = parse_arguments(sys.argv[1:])

# TODO Update exp-tag with updated exp parameters
exp_tag = '{mode}-{c1}-{c2}'
exp_tag = exp_tag.format(**vars(config))

if config.mode == "train":
    m_weights = pso_train("train/documents", "train/references")
    f = open("models/"+exp_tag +".txt", 'w')
    f.write(",".join([str(v) for v in m_weights]))
    f.close()
elif config.mode == "test":
    f = open(config.file,'r')
    w = [float(v) for v in f.readline().split(",")]
    f.close()
    pso_test("test/documents", "test/references", w)
