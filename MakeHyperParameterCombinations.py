import itertools as it
import numpy as np

param_dict = {

    # PSO PARAMS
    "w_max": [0.9],
    "w_min": [0.4],
    "v_max": [4],
    "v_min": [-4],
    "c1": [1],
    "c2": [1],
    "num_particles": [2, 4],
    "num_iterations": [20, 30],
    "num_features": [3],
    "summary_size": [75],

    # Feature Gen  Params
    "similarity_score": [0.06, 0.15],
    "n_grams": [3],
    "max_sent_thresh": [0.8, 0.9, 1.0],
    "min_sent_thresh": [0.2, 0.5],
    "use_stopwords": [True],
    "use_stemmer": [True],
}

output_file = open('list_of_terminal_commands_for_training', 'w+')
output_file2 = open('list_of_terminal_commands_for_testing', 'w+')
mapping_dict = dict()
index = 0
prefix = "python main.py"

ans_list = []
feature_vals = []

for key in param_dict:
    ans_list.append(key)
    feature_vals.append(param_dict[key])

combination_list = list(it.product(*feature_vals))
for q in combination_list:
    command = prefix
    for feature_name, feature_val in zip(ans_list, q):
        command = command + "  -" + str(feature_name) + " " + str(feature_val)

    # Add index at end
    command = command + "  -" + "index" + " " + str(index)

    # for each set  of hyperparameters  create a train and test command , and use index to create a dictionary mapping models with set of
    # hyperparameters to an index eg  1 : [num_particles =  3, c1 = -5]
    # will be saved as  model_1.txt

    train_command = command + "  -" + "mode" + " " + "train"
    test_command = command + "  -" + "mode" + " " + "test"

    mapping_dict[index] = command
    index += 1
    output_file.write(train_command)
    output_file.write("\n")

    output_file2.write(test_command)
    output_file2.write("\n")

np.save("mapping_dict.npy", mapping_dict)

# to load
# read_dictionary = np.load("mapping_dict.npy",allow_pickle='TRUE').item()
# print(read_dictionary[1])
# prints  python main.py  -w_max 0.9  -w_min 0.4  -v_max 4  -v_min -4  -c1 1  -c2 1  -num_particles 2  -num_iterations 20  -num_features 3  -summary_size 75  -similarity_score 0.12  -n_grams 1  -freq_thresh 0.3  -max_sent_thresh 0.8  -min_sent_thresh 0.2  -use_stopwords True  -use_lemmatizer False  -index 1
