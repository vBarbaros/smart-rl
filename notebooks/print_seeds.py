import process_helper as proc
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# Adjust display options
pd.set_option('display.max_rows', None)    # Show all rows
pd.set_option('display.max_columns', None) # Show all columns
pd.set_option('display.width', None)       # Adjust the width to avoid line breaks
pd.set_option('display.colheader_justify', 'center')  # Center align column headers

# per_seed = True
# subdir_pattern = 'seed_10/*/eval.csv'

per_seed = False
subdir_pattern = '*/*/eval.csv'


# EXP_TYPE = 'pad'
# EXP_NAME = 'shift'
# LIST_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


EXP_TYPE = 'rot'
EXP_NAME = 'rotate'
LIST_VALS = [
    # 0,
    # 0.1,
    0.2,
    # 0.3,
             # 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
             #    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
             #    2.1, 2.2, 2.3, 2.4, 2.5,
             #    3, 4, 5, 7, 9, 11, 13, 15
             ]


# EXP_TYPE = 'contrast'
# EXP_NAME = 'contrast'
# LIST_VALS = [0.1,
#              0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9
#              ]



column_name = 'episode_reward'
XLABEL_STATS_AUG = 'Padding (in pixels)'
USE_VARIANCE = False
TIMES_STDDEV = 4
DICTS_ALL_STATS = {}
STATS_DATA = {}
MAX_INDICES = {}

plot_performance = True
plot_stats_dists = True

stats_column_names = [
    "manhattan",
    "ssim_dist",
    "kl_div",
    "hamming",
    "euclidian",
    "chebyshev",
    "cosine_dist",
    "bhattacharyya"
]

env_names = [
    'cartpole_balance_sparse',
    #     'finger_turn_hard',
    'pendulum_swingup',
    # #     'quadruped_run',
    # #     'reacher_hard',
    'walker_run'
]

ENV_MAX_TOP = {}
MAX_INDICES_TOP = {}

all_stats_padding = {}
for env_name in env_names:
    root = '/Users/victor/Documents/python-projects/smart-rl/'
    exp_main_folder = 'exp-' + EXP_TYPE
    agent_name = 'drqv2_aug_pixels-True-' + EXP_NAME + '-'


    path = os.path.join(root, exp_main_folder, 'exp', env_name)
    agent_augment_types = []
    for i in LIST_VALS:
        agent_augment_types.append(agent_name + str(i))
    list_of_root_dirs = [os.path.join(path, i) for i in agent_augment_types]

    data = {}
    for root_directory in list_of_root_dirs:
        dataset = []
        found_files = proc.find_files(root_directory, subdir_pattern, show=False)  # Assuming function signature matches
        # print(found_files)
        for file in found_files:
            try:
                dataset.append(pd.read_csv(file))
            except Exception as e:
                print(f"Failed to load {file}: {e}")
        data[root_directory] = dataset

    #     print(data)
    stats_results = {}
    # print(data)
    for root_dir in list_of_root_dirs:
        # Call a function to compute the statistics across all DataFrames for the current root directory
        try:
            # Assume datasets_dict[root_dir] is a list of DataFrames
            if root_dir in data and data[root_dir]:
                stats_results[root_dir] = pd.concat([df.set_index('episode')[column_name] for df in data[root_dir]], axis=1)
            else:
                print(f"No data available for {root_dir}")
        except Exception as e:
            print(f"Error processing {root_dir}: {e}")
            stats_results[root_dir] = None

    print("\n\n", env_name, ":\n")
    for k in stats_results.keys():
        # print(k)
        if per_seed:
            print(list(stats_results[k]['episode_reward'].T)[:30])
            # print(stats_results[k]['episode_reward'])
        else:
            reward_data = stats_results[k].filter(like='episode_reward')
            print(reward_data)

