import fnmatch
import glob
import os
import sys
import argparse

import pandas as pd

import os
import glob


def find_files_and_check_lines(root_dir, subdir_pattern, seed_list, expected_lines=111):
    # Create a search pattern that includes the root directory and the subdirectory pattern
    search_pattern = os.path.join(root_dir, subdir_pattern)
    print("\n... searching files in {}".format(search_pattern))

    # Dictionary to store the file paths and their line counts
    files_found = {}

    # Use glob.glob to find all files that match the search pattern
    for file_path in glob.glob(search_pattern, recursive=True):
        if os.path.isfile(file_path):  # Ensure it's a file, not a directory
            with open(file_path, 'r') as file:
                lines = file.readlines()
                file_name = os.path.basename(file_path)
                files_found[file_path] = len(lines)

    # Generate the output for the list of seeds
    output_line = '|'
    # Organize files by experiment parameters
    experiment_files = {}
    for file_path in files_found:
        if files_found[file_path] == expected_lines:
            experiment_param = file_path.split('/')[3]
            if experiment_param not in experiment_files:
                experiment_files[experiment_param] = []
            experiment_files[experiment_param].append(file_path)

    sorted_experiment_files = {k: sorted(v) for k, v in sorted(experiment_files.items())}

    # Generate the output for each experiment parameter and seed
    for experiment_param, paths in sorted_experiment_files.items():
        output_line = '| | ' + experiment_param + '|'
        seed_ctr = 0
        paths_str = "".join(paths)
        for seed in seed_list:
            seed_pattern = f"/seed_{seed}/"
            # found = bool(seed_pattern in path for path in paths)
            found = bool(seed_pattern in paths_str)
            if found:
                output_line += str(seed) + '(+) |'
                seed_ctr += 1
            else:
                output_line += str(seed) + '(-) |'
        print(output_line + str(len(paths)) + '|')

    # for file_path in files_found:
    #     experiment_param = file_path.split('/')[3]
    #     output_line = experiment_param + '|'
    #     for seed in seed_list:
    #         seed_pattern = f"seed_{seed}"
    #         found = False
    #         for file_path in files_found:
    #             if seed_pattern in file_path and files_found[file_path] == expected_lines:
    #                 found = True
    #                 break
    #         if found:
    #             output_line = output_line + str(seed) + '(+) |'
    #         else:
    #             output_line = output_line + str(seed) + '(-) |'
    #     print(output_line)

    return files_found



def find_files(root_dir, subdir_pattern):
    # Create a search pattern that includes the root directory and the subdirectory pattern
    search_pattern = os.path.join(root_dir, subdir_pattern)
    print("\n... searching files in {}".format(search_pattern))

    # List to store the file paths
    files_found = []

    # Use glob.glob to find all files that match the search pattern
    for file_path in glob.glob(search_pattern, recursive=True):
        if os.path.isfile(file_path):  # Ensure it's a file, not a directory
            files_found.append(file_path)
            print(file_path)

    return files_found


def csv_to_dataframe(files_found):
    dataframes = {}
    for file in files_found:
        dataframes[file] = pd.read_csv(file)

    print("\n... dataframes loaded")
    for file_name, df_from_csv in dataframes.items():
        print(file_name)
        print(df_from_csv.head())
    return dataframes


def extract_column_from_csv(dataframe_dicts, column_name):
    combined_data = pd.DataFrame()

    for file_name, df_from_csv in dataframe_dicts.items():
        print()
        fl_lst = file_name.split("/")
        file_name_col = fl_lst[3]
        if column_name in df_from_csv.columns:
            combined_data[file_name_col] = df_from_csv[column_name]

    return combined_data


def save_to_csv(args, dataframe_columns):
    # Ensure the output directory exists
    output_dir = os.path.dirname(os.path.join(args.output_path, args.output_filename))
    os.makedirs(output_dir, exist_ok=True)
    # Save the combined DataFrame as CSV
    output_file = os.path.join(args.output_path, args.output_filename)
    dataframe_columns.to_csv(output_file, index=False)
    print(f"Combined DataFrame saved to {output_file}")


def compute_column_averages(files_found):
    averages_list = []

    for file in files_found:
        df = pd.read_csv(file)
        averages = df.mean().to_frame().T
        averages['file'] = file
        averages_list.append(averages)

    combined_averages = pd.concat(averages_list, ignore_index=True)
    return combined_averages


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Find files in subdirectories matching a pattern.')
    parser.add_argument('--root_directory', type=str, required=False, help='Relative path to the root directory')
    parser.add_argument('--subdir_pattern', type=str, required=False, help='Subdirectory pattern to match')
    parser.add_argument('--column_to_extract', type=str, required=False, help='Column to extract from each CSV')
    parser.add_argument('--compute_averages', type=str, required=False, help='Compute averages per column on the read CSV')
    parser.add_argument('--output_path', type=str, required=False, help='Path where the combined DataFrame will be saved')
    parser.add_argument('--output_filename', type=str, required=False, help='Filename for the combined DataFrame CSV')

    print("\n... starting processing")
    # Parse arguments
    args = parser.parse_args()
    # Call the function to find and print file paths
    # files_found = find_files(args.root_directory, args.subdir_pattern)
    seed_list = [2, 3, 4, 7, 10, 13, 23, 123, 1234, 1313]

    # root_directory = 'exp-pad/exp/cartpole_balance_sparse'
    # root_directory = 'exp-pad/exp/finger_turn_hard' #!!!
    # root_directory = 'exp-pad/exp/pendulum_swingup'
    # root_directory = 'exp-pad/exp/quadruped_run'
    # root_directory = 'exp-pad/exp/reacher_hard'
    # root_directory = 'exp-pad/exp/walker_run'

    # root_directory = 'exp-rot/exp/cartpole_balance_sparse'
    # root_directory = 'exp-rot/exp/pendulum_swingup'
    # root_directory = 'exp-rot/exp/walker_run'


    # root_directory = 'exp-contrast/exp/cartpole_balance_sparse'
    root_directory = 'exp-contrast/exp/pendulum_swingup'
    # root_directory = 'exp-contrast/exp/walker_run'
    subdir_pattern = '*/*/*/eval.csv'
    files_found = find_files_and_check_lines(root_directory, subdir_pattern, seed_list=seed_list)

    # dataframe_dicts = csv_to_dataframe(files_found)
    # if args.column_to_extract is not None:
    #     column_to_extract = args.column_to_extract
    #     print("\n... extracting column '{}'".format(column_to_extract))
    #     dfs_extracted_columns = extract_column_from_csv(dataframe_dicts, column_to_extract)
    #     print("\n... dataframes extracted")
    #     print(dfs_extracted_columns.head())
    #
    #     save_to_csv(args, dfs_extracted_columns)
    #
    # if args.compute_averages is not None:
    #     dfs_column_averages = compute_column_averages(files_found)
    #     save_to_csv(args, dfs_column_averages)


if __name__ == "__main__":
    main()

# ex. 1 - find all eval.csv files under one experiment, for all seeds of that experiment
# python process.py --root_directory 'exp-pad/exp/pendulum_swingup/drqv2_pixels-True' --subdir_pattern '*/*/eval.csv'
# python process.py --root_directory 'exp-pad/exp/pendulum_swingup/drqv2_pad_2_pixels-True' --subdir_pattern '*/*/eval.csv'

# ex. 2 - find all eval.csv files under all experiments, for all seeds of that experiment
# python process.py --root_directory 'exp-rot/exp/pendulum_swingup' --subdir_pattern '*/*/*/eval.csv'
# python process.py --root_directory 'exp-rot/exp/pendulum_swingup' --subdir_pattern '*/*/*/eval.csv' --column_to_extract episode_reward --output_path aggregate/rotate/ --output_filename episode-rewards-rotate-all.csv



# python process.py --root_directory 'exp-rot/pendulum_swingup_augment_stats' --subdir_pattern '*/*/*/augment.csv'
# python process.py --root_directory 'exp-rot/pendulum_swingup_augment_stats' --subdir_pattern '*/*/*/augment.csv' --compute_averages Y --output_path aggregate/rotate/ --output_filename all-stats-average-rotate-all.csv


# python process.py --root_directory 'exp-pad/exp/pendulum_swingup' --subdir_pattern '*/*/*/eval.csv'
# python process.py --root_directory 'exp-pad/exp/pendulum_swingup' --subdir_pattern '*/*/*/eval.csv' --column_to_extract episode_reward --output_path aggregate/shift/ --output_filename episode-rewards-shift-all.csv


# python process.py --root_directory 'exp-pad/pendulum_swingup_augment_stats' --subdir_pattern '*/*/*/augment.csv'
# python process.py --root_directory 'exp-pad/pendulum_swingup_augment_stats' --subdir_pattern '*/*/*/augment.csv' --compute_averages Y --output_path aggregate/shift/ --output_filename all-stats-average-shift-all.csv




# python process.py --root_directory 'exp-pad/exp/pendulum_swingup' --subdir_pattern '*/seed_2/*/eval.csv'
# python process.py --root_directory 'exp-pad/exp/pendulum_swingup' --subdir_pattern '*/seed_2/*/eval.csv' --column_to_extract episode_reward --output_path aggregate/shift/ --output_filename episode-rewards-shift-seed-2-all.csv

# python process.py --root_directory 'exp-pad/pendulum_swingup_augment_stats' --subdir_pattern '*/seed_2/*/augment.csv'
# python process.py --root_directory 'exp-pad/pendulum_swingup_augment_stats' --subdir_pattern '*/seed_2/*/augment.csv' --compute_averages Y --output_path aggregate/shift/ --output_filename all-stats-average-shift-all.csv




# python process.py --root_directory 'exp-contrast/exp/pendulum_swingup' --subdir_pattern '*/seed_2/*/eval.csv'
# python process.py --root_directory 'exp-contrast/exp/pendulum_swingup' --subdir_pattern '*/seed_2/*/eval.csv' --column_to_extract episode_reward --output_path aggregate/contrast/ --output_filename episode-rewards-contrast-seed-2-all.csv

# python process.py --root_directory 'exp-contrast/pendulum_swingup_augment_stats' --subdir_pattern '*/seed_2/*/augment.csv'
# python process.py --root_directory 'exp-contrast/pendulum_swingup_augment_stats' --subdir_pattern '*/seed_2/*/augment.csv' --compute_averages Y --output_path aggregate/contrast/ --output_filename all-stats-average-contrast-all.csv



# python process.py --root_directory 'exp-zoom/exp/pendulum_swingup' --subdir_pattern '*/seed_2/*/eval.csv'
# python process.py --root_directory 'exp-zoom/exp/pendulum_swingup' --subdir_pattern '*/seed_2/*/eval.csv' --column_to_extract episode_reward --output_path aggregate/zoom/ --output_filename episode-rewards-zoom-all.csv


# python process.py --root_directory 'exp-sharp/exp/pendulum_swingup' --subdir_pattern '*/seed_2/*/eval.csv'
# python process.py --root_directory 'exp-sharp/exp/pendulum_swingup' --subdir_pattern '*/seed_2/*/eval.csv' --column_to_extract episode_reward --output_path aggregate/sharp/ --output_filename episode-rewards-sharp-all.csv

# python process.py --root_directory 'exp-sharpmin/exp/pendulum_swingup' --subdir_pattern '*/seed_2/*/eval.csv'
# python process.py --root_directory 'exp-sharpmin/exp/pendulum_swingup' --subdir_pattern '*/seed_2/*/eval.csv' --column_to_extract episode_reward --output_path aggregate/sharpmin/ --output_filename episode-rewards-sharpmin-all.csv




# python process.py --root_directory 'exp-zoom/pendulum_swingup_augment_stats' --subdir_pattern '*/seed_2/*/augment.csv'
# python process.py --root_directory 'exp-zoom/pendulum_swingup_augment_stats' --subdir_pattern '*/seed_2/*/augment.csv' --compute_averages Y --output_path aggregate/zoom/ --output_filename all-stats-average-zoom-all.csv


# python process.py --root_directory 'exp-sharp/pendulum_swingup_augment_stats' --subdir_pattern '*/seed_2/*/augment.csv'
# python process.py --root_directory 'exp-sharp/pendulum_swingup_augment_stats' --subdir_pattern '*/seed_2/*/augment.csv' --compute_averages Y --output_path aggregate/sharp/ --output_filename all-stats-average-sharp-all.csv