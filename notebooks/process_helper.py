import fnmatch
import glob
import os
import sys
import argparse

import pandas as pd


def find_files(root_dir, subdir_pattern, show=False):
    # Create a search pattern that includes the root directory and the subdirectory pattern
    search_pattern = os.path.join(root_dir, subdir_pattern)
    if show:
        print("\n... searching files in {}".format(search_pattern))

    # List to store the file paths
    files_found = []

    # Use glob.glob to find all files that match the search pattern
    for file_path in glob.glob(search_pattern, recursive=True):
        if os.path.isfile(file_path):  # Ensure it's a file, not a directory
            files_found.append(file_path)
            if show:
                print(file_path)

    return files_found


def load_datasets_by_directory(list_of_root_dirs, subdir_pattern, show=False):
    data = {}
    for root_directory in list_of_root_dirs:
        if show:
            print(root_directory)
        dataset = []
        found_files = find_files(root_directory, subdir_pattern, show=show)  # Assuming function signature matches
        if show:
            print("Found files:")
        for file in found_files:
            if show:
                print(file)
            try:
                dataset.append(pd.read_csv(file))
            except Exception as e:
                print(f"Failed to load {file}: {e}")
        data[root_directory] = dataset
    return data


def generate_stats_for_directories(root_dirs, datasets_dict, column_name):
    stats_results = {}
    for root_dir in root_dirs:
        # Call a function to compute the statistics across all DataFrames for the current root directory
        try:
            # Assume datasets_dict[root_dir] is a list of DataFrames
            if root_dir in datasets_dict and datasets_dict[root_dir]:
                # Here we expect proc.row_wise_stats to be defined to handle a list of DataFrames and compute row-wise stats
                stats_df = row_wise_stats(datasets_dict[root_dir], column_name)
                stats_results[root_dir] = stats_df
            else:
                print(f"No data available for {root_dir}")
        except Exception as e:
            print(f"Error processing {root_dir}: {e}")
            stats_results[root_dir] = None

    return stats_results


def compute_summary_stats(stats_dict):
    summary_stats = {}
    for key, df in stats_dict.items():
        if df is not None and not df.empty:
            # Compute sum of 'Min', 'Max', and 'Mean' columns
            sum_stats = {
                'Sum Min': df['Min'].sum(),
                'Sum Max': df['Max'].sum(),
                'Sum Mean': df['Mean'].sum()
            }

            # Compute max of 'Min', 'Max', and 'Mean' columns
            max_stats = {
                'Max Min': df['Min'].max(),
                'Max Max': df['Max'].max(),
                'Max Mean': df['Mean'].max()
            }

            # Store results in a structured dictionary
            summary_stats[key] = {
                'Sum Statistics': sum_stats,
                'Max Statistics': max_stats
            }
        else:
            summary_stats[key] = {
                'Sum Statistics': {'Sum Min': None, 'Sum Max': None, 'Sum Mean': None},
                'Max Statistics': {'Max Min': None, 'Max Max': None, 'Max Mean': None}
            }
            print(f"No data or empty DataFrame for key: {key}")

    return summary_stats


def print_sorted_by_max_mean(summary_stats, stats_type='Max Statistics', stats_key='Max Mean'):
    # Create a list of tuples (key, max mean value)
    sortable_list = [(key, stats[stats_type][stats_key]) for key, stats in summary_stats.items() if
                     stats[stats_type][stats_key] is not None]

    # Sort the list by 'Max Mean' in descending order
    sorted_list = sorted(sortable_list, key=lambda x: x[1], reverse=True)

    # Print the sorted statistics
    for key, max_mean in sorted_list:
        print(f"Directory: {key}")
        print(f"{stats_type}:", summary_stats[key][stats_type])
        print()  # Adds a newline for better readability


def generate_root_dirs_by_experiment_and_augment_degree(
        exp_main_folder='exp-pad', exp_name='pendulum_swingup', agent_null='drqv2_pixels-True', agent_name='drqv2_pad_', min_val=1, max_val=10,
        increment=1):
    root = '/Users/victor/Documents/python-projects/smart-rl/'
    path = os.path.join(root, exp_main_folder, 'exp', exp_name)
    agent_augment_types = [agent_null]
    for i in range(min_val, max_val + 1, increment):
        agent_augment_types.append(agent_name + str(i) + '_pixels-True')

    return [os.path.join(path, i) for i in agent_augment_types]


def generate_root_dirs_by_experiment_and_augment_degree_new(
        exp_main_folder='exp-pad', exp_name='pendulum_swingup', agent_name='drqv2_aug_pixels-True-rotate-', list_vals=None):
    if list_vals is None:
        list_vals = [0.1]
    root = '/Users/victor/Documents/python-projects/smart-rl/'
    path = os.path.join(root, exp_main_folder, 'exp', exp_name)
    agent_augment_types = []
    for i in list_vals:
        agent_augment_types.append(agent_name + str(i))
    return [os.path.join(path, i) for i in agent_augment_types]


def row_wise_stats(dataframes, column_name):
    concatenated = pd.concat([df.set_index('episode')[column_name] for df in dataframes], axis=1)

    # Compute statistics
    max_values = concatenated.max(axis=1)
    min_values = concatenated.min(axis=1)
    mean_values = concatenated.mean(axis=1)

    # Combine into a single DataFrame
    stats_df = pd.DataFrame({
        'Max': max_values,
        'Min': min_values,
        'Mean': mean_values
    })

    return stats_df


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
    parser.add_argument('--root_directory', type=str, required=True, help='Relative path to the root directory')
    parser.add_argument('--subdir_pattern', type=str, required=True, help='Subdirectory pattern to match')
    parser.add_argument('--column_to_extract', type=str, required=False, help='Column to extract from each CSV')
    parser.add_argument('--compute_averages', type=str, required=False, help='Compute averages per column on the read CSV')
    parser.add_argument('--output_path', type=str, required=False, help='Path where the combined DataFrame will be saved')
    parser.add_argument('--output_filename', type=str, required=False, help='Filename for the combined DataFrame CSV')

    print("\n... starting processing")
    # Parse arguments
    args = parser.parse_args()
    # Call the function to find and print file paths
    files_found = find_files(args.root_directory, args.subdir_pattern)

    dataframe_dicts = csv_to_dataframe(files_found)
    if args.column_to_extract is not None:
        column_to_extract = args.column_to_extract
        print("\n... extracting column '{}'".format(column_to_extract))
        dfs_extracted_columns = extract_column_from_csv(dataframe_dicts, column_to_extract)
        print("\n... dataframes extracted")
        print(dfs_extracted_columns.head())

        save_to_csv(args, dfs_extracted_columns)

    if args.compute_averages is not None:
        dfs_column_averages = compute_column_averages(files_found)
        save_to_csv(args, dfs_column_averages)


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
