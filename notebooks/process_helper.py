import fnmatch
import glob
import os
import sys
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np


def plot_rewards_similar_to_stats_metrics(STATS_DATA, MAX_INDICES, stats_column_names, XLABEL_STATS, show=True):
    """
    Plots statistical distance metrics for multiple environments.
    Highlights the max value entry for each environment with a larger red dot.
{'cartpole_balance_sparse': {'perform': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, 'pendulum_swingup': {'perform': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, 'walker_run': {'perform': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}}
{'cartpole_balance_sparse_pad_max_at_3.0': {'manhattan': 3, 'ssim_dist': 3, 'kl_div': 3, 'hamming': 3, 'euclidian': 3, 'chebyshev': 3, 'cosine_dist': 3, 'bhattacharyya': 3}, 'pendulum_swingup_pad_max_at_5.0': {'manhattan': 5, 'ssim_dist': 5, 'kl_div': 5, 'hamming': 5, 'euclidian': 5, 'chebyshev': 5, 'cosine_dist': 5, 'bhattacharyya': 5}, 'walker_run_pad_max_at_5.0': {'manhattan': 5, 'ssim_dist': 5, 'kl_div': 5, 'hamming': 5, 'euclidian': 5, 'chebyshev': 5, 'cosine_dist': 5, 'bhattacharyya': 5}}
['perform']
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] cartpole_balance_sparse {'perform': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]} perform
    Parameters:
    STATS_DATA (dict): Dictionary where keys are environment names and values are dictionaries of stat names to lists of values.
    MAX_INDICES (dict): Dictionary where keys are environment names and values are dictionaries of stat names to max value indices.
    stats_column_names (list): List of stat names to process.
    """
    if show:
        for stat_name in stats_column_names:
            # Create the plot for each stat_name
            plt.figure(figsize=(12, 8))
            YLABEL_STATS = f'Augmentation parameters for {stat_name}'

            for i, (env, statvals) in enumerate(STATS_DATA.items()):
                values = statvals[stat_name]
                x_values = [i] * len(values)  # All values for this environment are plotted at the same x position

                for j, val in enumerate(values):
                    # print(j, MAX_INDICES[env][stat_name])
                    if values[j] == MAX_INDICES[env][stat_name]:
                        # print('equals_max', values[j], MAX_INDICES[env][stat_name])
                        plt.scatter(x_values[j], val, color='red', s=300, zorder=2)  # Highlight max entry
                    else:
                        # print('not_equal_max', j, MAX_INDICES[env][stat_name])
                        plt.scatter(x_values[j], val, color='blue', s=30, zorder=1)

                plt.plot([i] * len(values), values, linestyle=':', color='black')

            # Set x-ticks to environment names
            plt.xticks(range(len(STATS_DATA)), STATS_DATA.keys(), rotation=45, ha='right')
            TITLE_STATS = (f'{stat_name} Augment Params over Full Augmentation Range \n({XLABEL_STATS}) per Environment \nwith Highlighted Augment '
                           f'Parameter Corresponding to Max Environment Performance')
            plt.title(TITLE_STATS)
            plt.xlabel(XLABEL_STATS)
            plt.ylabel(YLABEL_STATS)
            plt.grid(True)
            plt.tight_layout()
            plt.show()


def plot_statistical_distance_metrics(STATS_DATA, MAX_INDICES, stats_column_names, XLABEL_STATS, show=True):
    """
    Plots statistical distance metrics for multiple environments.
    Highlights the max value entry for each environment with a larger red dot.

    Parameters:
    STATS_DATA (dict): Dictionary where keys are environment names and values are dictionaries of stat names to lists of values.
    MAX_INDICES (dict): Dictionary where keys are environment names and values are dictionaries of stat names to max value indices.
    stats_column_names (list): List of stat names to process.
    """
    if show:
        for stat_name in stats_column_names:
            # Create the plot for each stat_name
            plt.figure(figsize=(12, 8))
            YLABEL_STATS = f'Statistical Distance Units for {stat_name}'

            for i, (env, statvals) in enumerate(STATS_DATA.items()):
                values = statvals[stat_name]
                x_values = [i] * len(values)  # All values for this environment are plotted at the same x position

                for j, val in enumerate(values):
                    if j == MAX_INDICES[env][stat_name]:
                        plt.scatter(x_values[j], val, color='red', s=300, zorder=2)  # Highlight max entry
                    else:
                        plt.scatter(x_values[j], val, color='blue', s=30, zorder=1)

                plt.plot([i] * len(values), values, linestyle=':', color='black')

            # Set x-ticks to environment names
            plt.xticks(range(len(STATS_DATA)), STATS_DATA.keys(), rotation=45, ha='right')
            TITLE_STATS = f'{stat_name} Distance Values over Full Augmentation Range \n({XLABEL_STATS}) per Environment \nwith Highlighted Distance Data Point Corresponding to Max Environment Performance'
            plt.title(TITLE_STATS)
            plt.xlabel(XLABEL_STATS)
            plt.ylabel(YLABEL_STATS)
            plt.grid(True)
            plt.tight_layout()
            plt.show()


def process_and_generate_aug_stats(EXP_TYPE, ENV_NAME, EXP_NAME, LIST_VALS, MAX_TOP_FIVE, stats_column_names, STATS_DATA, MAX_INDICES):
    """
    Fetches augmentation stats, performs analysis, and updates STATS_DATA and MAX_INDICES.

    Parameters:
    EXP_TYPE (str): Experiment type.
    ENV_NAME (str): Environment name.
    EXP_NAME (str): Experiment name.
    LIST_VALS (list): List of values used in the experiment.
    MAX_TOP_FIVE (float): Maximum top value for the environment.
    stats_column_names (list): List of stat names to process.
    STATS_DATA (dict): Dictionary to store stats data.
    MAX_INDICES (dict): Dictionary to store max indices for each environment.
    """
    datasets_augstats_dict, list_of_root_dirs_by_augment_stats = fetch_data_aug_stats(
        EXP_TYPE, ENV_NAME, EXP_NAME, LIST_VALS, show=False)

    ENV_NAME_ST = ENV_NAME + '_' + EXP_TYPE + '_max_at_' + str(MAX_TOP_FIVE)
    STATS_DATA[ENV_NAME_ST], MAX_INDICES[ENV_NAME_ST] = {}, {}

    for stat_name in stats_column_names:
        result_stats = generate_stats_for_augment_stats_directories(
            list_of_root_dirs_by_augment_stats, datasets_augstats_dict, stat_name)
        mean_vals_statdistances = extract_stat(result_stats, stat_name='Mean', stat_type=None)
        sorted_mean_diststats = print_sorted(mean_vals_statdistances, sort_by='key', desc=False, print_it=False)

        STATS_DATA[ENV_NAME_ST][stat_name] = [s[1] for s in sorted_mean_diststats]
        MAX_INDICES[ENV_NAME_ST][stat_name] = [s[0] for s in sorted_mean_diststats].index(MAX_TOP_FIVE)


def process_and_analyze_data(
        EXP_TYPE, ENV_NAME, EXP_NAME, LIST_VALS, column_name, XLABEL_STATS, USE_VARIANCE, TIMES_STDDEV, DICTS_ALL_STATS, show=True):
    """
    Fetches data, performs analysis, and updates DICTS_ALL_STATS.

    Parameters:
    EXP_TYPE (str): Experiment type.
    ENV_NAME (str): Environment name.
    EXP_NAME (str): Experiment name.
    LIST_VALS (list): List of values used in the experiment.
    column_name (str): Column name for analysis.
    XLABEL_STATS (str): Label for x-axis in the plot.
    USE_VARIANCE (bool): Flag to indicate if variance should be used.
    TIMES_STDDEV (int): Multiplier for standard deviation.
    DICTS_ALL_STATS (dict): Dictionary to store all statistics.
    """
    datasets_dict, list_of_root_dirs_by_augment_degree = fetch_data(
        EXP_TYPE, ENV_NAME, EXP_NAME, LIST_VALS, show=False)

    MIN_TOP_FIVE, MAX_TOP_FIVE, summary_statistics, sorted_items = display_analysis(
        datasets_dict, list_of_root_dirs_by_augment_degree, ENV_NAME, column_name, show=False)

    MAX_TOP_FIVE = sorted_items[0][0]
    DICTS_ALL_STATS[ENV_NAME][column_name + '_sorted'] = sorted_items
    DICTS_ALL_STATS[ENV_NAME]['range_top_five'] = (MIN_TOP_FIVE, MAX_TOP_FIVE)
    if show:
        plot_performance_shaded_area(
            summary_statistics, ENV_NAME, column_name, XLABEL_STATS, use_var=USE_VARIANCE, times_stddev=TIMES_STDDEV)
    return MAX_TOP_FIVE


def fetch_data_aug_stats(exp_type, env_name, exp_name, list_vals, show=False):
    subdir_pattern = '*/*/augment.csv'
    list_of_root_dirs_by_augment_stats = generate_root_dirs_by_experiment_and_augment_degree_new_augment_stats(
        exp_main_folder='exp-' + exp_type,
        exp_name=env_name + '_augment_stats',
        agent_name='drqv2_aug_pixels-True-' + exp_name + '-',
        list_vals=list_vals)

    datasets_augstats_dict = load_datasets_by_directory(
        list_of_root_dirs_by_augment_stats, subdir_pattern, show=False)

    if show:
        for root_directory in list_of_root_dirs_by_augment_stats:
            print(root_directory, ' : ', len(datasets_augstats_dict[root_directory]))
    return datasets_augstats_dict, list_of_root_dirs_by_augment_stats


def fetch_data(exp_type, env_name, exp_name, list_vals, show=False):
    subdir_pattern = '*/*/eval.csv'  # This example finds all .txt files in all subdirectories

    list_of_root_dirs_by_augment_degree = generate_root_dirs_by_experiment_and_augment_degree_new(
        exp_main_folder='exp-' + exp_type,
        exp_name=env_name,
        agent_name='drqv2_aug_pixels-True-' + exp_name + '-',
        list_vals=list_vals)

    datasets_dict = load_datasets_by_directory(list_of_root_dirs_by_augment_degree, subdir_pattern, show=show)
    if show:
        for root_directory in list_of_root_dirs_by_augment_degree:
            print(root_directory, ' : ', len(datasets_dict[root_directory]))
    return datasets_dict, list_of_root_dirs_by_augment_degree


def display_analysis(datasets_dict, list_of_root_dirs_by_augment_degree, env_name, col_name, show=False):
    explainer = "Highest to Lowest"
    column_name = col_name
    result_stats = generate_stats_for_directories(list_of_root_dirs_by_augment_degree, datasets_dict, column_name)
    summary_statistics = compute_summary_stats(result_stats)

    if show:
        for key, stats in summary_statistics.items():
            print(f"Directory: {key}")
            print("Sum Statistics:", stats['Sum Statistics'])
            print("Max Statistics:", stats['Max Statistics'])
    if show:
        print("\n", env_name, "...printing Mean over Sums for", col_name, "(", explainer, ")")
    mean_vals_over_sums_performance = extract_stat(summary_statistics, stat_name='Sum Mean', stat_type='Sum Statistics')
    sum_mean_sorted_items = print_sorted(mean_vals_over_sums_performance, sort_by='value', desc=True, print_it=show)

    MIN_TOP_FIVE = min([i[0] for i in sum_mean_sorted_items[:5]])
    MAX_TOP_FIVE = max([i[0] for i in sum_mean_sorted_items[:5]])
    if show:
        print("\n", env_name, "...printing Max over Sums for", col_name, "(", explainer, ")")
    max_vals_over_sums_performance = extract_stat(summary_statistics, stat_name='Sum Max', stat_type='Sum Statistics')
    sorted_items = print_sorted(max_vals_over_sums_performance, sort_by='value', desc=True, print_it=show)
    if show:
        print("\n", env_name, "...printing Max over Maxes for", col_name, "(", explainer, ")")
    max_vals_over_max_performance = extract_stat(summary_statistics, stat_name='Max Max', stat_type='Max Statistics')
    sorted_items = print_sorted(max_vals_over_max_performance, sort_by='value', desc=show, print_it=show)
    if show:
        print("\n", env_name, "...printing Mean over Maxes for", col_name, "(", explainer, ")")
    mean_vals_over_max_performance = extract_stat(summary_statistics, stat_name='Max Mean', stat_type='Max Statistics')
    sorted_items = print_sorted(mean_vals_over_max_performance, sort_by='value', desc=show, print_it=show)
    return MIN_TOP_FIVE, MAX_TOP_FIVE, summary_statistics, sum_mean_sorted_items


def plot_aug_stats_dotplot(datasets_augstats_dict, list_of_root_dirs_by_augment_stats, stat_name, MIN_TOP_FIVE, MAX_TOP_FIVE, xlabels, show=True):
    print('PLOTTING AUGMENT STATS - ', stat_name)
    result_stats = generate_stats_for_augment_stats_directories(list_of_root_dirs_by_augment_stats, datasets_augstats_dict, stat_name)
    mean_vals_statdistances = extract_stat(result_stats, stat_name='Mean', stat_type=None)
    sorted_mean_diststats = print_sorted(mean_vals_statdistances, sort_by='key', desc=False, print_it=False)
    vals_to_str = " | ".join([str(t[1]) for t in sorted_mean_diststats])
    # print(vals_to_str)
    # print(sorted_mean_diststats)

    min_vals_statdistances = extract_stat(result_stats, stat_name='Min', stat_type=None)
    sorted_min_diststats = print_sorted(min_vals_statdistances, sort_by='key', desc=False, print_it=False)

    max_vals_statdistances = extract_stat(result_stats, stat_name='Max', stat_type=None)
    sorted_max_diststats = print_sorted(max_vals_statdistances, sort_by='key', desc=False, print_it=False)

    categories = [sm[0] for sm in sorted_mean_diststats]
    mean_sum_stats_vals = [sm[1] for sm in sorted_mean_diststats]
    min_sum_vals = [sm[1] for sm in sorted_min_diststats]
    max_sum_vals = [sm[1] for sm in sorted_max_diststats]

    # Calculate error values (difference between mean and min/max)
    lower_error = np.array(mean_sum_stats_vals) - np.array(min_sum_vals)
    upper_error = np.array(max_sum_vals) - np.array(mean_sum_stats_vals)
    errors = [lower_error, upper_error]

    ylabels = 'Statistical Distance Units'

    # Define the range for the vertical highlight
    highlight_xmin = MIN_TOP_FIVE  # Index of the second category
    highlight_xmax = MAX_TOP_FIVE  # Index of the fourth category

    # Plotting the dot plot with highlighted points for MAX TOP FIVE
    plt.figure(figsize=(12, 8))
    print(datasets_augstats_dict)
    for category in [int(x) for x in categories]:
        all_values = datasets_augstats_dict[category][stat_name]
        x_values = [category] * len(all_values)
        y_values = all_values

        plt.scatter(x_values, y_values, color='blue', s=10, alpha=0.6)

        # Highlight the MAX TOP FIVE points
        for value in y_values:
            if value >= highlight_xmin and value <= highlight_xmax:
                plt.scatter(category, value, color='red', s=100, alpha=0.8)

    # plt.title(f'{stat_name} Distribution with Highlights for {env_name}')
    plt.title(f'{stat_name} Distribution with Highlights for ...')
    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if show:
        plt.show()

    return sorted_mean_diststats


def plot_aug_stats(datasets_augstats_dict, list_of_root_dirs_by_augment_stats, stat_name, MIN_TOP_FIVE, MAX_TOP_FIVE, xlabels, show=True):
    print('PLOTTING AUGMENT STATS - ', stat_name)
    result_stats = generate_stats_for_augment_stats_directories(list_of_root_dirs_by_augment_stats, datasets_augstats_dict, stat_name)
    mean_vals_statdistances = extract_stat(result_stats, stat_name='Mean', stat_type=None)
    sorted_mean_diststats = print_sorted(mean_vals_statdistances, sort_by='key', desc=False, print_it=False)
    vals_to_str = " | ".join([str(t[1]) for t in sorted_mean_diststats])
    print(vals_to_str)
    print(sorted_mean_diststats)

    min_vals_statdistances = extract_stat(result_stats, stat_name='Min', stat_type=None)
    sorted_min_diststats = print_sorted(min_vals_statdistances, sort_by='key', desc=False, print_it=False)

    max_vals_statdistances = extract_stat(result_stats, stat_name='Max', stat_type=None)
    sorted_max_diststats = print_sorted(max_vals_statdistances, sort_by='key', desc=False, print_it=False)

    categories = [sm[0] for sm in sorted_mean_diststats]
    mean_sum_stats_vals = [sm[1] for sm in sorted_mean_diststats]
    min_sum_vals = [sm[1] for sm in sorted_min_diststats]
    max_sum_vals = [sm[1] for sm in sorted_max_diststats]

    # Calculate error values (difference between mean and min/max)
    lower_error = np.array(mean_sum_stats_vals) - np.array(min_sum_vals)
    upper_error = np.array(max_sum_vals) - np.array(mean_sum_stats_vals)
    errors = [lower_error, upper_error]

    ylabels = 'Statistical Distance Units'
    # Plotting the bar plot with error bars
    plt.figure(figsize=(10, 6))
    plt.bar(categories, mean_sum_stats_vals, yerr=errors, capsize=5, color='skyblue', alpha=0.7, ecolor='black')
    plt.title(stat_name + ' - Metrics Units with Error Bars vs ' + xlabels)
    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.show()

    # Scatter Plot with Error Bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(categories, mean_sum_stats_vals, yerr=errors, fmt='o', ecolor='black', capsize=5, label='Mean with Min-Max Range')
    plt.title(stat_name + ' - Scatter Plot of Metrics Units with Error Bars vs ' + xlabels)
    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.legend()
    plt.show()

    # Line Plot with Shaded Error Region
    plt.figure(figsize=(10, 6))
    plt.plot(categories, mean_sum_stats_vals, marker='o', color='b', label='Mean')
    plt.fill_between(categories, min_sum_vals, max_sum_vals, color='b', alpha=0.2, label='Min-Max Range')
    plt.title(stat_name + ' - Line Plot of Metrics Units with Shaded Error Region vs ' + xlabels)
    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.legend()
    plt.show()

    # Define the range for the vertical highlight
    highlight_xmin = MIN_TOP_FIVE  # Index of the second category
    highlight_xmax = MAX_TOP_FIVE  # Index of the fourth category

    if show:
        for i in range(len(sorted_mean_diststats)):
            if i <= highlight_xmax and i >= highlight_xmin:
                print('padding:', sorted_mean_diststats[i][0], stat_name + ':', sorted_mean_diststats[i][1])

    # Bar Plot with Error Bars and Vertical Highlight
    plt.figure(figsize=(10, 6))
    plt.bar(categories, mean_sum_stats_vals, yerr=errors, capsize=5, color='skyblue', alpha=0.7, ecolor='black')
    plt.axvspan(highlight_xmin - 0.5, highlight_xmax + 0.5, color='yellow', alpha=0.3)
    plt.title(stat_name + ' - Performance Metrics of Metrics Units with Error Bars vs ' + xlabels)
    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.show()

    # Scatter Plot with Error Bars and Vertical Highlight
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        categories, mean_sum_stats_vals, yerr=errors, fmt='o', ecolor='black', capsize=5, label='Mean Stat Distance Values with Min-Max Range')
    plt.axvspan(highlight_xmin - 0.5, highlight_xmax + 0.5, color='yellow', alpha=0.3)
    plt.title(stat_name + ' - Scatter Plot of Metrics Units with Error Bars vs ' + xlabels)
    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.legend()
    plt.show()

    # Line Plot with Shaded Error Region and Vertical Highlight
    plt.figure(figsize=(10, 6))
    plt.plot(categories, mean_sum_stats_vals, marker='o', color='b', label='Mean Stat Distance Values')
    plt.fill_between(categories, min_sum_vals, max_sum_vals, color='b', alpha=0.2, label='Min-Max Range')
    plt.axvspan(highlight_xmin - 0.5, highlight_xmax + 0.5, color='yellow', alpha=0.3)
    plt.title(stat_name + ' - Line Plot of Metrics Units with Shaded Error Region vs ' + xlabels)
    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.legend()
    plt.show()
    return sorted_mean_diststats


def plot_performance_all(summary_statistics, env_name, performance_param, x_label):
    # Sample data with mean, min, and max values for each category
    y_label = 'Total Sum of Episodic Rewards'

    print("\n", env_name, "...Plotting Sum Stats for", performance_param)
    min_vals_over_sum_performance = extract_stat(summary_statistics, stat_name='Sum Min', stat_type='Sum Statistics')
    mean_vals_over_sum_performance = extract_stat(summary_statistics, stat_name='Sum Mean', stat_type='Sum Statistics')
    max_vals_over_sum_performance = extract_stat(summary_statistics, stat_name='Sum Max', stat_type='Sum Statistics')

    categories = list(min_vals_over_sum_performance.keys())
    mean_sum_vals = list(mean_vals_over_sum_performance.values())
    min_sum_vals = list(min_vals_over_sum_performance.values())
    max_sum_vals = list(max_vals_over_sum_performance.values())

    # Calculate error values (difference between mean and min/max)
    lower_error = np.array(mean_sum_vals) - np.array(min_sum_vals)
    upper_error = np.array(max_sum_vals) - np.array(mean_sum_vals)
    errors = [lower_error, upper_error]

    # Plotting the bar plot with error bars
    plt.figure(figsize=(10, 6))
    plt.bar(categories, mean_sum_vals, yerr=errors, capsize=5, color='skyblue', alpha=0.8, ecolor='black')
    plt.title(env_name + ' - Performance Metrics with Error Bars')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0, max(max_sum_vals) + (0.05 * max(max_sum_vals)))
    plt.show()

    # Scatter Plot with Error Bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(categories, mean_sum_vals, yerr=errors, fmt='o', ecolor='black', capsize=5, label='Mean with Min-Max Range')
    plt.title(env_name + ' - Scatter Plot with Error Bars')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.ylim(0, max(max_sum_vals) + (0.05*max(max_sum_vals)))
    plt.legend()
    plt.show()

    # Line Plot with Shaded Error Region
    plt.figure(figsize=(10, 6))
    plt.plot(categories, mean_sum_vals, marker='o', color='b', label='Mean')
    plt.fill_between(categories, min_sum_vals, max_sum_vals, color='b', alpha=0.3, label='Min-Max Range')
    plt.title(env_name + ' - Line Plot with Shaded Error Region')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.ylim(0, max(max_sum_vals) + (0.05 * max(max_sum_vals)))
    plt.legend()
    plt.show()

    # Sample data with mean, min, and max values for each category
    print("\n", env_name, "...plotting Max Stats for", performance_param)
    y_label = 'Max Episodic Rewards'
    min_vals_over_max_performance = extract_stat(summary_statistics, stat_name='Max Min', stat_type='Max Statistics')
    mean_vals_over_max_performance = extract_stat(summary_statistics, stat_name='Max Mean', stat_type='Max Statistics')
    max_vals_over_max_performance = extract_stat(summary_statistics, stat_name='Max Max', stat_type='Max Statistics')

    categories = list(max_vals_over_max_performance.keys())
    mean_vals = list(mean_vals_over_max_performance.values())
    min_vals = list(min_vals_over_max_performance.values())
    max_vals = list(max_vals_over_max_performance.values())

    # Calculate error values (difference between mean and min/max)
    lower_error = np.array(mean_vals) - np.array(min_vals)
    upper_error = np.array(max_vals) - np.array(mean_vals)
    errors = [lower_error, upper_error]

    # Plotting the bar plot with error bars
    plt.figure(figsize=(10, 6))
    plt.bar(categories, mean_vals, yerr=errors, capsize=5, color='skyblue', alpha=0.7, ecolor='black')
    plt.title(env_name + ' - Performance Metrics with Error Bars')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

    # Scatter Plot with Error Bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(categories, mean_vals, yerr=errors, fmt='o', ecolor='black', capsize=5, label='Mean with Min-Max Range')
    plt.title(env_name + ' - Scatter Plot with Error Bars')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

    # Line Plot with Shaded Error Region
    plt.figure(figsize=(10, 6))
    plt.plot(categories, mean_vals, marker='o', color='b', label='Mean')
    plt.fill_between(categories, min_vals, max_vals, color='b', alpha=0.2, label='Min-Max Range')
    plt.title(env_name + ' - Line Plot with Shaded Error Region')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def compute_variance(data):
    mean = np.mean(data)
    variance = np.mean((data - mean) ** 2)
    return variance


def plot_performance_shaded_area(summary_statistics, env_name, performance_param, x_label, use_var=False, times_stddev=2):
    y_label = 'Total Sum of Episodic Rewards'

    # print("\n", env_name, "...Plotting Sum Stats for", performance_param)
    mean_vals_over_sum_performance = extract_stat(summary_statistics, stat_name='Sum Mean', stat_type='Sum Statistics')
    var_vals_over_sum_performance = extract_stat(summary_statistics, stat_name='Var Mean', stat_type='Var Statistics')
    # print(mean_vals_over_sum_performance)
    # print(var_vals_over_sum_performance)
    categories = list(mean_vals_over_sum_performance.keys())
    mean_sum_vals = list(mean_vals_over_sum_performance.values())

    std_sum_vals = [np.sqrt(var) for var in var_vals_over_sum_performance.values()]
    if use_var:
        var_sum_vals = list(var_vals_over_sum_performance.values())

    # Calculate the shaded area (mean ± standard deviation)
    lower_bound = [mean - (times_stddev * std) for mean, std in zip(mean_sum_vals, std_sum_vals)]
    upper_bound = [mean + (times_stddev * std) for mean, std in zip(mean_sum_vals, std_sum_vals)]
    if use_var:
        lower_bound = [mean - var for mean, var in zip(mean_sum_vals, var_sum_vals)]
        upper_bound = [mean + var for mean, var in zip(mean_sum_vals, var_sum_vals)]

    plot_label = 'Mean ± ' + str(times_stddev) + '*StdDev'
    if use_var:
        plot_label = 'Mean ± Var'

    # Line Plot with Shaded Error Region
    plt.figure(figsize=(10, 6))
    plt.plot(categories, mean_sum_vals, marker='o', color='b', label='Mean')
    plt.fill_between(categories, lower_bound, upper_bound, color='b', alpha=0.3, label=plot_label)
    plt.title(env_name + ' - Line Plot with Shaded Error Region')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def compute_correlation(data1, data2):
    if len(data1) != len(data2):
        raise ValueError("Data1 and Data2 must be of the same length to compute correlation.")

    # Extract y-values from both datasets
    y_values1 = [y for _, y in data1]
    y_values2 = [y for _, y in data2]

    # Compute the Pearson correlation coefficient
    correlation, _ = pearsonr(y_values1, y_values2)

    return correlation


def normalize(data):
    """
    Normalize the list of y-values using Min-Max scaling.

    Args:
    data (list): List of y-values.

    Returns:
    list: Normalized list of y-values.
    """
    min_val = min(data)
    max_val = max(data)
    return [(y - min_val) / (max_val - min_val) for y in data]


def compute_normalized_correlation(data1, data2):
    if len(data1) != len(data2):
        raise ValueError("Data1 and Data2 must be of the same length to compute correlation.")

    # Extract and normalize y-values from both datasets
    y_values1 = normalize([y for _, y in data1])
    y_values2 = normalize([y for _, y in data2])

    # Compute the Pearson correlation coefficient
    correlation, _ = pearsonr(y_values1, y_values2)

    return correlation


def plot_dual_dot_plots(data1, data2, data1_title, data2_title, augment_val, data1_ylabel, data2_ylabel):
    # Create a figure and two subplots (axes), stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))  # figsize controls the size of the overall figure

    # Plotting the first dataset on the first subplot
    x_values1, y_values1 = zip(*data1)
    ax1.scatter(x_values1, y_values1, color='blue', marker='o')  # Blue dots
    ax1.set_title(data1_title)
    ax1.set_xlabel(augment_val)
    ax1.set_ylabel(data1_ylabel)
    ax1.grid(True)  # Enable grid for better readability

    # Plotting the second dataset on the second subplot
    x_values2, y_values2 = zip(*data2)
    ax2.scatter(x_values2, y_values2, color='green', marker='o')  # Green dots
    ax2.set_title(data2_title)
    ax2.set_xlabel(augment_val)
    ax2.set_ylabel(data2_ylabel)
    ax2.grid(True)

    # Adjust layout to prevent overlap of subplots
    plt.tight_layout()

    # Display the plots
    plt.show()


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

            # Compute variance and standard deviation of 'Min', 'Max', and 'Mean' columns
            var_stats = {
                'Var Min': df['Min'].var(),
                'Var Max': df['Max'].var(),
                'Var Mean': df['Mean'].var()
            }

            stddev_stats = {
                'StdDev Min': np.sqrt(var_stats['Var Min']),
                'StdDev Max': np.sqrt(var_stats['Var Max']),
                'StdDev Mean': np.sqrt(var_stats['Var Mean'])
            }

            # Store results in a structured dictionary
            summary_stats[key] = {
                'Sum Statistics': sum_stats,
                'Max Statistics': max_stats,
                'Var Statistics': var_stats,
                'StdDev Statistics': stddev_stats
            }
        else:
            summary_stats[key] = {
                'Sum Statistics': {'Sum Min': None, 'Sum Max': None, 'Sum Mean': None},
                'Max Statistics': {'Max Min': None, 'Max Max': None, 'Max Mean': None},
                'Var Statistics': {'Var Min': None, 'Var Max': None, 'Var Mean': None},
                'StdDev Statistics': {'StdDev Min': None, 'StdDev Max': None, 'StdDev Mean': None}
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


def generate_root_dirs_by_experiment_and_augment_degree_new_augment_stats(exp_main_folder, exp_name, agent_name, list_vals=None):
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


def compute_stats_across_dfs(dataframes, column_name):
    concatenated = pd.Series()
    for df in dataframes:
        if column_name not in df.columns:
            raise ValueError(f"The column '{column_name}' is not in one of the DataFrames.")
        concatenated = pd.concat([concatenated, df[column_name]], ignore_index=True)

    # Calculate the statistics
    min_value = concatenated.min()
    max_value = concatenated.max()
    mean_value = concatenated.mean()
    var_value = concatenated.var()

    # Return the results as a dictionary
    return {
        'Min': min_value,
        'Max': max_value,
        'Mean': mean_value,
        'Var': var_value
    }


def generate_stats_for_augment_stats_directories(root_dirs, datasets_dict, column_name):
    stats_results = {}
    for root_dir in root_dirs:
        try:
            # Check if there are any DataFrames in the list for this directory
            if root_dir in datasets_dict and datasets_dict[root_dir]:
                # Compute aggregated statistics across all DataFrames for the current directory
                stats_dict = compute_stats_across_dfs(datasets_dict[root_dir], column_name)
                stats_results[root_dir] = stats_dict
            else:
                print(f"No data available for {root_dir}")
        except Exception as e:
            print(f"Error processing {root_dir}: {e}")
            stats_results[root_dir] = None

    return stats_results


def extract_stat(stats_dict, stat_name, stat_type=None):
    if stat_type:
        return {float(key.split("-")[-1]): stats[stat_type][stat_name] for key, stats in stats_dict.items()}
    return {float(key.split("-")[-1]): stats[stat_name] for key, stats in stats_dict.items()}


def print_sorted(dictionary, sort_by='key', desc=True, print_it=True):
    if sort_by == 'value':
        # Sorting dictionary by value in descending order
        sorted_items = sorted(dictionary.items(), key=lambda item: item[1], reverse=desc)
    else:
        # Sorting dictionary by key in ascending order
        sorted_items = sorted(dictionary.items(), key=lambda item: item[0], reverse=desc)

    # Print sorted dictionary
    if print_it:
        for key, value in sorted_items:
            print(f"{key}: {value}")

    return sorted_items


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
