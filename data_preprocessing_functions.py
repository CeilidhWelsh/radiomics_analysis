"""
Module for managing file dictionaries and processing CSV files related to structure analysis.

Functions:
    - construct_file_dictionaries: Constructs a nested dictionary structure to organise file paths based on trials and cohort subdivisions.
    - populate_file_dictionaries: Populates the file dictionary with file paths based on provided trial names, cohort subdivisions, and output directory.
    - combine_csv_files: Combines multiple CSV files into a single DataFrame for each category and saves the merged DataFrame to a specified directory.
    - add_recurrence_information: Adds a recurrence flag to the DataFrames based on the file name and rewrites the updated DataFrames.

Imports:
    - pandas: A library for data manipulation and analysis, especially useful for handling CSV files.
    - structure_analysis_functions: Custom functions used for structure analysis (assumed to be defined in another module).
"""

import pandas as pd 
from structure_analysis_functions import * 

def construct_file_dictionaries(trial_names, cohort_subdivisions, file_names):
    """
    Constructs a nested dictionary to organise file paths based on trial names and cohort subdivisions.

    Parameters:
        trial_names (list of str): List of trial names.
        cohort_subdivisions (list of str): List of cohort subdivisions.
        file_names (list of str): List of file names to include in the dictionary.

    Returns:
        dict: A nested dictionary with file names as keys, containing lists for 'all', each trial, and each cohort subdivision.
    """
    file_dict = {}
    for file_name in file_names:
        file_dict[file_name] = {}
        file_dict[file_name]['all'] = []
        for subdivision in cohort_subdivisions:
            file_dict[file_name][subdivision] = []
            for trial in trial_names:
                file_dict[file_name][trial] = []
    return file_dict

def populate_file_dictionaries(file_dict, trial_names, cohort_subdivisions, file_names, output_directory):
    """
    Populates the file dictionary with constructed file paths based on trial names, cohort subdivisions, and output directory.

    Parameters:
        file_dict (dict): The dictionary to populate with file paths.
        trial_names (list of str): List of trial names.
        cohort_subdivisions (list of str): List of cohort subdivisions.
        file_names (list of str): List of file names to populate.
        output_directory (str): Directory where the files are located.

    Returns:
        dict: The updated file dictionary populated with file paths.
    """
    print(file_names)
    for file_name in file_names:
        for subdivision in cohort_subdivisions: 
            for trial in trial_names:
                file_dict[file_name]['all'].append(f'{output_directory}/{trial}{subdivision}_{file_name}')
                file_dict[file_name][trial].append(f'{output_directory}/{trial}{subdivision}_{file_name}')
                file_dict[file_name][subdivision].append(f'{output_directory}/{trial}{subdivision}_{file_name}')
        print(file_dict)
    return file_dict

def combine_csv_files(file_dict, results_dir):
    """
    Combines CSV files listed in the file dictionary into a single DataFrame for each category and saves the result.

    Parameters:
        file_dict (dict): The dictionary containing file paths organised by categories.
        results_dir (str): The directory where the merged CSV files will be saved.

    Returns:
        None
    """
    for key, categories in file_dict.items():
        print('key:', key, 'categories:',  categories)
        
        for category, file_list in categories.items():
            dfs = []
            print('category:', category)
            print('file_list:', file_list)
            for filename in file_list:
                print(filename)
                df = pd.read_csv(filename)
                dfs.append(df)
        
            if dfs:  # Check if the list is not empty
                print('dfs check:', len(dfs))
                merged_df = pd.concat(dfs, ignore_index=True)
                
                directory_name = key.replace('.csv', '')
                ensure_directory_exists(f'{results_dir}/{directory_name}')

                merged_df.to_csv(f'{results_dir}/{directory_name}/{category}_{key}', index=False)
                print(f"Saved merged DataFrame to {category}_{key}")

def add_recurrence_information(file_dict): 
    """
    Adds recurrence information to DataFrames based on the file names and rewrites the updated DataFrames.

    Parameters:
        file_dict (dict): The dictionary containing file paths to CSV files.

    Returns:
        None
    """
    for key in file_dict.keys():
        for file in file_dict[key]['all']:
            print(file)
            df = pd.read_csv(f'{file}')
            if 'nonrecurrent' in file: 
                df['recurrence'] = 0
            else:
                df['recurrence'] = 1
                
            file_path_without_extension = file.rsplit('.', 1)[0]
            df.to_csv(f'{file_path_without_extension}.csv')
