"""
This script pulls patient files from a specified location and executes a series of image preprocessing steps to enhance the reproducibility of the workflow. The script extracts radiomic features from planning images and their associated structures and saves the results in a CSV format.

Functions:
    - ensure_directory_exists: Checks if a directory exists and creates it if it does not.
    - obtain_radiomic_masks: Retrieves the necessary masks for radiomic feature extraction.
    - extract_radiomics_features: Extracts features from medical images using the provided extractor.
    - find_matching_string: Searches for specific structures in the given directory.

Imports:
    - os: For interacting with the operating system.
    - sys: To access command-line arguments.
    - skrt: A library for handling radiotherapy datasets and processing images.
    - numpy: For numerical operations and array manipulations.
    - pandas: For data manipulation and CSV file generation.
    - ast: For evaluating string representations of Python expressions.
    - matplotlib.pyplot: For data visualisation (not used in the script but imported).
    - pathlib.Path: For handling filesystem paths.
    - radiomics: Library for extracting features from medical images.
    - radiomic_analysis_classes: Custom functions and classes for specific analyses.

Usage:
    Run the script from the command line with the following parameters:
    
    python <script_name.py> <dataset_name> <planning_data_dir> <data_dir> <local> <subset> <patient_list> <additional_dataset_name>
"""

# Import standard Python packages
import os
import sys
import skrt
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from pathlib import Path

# Import scikit-rt modules 
from skrt import Patient, ROI, StructureSet, Image
from skrt.registration import get_default_pfiles, Registration, set_elastix_dir
from skrt.better_viewer import BetterViewer
from skrt.simulation import SyntheticImage

from radiomics import featureextractor  
from radiomic_analysis_classes import * 

# Pull in the variables from the shell script 
dataset_name = sys.argv[1]                # Name of the dataset
planning_data_dir = sys.argv[2]           # Directory for planning data
data_dir = sys.argv[3]                    # Directory to save processed data
local = sys.argv[4]                       # Local machine flag
subset = sys.argv[5]                      # Subset flag
patient_list = sys.argv[6]                # List of patients to process
additional_dataset_name = sys.argv[7]     # Additional dataset name for output files

# Patient list processing
if subset == "False": 
    patient_list = os.listdir(data_dir)
    if '.DS_Store' in patient_list:
        patient_list.remove('.DS_Store')
else:
    patient_list = ast.literal_eval(patient_list) 

# Instantiate the extractor
extractor = featureextractor.RadiomicsFeatureExtractor()

# Initialise an empty DataFrame for storing radiomic features
radiomics_df = pd.DataFrame()

# Pull the planning image from the patient folder 
i = 1 
for patient in patient_list:
    print('> Patient:', patient, i, 'out of', len(patient_list))
    i += 1
    output_data_dir = f'{data_dir}/{patient}/'
    ensure_directory_exists(output_data_dir)

    results_dir = f'{output_data_dir}/results'
    structure_dir = f'{output_data_dir}/structures'
    radiomics_dir  = f'{output_data_dir}/radiomics_features'
    ensure_directory_exists(radiomics_dir)

    # Obtain the available planning structures as a list
    planning_files = os.listdir(results_dir)
    structure_files = os.listdir(structure_dir)
    bin_number_list = [0, 16, 32, 64, 128]

    # Iterate through each structure and bin number
    for bin_number in bin_number_list:
        for structure_file in structure_files:
            print(structure_file, bin_number)
            base_name, _ = os.path.splitext(structure_file)
            if '.' in base_name:
                base_name, _ = os.path.splitext(base_name)
            print('Base name:', base_name)
            ctv_mask, planning_im_mask = obtain_radiomic_masks(
                planning_search_string='planning_im',
                results_dir=results_dir,
                structure_file_name=base_name,
                bin_number=bin_number,
                base_name=base_name
            )
            
            # Extract radiomic features
            radiomics_dict = extract_radiomics_features(extractor, im_path=planning_im_mask, label_path=ctv_mask)

            # Prepare results DataFrame
            result_test_df = pd.DataFrame([radiomics_dict])
            result_test_df['patient'] = patient 
            result_test_df['ctv_structure'] = base_name
            result_test_df['bin_number'] = bin_number
            result_test_df['multiple_ctvs'] = find_matching_string(structure_dir, search_string='ctv2')
            result_test_df['nodes'] = find_matching_string(structure_dir, search_string='node')
            result_test_df['multiple_nodes'] = find_matching_string(structure_dir, search_string='node2')
            result_test_df['trial'] = dataset_name

            # Concatenate results
            radiomics_df = pd.concat([radiomics_df, result_test_df])

# Save the final structure DataFrame 
if subset == "True":
    # Check if the output CSV file already exists
    if os.path.exists(f'{dataset_name}{additional_dataset_name}_extracted_radiomic_features.csv'):
        # Load and save an updated version 
        loaded_radiomics_df = pd.read_csv(f'{dataset_name}{additional_dataset_name}_extracted_radiomic_features.csv')

        # Combine existing and new DataFrames
        concatenated_df = pd.concat([loaded_radiomics_df, radiomics_df], ignore_index=True)

        # Drop duplicates, keeping only the last occurrence
        updated_radiomics_df = concatenated_df.drop_duplicates(subset='patient', keep='last')
        # Save updated dataset 
        updated_radiomics_df.to_csv(f'{dataset_name}{additional_dataset_name}_extracted_radiomic_features.csv')
        
    else: 
        radiomics_df.to_csv(f'{dataset_name}{additional_dataset_name}_extracted_radiomic_features.csv')
else: 
    radiomics_df.to_csv(f'{dataset_name}{additional_dataset_name}_extracted_radiomic_features.csv')
