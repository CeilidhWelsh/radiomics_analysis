# 30-07-24
"""
This script pulls the patient file from a saved location and executes a series of image preprocessing 
steps to enhance workflow reproducibility. It loads patient data, processes images, extracts radiomic features, 
and saves the results into a CSV file.

Functions:
    - ensure_directory_exists(directory): Ensures that the specified directory exists; 
    creates it if not.
    - find_matching_string(directory, search_string): Searches for files in the given 
    directory that match the search string.
    - augment_images(results_dir, base_name, local): Performs image augmentation based 
    on input parameters.
    - extract_radiomics_features(extractor, im_path, label_path): Extracts radiomic 
    features from the provided image and label paths.


Usage:
    python <script_name.py> <dataset_name> <planning_data_dir> <data_dir> <local> <subset> <patient_list> <additional_dataset_name>
"""

# Import necessary packages
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import additional packages
import ast

# Importing custom radiomic analysis classes
from radiomic_analysis_classes import * 

# Pull in the variables from the shell script
dataset_name = sys.argv[1]
planning_data_dir = sys.argv[2]
data_dir = sys.argv[3]
local = sys.argv[4]
subset = sys.argv[5]
patient_list = sys.argv[6]
additional_dataset_name = sys.argv[7]

error_patients = []

# Check if the radiomics CSV file exists already
radiomics_df_filename = f'updating_{dataset_name}{additional_dataset_name}_extracted_radiomic_features_edited_images.csv'
if os.path.isfile(radiomics_df_filename):
    print('Radiomics DataFrame already exists, loading into the Python session.')
    radiomics_df = pd.read_csv(radiomics_df_filename)
else:
    radiomics_df = pd.DataFrame()

# Load the patient list
if subset == "False":
    patient_list = os.listdir(data_dir)
    print(patient_list)
    if '.DS_Store' in patient_list:
        patient_list = [item for item in patient_list if '.DS_Store' not in item]
else:
    patient_list = ast.literal_eval(patient_list)

print('> Patient list:', patient_list)

# Pull the planning image from the patient folder
for patient in patient_list:
    print('> Patient:', patient)
    output_data_dir = f'{data_dir}/{patient}/'
    ensure_directory_exists(output_data_dir)

    # Set up results and structure directories
    results_dir = f'{output_data_dir}/results'
    structure_dir = f'{output_data_dir}/structures'
    ensure_directory_exists(results_dir)

    # Determine if the patient has nodes and do the same image analysis
    nodes = find_matching_string(structure_dir, search_string='planning_node_ctv')
    ctv = find_matching_string(structure_dir, search_string='planning_ctv')

    # Define planning and structure files based on node and CTV presence
    if ctv:
        if nodes:
            planning_files = ['planning_im_planning_ctv.nii', 'planning_im_planning_node_ctv.nii']
            structure_files = ['planning_ctv_resegmented.nii', 'planning_node_ctv_resegmented.nii']
        else:
            planning_files = ['planning_im_planning_ctv.nii']
            structure_files = ['planning_ctv_resegmented.nii']
    else:
        planning_files = ['planning_im_planning_node_ctv.nii']
        structure_files = ['planning_node_ctv_resegmented.nii']

    if not ctv and not nodes:
        # Skip to the next loop if neither CTV nor nodes exist
        error_patients.append(patient)
        print('Error patients:', error_patients)
        continue

    # Use local as a surrogate flag for recurrence
    print('> Local:', local)
    if local == "False":
        augmented_files = ['scrambled_planning_im', 'randomised_planning_im']
    else:
        augmented_files = ['scrambled_planning_im', 'randomised_planning_im', 'augmented_planning_im']

    for structure_file in structure_files:
        print('> Structure File:', structure_file)
        base_name, _ = os.path.splitext(structure_file)
        base_name = base_name.replace('_resegmented', '')
        if '.' in base_name:
            base_name, _ = os.path.splitext(base_name)

        print('> Base name:', base_name)

        # Call function that employs the three subfunctions
        augment_images(results_dir=results_dir, base_name=base_name, local=local)

        # Instantiate the extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()

        # Extract the features by obtaining the correct file pairs
        for file in augmented_files:
            image_mask = f'{results_dir}/{file}_{base_name}.nii'
            ctv_mask = f'{results_dir}/{base_name}_resegmented.nii'
            print('Using image mask and CTV mask:', image_mask, ctv_mask)

            # Extract the features
            radiomics_dict = extract_radiomics_features(extractor, im_path=image_mask, label_path=ctv_mask)

            # Create a DataFrame for the results
            result_test_df = pd.DataFrame([radiomics_dict])

            # Append additional metadata to the results DataFrame
            result_test_df['patient'] = patient
            result_test_df['ctv_structure'] = base_name
            result_test_df['file'] = file
            result_test_df['multiple_ctvs'] = find_matching_string(structure_dir, search_string='ctv2')
            result_test_df['nodes'] = find_matching_string(structure_dir, search_string='node')
            result_test_df['multiple_nodes'] = find_matching_string(structure_dir, search_string='node2')
            result_test_df['trial'] = dataset_name

            # Concatenate the results and save to CSV
            radiomics_df = pd.concat([radiomics_df, result_test_df])
            radiomics_df.to_csv(radiomics_df_filename, index=False)
