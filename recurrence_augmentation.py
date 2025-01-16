"""
This script extracts radiomic features from augmented images associated with patients in a radiotherapy dataset. It performs image augmentation on a subset of patients to balance the classes of the recurrence/non-recurrence datasets. It randomly extracts voxels from within the CTV or nodal volume as a method for augmentation. 

Functions:
    - ensure_directory_exists: Checks if a directory exists and creates it if it does not.
    - find_matching_string: Searches for specific strings in file names within a directory.
    - recurrence_augment_images: Generates augmented images based on recurrence and other criteria.
    - extract_radiomics_features: Extracts features from medical images using the provided extractor.

Imports:
    - os: For interacting with the operating system.
    - sys: To access command-line arguments.
    - skrt: A library for handling radiotherapy datasets and processing images.
    - numpy: For numerical operations and array manipulations.
    - pandas: For data manipulation and CSV file generation.
    - ast: For evaluating string representations of Python expressions.
    - copy: For creating shallow or deep copies of objects (not used in the script).
    - matplotlib.pyplot: For data visualization (not used in the script but imported).
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
import copy
import matplotlib.pyplot as plt
from pathlib import Path

# Import scikit-rt modules 
from skrt import Patient, ROI, StructureSet, Image
from skrt.registration import get_default_pfiles, Registration, set_elastix_dir
from skrt.better_viewer import BetterViewer
from skrt.simulation import SyntheticImage

from radiomics import featureextractor  
from radiomic_features_functions import * 

# Pull in the variables from the shell script 
dataset_name = sys.argv[1]               # Name of the dataset
planning_data_dir = sys.argv[2]          # Directory for planning data
data_dir = sys.argv[3]                    # Directory to save processed data
local = sys.argv[4]                       # Local machine flag
subset = sys.argv[5]                      # Subset flag
patient_list = sys.argv[6]                # List of patients to process
additional_dataset_name = sys.argv[7]     # Additional dataset name for output files

# Check if the radiomics CSV file exists already
radiomics_df_filename = f'updating_{dataset_name}{additional_dataset_name}_extracted_radiomic_features_augmented_images.csv'
if os.path.isfile(radiomics_df_filename):
    print('Radiomics DataFrame already exists, loading into the Python session.') 
    radiomics_df = pd.read_csv(radiomics_df_filename)
else:
    radiomics_df = pd.DataFrame()

# Load the patient list 
if subset == "False":
    patient_list = os.listdir(data_dir)
    if '.DS_Store' in patient_list:     
        patient_list = [item for item in patient_list if '.DS_Store' not in item]
else: 
    patient_list = ast.literal_eval(patient_list)

print('> Patient list:', patient_list)

# Process each patient
for patient in patient_list: 
    print('> Patient:', patient)
    output_data_dir = f'{data_dir}/{patient}/'
    ensure_directory_exists(output_data_dir)

    # Set up results directory
    results_dir = f'{output_data_dir}/results'
    structure_dir = f'{output_data_dir}/structures'
    ensure_directory_exists(results_dir)

    # Check for nodes in the patient's structures
    nodes = find_matching_string(structure_dir, search_string='node')

    # Determine planning and structure files based on the presence of nodes
    if nodes: 
        planning_files = ['planning_im_planning_ctv.nii', 'planning_im_planning_node_ctv.nii']
        structure_files = ['planning_ctv_resegmented.nii', 'planning_node_ctv_resegmented.nii']
    else: 
        planning_files = ['planning_im_planning_ctv.nii']
        structure_files = ['planning_ctv_resegmented.nii']

    augmented_files = ['augmented_planning_im', 'augmented_planning_im1', 'augmented_planning_im2',        
                        'augmented_planning_im3', 'augmented_planning_im4']

    for structure_file in structure_files:
        print('> Structure File:', structure_file)
        base_name, _ = os.path.splitext(structure_file)
        base_name = base_name.replace('_resegmented', '')
        if '.' in base_name:
            base_name, _ = os.path.splitext(base_name)

        print('> Base name:', base_name)

        # Call function that employs the three subfunctions 
        recurrence_augment_images(results_dir=results_dir, base_name=base_name, local=local)

        # Instantiate the extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()

        # Extract features from augmented images
        for file in augmented_files: 
            image_mask = f'{results_dir}/{file}_{base_name}.nii'
            ctv_mask = f'{results_dir}/{base_name}_resegmented.nii'
            print('File paths for Image Mask:', image_mask, '\n', 'and CTV Mask', ctv_mask)
        
            # Extract the features 
            radiomics_dict = extract_radiomics_features(extractor, im_path=image_mask, label_path=ctv_mask)
            
            result_test_df = pd.DataFrame([radiomics_dict])
            
            result_test_df['patient'] = patient 
            result_test_df['ctv_structure'] = base_name
            result_test_df['file'] = file
            result_test_df['multiple_ctvs'] = find_matching_string(structure_dir, search_string='ctv2')
            result_test_df['nodes'] = find_matching_string(structure_dir, search_string='node')
            result_test_df['multiple_nodes'] = find_matching_string(structure_dir, search_string='node2')
            result_test_df['trial'] = dataset_name

            # Concatenate results and save to CSV
            radiomics_df = pd.concat([radiomics_df, result_test_df])
            radiomics_df.to_csv(f'updating_{dataset_name}{additional_dataset_name}_extracted_radiomic_features_augmented_images.csv')
