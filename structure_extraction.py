"""
Script to obtain the names of the planning structures obtained and saved for each patient from the structure extraction script. This script then saves these structures as NIfTI files and the original planning structure set as a NIfTI file in the specified results directory.

Called Functions:
    - ensure_directory_exists: Ensures that specified directories exist.

Imports:
    sys: Provides access to system-specific parameters and functions.
    os: Standard library for interacting with the operating system.
    skrt: Provides medical image processing tools for radiotherapy dataset standardization.
    pandas: For data manipulation and CSV file generation.
    structure_analysis_functions: Custom functions for structure processing and analysis 
    (script: structure_analysis_functions.py).

Usage:
    Run the script from the command line with the following parameters:
    
    python <script_name.py> <dataset_name> <planning_data_dir> <data_dir> <local> <subset> <patient_list> <additional_dataset_name>
"""

# Import packages
import sys
import os
from skrt import Patient
import pandas as pd
from structure_analysis_functions import *

# Pull in the variables from the shell script
dataset_name = sys.argv[1]               # Name of the dataset
data_dir = sys.argv[2]                   # Directory containing patient data
results_dir = sys.argv[3]                 # Directory to save results
local = sys.argv[4]                       # Local machine flag
subset = sys.argv[5]                      # Subset flag
patient_list = sys.argv[6]                # List of patients
additional_dataset_name = sys.argv[7]     # Additional dataset name

# Load the dataset from the same folder
ctv_df = pd.read_csv(f'{dataset_name}{additional_dataset_name}_structures.csv')
# Reduce dimensionality of the dataframe
ctv_df2 = ctv_df[['PATIENT_ID', 'Final CTV', 'Final CTV Original Name', 
                   'Final CTV Node', 'Final CTV Node Original Name', 'trial']].copy()

# Iterate through specified columns row by row
for index, row in ctv_df2.iterrows():
    print(f"Index: {index}")

    # Get patient ID and original name
    patient_id = row['PATIENT_ID']
    print(patient_id)
    original_ctv_name = row['Final CTV Original Name']
    new_name = row['Final CTV']
    original_ctv_node_name = row['Final CTV Node Original Name']
    new_name = row['Final CTV Node']
    print('Original CTV name:', original_ctv_name)
    print('Original CTV Node name:', original_ctv_node_name)

    # Set up patient and extract the ROI with the original name
    p = Patient(f'{data_dir}/{patient_id}')
    try: 
        for ss in p.studies[0].ct_structure_sets:
            if len(ss.filtered_copy(to_keep=['*ptv*']).get_roi_names()) != 0:
                planning_ss = ss
    except:
        for ss in p.studies[0].rtstruct_structure_sets:
            if len(ss.filtered_copy(to_keep=['*ptv*']).get_roi_names()) != 0:
                planning_ss = ss

    # Ensure the results and structure directories exist; if they don't, create them
    patient_dir = f'{results_dir}/{patient_id}'
    ensure_directory_exists(patient_dir)
    structure_dir = f'{patient_dir}/structures'
    ensure_directory_exists(structure_dir)

    # Check that there is a CTV structure to save
    if str(original_ctv_name) == 'nan':
        print('No CTV')
    else:
        print('CTV test')
        ctv_roi = planning_ss.get_roi(f'{original_ctv_name}')
        if '/' not in original_ctv_name:
            ctv_roi.write(f'{patient_dir}/{original_ctv_name}.nii.gz')
        else:
            continue

        # Save planning CTV structure
        if 'planning_ctv.nii.gz' not in os.listdir(structure_dir): 
            ctv_roi.write(f'{structure_dir}/planning_ctv.nii.gz')
        else:
            ctv_roi.write(f'{structure_dir}/planning_ctv2.nii.gz')

    # Check there is node structure to save
    if str(original_ctv_node_name) == 'nan':
        print('No node CTV')
    else: 
        print('Node test')
        if '/' not in original_ctv_node_name:
            ctv_node_roi = planning_ss.get_roi(f'{original_ctv_node_name}')
            ctv_node_roi.write(f'{patient_dir}/{original_ctv_node_name}.nii.gz')
        else:
            continue 

        # Save planning node CTV structure
        if 'planning_ctv_node.nii' not in os.listdir(structure_dir): 
            ctv_node_roi.write(f'{structure_dir}/planning_node_ctv.nii.gz')
        else:
            ctv_node_roi.write(f'{structure_dir}/planning_node_ctv2.nii.gz')
    
    # Save the planning image
    planning_im = p.studies[0].ct_images[0]
    planning_im.write(f'{patient_dir}/planning_im.nii.gz')
