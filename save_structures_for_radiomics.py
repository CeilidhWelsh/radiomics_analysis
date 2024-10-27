"""
This script processes the planning structure sets for each patient, extracting 
the highest dose CTV (Clinical Target Volume) and CTV Node (if applicable), 
and saves these structures as NIfTI (.nii.gz) files. Additionally, it saves 
the original planning structure as a NIfTI file.

Dependencies:
- sys: For command-line argument handling.
- os: For file and directory manipulation.
- pandas: For data manipulation and analysis.
- skrt: Custom module for handling patient data.
- structure_analysis_functions: Custom module containing helper functions.

Usage:
    Run the script from the command line with the following parameters:
    
    python <script_name.py> <dataset_name> <planning_data_dir> <data_dir> <local> <subset> <patient_list> <additional_dataset_name>
"""

import sys
import os
from skrt import Patient
import pandas as pd
from structure_analysis_functions import *  # Import custom functions

# Pull in the variables from the shell script
dataset_name = sys.argv[1]               # Name of the dataset
data_dir = sys.argv[2]                   # Directory containing patient data
results_dir = sys.argv[3]                 # Directory to save results
local = sys.argv[4]                       # Local settings (not used)
subset = sys.argv[5]                      # Subset parameter (not used)
patient_list = sys.argv[6]                # List of patients (not used)
additional_dataset_name = sys.argv[7]     # Additional dataset name

# Load the dataset from the specified folder
ctv_df = pd.read_csv(f'{dataset_name}{additional_dataset_name}_structures.csv')
# Reduce dimensionality of the dataframe
ctv_df2 = ctv_df[['PATIENT_ID', 'Final CTV', 'Final CTV Original Name', 'Final CTV Node', 
                   'Final CTV Node Original Name', 'trial']].copy()

# Iterate through specified columns row by row
for index, row in ctv_df2.iterrows():
    print(f"Index: {index}")

    # Get patient ID and original name 
    patient_id = row['PATIENT_ID']
    print(patient_id)
    original_ctv_name = row['Final CTV Original Name']
    new_ctv_name = row['Final CTV']  # Note: new_name is reassigned, using new_ctv_name for clarity
    original_ctv_node_name = row['Final CTV Node Original Name']
    new_ctv_node_name = row['Final CTV Node']  # Using new_ctv_node_name for clarity
    print('Original CTV name:', original_ctv_name)
    print('Original CTV Node name:', original_ctv_node_name)

    # Set up patient and extract the ROI (Region of Interest) with the original name 
    p = Patient(f'{data_dir}/{patient_id}')
    try: 
        for ss in p.studies[0].ct_structure_sets:
            if len(ss.filtered_copy(to_keep=['*ptv*']).get_roi_names()) != 0:
                planning_ss = ss
    except:
        for ss in p.studies[0].rtstruct_structure_sets:
            if len(ss.filtered_copy(to_keep=['*ptv*']).get_roi_names()) != 0:
                planning_ss = ss

    # Ensure the results and structure directories exist; create them if they don't
    patient_dir = f'{results_dir}/{patient_id}'
    ensure_directory_exists(patient_dir)
    structure_dir = f'{patient_dir}/structures'
    ensure_directory_exists(structure_dir)

    # Check that there is a CTV structure to save
    if str(original_ctv_name) == 'nan':
        print('No CTV structure found.')
    else:
        print('Saving CTV structure...')
        ctv_roi = planning_ss.get_roi(f'{original_ctv_name}')
        if '/' not in original_ctv_name:
            ctv_roi.write(f'{patient_dir}/{original_ctv_name}.nii.gz')
        else:
            continue  # Skip if the original CTV name is invalid

        # Save the planning CTV structure
        if 'planning_ctv.nii.gz' not in os.listdir(structure_dir): 
            ctv_roi.write(f'{structure_dir}/planning_ctv.nii.gz')
        else:
            ctv_roi.write(f'{structure_dir}/planning_ctv2.nii.gz')

    # Check if there is a node structure to save
    if str(original_ctv_node_name) == 'nan':
        print('No CTV Node structure found.')
    else: 
        print('Saving CTV Node structure...')
        if '/' not in original_ctv_node_name:
            ctv_node_roi = planning_ss.get_roi(f'{original_ctv_node_name}')
            ctv_node_roi.write(f'{patient_dir}/{original_ctv_node_name}.nii.gz')
        else:
            continue  # Skip if the original CTV node name is invalid 

        # Save the planning CTV Node structure
        if 'planning_ctv_node.nii.gz' not in os.listdir(structure_dir): 
            ctv_node_roi.write(f'{structure_dir}/planning_node_ctv.nii.gz')
        else:
            ctv_node_roi.write(f'{structure_dir}/planning_node_ctv2.nii.gz')
    
    # Save the planning image
    planning_im = p.studies[0].ct_images[0]
    planning_im.write(f'{patient_dir}/planning_im.nii.gz')
