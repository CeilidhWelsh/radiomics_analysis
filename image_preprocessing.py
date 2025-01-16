"""
This script pulls the patient file from the saved location and executes a series of image preprocessing steps to enhance the reproducibility of the workflow. The main steps include resampling the planning image, applying masks, and discretizing the image data into specified bins. The results are saved as NIfTI files, and a summary of voxel removal percentages is recorded.

Functions:
    - ensure_directory_exists: Checks if a directory exists, creates it if not.

Imports:
    os: Standard library for interacting with the operating system.
    sys: Provides access to system-specific parameters and functions.
    skrt: Provides medical image processing tools for radiotherapy dataset standardization.
    numpy: For numerical computations and array manipulations.
    pandas: For data manipulation and CSV file generation.
    matplotlib.pyplot: For visualizing data (not used in this script but imported).
    pathlib.Path: For handling filesystem paths.
    radiomics: Library for extracting features from medical images.
    sklearn: A library for machine learning.
    six: A Python 2 and 3 compatibility library.
    re: For regular expression operations.
    ast: For evaluating string representations of Python expressions.
    structure_analysis_functions: Custom functions for structure processing and analysis 
    (imported from radiomic_features_functions).

Usage:
    python <script_name.py> <dataset_name> <planning_data_dir> <data_dir> <local> <subset> <patient_list> <additional_dataset_name>
    
    Parameters:
        dataset_name: Name of the dataset.
        planning_data_dir: Directory containing planning data.
        data_dir: Directory for saving processed data.
        local: Flag indicating if the script is running on a local machine.
        subset: Flag indicating if a subset of patients should be processed.
        patient_list: List of patients to process.
        additional_dataset_name: Additional dataset name for output files.
"""

# Import standard Python packages
import os
import sys
import skrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import scikit-rt modules
from skrt import Patient, ROI, StructureSet, Image
from skrt.registration import get_default_pfiles, Registration, set_elastix_dir
from skrt.better_viewer import BetterViewer
from skrt.simulation import SyntheticImage

import radiomics
import sklearn
import six
import re
import ast

from radiomic_features_functions import * 

# Pull in the variables from the shell script 
dataset_name = sys.argv[1]               # Name of the dataset
planning_data_dir = sys.argv[2]          # Directory for planning data
data_dir = sys.argv[3]                    # Directory to save processed data
local = sys.argv[4]                       # Local machine flag
subset = sys.argv[5]                      # Subset flag
patient_list = sys.argv[6]                # List of patients to process
additional_dataset_name = sys.argv[7]     # Additional dataset name for output files

# Initialize patient variables
patient_flag = {}
if subset == "False":
    patient_list = os.listdir(planning_data_dir)
    if '.DS_Store' in patient_list:
        patient_list.remove('.DS_Store')
else: 
    patient_list = ast.literal_eval(patient_list)

# Pull the planning image from the patient folder 
for patient in patient_list: 
    print('> Patient:', patient)
    patient_flag[patient] = {}
    output_data_dir = f'{data_dir}/{patient}/'
    ensure_directory_exists(output_data_dir)

    # Set up results directory
    results_dir = f'{output_data_dir}/results'
    ensure_directory_exists(results_dir)
    
    # Load patient object 
    p = Patient(f'{planning_data_dir}/{patient}')
    planning_im = p.studies[0].ct_images[0]

    # 1. Take a 3D array (image) and a binary mask with the same dimensions.
    planning_im.resample(voxel_size=(1, 1, 3))

    # Run through each file in the structures directory 
    structure_dir = f'{output_data_dir}/structures'
    ensure_directory_exists(structure_dir)
    for file in os.listdir(structure_dir):  
        if file == '.DS_Store':
            continue
        else:
            base_name, _ = os.path.splitext(file)
            # If there is a double extension like .nii.gz then split again 
            if '.' in base_name:
                base_name, _ = os.path.splitext(base_name)
            print('> File Base Name:', base_name)
            print('> File Name for Analysis:', file)
            
            # Load the patient's previously saved planning CTV
            ctv = ROI(f'{structure_dir}/{file}')

            # Set to the same image parameters as the planning image 
            ctv.set_image(planning_im)
            
            # Visualize the planning scan and the CTV overlaid 
            # planning_im.view(rois=[ctv])

            # Ensure that the image and mask have the same dimensions and affine matrices
            assert planning_im.get_data().shape == ctv.get_mask().data.shape, "Image and mask must have the same dimensions"
            assert np.array_equal(planning_im.affine, ctv.affine), "Image and mask must have the same affine matrices"

            # 2. Recreate the mask to only include voxels > -150 and < 250.
            masked_planning_im = (planning_im.data > -150) & (planning_im.data < 250) & ctv.mask.data

            # 3. Reapply the new mask to the original image to get the values within the range and calculate mean and std.
            values_within_range = planning_im.data[masked_planning_im]
            mean_value = values_within_range.mean()
            # print(mean_value)
            std_value = values_within_range.std()
            # print(std_value)

            # 4. Update the mask to exclude values outside mean ± 3 * std.
            lower_bound = mean_value - 3 * std_value
            upper_bound = mean_value + 3 * std_value
            final_mask = (planning_im.data >= lower_bound) & (planning_im.data <= upper_bound) & masked_planning_im
                
            # 5. Apply the final mask to the original image.
            final_im_mask = planning_im.data * final_mask.astype(int)
            final_im = Image(final_im_mask, affine=planning_im.affine)
            # final_im.view(zoom=True)

            # 5A. Save the image before discretization
            final_im.write(f'{results_dir}/planning_im_{base_name}.nii')

            # 5B. Save the mask as a structure 
            ctv_resegmented = ROI(final_im_mask, image=planning_im)
            # planning_im.view(rois=[ctv_resegmented], zoom=True)
            ctv_resegmented.write(f'{results_dir}/{base_name}_resegmented.nii')

            # Additional step: flag system
            # Count the number of voxels removed after the filtering step
            count_true_original = ctv.get_mask().sum()
            count_true_resegmented = ctv_resegmented.get_mask().sum()
            # print(count_true_original, count_true_resegmented)
            percentage_removed = ((count_true_original - count_true_resegmented) / count_true_original) * 100
            print('> Percentage of Structure Removed:', percentage_removed)
            patient_flag[patient][file] = percentage_removed

            # 6. Take the values in the final_mask and discretize them in 16, 32, 64, and 128 bins 
            # Get the values after applying the mask
            final_values = planning_im.data[final_mask]
            
            # Define the number of bins 
            num_bins_list = [16, 32, 64, 128]

            for num_bins in num_bins_list:
                # Define the bin edges
                bin_edges = np.linspace(np.min(final_values), np.max(final_values), num_bins)
                # print(bin_edges)

                # For each bin edge create a dictionary with the number and the intensity associated
                bin_intensity_dict = {}
                i = 0 
                for edge in bin_edges:
                    bin_intensity_dict[i] = edge
                    i += 1

                # print(bin_intensity_dict)

                # Digitize the final image
                discretized_im = np.digitize(final_im_mask, bin_edges) - 1  # bins are 0-indexed
                # Convert the array to the corresponding values from the dictionary
                vectorized_lookup = np.vectorize(bin_intensity_dict.get)

                # Apply the vectorized lookup to replace values
                intensity_discretized_im = vectorized_lookup(discretized_im)
                intensity_discretized_im_test = Image(intensity_discretized_im, affine=planning_im.affine)
                # intensity_discretized_im_test.view(zoom=True, rois=[ctv_resegmented])

                # Save each binned image to the folder with the bin in the name 
                intensity_discretized_im_test.write(f'{results_dir}/planning_im_{base_name}_{num_bins}.nii')

    print('> Testing the percentage removed dictionary:', patient_flag) 

# Convert the patient flag dictionary to a DataFrame
patient_flag_df = pd.DataFrame(patient_flag)
patient_flag_df = patient_flag_df.transpose()
patient_flag_df['trial'] = dataset_name
patient_flag_df.reset_index(inplace=True)
patient_flag_df.rename(columns={'index': 'patient'}, inplace=True)

# Save the percentage of voxels removed 
# Update any existing spreadsheets and replace duplicates with the latest extracted values 
if subset == "True":
    # Check if the file already exists
    if os.path.exists(f'{dataset_name}{additional_dataset_name}_percentage_voxels_removed.csv'):
        # Load and save an updated version 
        loaded_voxel_df = pd.read_csv(f'{dataset_name}{additional_dataset_name}_percentage_voxels_removed.csv')

        # Combine the structure_df and updated_structure_csv 
        # Concatenate the DataFrames
        concatenated_df = pd.concat([loaded_voxel_df, patient_flag_df], ignore_index=True)

        # Drop duplicates, keeping only the last occurrence (which will be from df2)
        updated_voxel_df = concatenated_df.drop_duplicates(subset='patient', keep='last')
        # Save updated dataset 
        updated_voxel_df.to_csv(f'{dataset_name}{additional_dataset_name}_percentage_voxels_removed.csv')
        
    else: 
        patient_flag_df.to_csv(f'{dataset_name}{additional_dataset_name}_percentage_voxels_removed.csv')
else: 
    patient_flag_df.to_csv(f'{dataset_name}{additional_dataset_name}_percentage_voxels_removed.csv')
