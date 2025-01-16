import os
import radiomics
import numpy as np
import re
import pandas as pd 
import radiomics
import sklearn
import six
import re
from radiomics import featureextractor   

def normalize_ct_image(ct_image):
    ct_image_array = np.array(ct_image.get_data())
    # Assuming ct_image is a numpy array with values between -2048 and 2048
    min_val = ct_image_array.flatten().min()
    max_val = ct_image_array.flatten().max()
    print('normalised image min and max values:', min_val, max_val)
    
    # Shift the range from [-2048, 2048] to [0, 4096]
    shifted_image = ct_image_array - min_val  # This will shift the range to [0, 4096]
    
    # Normalize the shifted values to the range [0, 1]
    normalized_image = shifted_image / (max_val - min_val)
    
    return normalized_image

def zero_center_image(normalized_image):
    # Assuming normalized_image is a numpy array with values between 0 and 1
    mean_val = np.mean(normalized_image)
    zero_centered_image = normalized_image - mean_val

    return zero_centered_image
    
def standardise_image(centered_image):
    # Calculate the standard deviation of the zero-centered image
    std_val = np.std(centered_image)
    
    # Standardize the image
    standardised_image = centered_image / std_val
    
    return standardised_image


def ensure_directory_exists(directory_path):
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def get_radiomic_features(save_to_dir, extractor, patient):
    result = extractor.execute(f'{save_to_dir}/standardised_planning_im.nii', f'{save_to_dir}/ptv.nii')
    feature_df = pd.DataFrame([result])
    feature_df['patient'] = patient

    return feature_df


# Function to replace 'PTV T' with 'PTV' and clean the list while maintaining a mapping to the original names
def preprocess_ptv_list(ptv_list):
    cleaned_to_original = {}
    processed_list = []
    for ptv in ptv_list:

        original_ptv = ptv
        # Replace 'PTV T' with 'PTV'
        ptv = ptv.replace('PTV T', 'PTV')

        # Remove dashes, punctuation, and spaces and capitalise 
        cleaned_ptv = re.sub(r'[\s\-]+', '', ptv)
        cleaned_ptv = cleaned_ptv.upper()
        processed_list.append(cleaned_ptv)
    
        cleaned_to_original[cleaned_ptv] = original_ptv

    return processed_list, cleaned_to_original


# Function to replace and clean the list while maintaining a mapping to the original names
def preprocess_structure_list(structure_list, substring):
    cleaned_to_original = {}
    processed_list = []
    for structure in structure_list:

        original_structure = structure
        # Replace 'PTV T' with 'PTV'
        #ptv = ptv.replace('PTV T', 'PTV')

        # Remove dashes, punctuation, and spaces and capitalise 
        cleaned_structure = re.sub(r'[\s\-]+', '', substring)
        cleaned_structure = cleaned_structure.upper()
        processed_list.append(cleaned_structure)
    
        cleaned_to_original[cleaned_structure] = original_structure

    return processed_list, cleaned_to_original


# Function to extract the highest value PTV
def extract_ptv_values(ptv_list, prefix):
    max_value = -1
    max_ptv = None
    pattern = re.compile(f"{prefix}(\d+)")
    
    for ptv in ptv_list:
        match = pattern.search(ptv)
        if match:
            value = int(match.group(1))
            if value > max_value:
                max_value = value
                max_ptv = ptv

    return max_ptv


def ensure_directory_exists(directory_path):
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")



def obtain_radiomic_masks(planning_search_string, results_dir, structure_file_name, bin_number, base_name):
    #print(bin_number)

    planning_im_matching_files = find_files_with_string(results_dir, planning_search_string)
    #print(planning_im_matching_files)
    if bin_number == 0: 
        planning_im_mask = f'{results_dir}/planning_im_{structure_file_name}.nii'  
    else: 
        for planning_im in planning_im_matching_files:
            #print(planning_im)
            if str(bin_number) in planning_im: 
                planning_im_mask = f'{results_dir}/planning_im_{structure_file_name}_{bin_number}.nii'
    
    #print(planning_im_mask)

    ctv_mask = f'{results_dir}/{base_name}_resegmented.nii'
    #print(ctv_mask)

    return ctv_mask, planning_im_mask


# function for extracting radiomics features 
def extract_radiomics_features(extractor, im_path, label_path):

    radiomics_dict = {}
    print('Extraction parameters:\n\t', extractor.settings)
    print('Enabled filters:\n\t', extractor.enabledImagetypes)
    print('Enabled features:\n\t', extractor.enabledFeatures)

    # run the radiomic feature extraction 
    result = extractor.execute(im_path, label_path)

    # extract the feature 
    #print('Result type:', type(result))  # result is returned in a Python ordered dictionary)
    #print('')
    #print('Calculated features')
    for key, value in six.iteritems(result):
        #print('\t', key, ':', value)
        radiomics_dict[key] = value 

    return radiomics_dict


def find_files_with_string(directory, search_string):
    # Ensure the directory exists
    if not os.path.isdir(directory):
        raise ValueError(f"The directory {directory} does not exist.")
    
    # Get all entries in the directory
    all_entries = os.listdir(directory)
    
    # Filter out files containing the search string
    matching_files = [entry for entry in all_entries if os.path.isfile(os.path.join(directory, entry)) and search_string in entry]
     
    # return variable for search string being true or false 
    string_present = any(search_string in file_name for file_name in all_entries)

    return matching_files


def find_matching_string(directory, search_string):
    # Ensure the directory exists
    if not os.path.isdir(directory):
        raise ValueError(f"The directory {directory} does not exist.")
    
    # Get all entries in the directory
    all_entries = os.listdir(directory)
     
    # return variable for search string being true or false 
    string_present = any(search_string in file_name for file_name in all_entries)

    return string_present


def obtain_flagged_patients(dataset_name, column_name):
    data_dir = f'./{dataset_name}_percentage_voxels_removed.csv'
    flag_df = pd.read_csv(data_dir)

    if dataset_name == 'voxtox':
        flag_df.drop(1, inplace=True)


    flag_df[column_name] = flag_df[column_name].astype(float)
    mean_voxels_removed = flag_df[column_name].mean()
    std_voxels_removed = flag_df[column_name].std()

    # calculate an acceptable range of voxels removed 
    range = mean_voxels_removed + 3*std_voxels_removed
    print('Maximum allowable voxels:', range)
    # find any value > mean + 3*std 
    flagged_patients_df = flag_df[flag_df[column_name] > range]
    
    return flag_df, flagged_patients_df
