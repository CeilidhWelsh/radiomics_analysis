"""
Module for image processing and radiomics feature extraction.

Functions:
    - normalize_ct_image: Normalises a CT image to a range of [0, 1].
    - zero_center_image: Centers a normalised image around zero.
    - standardise_image: Standardises a zero-centered image.
    - ensure_directory_exists: Ensures that a directory exists, creating it if necessary.
    - get_radiomic_features: Extracts radiomic features for a given patient and saves them to a DataFrame.
    - preprocess_ptv_list: Cleans and processes a list of PTV names, mapping them to original names.
    - preprocess_structure_list: Cleans and processes a list of structures while maintaining a mapping to original names.
    - extract_ptv_values: Extracts the PTV with the highest numerical value from a list based on a given prefix.
    - obtain_radiomic_masks: Obtains file paths for radiomic masks based on given parameters.
    - extract_radiomics_features: Extracts radiomics features from images using a feature extractor.
    - find_files_with_string: Finds files in a directory that contain a specific substring in their names.
    - find_matching_string: Checks if a specific substring exists in any file names within a directory.
    - obtain_flagged_patients: Identifies patients whose voxel removal exceeds a specified threshold.
    - scramble_voxels: Scrambles the voxels within a given ROI of a CT image.
    - randomise_voxels: Randomizes voxel intensity values within a given ROI based on a normal distribution.
    - obtain_radiomic_augmented_im: Introduces NaN values to a percentage of voxels in a given ROI.
    - augment_images: Performs image augmentation techniques on CT images and saves the results.
    - recurrence_augment_images: Augments images specifically for recurrence patients and saves them.
"""

import os
import numpy as np
import re
import pandas as pd 
import six


def normalize_ct_image(ct_image):
    """
    Normalizes a CT image to a range of [0, 1].

    Parameters:
        ct_image: A CT image object that has a get_data() method returning pixel intensity values.

    Returns:
        numpy.ndarray: A normalized image array with values between 0 and 1.
    """
    ct_image_array = np.array(ct_image.get_data())
    min_val = ct_image_array.flatten().min()
    max_val = ct_image_array.flatten().max()
    print('normalized image min and max values:', min_val, max_val)
    
    shifted_image = ct_image_array - min_val  # Shift to [0, 4096]
    normalized_image = shifted_image / (max_val - min_val)  # Normalize to [0, 1]
    
    return normalized_image

def zero_center_image(normalized_image):
    """
    Centers a normalized image around zero.

    Parameters:
        normalized_image (numpy.ndarray): A normalized image array with values between 0 and 1.

    Returns:
        numpy.ndarray: A zero-centered image array.
    """
    mean_val = np.mean(normalized_image)
    zero_centered_image = normalized_image - mean_val

    return zero_centered_image
    
def standardise_image(centered_image):
    """
    Standardizes a zero-centered image by dividing by its standard deviation.

    Parameters:
        centered_image (numpy.ndarray): A zero-centered image array.

    Returns:
        numpy.ndarray: A standardized image array.
    """
    std_val = np.std(centered_image)
    standardised_image = centered_image / std_val
    
    return standardised_image

def ensure_directory_exists(directory_path):
    """
    Ensures that a directory exists, creating it if it does not.

    Parameters:
        directory_path (str): The path to the directory to check/create.

    Returns:
        None
    """
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def get_radiomic_features(save_to_dir, extractor, patient):
    """
    Extracts radiomic features from a CT image and a label mask for a specific patient.

    Parameters:
        save_to_dir (str): Directory where the images are saved.
        extractor: A feature extractor object from the radiomics library.
        patient: The patient identifier.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted features along with the patient identifier.
    """
    result = extractor.execute(f'{save_to_dir}/standardised_planning_im.nii', f'{save_to_dir}/ptv.nii')
    feature_df = pd.DataFrame([result])
    feature_df['patient'] = patient

    return feature_df

def preprocess_ptv_list(ptv_list):
    """
    Cleans and processes a list of PTV names, replacing 'PTV T' with 'PTV' and maintaining a mapping to original names.

    Parameters:
        ptv_list (list of str): A list of PTV names.

    Returns:
        tuple: A tuple containing:
            - list of str: The cleaned list of PTV names.
            - dict: A mapping of cleaned names to original names.
    """
    cleaned_to_original = {}
    processed_list = []
    for ptv in ptv_list:
        original_ptv = ptv
        ptv = ptv.replace('PTV T', 'PTV')
        cleaned_ptv = re.sub(r'[\s\-]+', '', ptv).upper()
        processed_list.append(cleaned_ptv)
        cleaned_to_original[cleaned_ptv] = original_ptv

    return processed_list, cleaned_to_original

def preprocess_structure_list(structure_list, substring):
    """
    Cleans and processes a list of structures while maintaining a mapping to original names.

    Parameters:
        structure_list (list of str): A list of structure names.
        substring (str): A substring to clean.

    Returns:
        tuple: A tuple containing:
            - list of str: The cleaned list of structure names.
            - dict: A mapping of cleaned names to original names.
    """
    cleaned_to_original = {}
    processed_list = []
    for structure in structure_list:
        original_structure = structure
        cleaned_structure = re.sub(r'[\s\-]+', '', substring).upper()
        processed_list.append(cleaned_structure)
        cleaned_to_original[cleaned_structure] = original_structure

    return processed_list, cleaned_to_original

def extract_ptv_values(ptv_list, prefix):
    """
    Extracts the PTV with the highest numerical value from a list based on a given prefix.

    Parameters:
        ptv_list (list of str): A list of PTV names.
        prefix (str): The prefix to filter PTV names.

    Returns:
        str: The PTV with the highest numerical value, or None if no match is found.
    """
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

def obtain_radiomic_masks(planning_search_string, results_dir, structure_file_name, bin_number, base_name):
    """
    Obtains file paths for radiomic masks based on given parameters.

    Parameters:
        planning_search_string (str): The string to search for planning images.
        results_dir (str): The directory where the results are stored.
        structure_file_name (str): The name of the structure file.
        bin_number (int): The bin number to filter the planning images.
        base_name (str): The base name for the CTV mask.

    Returns:
        tuple: A tuple containing:
            - str: The CTV mask file path.
            - str: The planning image mask file path.
    """
    planning_im_matching_files = find_files_with_string(results_dir, planning_search_string)
    if bin_number == 0: 
        planning_im_mask = f'{results_dir}/planning_im_{structure_file_name}.nii'  
    else: 
        for planning_im in planning_im_matching_files:
            if str(bin_number) in planning_im: 
                planning_im_mask = f'{results_dir}/planning_im_{structure_file_name}_{bin_number}.nii'
    
    ctv_mask = f'{results_dir}/{base_name}_resegmented.nii'
    return ctv_mask, planning_im_mask

def extract_radiomics_features(extractor, im_path, label_path):
    """
    Extracts radiomics features from images using a feature extractor.

    Parameters:
        extractor: A feature extractor object from the radiomics library.
        im_path (str): Path to the image file.
        label_path (str): Path to the label file.

    Returns:
        dict: A dictionary containing the extracted features.
    """
    radiomics_dict = {}
    result = extractor.execute(im_path, label_path)

    for key, value in six.iteritems(result):
        radiomics_dict[key] = value 

    return radiomics_dict

def find_files_with_string(directory, search_string):
    """
    Finds files in a directory that contain a specific substring in their names.

    Parameters:
        directory (str): The directory to search.
        search_string (str): The substring to search for in file names.

    Returns:
        list: A list of file paths matching the search criteria.
    """
    matching_files = []
    
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if search_string in filename:
                matching_files.append(os.path.join(dirpath, filename))

    return matching_files

def find_matching_string(directory, search_string):
    """
    Checks if a specific substring exists in any file names within a directory.

    Parameters:
        directory (str): The directory to search.
        search_string (str): The substring to check for in file names.

    Returns:
        bool: True if any file name contains the substring, False otherwise.
    """
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if search_string in filename:
                return True
    return False

def obtain_flagged_patients(patients, threshold):
    """
    Identifies patients whose voxel removal exceeds a specified threshold.

    Parameters:
        patients (list of str): A list of patient identifiers.
        threshold (float): The voxel removal threshold.

    Returns:
        list: A list of flagged patient identifiers.
    """
    flagged_patients = []
    for patient in patients:
        if patient.voxel_removal > threshold:
            flagged_patients.append(patient)

    return flagged_patients

def scramble_voxels(ct_image, roi):
    """
    Scrambles the voxels within a given ROI of a CT image.

    Parameters:
        ct_image: A CT image object.
        roi: The region of interest defining the area to scramble.

    Returns:
        numpy.ndarray: The image array with scrambled voxels within the ROI.
    """
    image_data = np.array(ct_image.get_data())
    mask = np.array(roi.get_data())
    voxel_indices = np.where(mask > 0)

    scrambled_voxels = np.random.permutation(image_data[voxel_indices])
    scrambled_image = image_data.copy()
    scrambled_image[voxel_indices] = scrambled_voxels

    return scrambled_image

def randomise_voxels(ct_image, roi, mean, std):
    """
    Randomizes voxel intensity values within a given ROI based on a normal distribution.

    Parameters:
        ct_image: A CT image object.
        roi: The region of interest defining the area to randomize.
        mean (float): The mean value for the normal distribution.
        std (float): The standard deviation for the normal distribution.

    Returns:
        numpy.ndarray: The image array with randomized voxels within the ROI.
    """
    image_data = np.array(ct_image.get_data())
    mask = np.array(roi.get_data())
    voxel_indices = np.where(mask > 0)

    randomized_values = np.random.normal(mean, std, size=len(voxel_indices[0]))
    randomized_image = image_data.copy()
    randomized_image[voxel_indices] = randomized_values

    return randomized_image

def obtain_radiomic_augmented_im(ct_image, roi, fraction_nan):
    """
    Introduces NaN values to a percentage of voxels in a given ROI.

    Parameters:
        ct_image: A CT image object.
        roi: The region of interest defining the area to augment.
        fraction_nan (float): The fraction of voxels to replace with NaN.

    Returns:
        numpy.ndarray: The augmented image array with NaN values introduced.
    """
    image_data = np.array(ct_image.get_data())
    mask = np.array(roi.get_data())
    voxel_indices = np.where(mask > 0)
    
    num_voxels = len(voxel_indices[0])
    num_nans = int(num_voxels * fraction_nan)
    random_indices = np.random.choice(num_voxels, num_nans, replace=False)

    augmented_image = image_data.copy()
    augmented_image[voxel_indices[0][random_indices], voxel_indices[1][random_indices], voxel_indices[2][random_indices]] = np.nan

    return augmented_image

def augment_images(directory, image_name, output_directory, augmentations):
    """
    Performs image augmentation techniques on CT images and saves the results.

    Parameters:
        directory (str): The directory containing the original images.
        image_name (str): The name of the image to augment.
        output_directory (str): The directory where the augmented images will be saved.
        augmentations (dict): A dictionary defining the augmentation techniques to apply.

    Returns:
        None
    """
    ensure_directory_exists(output_directory)
    
    image_path = os.path.join(directory, image_name)
    ct_image = load_ct_image(image_path)  # Assuming a function to load the CT image

    for aug_name, aug_func in augmentations.items():
        augmented_image = aug_func(ct_image)
        augmented_image_name = f"{image_name.split('.')[0]}_{aug_name}.nii"
        save_augmented_image(augmented_image, os.path.join(output_directory, augmented_image_name))  # Assuming a function to save images

def recurrence_augment_images(directory, image_name, output_directory, augmentations):
    """
    Augments images specifically for recurrence patients and saves them.

    Parameters:
        directory (str): The directory containing the original images.
        image_name (str): The name of the image to augment.
        output_directory (str): The directory where the augmented images will be saved.
        augmentations (dict): A dictionary defining the augmentation techniques to apply.

    Returns:
        None
    """
    ensure_directory_exists(output_directory)

    image_path = os.path.join(directory, image_name)
    ct_image = load_ct_image(image_path)  # Assuming a function to load the CT image

    for aug_name, aug_func in augmentations.items():
        if "recurrence" in aug_name:
            augmented_image = aug_func(ct_image)
            augmented_image_name = f"{image_name.split('.')[0]}_{aug_name}.nii"
            save_augmented_image(augmented_image, os.path.join(output_directory, augmented_image_name))  # Assuming a function to save images
