import os 
import re 
import string

def preprocess_structure_list(string_list, cleaned_to_original, patient):
    """
    Cleans and standardises a list of structure names by removing spaces, punctuation, and converting
    to uppercase, while maintaining a mapping to the original names for future reference.

    Parameters
    ----------
    string_list : list of str
        List of structure names to process.
    cleaned_to_original : dict
        Dictionary to store mappings of cleaned names to original names.
    patient : str
        Unique identifier for the patient, used as a key in the dictionary.

    Returns
    -------
    tuple
        Processed list of cleaned structure names and the updated `cleaned_to_original` dictionary.

    Example
    -------
    >>> preprocess_structure_list(["CTV 54", "GTV_32"], {}, "PatientA")
    (['CTV54', 'GTV32'], {'PatientA': {'CTV54': 'CTV 54', 'GTV32': 'GTV_32'}})
    """
    cleaned_to_original[patient] = {}
    processed_list = []
    for s in string_list:
        original_structure = s
        s = s.upper().replace(" ", "").translate(str.maketrans('', '', string.punctuation))
        processed_list.append(s)
        cleaned_to_original[patient][s] = original_structure
    return processed_list, cleaned_to_original


def sort_single_list(lst):
    """
    Sorts a list of strings based on descending dose value and left/right designation if present.

    Parameters
    ----------
    lst : list of str
        List of structure names to sort, which may include dose values and labels (e.g., "L" or "R").

    Returns
    -------
    list of str
        Sorted list of structure names in descending dose order.

    Example
    -------
    >>> sort_single_list(["CTV50L", "CTV54R", "CTV46"])
    ['CTV54R', 'CTV50L', 'CTV46']
    """
    def sort_key(name):
        match = re.match(r'(\D+)(\d+)(.*)', name)
        if match:
            prefix, number, suffix = match.groups()
            return (int(number), suffix, prefix)
        return (0, '', name)

    return sorted(lst, key=lambda x: sort_key(x), reverse=True)


def roi_name_check(roi_list=[], mistaken_strings=[]):
    """
    Filters out regions of interest (ROI) based on potentially mistaken naming schemes to avoid errors.

    Parameters
    ----------
    roi_list : list of str
        List of region names to check.
    mistaken_strings : list of str
        List of substrings that should not appear in region names.

    Returns
    -------
    list of str
        Filtered ROI list without mistakenly labeled regions.

    Example
    -------
    >>> roi_name_check(["NCTV54", "LARYNX_N"], ["LARYNX"])
    ["NCTV54"]
    """
    roi_list = [roi.upper() for roi in roi_list]
    mistaken_strings = [string.upper() for string in mistaken_strings]

    for roi in roi_list[:]:  # Use a copy of roi_list to avoid modifying while iterating
        for mistaken_string in mistaken_strings:
            if mistaken_string in roi:
                roi_list.remove(roi)
    
    return roi_list


def get_tumour_structure_and_nodes(structure_list, structure_dict, patient, structure_name=''):
    """
    Filters a list of structures to identify tumor-related structures, isolating primary and nodal structures, 
    and excluding structures based on specific strings. Adds processed lists to a structure dictionary.

    Parameters
    ----------
    structure_list : list of str
        List of all structure names to process.
    structure_dict : dict
        Dictionary to store patient-specific structure information.
    patient : str
        Patient identifier for updating the structure dictionary.
    structure_name : str, optional
        Type of structure to identify (e.g., "CTV", "PTV").

    Returns
    -------
    tuple
        List of primary tumor structures, nodal structures, updated structure dictionary, and backup list for structures.

    Example
    -------
    >>> get_tumour_structure_and_nodes(["CTV54", "PTV54"], {}, "PatientA", "CTV")
    (['CTV54'], [], {'PatientA': {'CTV Structures': ['CTV54']}}, ['CTV54'])
    """
    tv_list = []
    tv_node_list = []
    just_ctv_list = []
    mistaken_strings = ['gland', 'larynx', 'oropharynx', 'new', 'final']

    for roi in structure_list:
        if structure_name in roi:
            tv_list.append(roi)

    tv_list = [s for s in tv_list if 'DVH' not in s and 'AND' not in s and 'RING' not in s]
    back_up_list = tv_list

    if structure_name in ['CTV', 'PTV']:
        tv_list = [x for x in tv_list if any(c.isdigit() for c in x)]
    structure_dict[patient][f'{structure_name} Structures'] = tv_list

    for ctv_roi in tv_list:
        if 'N' in ctv_roi:
            tv_node_list.append(ctv_roi)
            for string in mistaken_strings:
                if string.upper() in ctv_roi:
                    just_ctv_list.append(ctv_roi)
        else:
            if structure_name == 'CTV':
                just_ctv_list.append(ctv_roi)

    if structure_name == 'CTV':
        structure_dict[patient][f'Just {structure_name} Structures'] = just_ctv_list

    tv_node_list = roi_name_check(tv_node_list, mistaken_strings)
    structure_dict[patient][f'{structure_name} Node Structures'] = tv_node_list

    return tv_list, tv_node_list, structure_dict, back_up_list


def prioritise_ctv(back_up_strings):
    """
    Prioritizes "CTV" as the first item in a list if present.

    Parameters
    ----------
    back_up_strings : list of str
        List of structure names to prioritize.

    Returns
    -------
    list of str
        List with "CTV" prioritized as the first element if it was in the original list.

    Example
    -------
    >>> prioritise_ctv(["CTV", "CTV54", "PTV"])
    ['CTV', 'CTV54', 'PTV']
    """
    if 'CTV' in back_up_strings:
        back_up_strings.remove('CTV')
        back_up_strings.insert(0, 'CTV')
    return back_up_strings


def ensure_directory_exists(directory_path):
    """
    Ensures that a specified directory exists; creates it if it does not.

    Parameters
    ----------
    directory_path : str
        Path of the directory to check or create.

    Returns
    -------
    None

    Example
    -------
    >>> ensure_directory_exists('/path/to/directory')
    Directory '/path/to/directory' created.
    """
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def check_entries(lst):
    """
    Checks if a list contains two CTV structures with matching dose but differing by left/right or A/B labeling.

    Parameters
    ----------
    lst : list of str
        List of structure names to check.

    Returns
    -------
    bool
        True if two matching CTV structures are found, False otherwise.

    Example
    -------
    >>> check_entries(["CTV54L", "CTV54R"])
    True
    """
    def extract_number_and_suffix(name):
        match = re.match(r'(\D+)(\d+)(.*)', name)
        if match:
            prefix, number, suffix = match.groups()
            return int(number), suffix
        return None, None

    if len(lst) < 2:
        return False

    num1, suffix1 = extract_number_and_suffix(lst[0])
    num2, suffix2 = extract_number_and_suffix(lst[1])

    if not num1 or not num2:
        return False

    if ('L' in suffix1 or 'R' in suffix1) and num1 == num2:
        if 'L' in suffix2 or 'R' in suffix2:
            return True

    if ('B' in suffix1) and num1 == num2:
        return True

    return False
