#!/bin/bash

# This script MUST be in the same local as the python files we wish to execute 
# Please activate the relevant environment that has the latest version of python 

# Define the variables: these depend on whether we are running locally or on the HEP cluster
# Local Machine: please input the relevant variables 
# 1: dataset name; 
# 2: path to the location of the patient data files; 
# 3: path to where you want to store the results; 
# 4: True or False depending on whether this is running to a local machine or on a cluster 
# 5: True or False depending on whether this a subset of patients being re-run 
# 6: A list of the subset of patient IDs to re-run the code on 
# 7: Additional string label for if there is a subdirectory of data - leave blank if unneccessary 

var1="<trial_name>"
var2="<path to dataset>"
var3="<path to results directory>"
var4="<local machine or cluster: this should stay set to False for external analysis>"
var5="<patient list, True or False>"
var6=[<patient list>]
var7="<additional string for subdirectories of data>"


# Function to run a script and report time and success
run_script() {
    script_name=$1

    echo "Running $script_name with dataset name:$var1, location of patient data: $var2, results directory: $var3, and local data flag: $var4, subset flag: $var5, and subset patient list if flag set to True $var6, with additional string set to $var7"
    start_time=$(date +%s)

    if python "$script_name" "$var1" "$var2" "$var3" "$var4" "$var5" "$var6" "$var7"; then
        end_time=$(date +%s)
        elapsed_time=$((end_time - start_time))
        echo "$script_name completed successfully in $elapsed_time seconds."
    else
        end_time=$(date +%s)
        elapsed_time=$((end_time - start_time))
        echo "$script_name failed after $elapsed_time seconds."
        exit 1  # Exit if the script fails
    fi
}

# Run the scripts sequentially
run_script structure_extraction.py
run_script save_structures_for_radiomics.py
run_script image_preprocessing.py
run_script radiomics_feature_extraction.py





