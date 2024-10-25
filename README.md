# Computational Framework for Predicting Radiotherapy Effect 
## Radiomics Analysis Pipeline for Predicting Locoregional Recurrence in Cancer

This repository contains a radiomics analysis pipeline implemented as a series of Python scripts designed to automate and standardise the extraction and analysis of radiomic features from radiotherapy images. The primary focus is on head and neck cancer, with a particular interest in using radiomic features to predict locoregional recurrence outcomes. 

## Overview

This pipeline processes patient radiotherapy data by:
1. **Loading Patient Data**: Data is imported in a standardised format for consistency and accuracy in analysis. The pipeline loads radiotherapy images and associated structure sets, specifically targeting the primary Clinical Target Volume.
2. **Extracting and Renaming CTVs**: The primary tumour volumes and nodal regions are extracted. The original clinical names are stored for reference, and the structures are renamed to standardised reference names to facilitate consistent processing across datasets.
3. **Saving Planning Scans**: Original planning CT scans are saved for image processing.

## Preprocessing

The pipeline applies a series of preprocessing steps to ensure robustness of the extracted radiomic features:
1. **Image Normalisation and Scaling**: Planning CT images are normalised and z-scaled. Discretisation is applied to enhance feature robustness.
2. **CTV Preprocessing**: CTV structures undergo further preprocessing to:
   - Remove regions potentially affected by image corruption, such as beam hardening artefacts, e.g. due to dental fillings.
   - Exclude anatomical regions that are not soft tissue, refining the CTV structure to focus on clinically relevant areas.
3. **Output**: The preprocessed CTV volume and the planning CT are prepared for feature extraction.

## Radiomic Feature Extraction

PyRadiomics is implemented to load the preprocessed planning CT images and CTV masks and extract radiomic features. This includes a comprehensive range of quantitative features that may provide insights into the tumour’s spatial, textural, and intensity-based characteristics.

## Machine Learning Analysis

The extracted radiomic features are then analysed using several machine learning models to investigate their predictive power for locoregional recurrence. This phase includes:
1. **Predictive Model Training**: Multiple machine learning models, including Logistic Regression, Support Vector Machines (SVM), and Decision Trees, are trained to determine if specific radiomic features are associated with patient outcomes.
2. **Few-Shot Learning and Deep Learning**: Development is ongoing to integrate few-shot learning approaches and pre-trained deep learning models for cases with limited sample sizes, aimed at enhancing prediction accuracy. Features deemed predictive of outcomes are cross-correlated with those extracted by PyRadiomics to ensure consistency and enhance interpretability in radiomic signature development.
3. **Feature Validation**: These features can then be checked in an additional dataset where the scripts for structure extraction, image preprocessing and radiomic feature extraction can be employed and the final data used in the pre-trained model to determine models metrics on a new datase

## Dependencies

- Python 3.10.13
- [pyRadiomics](https://pyradiomics.readthedocs.io/en/latest/)
- Additional Python libraries: numpy, pandas, scikit-learn, etc.

## Usage

This pipeline can be run as a sequence of scripts, each loaded in a shell script, with each stage designed to accept outputs from the previous stage. For detailed instructions on each stage and customisable parameters, please refer to the respective script documentation within each file.

## Future Development

Plans for future development include integrating additional machine learning models, refining preprocessing techniques, and expanding compatibility for other cancer types and imaging modalities.

---

This pipeline is designed to streamline and standardise radiomics analysis, from data import through feature extraction and machine learning analysis, in a reproducible and scalable framework for clinical research applications.
