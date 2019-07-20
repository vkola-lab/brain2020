# nm2019

## Title
Neurologist-level Alzheimer's disease classification using an explainable deep learning framework

## Background
This repo contains the pytorch implementation of a novel explainable diagnositic deep learning framework for Alzheimer's diease operating on brain magnetic resonance images. This framework comprises a fully convolutinal network (FCN) and a multilayer perceptron (MLP) model. The FCN generates a novel 3D visiaization of the brain disease affected regions which we refered as disease probabiliy map (DPM) in our paper. Features taken from the DPMs were then sent into the MLP model to achieve an overall classification of disease status. 

## Usage
### requirement
pytorch 0.4.0 \
numpy   1.16.2 \
pandas  0.24.2 

### data preparation 
In this work, we trained, validated and tested the framework on the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. To investigate the generalizability of the framework, we externally tested the framework on the National Alzheimer's Coordinating Center (NACC), the Australian Imaging Biomarkers and Lifestyle Study of Ageing (AIBL) and Framingham Heart Study (FHS) datasets. 

To download the data, please contact those affiliations directly. We only provided several data samples in the repo for the purpose of illustration. The framework should work on your own MRI data if proper data preprocessing steps were done.



