## Neurologist-level Alzheimer's disease classification using an explainable deep learning framework

## Background
This repo contains the pytorch implementation of a novel explainable diagnositic deep learning framework for Alzheimer's diease operating on brain magnetic resonance images. This framework comprises a fully convolutinal network (FCN) and a multilayer perceptron (MLP) model. The FCN generates a novel 3D visiaization of the brain disease affected regions which we refered as disease probabiliy map (DPM) in our paper. Features taken from the DPMs were then sent into the MLP model to achieve an overall classification of disease status. 

## Usage
### requirement
pytorch 0.4.0 \
numpy   1.16.2 \
pandas  0.24.2 

### data preparation 
In this work, we trained, validated and tested the framework on the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. To investigate the generalizability of the framework, we externally tested the framework on the National Alzheimer's Coordinating Center (NACC), the Australian Imaging Biomarkers and Lifestyle Study of Ageing (AIBL) and Framingham Heart Study (FHS) datasets.

To download the data, please contact those affiliations directly. We only provided several data samples in the repo for the purpose of illustration. We organized the 3 datasets into 3 folders:

data/ADNI/ \
data/NACC/ \
data/AIBL/ 

Inside each of the above folder, we provided a table of meta-information of the subjects for all MRI scans that we used for model development and validation. To use the scirpts, you need to store the MRIs collected from those affilications in above corresponding folders. Note that this framework should work on your own T1-weighted MRI data if following data preprocessing pipeline were done properly:

1. linear register the MRI to template, i.e, any one of the sample MRIs provided.
2. normalize the data by data = (data - data.mean()) / data.std()
3. clip the data to the range of -1 to 2.5





