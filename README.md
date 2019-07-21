## Neurologist-level Alzheimer's disease classification using an explainable deep learning framework

## Background
This repo contains a PyTorch implementation of a deep learning framework for Alzheimer's disease classification using volumetric brain MR images. This framework comprises a fully convolutional network (FCN) and a multilayer perceptron (MLP) model. The FCN generates a novel 3D visualization of the disease affected regions. We refer to them as disease probability map (DPM) in our paper. Features taken from the DPMs were then sent into the MLP model to achieve an overall classification of AD status. 

## Usage
### Requirements
PyTorch 0.4.0 \
Numpy   1.16.2 \
Pandas  0.24.2 

### Data preparation 
We trained, validated and tested the framework using the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. To investigate the generalizability of the framework, we externally tested the framework on the National Alzheimer's Coordinating Center (NACC), the Australian Imaging Biomarkers and Lifestyle Study of Ageing (AIBL) and Framingham Heart Study (FHS) datasets.

To download the data, please contact those affiliations directly. We only provided few data samples in the repo for the purpose of illustration. We organized the data into 3 folders:

data/ADNI/ \
data/NACC/ \
data/AIBL/ 

Inside each of the folders, we provided a table of metadata of the subjects for all MRI scans that we used for model development and validation. Note that this framework should also work on your own T1-weighted MRI data if the following data preprocessing pipeline is done properly.

1. Register the MRI to template, i.e, any one of the sample MRIs provided.
2. Normalize the data using: data = (data - data.mean()) / data.std()
3. Clip the data to the range of -1 to 2.5

### Configuration settings
The configuration file is a json file, which looks like this:

    {
          "model" : {
              "channels":             20,
              "dropout":              0.6
          },

          "train": {
              "train_data_folder":   "./data/ADNI/",   
              "train_epoches":         2000,             
              "pretrained_weights":   "",             
              "batch_size":           16,             
              "learning_rate":        1e-4
          },

          "valid": {
              "valid_data_folder":   ["./data/NACC/", "./data/AIBL/"]
          }
    }

### Train the FCN model and generate DPMs
    python FCN_main.py -c config.json 
After running above command, the following things will be performed:
1. An FCN model will be trained from scratch, optimal model weights based on validation accuracy will be stored in the FCN_model/ folder.  
2. After FCN training, DPMs for all subjects will be generated in the ./Riskmap/ folder with the same organization as ./data/ folder.
3. To evaluate performance of the FCN model, heatmaps of various metrics will be generated (MCC, F-1, Accuracy, True Positive, False Postive, True negative, False negative) in the ./Heatmap/ folder.

### Train the MLP with DPM features with or without other features. 


### Model visualization



