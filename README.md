# Development and validation of an interpretable deep learning framework for Alzheimer’s disease classification

## Introduction

This repo contains a PyTorch implementation of a deep learning framework that delineates explainable Alzheimer’s disease signatures (3D disease risk map) from magnetic resonance imaging which are then integrated with multimodal inputs, including age, gender, and mini-mental state examination score. Our framework links a fully convolutional network (FCN) to a multilayer perceptron. The FCN generates patient specific 3D disease risk map (dense local predictions) see below. 

<img src="plot/riskmap.png" width="425"/> 

The FCN model was developed on ADNI training and validation sets and its performance was evaluated on ADNI testing set, 3 external testing datasets, including NACC, AIBL and FHS datasets. The Matthews correlation coefficient (MCC) values for all locations are shown as MCC heat map to visualize how accurate the FCN is on every locations.  

<img src="plot/heatmap.png" width="425"/> 

The MLP makes final global prediction on the diagnosis results. The predicted high risk regions were compared to neuropath findings as a purpose of verification and correlations between the model predicted regions with the neuropath findings were demonstrated in this work. see below

<img src="plot/neuropath.png" width="395"/>

The performance of the final global prediction from MLP model was compared with 11 neurologists. For the comprehensive comparision of our deep learning framework with other standard models, CNN and random forest models are also included in this repo. See below our FCN + MLP model roc curve.

<img src="plot/roc.png" width="695"/>

Please refer to our paper for more details. 

## How to use

These instructions will help you properly configure and use the tool. For detailed usage of the parameters, you may refer to Documentation section.

### Data

We trained, validated and tested the framework using the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. To investigate the generalizability of the framework, we externally tested the framework on the National Alzheimer's Coordinating Center (NACC), the Australian Imaging Biomarkers and Lifestyle Study of Ageing (AIBL) and Framingham Heart Study (FHS) datasets.

To download the data, please contact those affiliations directly. We only provide few data samples in the repo for the purpose of illustration. We organized the data into 3 folders:

```
<repository_root>/data/ADNI/
<repository_root>/data/NACC/
<repository_root>/data/AIBL/
```

Inside each of the folders, we provide a table of metadata of the subjects for all MRI scans that we used for model development and validation. Note that this framework should also work on your own T1-weighted MRI data if the following data preprocessing pipeline is done properly.

1. Register the MRI to template, i.e, any one of the sample MRIs provided.
2. Normalize the data using: data = (data - data.mean()) / data.std()
3. Clip the data to the range [-1, 2.5]

### Code dependencies

The tool was developped based on the following packages:

1. PyTorch (1.1 or greater).
2. NumPy (1.16 or greater).
3. pandas (0.24 or greater).
4. tqdm (4.31 or greater).

Please note that the dependencies may require Python 3.6 or greater. It is recommemded to install and maintain all packages by using [`conda`](https://www.anaconda.com/) or [`pip`](https://pypi.org/project/pip/). For the installation of GPU accelerated PyTorch, additional effort may be required. Please check the official websites of [PyTorch](https://pytorch.org/get-started/locally/) and [CUDA](https://developer.nvidia.com/cuda-downloads) for detailed instructions.

### Configuration file

The configuration file is a json file, which looks like this:

```json
{
    "fcn": {
        "__init__": {
            "channels":             20,
            "dropout":              0.6
        },
        "fit": {
            "trn_mri_fp":           ["./data/ADNI/"],
            "vld_mri_fp":           ["./data/NACC/", "./data/AIBL/"],
            "n_epoch":              2000,             
            "pretrained_weights":   "",             
            "batch_size":           16,             
            "learning_rate":        0.0001
        },
        "predict": {
        }
    },
    "mlp": {
        "__init__": {
            "hmap_fn":              "./dpm_statistics/ADNI/MCC.npy",
            "features":             ["rmap"],
            "hidden_dim":           64,
            "dropout":              0.5
        },
        "fit": {
            "dpm_fp":               ["./dpm/ADNI/"],
            "info_fn":              ["./data/ADNI/ADNI.csv"],
            "loss_balance":         2.3,
            "n_epoch":              200, 
            "batch_size":           10,
            "learning_rate":        0.01
        },
        "predict": {
            "dpm_fp":               ["./dpm/NACC/", "./dpm/AIBL/"],
            "info_fn":              ["./data/NACC/NACC.csv", "./data/AIBL/NACC.csv"],
            "batch_size":           -1
        }
}
```

### Train FCN and generate DPM

```
python FCN_main.py -c config.json
```

After running above command, the following steps will be performed:

1. An FCN model will be trained from scratch, optimal model weights based on validation accuracy will be stored in the `<repository_root>/FCN_model/` folder.

2. After FCN training, DPMs for all subjects will be generated in the `<repository_root>/dpm/` folder with the same organization as `<repository_root>/data/` folder.

3. To evaluate performance of the FCN model, heatmaps of various metrics will be generated (MCC, F-1, Accuracy, True Positive, False Postive, True negative, False negative) in the ./Heatmap/ folder.

### Train MLP with DPM and/or other non-imaging features

Multilayer Perceptron (MLP) network is a densely connected neural network with one hidden layer. It utilizes DPM along with other non-imaging factors (i.e. age, gender, MMSE etc.) to do AD/NL classification. The MLP routine contains the following steps.

1. Filter DPM. DPM may contain millions of voxels. Dumping all of them into the MLP is unefficient and may even have negative impact on the classification performance. We selected only ~1% voxels that were highly correlated with AD/NL classfication. As to the definition of the "correlation", we believe MCC is practically an ideal metric.

2. Choose non-imaging features. Besides DPM, arbitrary combination of the available non-imaging features can also be fed into the MLP. Our experiments shown that DPM with age, gender and MMSE was sufficient to achieve neurologist-level diagnosis.

3. Train model.
