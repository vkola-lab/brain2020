import numpy as np
import nibabel as nib
import sys
from glob import glob 

def nifti_to_numpy(file):
    data = nib.load(file).get_data()[:181, :217, :181]
    return data

def normalization(scan):
    scan = (scan - np.mean(scan)) / np.std(scan)
    return scan

def clip(scan):
    return np.clip(scan, -1, 2.5)

if __name__ == "__main__":
    folder = sys.argv[1]
    for file in glob(folder + '*.nii'):
        data = nifti_to_numpy(file)
        data = normalization(data)
        data = clip(data)
        np.save(file.replace('.nii', '.npy'), data)
    


