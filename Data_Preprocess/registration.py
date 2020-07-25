import nibabel as nib
import numpy as np
from glob import glob
import nipype
from nipype.interfaces import fsl
from subprocess import call
import sys
import os

def registration(in_file, out_file, reference):
    fsl.FSLCommand.set_default_output_type('NIFTI')
    flt = fsl.FLIRT(bins=640, cost_func='mutualinfo')
    flt.inputs.in_file = in_file
    flt.inputs.out_file = out_file
    flt.inputs.reference = reference
    result = flt.run()

if __name__ == "__main__":
    choose_robust_bash_pipeline = True

    raw_folder = sys.argv[1]  # where the raw data is saved
    out_folder = sys.argv[2]  # where you want to save processed scans
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    if choose_robust_bash_pipeline: # recommend to use robust bash pipeline
        if not os.path.exists(out_folder+'tmp/'): # tmp is used to store intermediate results from the bash pipeline
            os.mkdir(out_folder+'tmp/')
        for fullPath in glob(raw_folder+'/*.nii'):
            filename = fullPath.split('/')[-1]
            call('bash registration.sh ' + raw_folder + ' ' + filename + ' ' + out_folder, shell=True)

    else: # you can also choose to use simple registration function defined above
        # see reference: https://nipype.readthedocs.io/en/0.12.0/interfaces/generated/nipype.interfaces.fsl.preprocess.html#flirt
        for file in glob(raw_folder+'/*.nii'):
            reference = 'MNI152_T1_1mm.nii.gz'
            registration(file, out_folder, reference)


