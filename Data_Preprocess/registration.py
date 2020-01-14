import nibabel as nib
import numpy as np
from glob import glob
import nipype
from nipype.interfaces import fsl

def registration(in_file, out_file, reference):
    fsl.FSLCommand.set_default_output_type('NIFTI')
    flt = fsl.FLIRT(bins=640, cost_func='mutualinfo')
    flt.inputs.in_file = in_file
    flt.inputs.out_file = out_file
    flt.inputs.reference = reference
    result = flt.run()

if __name__ == "__main__":
    for file in glob('path_for_nifti_files/*.nii'):
        in_file = file
        out_file = "registered_path"
        reference = './template/temp.nii'
        registration(in_file, out_file, reference)
