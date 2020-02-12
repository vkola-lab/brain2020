import nibabel as nib
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--subject", "-s", type=str, help="Name of subject", required=True)
parser.add_argument("--subdir", "-d", type=str, help="Name of subjects directory", required=True)
args = vars(parser.parse_args())
print(args.keys())

subcort = nib.load(""+args["subdir"]+"/"+args["subject"]+"/mri/aseg.mgz")

rejects = [1, 2, 3, 40, 41, 42, 77]

subcort_data = np.asarray(subcort.dataobj)

subcort_only = subcort_data.copy()
for value in rejects:
    subcort_only = np.where(subcort_only == value, 0, subcort_only)

subcort_clean = nib.MGHImage(dataobj=subcort_only, affine=subcort.affine, header=subcort.header)
nib.save(subcort_clean, ""+args["subdir"]+"/"+args["subject"]+"/mri/aseg.onlysubcort.mgz")
