#!/bin/sh
#this bash script use fsl to process brain MRI into MNI template

export FSLOUTPUTTYPE='NIFTI'

# $1 is the folder of the raw data, $2 is the filename of the raw data, $3 is output folder to save processed data
echo "processing file from $1: $2"

cd $1

# perform all intermediate calculations in tmp/ folder
cp $2 $3/tmp/

cd $3/tmp/

#step1 is to swap axes so that the brain is in the same direction as MNI template.
${FSLDIR}/bin/fslreorient2std $2 T1.nii

#step2 is to estimate robust field of view
line=`${FSLDIR}/bin/robustfov -i T1.nii | grep -v Final | head -n 1`

x1=`echo ${line} | awk '{print $1}'`
x2=`echo ${line} | awk '{print $2}'`
y1=`echo ${line} | awk '{print $3}'`
y2=`echo ${line} | awk '{print $4}'`
z1=`echo ${line} | awk '{print $5}'`
z2=`echo ${line} | awk '{print $6}'`

x1=`printf "%.0f", $x1`
x2=`printf "%.0f", $x2`
y1=`printf "%.0f", $y1`
y2=`printf "%.0f", $y2`
z1=`printf "%.0f", $z1`
z2=`printf "%.0f", $z2`

#step3 is to cut the brain to get area of interest (roi), sometimes it cuts part of the brain
${FSLDIR}/bin/fslmaths T1.nii -roi $x1 $x2 $y1 $y2 $z1 $z2 0 1 T1_roi.nii

#step4: remove skull -g 0.1 -f 0.45
${FSLDIR}/bin/bet T1_roi.nii T1_brain.nii -R

#step5: registration from cut to MNI
${FSLDIR}/bin/flirt -in T1_brain.nii -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain -omat orig_to_MNI.mat

#step6: apply matrix onto original image
${FSLDIR}/bin/flirt -in T1.nii -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain -applyxfm -init orig_to_MNI.mat -out T1_MNI.nii

#step7: rename and move final file
echo "saving processed file into $3:$2"
mv T1_MNI.nii $3/$2

# clear tmp folder
rm -f $3/tmp/*
