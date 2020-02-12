help () {
    echo " Performs volumetric brain segmentation on a freesurfer processed subject"
    echo " (Prerequisite: recon-all must have run for the subject)"
    echo " The combined files are stored in \"\$SUBJECTS_DIR/$SUBJECT/segments/\""
    echo "\nArguments:\n\t-h : Prints this messaage"
    echo "\t-s : Name of subject\n\t-d : Subjects Directory"
    echo "\t-v : Verbose output"
    exit
}

printv () {
    if [ -n "$VERBOSE" ]; then
	echo $1
    fi
}

evalv () {
    if [ -n "$VERBOSE" ]; 
	then
	    eval $1
	else
	    eval $1 >/dev/null
    fi
}

label_and_register () {
    evalv "mri_annotation2label --subject ${SUBJECT} --hemi lh --outdir ${SUBJECTS_DIR}/${SUBJECT}/labels/" >/dev/null
    evalv "mri_annotation2label --subject ${SUBJECT} --hemi rh --outdir ${SUBJECTS_DIR}/${SUBJECT}/labels/" >/dev/null
    printv "Converted Annotations to labels"
    evalv "tkregister2 --mov ${SUBJECTS_DIR}/${SUBJECT}/mri/rawavg.mgz --noedit --s ${SUBJECT} --regheader --reg ${SUBJECTS_DIR}/${SUBJECT}/register.dat"
    printv "Registered MRI"
}

label_to_volume () {
    local cmd="mri_label2vol "
    for label in ${SUBJECTS_DIR}/${SUBJECT}/labels/$1*.label; do
	cmd=$cmd" --label $label "
    done
    cmd=$cmd" --temp ${SUBJECTS_DIR}/${SUBJECT}/mri/rawavg.mgz"
    cmd=$cmd" --subject ${SUBJECT}"
    cmd=$cmd" --hemi $1"
    cmd=$cmd" --o ${SUBJECTS_DIR}/${SUBJECT}/segments/$1_combined.nii.gz"
    #cmd=$cmd" --fillthresh "
    cmd=$cmd" --proj frac -3 1 .01"
    cmd=$cmd" --reg ${SUBJECTS_DIR}/${SUBJECT}/register.dat"
    if [ ! -d ${SUBJECTS_DIR}/${SUBJECT}/"segments" ]; then
	mkdir ${SUBJECTS_DIR}/${SUBJECT}/"segments"
    fi
    if [ -z "$VERBOSE" ]; then
	cmd=$cmd" >/dev/null"
    fi
    chmod +w ${SUBJECTS_DIR}/${SUBJECT}/"segments"
    printv "Converting $1"
    eval $cmd
}

correct_aseg () {
    local cmd="mri_convert -rl $SUBJECTS_DIR/$SUBJECT/segments/lh_combined.nii.gz -o $SUBJECTS_DIR/$SUBJECT/mri/aseg.onlysubcort.mgz -i $SUBJECTS_DIR/$SUBJECT/mri/aseg.onlysubcort.mgz"
    if [ -n "$VERBOSE" ]; then
	cmd=$cmd" >/dev/null"
    fi
    printv "Correcting aseg"
    eval $cmd
}

combine_segments () {
    local cmd1="mris_calc -o $SUBJECTS_DIR/$SUBJECT/segments/combined_hemis.mgz $SUBJECTS_DIR/$SUBJECT/segments/lh_combined.nii.gz add $SUBJECTS_DIR/$SUBJECT/segments/rh_combined.nii.gz"
    local cmd0="mri_convert --dil-seg 5 -i $SUBJECTS_DIR/$SUBJECT/segments/combined_hemis.mgz -o $SUBJECTS_DIR/$SUBJECT/segments/combined_hemis.mgz"
    local cmd2="mris_calc -o $SUBJECTS_DIR/$SUBJECT/segments/combined_full.mgz $SUBJECTS_DIR/$SUBJECT/segments/combined_hemis.mgz add $SUBJECTS_DIR/$SUBJECT/mri/aseg.onlysubcort.mgz"
    if [ -n "$VERBOSE" ]; then
	cmd1=$cmd1" >/dev/null"
    cmd0=$cmd0" >/dev/null"
	cmd2=$cmd2" >/dev/null"
    fi
    printv "Combining labels"
    eval $cmd1
    eval $cmd0
    eval $cmd2
}

while getopts 's:d:v?h' c;
do
  case $c in
    s) SUBJECT=$OPTARG ;;
    d) SUBJECTS_DIR=$OPTARG ;;
    v) VERBOSE=true ;;
    h|?) help ;;
  esac
done

if [ -z "$SUBJECT" ]; then
    echo "Subject name is required"
    return 1
fi

if [ -z "$SUBJECTS_DIR" ]; then
    echo "Subjects directory not provided"
    return 1
fi

label_and_register

label_to_volume lh
label_to_volume rh

python clean_aseg.py -s $SUBJECT -d $SUBJECTS_DIR
correct_aseg
combine_segments


