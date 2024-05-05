#!/bin/bash

DATA_ROOT_DIR="/home/zhany0x/Documents/data/ctranspath-pibd" # where are the TCGA features stored?
declare -A RESULTS_DIRS=(["coadread"]="/home/zhany0x/Documents/experiment/PIBD/tcga_coadread_b32_survival_months_dss_wsiDim_256_epochs_30_omics_pathways_pathT_combine_s3/") # where is the results stored?
TYPE_OF_PATH="combine" # what type of pathways?
STUDIES=("coadread")

for STUDY in ${STUDIES[@]}; do
    python main.py \
        --only_test \
        --study tcga_${STUDY} --task survival --which_splits 5foldcv \
        --type_of_path $TYPE_OF_PATH  \
        --data_root_dir "$DATA_ROOT_DIR/$STUDY/tiles-l1-s224/feats-l1-s224-CTransPath-sampler/" \
        --label_file "datasets_csv/metadata/tcga_${STUDY}.csv" \
        --omics_dir "datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY}" --results_dir "${RESULTS_DIRS[${STUDY}]}" \
        &
done
wait