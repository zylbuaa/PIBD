#!/bin/bash

DATA_ROOT_DIR="/ibex/user/zhany0x/data/CTranPath" # where are the TCGA features stored?
RESULTS_DIR="/ibex/user/zhany0x/experiment/PIBD/" # where is the results stored?
TYPE_OF_PATH="combine" # what type of pathways?
SEEDS=(1 2 3)
STUDIES=("blca")

for SEED in ${SEEDS[@]}; do
  for STUDY in ${STUDIES[@]}; do
      python main.py \
          --study tcga_${STUDY} --task survival --which_splits 5foldcv \
          --type_of_path $TYPE_OF_PATH --seed ${SEED}\
          --data_root_dir "$DATA_ROOT_DIR/$STUDY/tiles-l1-s224/feats-l1-s224-CTransPath-sampler/" \
          --label_file "datasets_csv/metadata/tcga_${STUDY}.csv" \
          --omics_dir "datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY}" --results_dir "$RESULTS_DIR" \
          &
  done
done
wait