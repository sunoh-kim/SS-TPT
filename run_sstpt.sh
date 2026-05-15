#!/bin/bash
BASE_CMD_CLEAN="python sstpt.py /home/work/TPT/data -a RN50 -b 16 --gpu 0 --ctx_init a_photo_of_a -p 50 --eps 0.0" 
BASE_CMD_ADV="python sstpt.py /home/work/TPT/data -a RN50 -b 16 --gpu 0 --ctx_init a_photo_of_a -p 50 --eps 1.0 --steps 7" 

DATASETS=("Caltech101")

for DATASET in "${DATASETS[@]}"; do
  ${BASE_CMD_CLEAN} \
    --test_sets "${DATASET}" \
    --output_dir output_results/sstpt

  ${BASE_CMD_ADV} \
    --test_sets "${DATASET}" \
    --output_dir output_results/sstpt
done

