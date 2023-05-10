#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSlhch
PYTHON_VIRTUAL_ENVIRONMENT=tim
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

for thres in 0.5 1.0
do
python \
      $HOME2/scratch/supervised-CL/imagenet/simclr/launcher.py \
      --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ \
      --workers 32 \
      --epochs 200 \
      --batch-size 4096 \
      --learning-rate 4.8 \
      --checkpoint-dir $HOME2/scratch/supervised-CL/imagenet/simclr/saved_models/ \
      --log-dir $HOME2/scratch/supervised-CL/imagenet/simclr/logs/ \
      --rotation 0.0 \
      --ngpus-per-node 4 \
      --nodes 8 \
      --exp Supcon_new_thres${thres}_ep200 \
      --supcon_new \
      --threshold ${thres}
done
echo "Run completed at:- "
date

