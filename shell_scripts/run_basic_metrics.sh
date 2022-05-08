#!/bin/bash
export TOXIC_DIR=/scratch/sbp354/DSGA1012/Final_Project/data
export FILE=PAPI_BiBiFi_test_evaluated.csv
export LABEL=true_labels
export SCORE=scores
export P = True

python /scratch/pg2255/nlu/Toxic_Debias/src/basic_metrics.py \
  --data_dir $TOXIC_DIR \
  --results_csv $FILE \
  --label_name $LABEL \
  --score_name $SCORE
