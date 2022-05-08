#!/bin/bash
export TOXIC_DIR=/scratch/sbp354/DSGA1012/Final_Project/data
export FILE=PAPI_civil_test_evaluated.csv
export LABEL=label
export SCORE=perspective_score
export P=True

python /scratch/pg2255/nlu/Toxic_Debias/src/basic_metrics.py \
  --data_dir $TOXIC_DIR \
  --results_csv $FILE \
  --label_name $LABEL \
  --score_name $SCORE \
  --pAPI $P
