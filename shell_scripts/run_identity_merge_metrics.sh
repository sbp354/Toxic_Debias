export TOXIC_DIR=/scratch/sbp354/DSGA1012/Final_Project/models/civil_comments_0.5/roberta-large
export FILE=finetune_civil_comments_0.5_challenge_civil_comments_0.5_results.csv
export IDIR=/scratch/sbp354/DSGA1012/Final_Project/data
export IFILE=civil_test.csv

python /scratch/pg2255/nlu/Toxic_Debias/src/basic_metrics.py \
  --model_dir $TOXIC_DIR \
  --results_csv $FILE \
  --identity_dir $IDIR \ 
  --identities_csv $IFILE \