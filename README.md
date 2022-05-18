## Setup 

### Dependencies

We require pytorch>=1.2 and transformers=2.3.0  Additional requirements are are
in

`requirements.txt`

### NLU Finetuning Instructions

We have set up the repo to allow for finetuning on two datasets and testing/ eval on 6 datasets. Combinations of finetune/ test datasets are in the table below:

| Finetune Dataset     | Challenge/Eval Datasets                                         |
|----------------------|-----------------------------------------------------------------|
| Civil identitites    | Civil identities                                                |
|                      | Founta test                                                     |
|                      | SBIC                                                            |
|                      | BiBiFi                                                |
|                      | Covert comments                                             |
|                      | TwitterAAE                                                  |
| Founta train         | Founta test  |
|                      | Civil identities                                             |
|                      | SBIC                                                            |
|                      | BiBiFi                                                |
|                      | Covert comments                                             |
|                      | TwitterAAE                                                  |

We run finetuning and eval by updating different shell scripts found in the shell_scripts/ folder of the parent directory. The relevant arguments to update are below:

* TOXIC_DIR: Parent directory where different datasets are read in and where tokenized forms of datasets are cached (assumption is that there are dataset-specific subdirectories)
* TRAIN_DATASET : name of the finetuning dataset to use. Options allowed in current iteration of the repository are:
  *  founta
  *  civil_identities
* DEV_DATASET : name of the challenge dataset on which the finetuned model is to be scores. Options allowed in current iteration of the repository are:
  *  founta
  *  civil_identities
  *  SBIC
  *  bibifi
  *  covert_comments 
  *  twitter_aae
* MODEL_DIR : directory where model checkpoints/results will get output 
* do_train / no-do_train: when no_train finetuning will run; when no-do_train only eval will run

## Code modifications for debiasing

We have heavily modified run_toxic.py in order to allow for the new loss functions introduced in src/clf_loss_functions.py

To run these debiasing methods you must first train a shallow model. Do this by calling src/shallow_subsample.py

Take the shallow subsamples created and finetune the model by setting the --debias argument to "shallow". Finetune with the training dataset as the 0.5% portion and evaluate on the remainder. An example for this can be found in shell_scripts/shallow_example.sh

Examples for running debiasing can be found in the shell_scripts folder and should be fairly self-explanatory to run with your own folder structure.

### Original Readme

Please see the original readme at https://github.com/XuhuiZhou/Toxic_Debias/blob/main/README.md in order to understand the original intent of the code we've modified.

### Report

 Our report can be found within the same github Project folder titled NLU_final_paper.pdf
