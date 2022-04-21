# Toxic Language Debiasing
This repo contains the code for our paper "[Challenges in Automated Debiasing
for Toxic Language Detection](https://arxiv.org/pdf/2102.00086.pdf)". In particular, it contains the
code to fine-tune RoBERTa and RoBERTa with the ensemble-based method in the
task of toxic language prediction. It also contains the index of data points
that we used in the experiments. 
<!---
## Citation:

```bibtex
@inproceedings{Zhou2021ToxicDebias,
            author={Xuhui Zhou, Maarten Sap, Swabha Swayamdipta, Noah A. Smith
            and Yejin Choi},
            title={Challenges in Automated Debiasing for Toxic Language
            Detection},
            booktitle={EACL},
            year={2021}
        }
```
-->
## Overview
### Tasks
This repo contains code to detect toxic language with RoBERT/ ensemble-based
ROBERTa. Our experiments mainly focus on the dataset from 
["Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior"](https://ojs.aaai.org/index.php/ICWSM/article/view/14991).

### Code
Our implementation exists in the `.\src` folder. The `run_toxic.py` file
organize the classifier, and the `modeling_roberta_debias.py` builds the
ensemble-based model.

## Setup 

### Dependencies

We require pytorch>=1.2 and transformers=2.3.0  Additional requirements are are
in

`requirements.txt`

### NLU Finetuning Instructions

We have set up the repo to allow for finetuning on two datasets and testing/ eval on 5 datas (4 for each of the finetuning datasets). Combinations of finetune/ test datasets are in the table below:

| Finetune Dataset     | Challenge/Eval Datasets                                         |
|----------------------|-----------------------------------------------------------------|
| Civil comments train | Civil comments test                                           |
|                      | SBIC                                                            |
|                      | BiBiFi                                                |
|                      | Covert comments                                             |
|                      | TwitterAAE                                                  |
| Founta train         | Founta test 
|                      | SBIC                                                            |
|                      | BiBiFi                                                |
|                      | Covert comments                                             |
|                      | TwitterAAE                                                  |

We run finetuning and eval by updating different shell scripts found in the shell_scripts/ folder of the parent directory. The relevant arguments to update are below:

* TOXIC_DIR: Parent directory where different datasets are read in and where tokenized forms of datasets are cached (assumption is that there are dataset-specific subdirectories)
* TRAIN_DATASET : name of the finetuning dataset to use. Options allowed in current iteration of the repository are:
  *  founta
  *  civil_comments_0.5
* DEV_DATASET : name of the challenge dataset on which the finetuned model is to be scores. Options allowed in current iteration of the repository are:
  *  founta (only use when running founta finetuned model)
  *  civil_comments_0.5 (only use whne running civil comments finetuned model
  *  SBIC
  *  bibifi
  *  covert_comments 
* MODEL_DIR : directory where model checkpoints/results will get output 
* do_train / no-do_train: when no_train finetuning will run; when no-do_train only eval will run

### Data

* You can find the index of the training data with different data selection
  methods in `data/founta/train`
* You can find a complete list of entries of data that we need for experiments
  in `data/demo.csv`
* Out-of-distribution (OOD) data, the two OOD datasets we use are publicly
  available:
    * ONI-adv: This dataset is the test set of the work ["Build it Break it Fix
it for Dialogue Safety: Robustness from Adversarial Human
Attack"](https://www.aclweb.org/anthology/D19-1461/)
    * User-reported: This dataset is from the work ['User-Level Race and Ethnicity Predictors from Twitter Text'](https://www.aclweb.org/anthology/C18-1130/)

* Our word list for lexical bias is in the file: `./data/word_based_bias_list.csv`
* Since we do not encourage building systems based on our relabeling dataset,
  we decide not to release the relabeling dataset publicly. For research purpose, please
  contact the first author for the access of the dataset.

## Experiments

### Measure Dataset Bias
Run 
```python 
python ./tools/get_stats.py /location/of/your/data_file.csv

```
To obtain the Peasonr correlation between toxicity and Tox-Trig words/ aav
probabilities.

### Fine-tune a Vanilla RoBERTa
Run 
```bash
sh run_toxic.sh 
```

### Fine-tune a Ensemble-based RoBERTa
Run 
```bash
sh run_toxic_debias.sh
```

You need to obtain the bias-only model first in order to train the ensemble
model. Feel free to use files we provided in the folder `tools`.

### Model Evaluation & Measuring Models' Bias

You can use the same fine-tuning script to obtain predictions from models. 

The measuring bias script takes the predictions as input and output models'
performance and lexical/dialectal bias scores. The script is available in the
`src` folder.
