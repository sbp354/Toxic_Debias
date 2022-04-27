import os
import pandas as pd
import argparse
import pandas as pd
import random

"""
The Multi-Genre Natural Language Inference (MultiNLI) corpus is 433k sentence pairs.
https://github.com/UKPLab/emnlp2020-debiasing-unknown ran a shallow model on a 
2K sub-sampled set of MNLI for 5 epochs. 2/433 = .0046.
We'll run it on a .5% sub-sample 
"""


def set_seed(args):
    random.seed(args.seed)

def save_csvs(args,df_shallow, df_remainder):
    df_shallow.to_csv(path= os.path.join(args.data_dir, args.train_dataset + '_train_shallow_' + args.mode + '.csv'),
    index = True,
    index_label = 'ind')
    
    df_remainder.to_csv(path= os.path.join(args.data_dir, args.train_dataset + '_train_shallow_remainder_' + args.mode + '.csv'),
    index = True,
    index_label = 'ind')

def random_perc(args):
    set_seed(args)
    if args.mode == "random_005":
        perc = .005
    df = pd.read_csv(
         os.path.join(args.data_dir, args.train_dataset + '_train.csv'),
         header=0) #, 
         #skiprows=lambda i: i>0 and random.random() > perc)

    # We need indices of the original rows for teacher predictions
    df['ind'] = [x for x in range(0, len(df.values))]

    df_shallow = df.sample(frac = perc)
    df_remainder = df.drop(df_shallow.index)

    save_csvs(args,df_shallow,df_remainder)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .csv files (or other data files) for the task.",
    )

    parser.add_argument(
        "--train_dataset",
        default=None,
        type=str,
        required=True,
        help="Which finetuning training dataset to use",
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="random seed for initialization"
    )

    parser.add_argument("--mode", 
                        choices=["random_005"],
                        help = "Different subsampling methods",
                        default="random_005")

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the shallow train data file will be written.",
    )

    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Set seed
    set_seed(args)

    if args.mode == "random_005":
        random_perc(args)


if __name__ == "__main__":
    main()
