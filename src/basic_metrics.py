import os
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score 


def get_scores(df, label_name='true_labels', pred_name='predictions', score_name='scores', binary=True):
    """
    calculate relevant statistics for measuring bias.
    
    df: contains predictions and true_labels
    Depending on the dataset df should also have associations of identity 
    related to the id which will be tied to the original index 'Unnamed: 0'
    
    
    binary: boolean for if its only binary toxicity
    """
    
    
    labels = df[label_name]
    predictions = df[pred_name]
    scores = df[score_name]
    
    if binary:
        f1_avg = 'binary'
        recall_pos = recall_score(labels, predictions) #Not sure about how to use the threshold for these for now
        #recall_neg = recall_score(labels, predictions, pos_label=0) # Not sure about this 
        
    predicted_prevalence = predictions.mean()
    avg_scores = scores.mean()
    f1 = f1_score(y_true = labels, y_pred = predictions, average=f1_avg)
    
    auc_roc = roc_auc_score(labels, scores)
    
    metrics  = {'avg-scores': avg_scores,
                'predicted-prevalence': predicted_prevalence,
                'f1': f1,
                'auc-roc': auc_roc,
                'fpr': recall_pos,  # FPR
               }
    #return [acc,f1, 1-recall_neg] Need to understand why they do 1- recall_neg before adding it
    return metrics



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .csv files that have model results.",
    )

    parser.add_argument(
        "--results_csv",
        default=None,
        type=str,
        required=True,
        help="Output of a model with predictions, scores, true label on an eval dataset",
    )

    parser.add_argument("--label_name", 
                    help = "The column name for the ground truth",
                    required=False,
                    default="true_labels")

    parser.add_argument("--pred_name", 
                    help = "The column name for the model predictions",
                    required=False,
                    default="predictions")

    parser.add_argument("--score_name", 
                    help = "The column name for the model predictions",
                    required=False,
                    default="scores")

    parser.add_argument("--output_dir",
                    default="/scratch/sbp354/DSGA1012/Final_Project/models/results",
                    type=str,
                    required=False,
                    help="metrics out put file name")

    parser.add_argument("--output_name",
                    default=None,
                    type=str,
                    required=False,
                    help="Metrics out put file name")
    
    args = parser.parse_args()


    df = pd.read_csv(os.path.join(args.data_dir, args.results_csv))
    if args.pred_name:
        pred_name = args.pred_name
    else:
        pred_name = 'predictions'
        df[pred_name] = (df[args.score_name] > .5).astype(int).values

    metrics = get_scores(df, args.label_name, pred_name, args.score_name)

    if args.output_name:
        output_path =  os.path.join(args.output_dir,args.output_name)
    else:
        if not args.pred_name:
            output_name = args.results_csv[:-4] + '_basic_metrics.txt'
        else: 
            datasets = ['founta','civil_comments','civil_comments_0.5']
            struct = ' '.join(args.model_dir.split('/')).split()  # This makes sure if there is / at the end its fine
            model = struct[-2]
            loss = struct[-1]
            if model in datasets:
                output_name = loss + args.results_csv[:-4]  # If no custom loss function then model name is here
            else:
                output_name = model + loss + args.results_csv[:-4]

        output_path = os.path.join(args.output_dir,output_name)


    # with open(os.path.join(args.data_dir, file_name), 'w') as f: 
    #     for key, value in metrics.items(): 
    #         f.write('%s:%s\n' % (key, value))
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)
    metrics_df.to_csv(output_path)


if __name__ == "__main__":
    main()
