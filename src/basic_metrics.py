import os
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import roc_curve, accuracy_score, f1_score, recall_score, roc_auc_score 
from sklearn.metrics import confusion_matrix

def get_scores(df, label_name='true_labels', pred_name='predictions', score_name='scores', proba_name='proba', binary=True):
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
    proba = df[proba_name]
    
    if binary:
        f1_avg = 'binary'
        recall_pos = recall_score(labels, predictions) #Not sure about how to use the threshold for these for now
        #recall_neg = recall_score(labels, predictions, pos_label=0) # Not sure about this 
        
    predicted_prevalence = predictions.mean()
    avg_scores = scores.mean()
    f1 = f1_score(y_true = labels, y_pred = predictions, average=f1_avg)
    
    try:
        auc_roc = roc_auc_score(labels, proba)
    except:
        auc_roc = np.nan

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = confusion_matrix(list(labels), list(predictions), labels=[0, 1]).ravel()
    tot = tn+tp+fp+fn
    #print('False Positive Rate', fp/tot, recall_pos)

    metrics  = {'avg-scores': avg_scores,
                'predicted-prevalence': predicted_prevalence,
                'f1': f1,
                'auc-roc': auc_roc,
                'fpr': fp/tot,  # FPR
                'recall_pos': recall_pos
               }
    #return [acc,f1, 1-recall_neg] Need to understand why they do 1- recall_neg before adding it
    return metrics



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
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
    
    parser.add_argument("--output_suffix",
            default="_basic_metrics.csv",
            type=str,
            required=False)

    parser.add_argument("--pAPI",
                    default=False,
                    type=str,
                    required=False,
                    help="modifications for pAPI")

    args = parser.parse_args()


    df = pd.read_csv(os.path.join(args.model_dir, args.results_csv))


    pred_name = args.pred_name
    if args.pAPI == "True": 
        df[pred_name] = (df[args.score_name] > .5).astype(int).values
    
    if args.results_csv == "PAPI_challenge_civil_results.csv":
        toxic_labels = ['toxicity', 'severe_toxicity','obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat']
        df['true_labels'] = df[toxic_labels].max(axis = 1)
        df['true_labels'] = np.where(df['true_labels']>.5, 1, 0)
    #     pred_name = 'predictions'
    #     df = df[~df['male'].isnull()]
    #     df[pred_name] = (df[args.score_name] > .5).astype(int).values
    #     df[args.label_name] = (df['toxicity'] > .5).astype(int).values
    
    if args.pAPI == "False":
        df['proba'] = np.where(df[pred_name]==1, df[args.score_name],1- df[args.score_name])
        metrics = get_scores(df, args.label_name, pred_name, args.score_name, 'proba')
    else:
        df['scores'] =  np.where(df[pred_name]==1, df[args.score_name],1- df[args.score_name])
        metrics = get_scores(df, args.label_name, pred_name, 'scores', args.score_name)

    # if args.output_name:
    #     output_path =  os.path.join(args.output_dir,args.output_name)
    # else:
    #     if args.pAPI == True:
    #         output_name = args.results_csv[:-4] + '_basic_metrics.csv'
    #     else: 
    #         datasets = ['founta','civil_comments','civil_comments_0.5']
    #         struct = ' '.join(args.model_dir.split('/')).split()  # This makes sure if there is / at the end its fine
    #         model = struct[-2]
    #         loss = struct[-1]
    #         if model in datasets:
    #             output_name = loss + args.results_csv[:-4] + '_basic_metrics.csv' # If no custom loss function then model name is here
    #         else:
    #             output_name = model + loss + args.results_csv[:-4] + '_basic_metrics.csv'

    #     output_path = os.path.join(args.output_dir,output_name)

    struct = ' '.join(args.model_dir.split('/')).split()  # This makes sure if there is / at the end its fine
    finetune_dataset = struct[-3].split('_')[0]
    results_struct = args.results_csv.split('_')
    eval_dataset = '_'.join(results_struct[results_struct.index('challenge')+1:-1])
    if args.output_name:
        output_path =  os.path.join(args.output_dir,args.output_name)
    else:
        if args.pAPI == "True":
            output_name = args.results_csv[:-4] + '_basic_metrics.csv'
            model = "pAPI"
            loss = "N/A"
            finetune_dataset = "N/A"
        else:
            datasets = ['founta','civil_comments','civil_comments_0.5', 'civil_identities']
            # struct = ' '.join(args.model_dir.split('/')).split()  # This makes sure if there is / at the end its fine
            #print(struct)
            # finetune_dataset = struct[0].split('_')[0]
            model = struct[-2]
            loss = struct[-1]
            if model in datasets:
                finetune_dataset = model
                model = loss
                loss = "plain"
                output_name = loss + args.results_csv[:-4] + args.output_suffix  # If no custom loss function then model name is here
            else:
                output_name = '{}_{}_{}_{}'.format(model,loss,args.results_csv[:-12],args.output_suffix)
            output_path =  os.path.join(args.output_dir,output_name)
    
    print(output_name)
    print(output_path)


    # with open(os.path.join(args.data_dir, file_name), 'w') as f: 
    #     for key, value in metrics.items(): 
    #         f.write('%s:%s\n' % (key, value))
    
    metrics_df = pd.DataFrame.from_dict([metrics])
    metrics_df['model'] = model
    metrics_df['loss'] = loss
    metrics_df['fine_tune_data'] = finetune_dataset
    metrics_df['eval_data'] = eval_dataset
    print(metrics_df)
    metrics_df.to_csv(output_path)


if __name__ == "__main__":
    main()
