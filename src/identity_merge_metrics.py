import os
import pandas as pd
import numpy as np
import argparse

from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
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
        recall_pos = recall_score(labels, predictions)
        
    avg_scores = scores.mean()
    predicted_prevalence = predictions.mean()

    f1 = f1_score(y_true = labels, y_pred = predictions, average=f1_avg)
    
    try:
        auc_roc = roc_auc_score(labels, proba)
    except:
        auc_roc = np.nan
    
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = confusion_matrix(list(labels), list(predictions), labels=[0, 1]).ravel()
    tot = tn+tp+fp+fn

    metrics  = {'metrics_condition' : '',
                'avg-scores': avg_scores,
                'predicted-prevalence': predicted_prevalence,
                'f1': f1,
                'auc-roc': auc_roc,
                'fpr': fp/tot, # FPR
                'recall_pos': recall_pos
               }
    return metrics

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--identities_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The input data dir. Should contain the .csv files with additional identitiy labels with each row corresponding to outputs.")

    parser.add_argument("--identities_csv",
                    default=None,
                    type=str,
                    required=True,
                    help="Original data frame from eval file containing labels of interest for post inference analysis")

    parser.add_argument("--model_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="directory wehre the desired model results file is contained")

    
    parser.add_argument("--results_csv",
                    default=None,
                    type=str,
                    required=True,
                    help="Output of a model with predictions, scores, true label on an eval dataset",)
    
    parser.add_argument("--output_dir",
                    default="/scratch/sbp354/DSGA1012/Final_Project/models/merged_results",
                    type=str,
                    required=False,
                    help="metrics out put file name")

    parser.add_argument("--output_name",
                    default=None,
                    type=str,
                    required=False,
                    help="metrics out put file name")


    parser.add_argument("--output_suffix",
                    default='merged_metrics.csv',
                    type=str,
                    required=False,
                    help="suffix to append to each file name.")

    parser.add_argument("--label_name", 
                    help = "The column name for the ground truth",
                    required=False,
                    default="true_labels")    

    parser.add_argument("--pred_name", 
                    help = "The column name for the model predictions",
                    required=False,
                    default="predictions")

    parser.add_argument("--score_name", 
                    help = "The column name for the model scores",
                    required=False,
                    default="scores")

    parser.add_argument("--identities_list", 
                    help = "list describing columns with hot encoded identities to merge and conduct metrics on",
                    type = str,
                    required=False,
                    default="race")  

    parser.add_argument("--pAPI",
                    default="False",
                    type=str,
                    required=False,
                    help="modifications for pAPI")                  
    
    args = parser.parse_args()

    identities_list = args.identities_list.split(',')
    eval_dataset = ' '.join(args.identities_dir.split('/')).split()[-1]

    struct = ' '.join(args.model_dir.split('/')).split()  # This makes sure if there is / at the end its fine
    finetune_dataset = struct[-3].split('_')[0]

    if args.output_name:
        output_name = args.output_name
        output_path =  os.path.join(args.output_dir,args.output_name)
        model = struct[-2]
        loss = struct[-1]
        datasets = ['founta','civil_comments','civil_comments_0.5', 'civil_identities']
        if model in datasets:
            finetune_dataset = model
            model = loss
            loss = "plain"
    else:
        if args.results_csv == "PAPI_challenge_civil_results.csv":
            output_name = args.results_csv[:-4] + args.output_suffix
            output_path = os.path.join(args.output_dir,output_name)
            model = "pAPI"
            loss = "N/A"
            finetune_dataset = "N/A"
        else:
            datasets = ['founta','civil_comments','civil_comments_0.5', 'civil_identities']
        # struct = ' '.join(args.model_dir.split('/')).split()  # This makes sure if there is / at the end its fine
        # print(struct)
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

    results_df = pd.read_csv(os.path.join(args.model_dir, args.results_csv))
    
    
    
    identities_df =  pd.read_csv(os.path.join(args.identities_dir, args.identities_csv))
    merged_df = pd.concat([results_df, identities_df],axis=1)
    # Civil identites requires binarization from floats and filtering
    if args.identities_csv == "civil_test.csv":
        if args.results_csv == "PAPI_challenge_civil_results.csv":
            merged_df = results_df
            merged_df[args.pred_name] = (merged_df[args.score_name] > .5).astype(int).values
            toxic_labels = ['toxicity', 'severe_toxicity','obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat']
            merged_df['true_labels'] = merged_df[toxic_labels].max(axis = 1)
            merged_df['true_labels'] = np.where(merged_df['true_labels']>.5, 1, 0)
            merged_df = merged_df[~merged_df['male'].isnull()].reset_index()
            #merged_df = merged_df.drop(['index' , 'Unnamed: 0'],axis=1)
        else:
            identities_df = identities_df[~identities_df['male'].isnull()].reset_index().drop(['index'],axis=1)
            merged_df = pd.concat([results_df, identities_df],axis=1)
        #merged_df = merged_df.drop(['index', 'Unnamed: 0'],axis=1)
        df_civil_test_full = merged_df#pd.read_csv(os.path.join(args.identities_dir, args.identities_csv))
        df_civil_identities = df_civil_test_full[
            (df_civil_test_full.male >= .5) |
            (df_civil_test_full.female >= .5) |
            (df_civil_test_full.white >= .5) |
           (df_civil_test_full.black >= .5)]
        # Throw away equal examples
        identities = df_civil_identities[
            ~(df_civil_identities.male == df_civil_identities.female) | 
            ~(df_civil_identities.white == df_civil_identities.black)]
        identities['is_female'] = np.where(identities.female > identities.male, 1.0, 0.0)
        identities['black'] = np.where(identities.black > identities.white, 1.0, 0.0)
        identities_list = ['is_female', 'black']
        merged_df = identities

    merged_df = merged_df[[ args.label_name,  args.pred_name,args.score_name] + identities_list]
    metrics_dict_list = []
    print('Calculating Aggretage Metrics...')
    if args.pAPI == "False":
        merged_df['proba'] = np.where(merged_df[args.pred_name]==1, merged_df[args.score_name],1- merged_df[args.score_name])
        metrics = get_scores(merged_df, args.label_name, args.pred_name, args.score_name,'proba')
    else:
        merged_df['scores'] = np.where(merged_df[args.pred_name]==1, merged_df[args.score_name], 1- merged_df[args.score_name])
        metrics = get_scores(merged_df, args.label_name, args.pred_name, 'scores', args.score_name) # Because we get proba as defualt so its swapped here
    metrics = get_scores(merged_df, args.label_name, args.pred_name, args.score_name)
    metrics['metrics_condition'] = 'none'
    metrics_dict_list.append(metrics)
    
    #print(metrics)
    #print(merged_df.head(5))

    for identity in identities_list:
        if args.pAPI == "False":
            print('Calculating {} = 1 Metrics...'.format(identity))
            metrics = get_scores(merged_df[merged_df[identity]==1], args.label_name, args.pred_name, args.score_name,'proba')
            metrics['metrics_condition'] = '{}_1'.format(identity)
            metrics_dict_list.append(metrics)
            #print(metrics)
    #
            print('Calculating {} = 0 Metrics...'.format(identity))
            metrics = get_scores(merged_df[merged_df[identity]==0], args.label_name, args.pred_name, args.score_name, 'proba')
            metrics['metrics_condition'] = '{}_0'.format(identity)
            metrics_dict_list.append(metrics)
           # print(metrics)
        else:
            print('Calculating {} = 1 Metrics...'.format(identity))
            metrics = get_scores(merged_df[merged_df[identity]==1], args.label_name, args.pred_name,'proba', args.score_name)
            metrics['metrics_condition'] = '{}_1'.format(identity)
            metrics_dict_list.append(metrics)
            # print(metrics)
    
            print('Calculating {} = 0 Metrics...'.format(identity))
            metrics = get_scores(merged_df[merged_df[identity]==0], args.label_name, args.pred_name, 'proba', args.score_name)
            metrics['metrics_condition'] = '{}_0'.format(identity)
            metrics_dict_list.append(metrics)
                        
    metrics_df = pd.DataFrame(metrics_dict_list)
    metrics_df['model'] = model
    metrics_df['loss'] = loss
    metrics_df['fine_tune_data'] = finetune_dataset
    metrics['eval_data'] = eval_dataset
    
    
    #print(results_df)
    print(metrics_df)
    metrics_df.to_csv(output_path)

if __name__ == "__main__":
    main()    
