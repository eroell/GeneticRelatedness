import argparse
import logging
import os
import pathlib
import numpy as np
import pandas as pd

from src.KNN_PCA_enabled import bigKNN_PCA_enabled
from torch.utils.data import Subset

from src.utils import Height_Statistics_np, hdf5Dataset, dataset_split

'''
Script for testing the custom KNN regression implementations.
This script tests this implementations on synthetic dataset 1.
Consult the dataset creation scripts in this folder for details on synthetic dataset 1.

This testing script demonstrates that the custom KNN regression works as
expected without error on a synthetic dataset of well-separated clusters.
'''

def get_df_from_cv_results(crossvalscore):
    # Writes convenient CVs from the crossvalidation results
    # Is a bit hacky, but does not need to change interfaces of the established
    # functions
    
    df = pd.DataFrame({'Split':[],'MSE':[],'MAE':[],'r': [],'r2': []})
    split_nr = 1
    for dic in crossvalscore:
        mse_mae_r_r2 = [str(split_nr)]
        split_nr += 1
        for key in dic:
            mse_mae_r_r2.append(dic[key])
        df.loc[len(df.index)] = mse_mae_r_r2
        
    return df

def cross_val_score_pca(model,dataset,n_splits=5,random_state=47,
                        store_dist_mat=False,n_components=None,data_set_name=None,
                        n_neighbors=None,w_state=None,metric=None,p=None,
                        adjusted=None, device=None, intermediate_dir=None, distance_matrix_dir=None,
                        ):
    '''
    perform crossvalidation of K-nearest-neighbours estimator for the Biobank data and height.
    Input:
    - model: A bigKNN object
    - dataset: a hdf5File
    - n_splits: n_splits-Fold crossvalidation
    -random_state: the random state used for the split generation
    -rest: arguments to store the results in uniquely named filenames
    '''
    # for each of the n_splits splits, create the train-indices and test-indices 
    split_lys = dataset_split(dataset,n_splits,random_state)
    # create list to store the evaluation of each of the n_splits splits
    evaluate_metrics_lys = []
    # run n_splits-crossvalidation
    for i,split in enumerate(split_lys):
        
        train_dataset = Subset(dataset, split['train_idxs'])
        test_dataset = Subset(dataset, split['test_idxs'])
        print("split train_idxs: ", split['train_idxs'])
        print("split test_idxs: ", split['test_idxs'])
        
        model.fit(train_dataset,test_dataset)#,components=components)
        model.predict()
        evaluate_metrics_lys.append(model.evaluate_metrics())
        
    return evaluate_metrics_lys

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_components',
        type=int,
        default=0,
        help='number of principal components to be considered'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='synthetic_1',
        help='Which dataset to use',
        required=False,
    )

    parser.add_argument(
        '-nn', '--n_neighbors',
        type=int,
        default=2,
        help='The number of nearest neighbours'
    )

    parser.add_argument(
        '--w_state',
        type=str,
        default='None',
        help='Use to weight the positions'
    )

    parser.add_argument(
        '--metric',
        type=str,
        default='euclidean',
        help='The metric used to find the KNN'
    )

    parser.add_argument(
        '--p',
        type=float,
        default=2,
        help='Parameter p used for fractional distance measure'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default='uniform',
        help='Use uniform or distance weighted KNN'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='Batch size for data loader. For 10000 SNPs, 400 should not be exceeded.'
    )
  
    args = parser.parse_args()

    logging.info(f'Dataset: {args.dataset}')
    logging.info(f'n_neighbors: {args.n_neighbors}')
    logging.info(f'w_state: {args.w_state}')
    logging.info(f'Metric: {args.metric}')
    logging.info(f'p: {args.p}')
    logging.info(f'weights: {args.weights}')
    logging.info(f'Batch size for DataLoader: {args.batch_size}')
        
    data_path = pathlib.Path(__file__).resolve().parents[0]
    
    X_file = f'{args.dataset}_train.hdf5'
    y_file = f'{args.dataset}_val.hdf5'

    model = bigKNN_PCA_enabled(n_neighbors=args.n_neighbors,
                metric=args.metric,
                p=args.p,
                weights=args.weights,
                # adjusted=args.unadjusted,
                data_path=data_path,
                batch_size=args.batch_size,
                # device=args.device,
                # n_jobs=args.n_jobs,
                scorer=Height_Statistics_np()#,
                # w=w
                )

    ### Load dataset
    train_dataset = hdf5Dataset(data_path,X_file)

    test_dataset = hdf5Dataset(data_path,y_file)

    ### Doing n-fold Crossvalidation
    crossvalscore = cross_val_score_pca(model,
                                    train_dataset,
                                    n_splits = 5,
                                    # store_dist_mat = True,
                                    # n_components=args.n_components,
                                    data_set_name=args.dataset,
                                    n_neighbors=args.n_neighbors,
                                    w_state=args.w_state,
                                    metric=args.metric,
                                    p = args.p,
                                    # adjusted = args.unadjusted,
                                    # device = args.device,
                                    # intermediate_dir = args.intermediate_dir,
                                    # distance_matrix_dir=args.distance_matrix_dir,
                                    )
    
    ### Print crossvalscore
    logging.info(crossvalscore)    

