#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:41:10 2021

@author: eljas
"""
##############################################################################
'''
Script for how accurate the "mean prediction" is.
Notice that mean here refers to the normalized data; this mean should be roughly 0 (as we draw this
sample from the overall population, which has mean of 0 by constructon).
You could also call this "0 prediction", indeed using 0 instead of the mean per crossval-split hardly any difference
'''
##############################################################################
# import argparse
import logging
import os
import h5py as h5
import numpy as np
# import pandas as pd
# import time as t
# import pathlib
import json
import pathlib
import pandas as pd

# from src.KNN_PCA_enabled import bigKNN_PCA_enabled
# from sklearn.neighbors import KNeighborsRegressor#, NearestNeighbors
# from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from torch.utils.data import Subset
from src.utils import Height_Statistics_np, hdf5Dataset, dataset_split

def get_SNPs(data_path,file):
    hf = h5.File(os.path.join(data_path,file),'r')
    return hf['X/SNP'][:,:]

def get_metadata(data_path,file):
    hf = h5.File(os.path.join(data_path,file),'r')
    sex = hf['X/Sex'][:]
    age = hf['X/Age'][:]
    return np.stack([sex,age],axis=0)

def get_phenotype(data_path,file):
    hf = h5.File(os.path.join(data_path,file),'r')
    return hf['y/Height'][:]

def prepare_hdf5Data(data_path,X_file,y_file):
    y_metadata = get_metadata(data_path,y_file)
    y_true = get_phenotype(data_path,y_file)
    X = get_SNPs(data_path,X_file)
    y = get_SNPs(data_path,y_file)
    X_phenotype = get_phenotype(data_path,X_file)
    X_metadata = get_metadata(data_path,X_file)
    scorer = Height_Statistics_np()
    # if not True:
    # X_phenotype = scorer.calculate_height(X_phenotype,X_metadata[0],X_metadata[1])
    
    return y_metadata, y_true, X, y, X_phenotype, X_metadata

def evaluate_metrics(y_true,predictions,scorer=None,y_metadata=None):
    if y_metadata is not None:
        phenotype = scorer.calculate_height(y_true,y_metadata[0],y_metadata[1])
    return({'MSE':mean_squared_error(phenotype,predictions),
            'MAE':mean_absolute_error(phenotype,predictions),
            'r':np.corrcoef(phenotype,predictions)[0,1],
            'r2':r2_score(phenotype,predictions)})

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

if __name__ == "__main__":
    n_splits = 5
    random_state = 47
    print("mean prediction accuracy evaluation")
    logging.info("mean prediction accuracy evaluation")
    logging.info("dataset: 10k_bellot")
    logging.info(f"n_splits: {n_splits}")
    logging.info(f"random_state: {random_state}")

    # Data directory
    # data_path = '/home/eljas/ownCloud/SLR/HeightPrediction/eljas_2/toydata'
    # data_path = '/local0/scratch/roelline/files'
    data_path = '/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data/'

    # Output directory
    out_path = pathlib.Path(__file__).resolve().parents[1] / 'output'
    
    # Files
    # X_file = 'Xy_small_toy_train.hdf5'
    # y_file = 'Xy_small_toy_val.hdf5'
    X_file = f'Xy_train_10k_bellot.hdf5'
    y_file = f'Xy_val_10k_bellot.hdf5'
    
    
    scorer = Height_Statistics_np()    
    # for each of the n_splits splits, create the train-indices ad test-indices 
    # import dataset to make nice split
    dataset = hdf5Dataset(data_path,X_file)
    split_lys = dataset_split(dataset,n_splits,random_state)
    
    # Document these to check if it matches with other experiments
    logging.info(f"split_lys[0]['train_idxs'][:5]: {split_lys[0]['train_idxs'][:5]}")
    logging.info(f"split_lys[0]['test_idxs'][:5]: {split_lys[0]['test_idxs'][:5]}")
    
    # get dataset as np array
    y_metadata, y_true, X, y, X_phenotype, X_metadata = prepare_hdf5Data(data_path, X_file, y_file)
    del y_metadata, y_true,y
    # create list to store the evaluation of each of the n_splits splits
    evaluate_metrics_lys = []
    
    # run n_splits-crossvalidation
    for i,split in enumerate(split_lys):
        print(f"split: {i}")
        # Split to training and test dataset
        train_dataset = Subset(dataset, split['train_idxs'])
        test_dataset = Subset(dataset, split['test_idxs'])
        
        # Split to training and test dataste
        train_X = X[split['train_idxs']]
        train_X_phenotype = X_phenotype[split['train_idxs']]
        train_X_metadata = X_metadata[:,split['train_idxs']]
        
        test_X = X[split['test_idxs']]
        test_X_phenotype = X_phenotype[split['test_idxs']]
        test_X_metadata = X_metadata[:,split['test_idxs']]        
        
        # mean of train dataset as prediction (roughly 0)
        train_mean = np.mean(train_X_phenotype)
        
        # Predictions
        predictions = np.full(len(test_X_phenotype),train_mean)
        
        # if args.unadjusted:
        predictions = scorer.calculate_height(predictions,test_X_metadata[0],test_X_metadata[1])
        
        # evaluate metric using test dataset and predicted mean
        evaluate_metrics_lys.append(evaluate_metrics(test_X_phenotype,predictions,scorer,test_X_metadata))
        
    logging.info(evaluate_metrics_lys)
    print(evaluate_metrics_lys)

    df = get_df_from_cv_results(evaluate_metrics_lys)
    
    with open(os.path.join(out_path,"evaluate_metrics_10k_bellot_mean_prediction.csv"), 'w') as f:
        df.to_csv(f,index=False)
    f.close()