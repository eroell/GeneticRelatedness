#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 20:21:26 2021

@author: eljas
"""

##############################################################################
##############################################################################
'''
A main script for running a dataset at once (loading it entirely into memory)
'''

##############################################################################
##############################################################################
import argparse
import logging
import os
import h5py as h5
import numpy as np
import pandas as pd
import time as t
import pathlib

from src.KNN_PCA_enabled import bigKNN_PCA_enabled
from sklearn.neighbors import KNeighborsRegressor#, NearestNeighbors
# from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


from src.utils import Height_Statistics_np, hdf5Dataset

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

def evaluate_metrics(y_true,predictions,scorer=None,y_metadata=None):
    if y_metadata is not None:
        phenotype = scorer.calculate_height(y_true,y_metadata[0],y_metadata[1])
    return({'MSE':mean_squared_error(phenotype,predictions),
            'MAE':mean_absolute_error(phenotype,predictions),
            'r':np.corrcoef(phenotype,predictions)[0,1],
            'r2':r2_score(phenotype,predictions)})

def prepare_hdf5Data(data_path,X_file,y_file):
    y_metadata = get_metadata(data_path,y_file)
    y_true = get_phenotype(data_path,y_file)
    X = get_SNPs(data_path,X_file)
    y = get_SNPs(data_path,y_file)
    X_phenotype = get_phenotype(data_path,X_file)
    X_metadata = get_metadata(data_path,X_file)
    scorer = Height_Statistics_np()
    if not args.unadjusted:
        X_phenotype = scorer.calculate_height(X_phenotype,X_metadata[0],X_metadata[1])
    return y_metadata, y_true, X, y, X_phenotype, X_metadata

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--dataset',
      type=str,
      default='small_toy',
      help='Which dataset to use [small_toy, toy, 10k_bellot]',
      required=False,
    )

    parser.add_argument(
        '-nn', '--n_neighbors',
        type=int,
        default=1000,
        help='The number of nearest neighbours'
    )

    parser.add_argument(
        '--w_state',
        type=str,
        default=None,
        help='BETA p or None'
    )

    parser.add_argument(
        '--metric',
        type=str,
        default='manhattan',
        help='The metric used to find the KNN'
    )

    parser.add_argument(
        '--p',
        type=float,
        default=1,
        help='Parameter p used for fractional distance measure'
    )
    
    parser.add_argument(
        '--unadjusted',
        action='store_false',
        help='Average the adjusted or raw phenotypes'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='cpu or gpu to use. make sure gpu is available if wanting one.'
    )
    
    parser.add_argument(
        '--intermediate_dir',
        default=pathlib.Path(__file__).resolve().parents[1] / 'intermediates',
        type=str,
        help='path to precomputed pca intermediates (PCs)'
    )
    
    #parser.add_argument(
    #    '-nj', '--n_jobs',
    #    type=int,
    #    default=3,
    #    help='The number of parallel processes to load the data. Default is 3.'
    #)
    ##############################################################
    ### These arguments are used to run this code easily on my
    ### machine and the borgwardt server; only 1 extra argument must be
    ### given when run on the borgwardt server.
    parser.add_argument(
        '--local',
        dest = "locality",
        action='store_true',
        help='If local, ie my machine'
    )
    parser.add_argument(
        '--non-local',
        dest = "locality",
        action='store_false',
        help='If non-local, ie server'
    )
    parser.set_defaults(locality=True)
	##############################################################

    args = parser.parse_args()

    # Basic log configuration to ensure that we see where the process
    # spends most of its time.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )

    logging.info(f'Dataset: {args.dataset}')
    logging.info(f'Neighbors: {args.n_neighbors}')
    logging.info(f'w_state: {args.w_state}')
    logging.info(f'Metric: {args.metric}')
    logging.info(f'p: {args.p}')
    logging.info(f'Device: {args.device}')
    logging.info(f'Adjusted: {args.unadjusted}')
    #logging.info(f'n_jobs: {args.n_jobs}')

    tick_initial = t.time()
    ### Set data paths. Modified to run on my machine
    if args.locality:
        data_path = '/home/eljas/ownCloud/SLR/HeightPrediction/eljas_2/toydata'
    else:
        #data_path = '/home/roelline/HeightPrediction/eljas_2/toydata'
        #data_path = '/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data'
        data_path = '/local0/scratch/roelline/files'
        
    scorer = Height_Statistics_np()
        
    if args.dataset == "small_toy":
        X_file = 'Xy_small_toy_train.hdf5'
        y_file = 'Xy_small_toy_val.hdf5'
    elif args.dataset == "toy":
        X_file = 'Xy_toy_train.hdf5'
        y_file = 'Xy_toy_val.hdf5'      
    elif args.dataset == "10k_bellot":    
        X_file = f'Xy_train_{args.dataset}.hdf5'
        y_file = f'Xy_val_{args.dataset}.hdf5'
        
    # Prepare data for NearestNeighbors
    logging.info("Loading Data")
    tick_before_loading = t.time()
    y_metadata, y_true, X, y, X_phenotype, X_metadata = prepare_hdf5Data(data_path, X_file, y_file)
    time_loading = t.time() - tick_before_loading
    
    # Load inverse Covariance Matrix for Mahalanobis
    # intermediate_dir = pathlib.Path(__file__).resolve().parents[1] / 'intermediates'
    # pca_fyle = 'pca_intermediates_small_device_cpu.hdf5'
    # with h5.File(os.path.join(intermediate_dir, pca_fyle), 'r') as hf:
        # hf = h5.File(os.path.join(intermediates_path, "cov_mat_10k_bellot_device_cpu_batchsize_400.hdf5"), 'r')
        # cov_loaded = np.array(hf['/covariance_matrix'])
        # VI = np.array(hf['/inverted_covariance_matrix'])
        # expl_var_loaded = np.array(hf['/explained_variance'])
        # components = np.array(hf['/components'])[:,:n_components]
        # hf.close()
    
    if not args.locality and args.w_state == "BETA":
        if args.dataset == "small_toy" or args.dataset == "toy":
            df = pd.read_csv(os.path.join(data_path,'GWAS_p_vals.csv'),index_col='SNP')
            top100 = np.loadtxt(os.path.join(data_path,'top100snps.txt'),dtype=str)
        elif args.dataset == "10k_bellot":
            df = pd.read_csv(os.path.join(data_path,'GWAS_p_vals.csv'),index_col='SNP')
            top100 = np.loadtxt(os.path.join(data_path,'SNP_ids.txt'),dtype=str)
            
        w = df[df.index.isin(top100)]['BETA'].values
        # w = np.abs(w)/np.sum(np.abs(w))
        # w = (1./w)/np.sum(1./w) ### Notice this makes all values extremely small: 10e-20 for 100 SNPs!
        
    elif args.w_state is None :
        if args.dataset == "small_toy" or args.dataset == "toy":
            w = np.ones(100)
        elif args.dataset == "10k_bellot":
            w = np.ones(10000)

    if args.w_state == "distance" and (args.metric == "minkowski" or args.metric == "manhattan" or args.metric == "euclidean"):
        p = args.p
        if args.metric == "manhattan":
            p = 1
        elif args.metric == "euclidean":
            p = 2
        logging.info(f"wminkowski, w_state: {args.w_state}, p: {p}")
        model_base = KNeighborsRegressor(n_neighbors = args.n_neighbors, metric="wminkowski",p=p,metric_params={'w':w})
    else:
        model_base = KNeighborsRegressor(n_neighbors = args.n_neighbors, metric=args.metric)#,p=args.p)

    logging.info("Fitting")
    tick_before_fit = t.time()
    model_base.fit(X,X_phenotype)
    time_fit = t.time()-tick_before_fit

    logging.info("Predicting")
    tick_before_predict = t.time()
    predictions = model_base.predict(y)
    time_predict = t.time()-tick_before_predict
    
    if args.unadjusted:
        predictions = scorer.calculate_height(predictions,y_metadata[0],y_metadata[1])
        
    ### Evaluate metrics of this prediction
    print(evaluate_metrics(y_true,predictions,scorer,y_metadata))
    tock_end = t.time()
    
    logging.info(f"Total time: {tock_end - tick_initial}")
    logging.info(f"Time for fitting: {time_fit}")
    logging.info(f"Time for prediction: {time_predict}")
    
