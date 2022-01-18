#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 18:42:55 2021

@author: eljas
"""

'''
Compute here intermediates of a PCA workflow:
    -covariance_matrix
    -inverted_covariance_matrix (just for mahalanobis distance)
    -explained_variance
    -components
'''

import numpy as np
import h5py as h5
import os
import argparse
import pathlib
import logging
import torch
from torch.utils.data import Subset

from src.utils import hdf5Dataset, dataset_split
from src.PCA import  bigPCA

def get_SNPs(data_path,file):
    hf = h5.File(os.path.join(data_path,file),'r')
    return hf['X/SNP'][:,:]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--dataset',
      type=str,
      default='toy',
      help='Which dataset to use',
      required=False,
    )    

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='cpu or gpu to use. make sure gpu is available if wanting one.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=400,
        help='Batch size for data loader. For 10000 SNPs, 400 should not be exceeded.'
    )
    
    parser.add_argument(
        '-o', '--intermediate_dir',
        default=pathlib.Path(__file__).resolve().parents[1] / 'intermediates',
                                                               #/ name,
        type=str,
        help='intermediate_dir for storing the results.'
    )

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
    
    ### Set data paths
    if args.locality:
        data_path = '/home/eljas/ownCloud/SLR/HeightPrediction/eljas_2/toydata'
    else:
        # data_path = '/home/roelline/HeightPrediction/eljas_2/toydata'
        data_path = '/local0/scratch/roelline/files'

    logging.info(f'Dataset: {args.dataset}')
    logging.info(f'Device: {args.device}')    
    logging.info(f'Batch size for DataLoader: {args.batch_size}')   
  
    ### IMPORTANT: these parameters must be set like this to ensure the split
    ### of the data is the same as will be used for KNN 5-Fold CV
    N_SPLITS = 5
    RANDOM_STATE = 47

    # X_file = 'Xy_small_toy_train.hdf5'
    # X_file = 'Xy_train_10k_bellot.hdf5'
    X_file = 'Xy_toy_train.hdf5'
    dataset = hdf5Dataset(data_path,X_file)

    split_lys = dataset_split(dataset,N_SPLITS,random_state=RANDOM_STATE)
    
    for i,split in enumerate(split_lys):
        
        train_dataset = Subset(dataset, split['train_idxs'])
        # test_dataset = Subset(dataset, split['test_idxs'])
        
        ### Create Files for Covariance Matrix, Inverse Covariance Matrix, Variance Explained, PC's
        intermediate_dir_filename = f'pca_intermediates_cvsplit_{i}_dataset_{args.dataset}_random_state_{RANDOM_STATE}_device_{args.device}.hdf5'
        
        if not os.path.exists(os.path.join(args.intermediate_dir, intermediate_dir_filename)):
    
            ### Create hdf5Dataset
                 
            ### Instantiate and fit bigPCA
            model = bigPCA(batch_size = args.batch_size, device = args.device)
            model.fit(train_dataset)
            
            ### Covariance Matrix
            cov_bigPCA = model.covariance_matrix_.numpy()
            
            ### For Mahalanobis: also go for the inverse!
            inv_cov_bigPCA = torch.linalg.inv(model.covariance_matrix_).numpy()
        
            ### eigenvalues
            expl_var_bigPCA = model.explained_variance_.numpy()
        
            ### eigenvectors
            components_bigPCA = model.components_.numpy()
            
            ### Save to compressed file
            with h5.File(os.path.join(args.intermediate_dir, intermediate_dir_filename), 'w') as f:
                f.create_dataset("covariance_matrix", data=cov_bigPCA)#, dtype='float64')
                f.create_dataset("inverted_covariance_matrix", data=inv_cov_bigPCA)
                f.create_dataset("explained_variance", data=expl_var_bigPCA)
                f.create_dataset("components", data=components_bigPCA)
    
                f.close()
            
        else:
            logging.warning(
                f'Skipping {intermediate_dir_filename} and others because some already exist.'
            )
    


