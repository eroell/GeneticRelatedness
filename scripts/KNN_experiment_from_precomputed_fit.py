#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 09:32:25 2021

@author: eljas
"""

'''
Use (expensive) precomputed sorted distance matrices to do KNN-Regression
with different number of neighbors, and uniform/distance input.

This is now a very cheap operation
'''

import argparse
import logging
import os
import pathlib
import h5py as h5
import numpy as np
import pandas as pd
import time as t

from src.KNN_PCA_enabled import bigKNN_PCA_enabled
from src.utils import Height_Statistics_np, hdf5Dataset, dataset_split, cross_val_score

def load_data(intermediate_dir, fylename):
    ### Load expensive precomputed data; this code below retrieves it.
    precomputed = np.load(os.path.join(intermediate_dir,fylename))
    distance_matrix_ = precomputed['distance_matrix_']
    neighbor_metadata_ = precomputed['neighbor_metadata_']
    y_metadata_ = precomputed['y_metadata_']
    
    return distance_matrix_, neighbor_metadata_, y_metadata_
    
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

if __name__ == '__main__':
    ### Read in Parameters: Which Setting, how many neighbors, uniform/weighted/both
    parser = argparse.ArgumentParser()

    ### NEW:
    parser.add_argument(
        '--n_neighbors_reg',
        type = list,
        default = [1,10,20,100,500],
        help='number of neighbors to be selected now'
    )
    parser.add_argument(
        '--weight_reg',
        type = str,
        default = 'both',
        help='weight to be selected now: "uniform", "distance", or "both" '
    )    
    
    parser.add_argument(
        '--dataset_name',
        type = str,
        default = "10k_bellot",
        help="name of dataset, toy or 10k_bellot"
    )
    parser.add_argument(
        '--n_components',
        type=int,
        default=10,
        help='number of principal components to be considered'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='10k_bellot',
        help='Which dataset to use',
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
        default='BETA',
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
        '--batch_size',
        type=int,
        default=400,
        help='Batch size for data loader. For 10000 SNPs, 400 should not be exceeded.'
    )
    
    parser.add_argument(
        '-nj', '--n_jobs',
        type=int,
        default=4,
        help='The number of parallel processes to load the data. Default is 4.'
    )
    
    
    parser.add_argument(
        '--intermediate_dir',
        default=pathlib.Path(__file__).resolve().parents[1] / 'intermediates',
        type=str,
        help='path to precomputed pca intermediates (PCs)'
    )
    
    parser.add_argument(
        '--distance_matrix_dir',
        default=pathlib.Path(__file__).resolve().parents[1] / 'distance_matrices',
        type=str,
        help='where to store the computed distance matrices'
    )

    parser.add_argument(
        '-o', '--output',
        default=pathlib.Path(__file__).resolve().parents[1] / 'output',
                                                               #/ name,
        type=str,
        help='Output path for storing the evaluation metrics.'
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
    parser.set_defaults(localn_neighbors_regity=True)
	##############################################################
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    
    logging.info(f'n_components: {args.n_components}')
    logging.info(f'Dataset: {args.dataset}')
    logging.info(f'n_neighbors: {args.n_neighbors}')
    logging.info(f'w_state: {args.w_state} (will be set to None internally if metric==mahalanobis')
    logging.info(f'Metric: {args.metric}')
    logging.info(f'p: {args.p}')
    logging.info(f'weights: {args.weights}')
    logging.info(f'Adjusted: {args.unadjusted}')
    logging.info(f'Device: {args.device}')
    logging.info(f'Batch size for DataLoader: {args.batch_size}')
    logging.info(f'n_jobs: {args.n_jobs}')
    logging.info(f'n_neighbors_reg: {args.n_neighbors_reg}')
    logging.info(f'weight_reg: {args.weight_reg}')

    # Select weight list
    if args.weight_reg == "uniform":
        weight_reg = ["uniform"]
    elif args.weight_reg == "distance":
        weight_reg = ["distance"]
    elif args.weight_reg == "both":
        weight_reg = ["uniform", "distance"]
        
    ### For each n_neighbors
    for n_neighbors in args.n_neighbors_reg:
        
        ### For each weight
        for weight in weight_reg:
            
            tick = t.time()
            ### For each split do
            evaluate_metrics_lys = []
            for split in [0,1,2,3,4]:
                fylename = f"distance_matrix_cv_split_{split}_n_components_{args.n_components}_dataset_{args.dataset_name}_n_neighbors_{args.n_neighbors}_wstate_{args.w_state}_metric_{args.metric}_p_{args.p}_adjusted_{args.unadjusted}_device_{args.device}.npz"
                
                # Load data
                distance_matrix_, neighbor_metadata_, y_metadata_ = load_data(args.distance_matrix_dir,fylename)
        
                ### "Fit" KNN Regressor with this data
                # I only want the predict and evaluate_metrics method of bigKNN_PCA_enabled.
                # Notice only adjusted and scorer are really influencing the computation.
                model = bigKNN_PCA_enabled(n_neighbors=args.n_neighbors,
                                           metric=args.metric,
                                           p=args.p,
                                           weights=args.weights,
                                           adjusted=args.unadjusted,
                                           # data_path=data_path,
                                           batch_size=args.batch_size,
                                           device=args.device,
                                           n_jobs=args.n_jobs,
                                           scorer=Height_Statistics_np(),
                                           # w=w
                )
                
                ### "Fit" model with the data
                model.distance_matrix_ = distance_matrix_[:,:n_neighbors]
                model.neighbor_metadata_ = neighbor_metadata_[:,:,:n_neighbors]
                model.y_metadata_ = y_metadata_
                
                ### Do prediction
                model.predict()
                
                ### Do metric evaluation
                evaluate_metrics_lys.append(model.evaluate_metrics())
                
            ### Write csv
            filename = f'evaluate_metrics_{args.dataset}_n_components_{args.n_components}_n_neighbours_{n_neighbors}_wstate_{args.w_state}_weights_{weight}_metric_{args.metric}_p_{args.p}_adjusted_{args.unadjusted}_njobs_{args.n_jobs}_batchsize_{args.batch_size}_device_{args.device}.csv'
            output_filename = os.path.join(args.output, filename)

            df = get_df_from_cv_results(evaluate_metrics_lys)  
            # print(output_filename)              
            # print(df)
            tock = t.time()
            logging.info(f'Clock time for {n_neighbors} neighbors, weight {weight}: {tock-tick}')
            if not os.path.exists(output_filename):
                with open(output_filename, 'w') as file:
                    file.write(f'cv_{args.dataset}_n_components_{args.n_components}_n_neighbours_{n_neighbors}_weights_{weight}_metric_{args.metric}_p_{args.p}_adjusted_{args.unadjusted}_njobs_{args.n_jobs}_batchsize_{args.batch_size}_device_{args.device}\n')
                
                    df.to_csv(file,index=False)
                
                file.close()
            else:
                logging.info(f'skipped, since file already exists: {output_filename}')
                print(f'skipped, since file already exists: {output_filename}')
                
                
                
                
