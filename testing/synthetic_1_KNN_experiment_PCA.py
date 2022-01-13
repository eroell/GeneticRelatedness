import argparse
import logging
import pathlib
import matplotlib.pyplot as plt

from src.KNN_PCA_enabled import bigKNN_PCA_enabled

from src.utils import Height_Statistics_np, hdf5Dataset
from src.PCA import bigPCA

'''
Script for testing the custom PCA and custom KNN regression implementations.
This script tests these two implementations on synthetic dataset 1.
Consult the dataset creation scripts in this folder for details on synthetic dataset 1.

This testing script demonstrates that custom PCA and custom KNN regression
interact as they should. That is, the concatenation of dimensionality reduction
prior to KNN regression retains correct KNN regression on a synthetic dataset
with separable clusters even in less than full dimensions.
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
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
        default=10,
        help='The number of nearest neighbours'
    )

    parser.add_argument(
        '--w_state',
        type=str,
        default=None,
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
        default=15,
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

    ### run on my machine
    data_path = '/home/eljas/ownCloud/SLR/HeightPrediction/eljas_2/testing'
    data_path = pathlib.Path(__file__).resolve().parents[0]

    ##########################################################################
    ########################## PCA ###########################################
    ##########################################################################
    X_file = f'{args.dataset}_train.hdf5'
    y_file = f'{args.dataset}_val.hdf5'  

    train_dataset = hdf5Dataset(data_path,X_file)
             
    ### Instantiate and fit bigPCA
    model = bigPCA(batch_size = args.batch_size)
    model.fit(train_dataset)
        
    ### Means
    # means_bigPCA = model.means_.cpu().numpy()
        
    ### Covariance Matrix
    # cov_bigPCA = model.covariance_matrix_.cpu().numpy()
    
    ### eigenvalues
    # expl_var_bigPCA = model.explained_variance_.cpu().numpy()
    
    ### eigenvectors
    components_bigPCA = model.components_.cpu().numpy()
        
    ##########################################################################
    ##########################################################################
    ##########################################################################
    
    ##########################################################################
    #################### KNN with precomputed PCA ############################
    ##########################################################################    
    
    for n_components in [2,1,0]:
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
    
        model.fit(train_dataset,test_dataset,components=components_bigPCA[:,:n_components])
        
        model.predict()
        print('\n----------------------------\n',
              f'KNN Prediction Performance on dataset {args.dataset} using {n_components} principal component(s):\n',
              model.evaluate_metrics(),
              '\n----------------------------\n')
    