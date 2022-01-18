import argparse
import logging
import os
# import pathlib
# import json
import h5py as h5
import numpy as np
# import pandas as pd

from src.KNN import bigKNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from src.utils import Height_Statistics_np, hdf5Dataset, dataset_split, cross_val_score

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--dataset',
      type=str,
      default='toy_small',
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
        '--weights',
        type=str,
        default='uniform',
        help='Use uniform or distance weighted KNN'
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
        '-f', '--force',
        action='store_true',
        help='If set, overwrites all files. Else, skips existing files.'
    )

    parser.add_argument(
        '-nj', '--n_jobs',
        type=int,
        default=4,
        help='The number of parallel processes to load the data. Default is 4.'
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

    # Basic log configuration to ensure that we see where the process
    # spends most of its time.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )

    logging.info(f'Dataset: {args.dataset}')
    logging.info(f'Neighbors: {args.n_neighbors}')
    logging.info(f'Weight: {args.weights}')
    logging.info(f'Metric: {args.metric}')
    logging.info(f'p: {args.p}')
    logging.info(f'Adjusted: {args.unadjusted}')

    ### Set data paths. Modified to run on my machine
    # data_path = '/home/michael/ETH/data/HeightPrediction'
    if args.locality:
        data_path = '/home/eljas/ownCloud/SLR/HeightPrediction/eljas_2/toydata'
    else:
        data_path = '/home/roelline/HeightPrediction/eljas_2/toydata'
        
    X_file = 'Xy_small_toy_train.hdf5'
    y_file = 'Xy_small_toy_val.hdf5'
    
    ## Run sklearn.neighbors.KNeighborsRegressor
    # Import data, prepare for KNeighborsRegressor compatibility
    # KNeighborsRegressor accepts for minkowski accepts only values of p >= 1
    if args.metric == 'fractional':
        logging.info('Skip KNeighborsRegressor for fractional distance')

    else:
        # Load data for sklearn.neighbors.KNeighborsRegressor
        X = get_SNPs(data_path,X_file)
        X_phenotype = get_phenotype(data_path,X_file)
        X_metadata = get_metadata(data_path,X_file)
        scorer = Height_Statistics_np()
        # recalculate height in cm
        if not args.unadjusted:
            X_phenotype = scorer.calculate_height(X_phenotype,X_metadata[0],X_metadata[1])
        # Instantiate model
        model_base = KNeighborsRegressor(n_neighbors=args.n_neighbors,
                                          weights=args.weights,
                                          metric=args.metric)
        # create dataset split (same as used for cross_val_score of bigKNN)
        split_lys = dataset_split(X_phenotype, n_splits = 2)
        
        # Run thorugh all n splits
        for split in split_lys:
            X_train = X[split['train_idxs'],:]
            X_test = X[split['test_idxs'],:]
            
            X_train_phenotype = X_phenotype[split['train_idxs']]
            X_test_phenotype = X_phenotype[split['test_idxs']]
            
            X_train_metadata = X_metadata[:,split['train_idxs']]
            X_test_metadata = X_metadata[:,split['test_idxs']] 
            
            model_base.fit(X_train,X_train_phenotype)
            predictions = model_base.predict(X_test)

            if args.unadjusted:
                predictions = scorer.calculate_height(predictions,X_test_metadata[0],X_test_metadata[1])
            print('KNN: ',evaluate_metrics(X_test_phenotype,predictions,scorer=scorer,y_metadata=X_test_metadata))


    
    ### Prepare src.bigKNN    
    w = np.ones(100)#/100
    # df = pd.read_csv(os.path.join(data_path,'GWAS_p_vals.csv'),index_col='SNP')
    # top100 = np.loadtxt(os.path.join(data_path,'top100snps.txt'),dtype=str)
    # w = df[df.index.isin(top100)]['P'].values
    # # w = np.abs(w)/np.sum(np.abs(w))
    # w = (1./w)/np.sum(1./w)
    
    model = bigKNN(n_neighbors=args.n_neighbors,
                weights=args.weights,
                metric=args.metric,
                n_jobs=args.n_jobs,
                adjusted=args.unadjusted,
                scorer=Height_Statistics_np(),
                data_path=data_path,
                w=w,
                p=args.p)
    
    ### Load dataset
    train_dataset = hdf5Dataset(data_path,X_file)

    
    crossvalscore = cross_val_score(model, train_dataset,n_splits = 2)
    
    print(crossvalscore)


