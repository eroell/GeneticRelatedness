##############################################################################
##############################################################################
'''
A main script for benchmarking the KNN.py module.
Mainly used this for finding slow actions in the KNN.py module.

Examines:
    -Runtime

For (all combinations of)
    -Metric: Minkowski (p=2), Minkowski(p=1), Minkowski(p=0.5), cosine
    -Device: CPU, GPU
        -GPU only testable on slurm
    -Weights: Uniform, Distance
        -Weighted only testable on slurm or bs-borgwardt01 server

Using the dataset
    -Bellot (only a small subset of it; Bellot because 10'000 features)
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

from src.KNN_benchmark import bigKNN
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
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

    # parser.add_argument(
    #     '--weights',
    #     type=str,
    #     default='uniform',
    #     help='Use uniform or distance weighted KNN'
    # )

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
        default=3,
        help='The number of parallel processes to load the data. Default is 3.'
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

    # logging.info(f'Dataset: {args.dataset}')
    # logging.info(f'Neighbors: {args.n_neighbors}')
    # logging.info(f'Weight: {args.weights}')
    # logging.info(f'Metric: {args.metric}')
    # logging.info(f'p: {args.p}')
    # logging.info(f'Adjusted: {args.unadjusted}')

    ### Set data paths. Modified to run on my machine
    if args.locality:
        data_path = '/home/eljas/ownCloud/SLR/HeightPrediction/eljas_2/toydata'
    else:
        #data_path = '/home/roelline/HeightPrediction/eljas_2/toydata'
        data_path = '/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data'
        #data_path = '/local0/scratch/roelline/files'
        
    #X_file = 'Xy_toy_train.hdf5'
    #y_file = 'Xy_toy_val.hdf5'
    X_file = "Xy_train_10k_bellot.hdf5"
    y_file = "Xy_val_10k_bellot.hdf5"
    
    # Prepare data for NearestNeighbors
    #y_metadata, y_true, X, y, X_phenotype, X_metadata = prepare_hdf5Data(data_path, X_file, y_file)
    
    for weights in ["uniform", "distance"]:
        # logging.info(f"weights: {weights}")
        if not args.locality and weights == "distance":
            df = pd.read_csv(os.path.join(data_path,'GWAS_p_vals.csv'),index_col='SNP')
            #top100 = np.loadtxt(os.path.join(data_path,'top100snps.txt'),dtype=str)
            top = np.loadtxt(os.path.join(data_path,'SNP_ids.txt'),dtype=str)
            
            w = df[df.index.isin(top)]['P'].values
            # w = np.abs(w)/np.sum(np.abs(w))
            # w = (1./w)/np.sum(1./w) ### Notice this makes all values extremely small: 10e-20 for 100 SNPs!
        elif weights == "uniform":
            w = np.ones(10000)
        else: # args.locality and weights == "distance" is True
            break
        for metric in ["minkowski","cosine"]:
            # logging.info(f"metric: {metric}")         
            # print("nnnn",weights)
            for device in ['cpu', 'gpu']:
                if metric == "minkowski":
                    for p in [2,1]: #0.5 missing
                        ### NearestNeighbors as Gold Standard
                        #if weights == "distance" and metric == "minkowski" and p >= 1:
                        #    model_base = NearestNeighbors(n_neighbors = 14, metric="wminkowski",p=p,metric_params={'w':w})
                        #elif p >= 1:
                        #    model_base = NearestNeighbors(n_neighbors = 14, metric=metric,p=p)
                        #if p < 1:
                        #    model_base = NearestNeighbors(n_neighbors = 14, metric="minkowski",p=1)
                        #   	                    
                        #model_base.fit(X)
                        #distance_matrix,nearest_neighbors = model_base.kneighbors(y,return_distance=True)
                           	            
                        ### BigKNN to be tested
                        model = bigKNN(n_neighbors=14,
                       	                            weights=weights,
                       	                            metric=metric,
                       	                            n_jobs=args.n_jobs,
                       	                            adjusted=args.unadjusted,
                       	                            scorer=Height_Statistics_np(),
                       	                            data_path=data_path,
                       	                            device = device,
                       	                            batch_size=500,
                       	                            w=w,
                       	                            p=p)
                        ### Load dataset
                        print("----------------------------------------")
                        train_dataset = hdf5Dataset(data_path,X_file)
                        test_dataset = hdf5Dataset(data_path,y_file)
                        ### Fit model
                        start = t.time()
                        model.fit(train_dataset,test_dataset)
                        end = t.time()
                        nearest_neighbors_bigKNN = model.get_neighbor_idxs_()
                        distance_matrix_bigKNN = model.get_distance_matrix_()
                        
                        print(f"\nTime {metric} (p={p}), weights {weights}, device: {device}: {end-start}")
                        #print(f"\nDistance Matrix Test {metric} (p={p}), weights {weights}, device: {device}: ", np.allclose(distance_matrix,distance_matrix_bigKNN))
                        #print(f"\nNearest Neighbors Test {metric} (p={p}), weights {weights}, device: {device}: ", np.allclose(nearest_neighbors,nearest_neighbors_bigKNN))
                        #print("4x4 sample of the sklearn.neighbors.NearestNeighbors distance matrix:")
                        #print(distance_matrix[:4,:4])
                        #print("4x4 Sample of the bigKNN distance matrix:")
                        #print(distance_matrix_bigKNN[:4,:4])
                        print("----------------------------------------")

                            
                elif metric == "cosine":
                           	            	
                    #model_base = NearestNeighbors(n_neighbors = 14, metric=metric)
                    #if weights == "distance":
                    #    model_base.fit(np.multiply(X,np.power(w,0.5)))
                    #    distance_matrix,nearest_neighbors = model_base.kneighbors(np.multiply(y,np.power(w,0.5)),return_distance=True)
                    #else:
                    #    model_base.fit(X)
                    #    distance_matrix,nearest_neighbors = model_base.kneighbors(y,return_distance=True)
   	
                    ### BigKNN to be tested
                    model = bigKNN(n_neighbors=14,
                       	                            weights=weights,
                       	                            metric=metric,
                       	                            n_jobs=args.n_jobs,
                       	                            adjusted=args.unadjusted,
                       	                            scorer=Height_Statistics_np(),
                       	                            data_path=data_path,
                       	                            device = device,
                       	                            batch_size=500,
                       	                            w=w,
                       	                            p=p)
                    print("----------------------------------------")
                    ### Load dataset
                    train_dataset = hdf5Dataset(data_path,X_file)
                    test_dataset = hdf5Dataset(data_path,y_file)
                    ### Fit model
                    start = t.time()
                    model.fit(train_dataset,test_dataset)
                    end = t.time()
                    nearest_neighbors_bigKNN = model.get_neighbor_idxs_()
                    distance_matrix_bigKNN = model.get_distance_matrix_()
                    print(f"\nTime {metric}, weights {weights}, device: {device}: {end-start}")
                    #print(f"\nDistance Matrix Test {metric}, weights: {weights}, device: {device}: ", np.allclose(distance_matrix,distance_matrix_bigKNN))
                    #print(f"\nNearest Neighbors Test {metric}, weights {weights}, device: {device}: ", np.allclose(nearest_neighbors,nearest_neighbors_bigKNN))
                    #print("4x4 sample of the sklearn.neighbors.NearestNeighbors distance matrix:")
                    #print(distance_matrix[:4,:4])
                    #print("4x4 Sample of the bigKNN distance matrix:")
                    #print(distance_matrix_bigKNN[:4,:4])
                    print("----------------------------------------")
