##############################################################################
##############################################################################
'''
A main script for testing the KNN_PCA_enabled.py module.
Examines:
    -Distance Matrix with KNN.py (True model: sklearn.neighbors.NearestNeighbors)
    -Nearest Neighbors with KNN.py (True model: sklearn.neighbors.NearestNeighbors)
    (Note this is not meaningful for only 100 SNPs, since here multiple
     neighbors are having the same distance, and the (for our purposes equivalent)
     ordering of these is may be different for KNN.py and NearestNeighbors)

For (all combinations of)
    -Metric: Minkowski (p=2), Minkowski(p=1), Minkowski(p=0.5), cosine
    -Device: CPU, GPU
        -GPU only testable on slurm
    -Weights: Uniform, Distance
        -Weighted only testable on slurm or bs-borgwardt01 server

Using the dataset
    -Small Toy
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
from sklearn.decomposition import PCA

from src.KNN_PCA_enabled import bigKNN_PCA_enabled
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

def load_pca_intermediates(full_file_name):
    ### Load PCA intermediates
    with h5.File(full_file_name, 'r') as hf:
        # hf = h5.File(os.path.join(intermediates_path, "cov_mat_small_device_cpu_batchsize_400.hdf5"), 'r')
        means_loaded = np.array(hf['/means'])
        cov_loaded = np.array(hf['/covariance_matrix'])
        expl_var_loaded = np.array(hf['/explained_variance'])
        components_loaded = np.array(hf['/components'])
        hf.close()
        
    return means_loaded, cov_loaded, expl_var_loaded, components_loaded

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

    # logging.info(f'Testing KNN_PCA_enabled on dataset {args.dataset}:')

    ### Set data paths. Modified to run on my machine
    if args.locality:
        data_path = '/home/eljas/ownCloud/SLR/HeightPrediction/eljas_2/toydata'
        intermediates_path = '/home/eljas/ownCloud/SLR/HeightPrediction/eljas_2/intermediates'
    else:
        #data_path = '/home/roelline/HeightPrediction/eljas_2/toydata'
        #data_path = '/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data'
        data_path = '/local0/scratch/roelline/files'
        intermediates_path = '/home/roelline/HeightPrediction/eljas_2/intermediates'
    
    
    X_file = 'Xy_small_toy_train.hdf5'
    y_file = 'Xy_small_toy_val.hdf5'
    batch_size = 400
    
    print("-------------------------------------------------------")
    print("Tests without Dimensionality Reduction:")
    print("-------------------------------------------------------")
    # Prepare data for NearestNeighbors
    y_metadata, y_true, X, y, X_phenotype, X_metadata = prepare_hdf5Data(data_path, X_file, y_file)
    print(f"X.shape: {X.shape}")
    # for weights in ["uniform", "distance"]:
    #     logging.info(f"weights: {weights}")
    #     if not args.locality and weights == "distance":
    #         df = pd.read_csv(os.path.join(data_path,'GWAS_p_vals.csv'),index_col='SNP')
    #         top100 = np.loadtxt(os.path.join(data_path,'top100snps.txt'),dtype=str)
            
    #         w = df[df.index.isin(top100)]['P'].values
    #         # w = np.abs(w)/np.sum(np.abs(w))
    #         # w = (1./w)/np.sum(1./w) ### Notice this makes all values extremely small: 10e-20 for 100 SNPs!
    #     elif weights == "uniform":
    #         w = np.ones(100)
    #     else: # args.locality and weights == "distance" is True
    #         break
    #     for metric in ["minkowski","cosine"]:
    #         logging.info(f"metric: {metric}")         
    #         # print("nnnn",weights)
    #         for device in ['cpu', 'gpu']:
    #             if metric == "minkowski":
    #                 for p in [2,1,0.5]:
    #                     ### NearestNeighbors as Gold Standard
    #                     if weights == "distance" and metric == "minkowski" and p >= 1:
    #                         model_base = NearestNeighbors(n_neighbors = 14, metric="wminkowski",p=p,metric_params={'w':w})
    #                     elif p >= 1:
    #                         model_base = NearestNeighbors(n_neighbors = 14, metric=metric,p=p)
    #                     if p < 1:
    #                         model_base = NearestNeighbors(n_neighbors = 14, metric="minkowski",p=1)
                           	                    
    #                     model_base.fit(X)
    #                     distance_matrix,nearest_neighbors = model_base.kneighbors(y,return_distance=True)
                           	            
    #                     ### bigKNN_PCA_enabled to be tested
    #                     model = bigKNN_PCA_enabled(n_neighbors=14,
    #                    	                            weights=weights,
    #                    	                            metric=metric,
    #                    	                            n_jobs=args.n_jobs,
    #                    	                            adjusted=args.unadjusted,
    #                    	                            scorer=Height_Statistics_np(),
    #                    	                            data_path=data_path,
    #                    	                            device = device,
    #                    	                            batch_size=batch_size,
    #                    	                            w=w,
    #                    	                            p=p)
    #                     ### Load dataset
    #                     train_dataset = hdf5Dataset(data_path,X_file)
    #                     test_dataset = hdf5Dataset(data_path,y_file)
    #                     ### Fit model
    #                     start = t.time()
    #                     model.fit(train_dataset,test_dataset)
    #                     end = t.time()
    #                     #nearest_neighbors_bigKNN = model.get_neighbor_idxs_()
    #                     distance_matrix_bigKNN = model.get_distance_matrix_()

    #                     print(distance_matrix.shape)
    #                     print(distance_matrix_bigKNN.shape)
    #                     print(f"\Time {metric} (p={p}), weights {weights}, device: {device}: {end-start}")
    #                     print(f"\nDistance Matrix Test {metric} (p={p}), weights {weights}, device: {device}: ", np.allclose(distance_matrix,distance_matrix_bigKNN))
    #                     #print(f"\nNearest Neighbors Test {metric} (p={p}), weights {weights}, device: {device}: ", np.allclose(nearest_neighbors,nearest_neighbors_bigKNN))
    #                     print("4x4 sample of the sklearn.neighbors.NearestNeighbors distance matrix:")
    #                     print(distance_matrix[:4,:4])
    #                     print("4x4 Sample of the bigKNN distance matrix:")
    #                     print(distance_matrix_bigKNN[:4,:4])
    #                     del model
                            
    #             elif metric == "cosine":  	            	
    #                 model_base = NearestNeighbors(n_neighbors = 14, metric=metric)
    #                 if weights == "distance":
    #                     model_base.fit(np.multiply(X,np.power(w,0.5)))
    #                     distance_matrix,nearest_neighbors = model_base.kneighbors(np.multiply(y,np.power(w,0.5)),return_distance=True)
    #                 else:
    #                     model_base.fit(X)
    #                     distance_matrix,nearest_neighbors = model_base.kneighbors(y,return_distance=True)
   	
    #                 ### BigKNN to be tested
    #                 model = bigKNN_PCA_enabled(n_neighbors=14,
    #                    	                            weights=weights,
    #                    	                            metric=metric,
    #                    	                            n_jobs=args.n_jobs,
    #                    	                            adjusted=args.unadjusted,
    #                    	                            scorer=Height_Statistics_np(),
    #                    	                            data_path=data_path,
    #                    	                            device = device,
    #                    	                            batch_size=batch_size,
    #                    	                            w=w,
    #                    	                            p=p)
    #                 ### Load dataset
    #                 train_dataset = hdf5Dataset(data_path,X_file)
    #                 test_dataset = hdf5Dataset(data_path,y_file)
    #                 ### Fit model
    #                 start = t.time()
    #                 model.fit(train_dataset,test_dataset)
    #                 end = t.time()
    #                 #nearest_neighbors_bigKNN = model.get_neighbor_idxs_()
    #                 distance_matrix_bigKNN = model.get_distance_matrix_()

    #                 print(f"\Time {metric} (p={p}), weights {weights}, device: {device}: {end-start}")
    #                 print(f"\nDistance Matrix Test {metric}, weights: {weights}, device: {device}: ", np.allclose(distance_matrix,distance_matrix_bigKNN))
    #                 #print(f"\nNearest Neighbors Test {metric}, weights {weights}, device: {device}: ", np.allclose(nearest_neighbors,nearest_neighbors_bigKNN))
    #                 print("4x4 sample of the sklearn.neighbors.NearestNeighbors distance matrix:")
    #                 print(distance_matrix[:4,:4])
    #                 print("4x4 Sample of the bigKNN distance matrix:")
    #                 print(distance_matrix_bigKNN[:4,:4])
    #                 del model
                    
    print("-------------------------------------------------------")
    print("Tests with Dimensionality Reduction")
    print("-------------------------------------------------------")
    means_loaded, cov_loaded, expl_var_loaded, components_loaded = load_pca_intermediates(os.path.join(intermediates_path,
                                                                                        "pca_intermediates_small_device_cpu_batchsize_400.hdf5"))
    ### bigKNN_PCA_enabled to be tested
    metric = "euclidean"
    # metric = "manhattan"
    p = 2
    for device in ["cpu"]:#,"gpu"):
        for n_components in [1,2,10,100]:
            model = bigKNN_PCA_enabled(n_neighbors=14,
                                	       weights="uniform",
                                	       metric=metric,
                                	       jobs=1,
                                	       adjusted=args.unadjusted,
                                	       scorer=Height_Statistics_np(),
                                	       data_path=data_path,
                                	       device = device,
                                	       batch_size=batch_size,
                                	       w=None,
                                	       p=p)            
            
            ### Load dataset
            train_dataset = hdf5Dataset(data_path,X_file)
            test_dataset = hdf5Dataset(data_path,y_file)
            ### Fit model
            start = t.time()
            model.fit(train_dataset,test_dataset,means=means_loaded,components = components_loaded[:,:n_components])
            end = t.time()
            #nearest_neighbors_bigKNN = model.get_neighbor_idxs_()
            distance_matrix_bigKNN = model.get_distance_matrix_()
            
            model.predict()
            ### Observe results
            results = model.evaluate_metrics()
            print(f'Results {device}, n_components {n_components}:{results}')
            
            # Full components KNN should be equal to non-dimensionality-reduced KNN
            if n_components == 100:
                ### Instantiate and fit sklearn.decomposition.PCA
                model = bigKNN_PCA_enabled(n_neighbors=100,
                               	       weights="uniform",
                               	       metric=metric,
                               	       jobs=1,
                               	       adjusted=args.unadjusted,
                               	       scorer=Height_Statistics_np(),
                               	       data_path=data_path,
                               	       device = device,
                               	       batch_size=batch_size,
                               	       w=None,
                               	       p=p)            

                train_dataset = hdf5Dataset(data_path,X_file)
                test_dataset = hdf5Dataset(data_path,y_file)

                model.fit(train_dataset,test_dataset,means=means_loaded,components = components_loaded[:,:n_components])

                distance_matrix_bigKNN = model.get_distance_matrix_()
            
                ### Load dataset
                X = get_SNPs(data_path, X_file)
                
                ### Reference model
                model_base = NearestNeighbors(n_neighbors = 100, metric=metric)
                model_base.fit(X)
                distance_matrix,nearest_neighbors = model_base.kneighbors(y,return_distance=True)
                
                print(f"\nDistance Matrix Test {metric}, weights uniform, device: {device}: ", np.allclose(distance_matrix,distance_matrix_bigKNN))
                print("4x4 sample of the sklearn.neighbors.NearestNeighbors distance matrix:")
                print(distance_matrix[:4,:4])
                print("4x4 Sample of bigKNN distance matrix of data transformed with the full set of PC's':")
                print(distance_matrix_bigKNN[:4,:4])               
                
                