import os
import h5py as h5
import numpy as np
import pandas as pd
from tqdm import tqdm
import time as t

from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset,DataLoader,random_split
import torch
from torch import Generator

from src.utils import Height_Statistics,Height_Statistics_big#, hdf5Dataset

class bigKNN_PCA_enabled(BaseEstimator):
    '''
    A scikit-learn class to calculate a K-nearest-neighbours estimator for the Biobank data and height.
    Input:
    - n_neighbors: The number of nearest neighbours
    - weights = 'uniform' or 'distance'; weigh the nearest neighbours; currently only 'uniform' is implemented.
    - metric: the metric to compare SNPs by; currently all metrics from sklearn.neighbors.DistanceMetric are supported
    - n_jobs: the number or parallel jobs
    - adjusted: calculate the distance based on the raw data or the stratification adjusted data
    - scorer: a scoring class which can calculate the original phenotype from the scores if adjusted=False
    - data_path: the path of the hdf5 files containing the data
    - batch_size: the batch size for the data loading
    - seed: the seed for the random number generator for splitting into train and val set
    - kwargs: the kwargs for scipy.spatial.distance.cdist
    '''
    def __init__(self,n_neighbors=10, metric='manhattan',weights='uniform',
                 adjusted=False,device='cpu',batch_size=400,n_jobs=1,
                 scorer=None,**kwargs):
        
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.adjusted = adjusted
        self.device = device
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.scorer = scorer
        # self.dist_matrix_save_dir = dist_matrix_save_dir
        self.p = kwargs.pop('p',None)
        self.w = kwargs.pop('w',None)
        self.V = kwargs.pop('V',None) #not implemented yet
        self.VI = kwargs.pop('VI',None) #not implemented yet
        # weighted euclidean/manhattan distance is very slow in scipy
        # however, weighted minkowski is optimised, therefore use minkowski
        if metric == 'manhattan':
            self.metric = 'minkowski'
            self.p = 1
        elif metric == 'euclidean':
            self.metric = 'minkowski'
            self.p = 2
            if self.w is not None:
                self.w = np.power(self.w,0.5)
        elif metric == 'fractional':
            self.metric = 'minkowski'
            if self.w is not None:
                self.w = np.power(self.w, 1/self.p)
        elif metric == 'cosine':
            self.metric = metric
            if self.w is not None:
                self.w = np.power(self.w,0.5)
                self.p = 1 # completely ignored, for performance test only
        else:
            self.metric = metric
        # Mahalanobis explicitly ignores w, even if it is accidentally set
        if metric == "mahalanobis":
            self.w = None
        

    def fit(self,train_dataset,test_dataset,X_phenotype=None,y_phenotype=None,means = None, components = None,):
        '''
        Calculate the distance matrix and find the nearest neighbours for each sample in the test set.
        Input:
        train_dataset: A hdf5Dataset containing the training points (hdf5Dataset implements torch.utils.data.Dataset)
        test_dataset: A hdf5Dataset containing the training points (hdf5Dataset implements torch.utils.data.Dataset)
        X_phenotype: A matrix of the phenotype of X. If None, it is assumed to be "y/height" in the hdf5 file.
        y_phenotype: A matrix of the phenotype of y. If None, it is assumed to be "y/height" in the hdf5 file.
        - means: component-wise means (from a previously run PCA of the data) [is this required for pairwise distance?]
        - components: Principal Components (from a previously run PCA of the data)

        Fits:
         - The distances of the k nearst neighbours of y, a (Ny x n_beighbours) matrix. Stored in self.distance_matrix_
         - The metadata containing phenotype, sex and age of the nearest neighbours, an (3 x Ny x n_neighbours) matrix. Stored in self.neighbor_metadata_
        '''
        if self.metric == 'precomputed':
            pass
        else:
            self.x_len = len(train_dataset)

            if self.metric == 'minkowski':
                self.dist_fun = torch.cdist
            elif self.metric == 'cosine':
                self.dist_fun = self._own_cosine
            elif self.metric == 'mahalanobis':
                self.dist_fun = self._own_mahalanobis

            train_dataloader = DataLoader(train_dataset,self.batch_size,num_workers=self.n_jobs)
            test_dataloader = DataLoader(test_dataset,self.batch_size,num_workers=self.n_jobs)

            # BaseEstimator API: Attributes that have been estimated from the data must always have a name ending with trailing underscore
            self.distance_matrix_ = np.zeros((len(test_dataset),self.n_neighbors))
            self.neighbor_metadata_ = np.zeros((3,len(test_dataset),self.n_neighbors))
            self.y_metadata_ = np.zeros((3,len(test_dataset)))
            
            start_idx = 0
            for y_batch in tqdm(test_dataloader,desc='Y progress',leave=True):
                y_SNP,y_true,y_sex,y_age = y_batch
                y_SNP = y_SNP.numpy()
                
                if self.w is not None:
                    y_SNP = np.multiply(y_SNP,self.w)
                # Dimensionality reduction of y_SNP (y_SNP is of dimension nxd)
                if components is not None:
                    # Subtracting constants from every dimension does not impact two point's distance
                    # y_SNP = np.subtract(y_SNP, means)
                    y_SNP = np.matmul(y_SNP, components)
                    # None
                    
                output, output_metadata = self._neighbors(y_SNP,train_dataloader,X_phenotype,means=means,components=components)
                self.distance_matrix_[start_idx:start_idx+y_SNP.shape[0],:] = output
                self.neighbor_metadata_[:,start_idx:start_idx+y_SNP.shape[0],:] = output_metadata
                self.y_metadata_[0,start_idx:start_idx+y_SNP.shape[0]] = y_true
                self.y_metadata_[1,start_idx:start_idx+y_SNP.shape[0]] = y_sex
                self.y_metadata_[2,start_idx:start_idx+y_SNP.shape[0]] = y_age
                start_idx += y_SNP.shape[0]
            
            ### Fractional Distance Metric: Frobenius norm, not Fractinal distance Metric as in Aggarwal et al.
            if self.metric == 'minkowski' and self.p < 1:
                self.distance_matrix_ = np.power(self.distance_matrix_, self.p)
                

    ### Own Minkowski implementation: Faster than cdist-minkowski with p < 1.
    ### Note that weights are handled elsewhere
    def _own_minkowski(self, Xtrain,Xval,p):
        ds_mink = np.zeros((Xtrain.shape[0], Xval.shape[0]))
        for j in range(Xval.shape[0]):
            ds_mink[:,j] = np.sum(np.abs((Xtrain[:,:]-Xval[j,:]))**p, axis = 1)**(1/p)
            
        return ds_mink
        
    ### Own cosine implementation. Note that weights are handled elsewhere
    ### Completely ignores p: only for performance
    def _own_cosine(self,Xtrain,Xval,p):
        #print("calculating cosine")
        if torch.cuda.is_available() and self.device == 'gpu':
            normalized_input_a = torch.nn.functional.normalize(Xtrain,dim=1)  
            normalized_input_b = torch.nn.functional.normalize(Xval,dim=1)
            res = torch.mm(normalized_input_a, normalized_input_b.T)
            res *= -1 # 1-res without copy
            res += 1

        else:
            normalized_input_a =  np.multiply(Xtrain,np.linalg.norm(Xtrain,axis=1)[:, np.newaxis]**-1)
            normalized_input_b = np.multiply(Xval,np.linalg.norm(Xval,axis=1)[:, np.newaxis]**-1)
            res = np.matmul(normalized_input_a,normalized_input_b.T)
            res *= -1 # 1-res without copy
            res += 1

        return res
    ### Own mahalanobis implementation. Note that weights are not valid with this
    ### Completely ignores p: only for performance
    def _own_mahalanobis(self,Xtrain,Xval,p=None):
        ### is neither "clean" nor optimal, but suffices the purpose for 500x100 matrices
        # if cuda: Xtrain and Xval were put to cuda before
        if torch.cuda.is_available() and self.device == 'gpu':
            mahal_dist_mat = torch.zeros((Xval.shape[0],Xtrain.shape[0]))
            #Xtrain = torch.tensor(Xtrain).float()
            #Xval = torch.tensor(Xval).float()
            #print("OOOOOOOOOOOOOOOOO",self.VI)
            inv_cov_mat = torch.tensor(self.VI).cuda().float()
            for i in range(Xtrain.shape[0]):
                delta = torch.subtract(Xval, Xtrain[i,:])
                a = torch.matmul(torch.matmul(delta, inv_cov_mat),delta.T)
                mahal_dist_mat[:,i] = torch.diag(a).cuda()
            mahal_dist_mat = torch.sqrt(mahal_dist_mat)
            return mahal_dist_mat.T
        else:
        # if not cuda: same implementation, but have to convert input from numpy
            Xtrain = torch.tensor(Xtrain).float()
            Xval = torch.tensor(Xval).float()
            #print("OOOOOOOOOOOOOOOOO",self.VI)
            inv_cov_mat = torch.tensor(self.VI).float()
            mahal_dist_mat = torch.zeros((Xval.shape[0],Xtrain.shape[0]))
        
            for i in range(Xtrain.shape[0]):
                delta = torch.subtract(Xval, Xtrain[i,:])
                a = torch.matmul(torch.matmul(delta, inv_cov_mat),delta.T)
                mahal_dist_mat[:,i] = torch.diag(a)
            mahal_dist_mat = torch.sqrt(mahal_dist_mat)
            return mahal_dist_mat.T

    def _neighbors(self,SNP,dataloader,X_phenotype,means=None,components=None):
        line_SNP = np.zeros((SNP.shape[0],self.x_len))
        line_metadata = np.zeros((3,SNP.shape[0],self.x_len))
        start_idx = 0
        for x_batch in tqdm(dataloader,desc='X for y progress',leave=False):
            x_SNP,x_pheno,x_Sex,x_Age = x_batch

            x_SNP = x_SNP.numpy()
            x_pheno = x_pheno.numpy()
            x_Sex = x_Sex.numpy()
            x_Age = x_Age.numpy()
            
            if self.w is not None:
                x_SNP = np.multiply(x_SNP,self.w)
            
            # Dimensionality reduction of y_SNP (y_SNP is of dimension nxd)
            if components is not None:
                # Subtracting constants from every dimension does not impact two point's distance
                # x_SNP = np.subtract(x_SNP, means)
                x_SNP = np.matmul(x_SNP,components)
                # None

                
            ### Check GPU instruction and availability
            if torch.cuda.is_available() and self.device == 'gpu':

                SNP_t = torch.from_numpy(SNP).cuda().float()
                x_SNP_t = torch.from_numpy(x_SNP).cuda().float()
                
                distmatrix = self.dist_fun(SNP_t, x_SNP_t,self.p)
                
                line_SNP[:,start_idx:start_idx+x_SNP.shape[0]] = distmatrix.cpu().numpy()
                
 
            elif not torch.cuda.is_available() and self.device == 'gpu':
                print('device == "gpu": no gpu available')
            else:     
                if self.metric == 'minkowski':
                    line_SNP[:,start_idx:start_idx+x_SNP.shape[0]] = cdist(SNP,x_SNP,metric=self.metric,p=self.p)
                elif self.metric == 'hamming':
                    line_SNP[:,start_idx:start_idx+x_SNP.shape[0]] = cdist(SNP,x_SNP,metric=self.metric)
                elif self.metric == 'cosine':
                    line_SNP[:,start_idx:start_idx+x_SNP.shape[0]] = self._own_cosine(SNP, x_SNP,self.p)
                elif self.metric == 'mahalanobis':
                    line_SNP[:,start_idx:start_idx+x_SNP.shape[0]] = self._own_mahalanobis(SNP, x_SNP)
                else:
                    print('Not implemented yet!')
                    break

            line_metadata[0,:,start_idx:start_idx+x_SNP.shape[0]] = x_pheno
            line_metadata[1,:,start_idx:start_idx+x_SNP.shape[0]] = x_Sex
            line_metadata[2,:,start_idx:start_idx+x_SNP.shape[0]] = x_Age
            start_idx += x_SNP.shape[0]

        if self.n_neighbors == self.x_len:
            return line_SNP,line_metadata
        else:
            #neighbor_idxs = np.argsort(line_SNP,axis=1)[:,:self.n_neighbors]
            neighbor_idxs = np.argpartition(line_SNP,self.n_neighbors,axis=1)[:,:self.n_neighbors]
            #self.neighbor_idxs_ = neighbor_idxs
            return np.array([line_SNP[i,idxs] for i,idxs in enumerate(neighbor_idxs)]),np.stack([line_metadata[:,i,idxs] for i,idxs in enumerate(neighbor_idxs)],axis=1)

    def predict(self):
        '''
        Predict the pheotpye
        Input:
        y_metadata: A matrix of the metadata (sex,age) of y.

        Output:
         - The predictions of the phenotypes
        '''

        metadata = self.neighbor_metadata_
        if not self.adjusted:
            for n in range(self.n_neighbors):
                metadata[0,:,n] = self.scorer.calculate_height(self.neighbor_metadata_[0,:,n],self.neighbor_metadata_[1,:,n],self.neighbor_metadata_[2,:,n])
        if self.weights == 'uniform':
            predictions = np.mean(metadata[0],axis=1)
        elif self.weights == 'distance':
            # Add a small number for numerical stability
            weights = np.divide(1./(self.distance_matrix_+1e-8),np.sum(1./(self.distance_matrix_+1e-8),axis=1,keepdims=True))
            predictions = np.sum(np.multiply(weights,metadata[0]),axis=1)
        if self.adjusted:
            self.predictions_ = self.scorer.calculate_height(predictions,self.y_metadata_[1],self.y_metadata_[2])
        else:
            self.predictions_ = predictions

    def fit_predict(self,X_file,y_file,X_phenotype=None,y_phenotype=None):
        '''
        Fit the model and predict
        '''
        self.fit(X_file,y_file,X_phenotype=X_phenotype,y_phenotype=None)
        self.predict()
        
    def get_distance_matrix_(self):
        '''
        Get distance matrix if the model has been fitted once.
        Throws name-error if model has not been fitted yet: this is ok.
        '''
        return self.distance_matrix_
    
    #def get_neighbor_idxs_(self):
    #    '''
    #    Get distance matrix if the model has been fitted once.
    #    Throws name-error if model has not been fitted yet: this is ok.
    #    '''
    #    return self.neighbor_idxs_
    
    # def _save_distance_matrix():
    #     '''
    #     Store distance matrix and metadata:
    #     This can be loaded then for experiments that start from the
    #     same distance matrix
    #     '''
    #     #         if store_dist_mat:
    #         fylename = f"dist_matrix_cv_split_{i}_dataset_{data_name}_metric_{metric}_p_{p}_n_neighbors_{n_neighbors}_seed_{random_state}_adjusted_device_{device}"
    #         np.savez_compressed(os.path.join(outdir, fylename),
    #                             distance_matrix_ = model.distance_matrix_,
    #                             neighbor_metadata_ = model.neighbor_metadata_,
    #                             y_metadata_ = model.y_metadata_)

    def evaluate_metrics(self):
        '''
        Evaluate common metrics (MSE,MAE,r,r2) on the prediction
        If y_metadata (sex,age) is given then the phenotype is first adjusted

        Output:
        - A dictionary with the scores
        '''
        phenotype = self.scorer.calculate_height(self.y_metadata_[0],self.y_metadata_[1],self.y_metadata_[2])
        return({'MSE':mean_squared_error(phenotype,self.predictions_),
                'MAE':mean_absolute_error(phenotype,self.predictions_),
                'r':np.corrcoef(phenotype,self.predictions_)[0,1],
                'r2':r2_score(phenotype,self.predictions_)})
