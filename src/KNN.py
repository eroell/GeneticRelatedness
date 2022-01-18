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

class bigKNN(BaseEstimator):
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
    def __init__(self,n_neighbors=10,weights='uniform',metric='manhattan',n_jobs=1,adjusted=False,scorer=None,data_path='.',batch_size=1000,device='cpu',output="",**kwargs):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.n_jobs = n_jobs
        self.adjusted = adjusted
        self.data_path = data_path
        self.scorer = scorer
        self.batch_size = batch_size
        # self.seed = seed
        self.device = device
        self.output = output
        # self.val_percent = val_percent
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
        

    def fit(self,train_dataset,test_dataset,X_phenotype=None,y_phenotype=None): #fit(self,X_file,y_file=None,X_phenotype=None,y_phenotype=None):
        '''
        Calculate the distance matrix and find the nearest neighbours for each sample in the test set.
        Input:
        train_dataset: A hdf5Dataset containing the training points (hdf5Dataset implements torch.utils.data.Dataset)
        test_dataset: A hdf5Dataset containing the training points (hdf5Dataset implements torch.utils.data.Dataset)
        X_phenotype: A matrix of the phenotype of X. If None, it is assumed to be "y/height" in the hdf5 file.
        y_phenotype: A matrix of the phenotype of y. If None, it is assumed to be "y/height" in the hdf5 file.

        Fits:
         - The distances of the k nearst neighbours of y, a (Ny x n_beighbours) matrix. Stored in self.distance_matrix_
         - The metadata containing phenotype, sex and age of the nearest neighbours, an (3 x Ny x n_neighbours) matrix. Stored in self.neighbor_metadata_
        '''
        if self.metric == 'precomputed':
            pass
        else:
            # if y_file is not None:
            #     train_dataset = hdf5Dataset(self.data_path,X_file,phenotype=X_phenotype)
            #     test_dataset = hdf5Dataset(self.data_path,y_file,phenotype=y_phenotype)
            self.x_len = len(train_dataset)
            # else:
            #     train_test_dataset = hdf5Dataset(self.data_path,X_file,phenotype=X_phenotype)
            #     train_test_split = [int((1-self.val_percent)*len(train_test_dataset)),int(len(train_test_dataset))-int((1-self.val_percent)*len(train_test_dataset))]
            #     train_dataset,test_dataset = random_split(train_test_dataset,train_test_split,generator=Generator().manual_seed(self.seed))
            #     self.x_len = len(train_dataset)
            if self.metric == 'minkowski':
                self.dist_fun = torch.cdist
            elif self.metric == 'cosine':
                self.dist_fun = self._own_cosine

            train_dataloader = DataLoader(train_dataset,self.batch_size,num_workers=self.n_jobs)
            test_dataloader = DataLoader(test_dataset,self.batch_size,num_workers=self.n_jobs)

            # BaseEstimator API: Attributes that have been estimated from the data must always have a name ending with trailing underscore
            self.distance_matrix_ = np.zeros((len(test_dataset),self.n_neighbors))
            self.neighbor_metadata_ = np.zeros((3,len(test_dataset),self.n_neighbors))
            self.y_metadata_ = np.zeros((3,len(test_dataset)))
            
            start_idx = 0
            for y_batch in tqdm(test_dataloader,desc='Y progress',leave=True):
                y_SNP,y_true,y_sex,y_age = y_batch
                ### If GPU available, move tensor there
                # if torch.cuda.is_available():
                #     y_SNP = y_SNP.cuda()
                # else:
                y_SNP = y_SNP.numpy()
                if self.w is not None:
                    y_SNP = np.multiply(y_SNP,self.w)
                output, output_metadata = self._neighbors(y_SNP,train_dataloader,X_phenotype)
                self.distance_matrix_[start_idx:start_idx+y_SNP.shape[0],:] = output
                self.neighbor_metadata_[:,start_idx:start_idx+y_SNP.shape[0],:] = output_metadata
                self.y_metadata_[0,start_idx:start_idx+y_SNP.shape[0]] = y_true
                self.y_metadata_[1,start_idx:start_idx+y_SNP.shape[0]] = y_sex
                self.y_metadata_[2,start_idx:start_idx+y_SNP.shape[0]] = y_age
                start_idx += y_SNP.shape[0]
            
            ### Fractional Distance Metric: Frobenius norm, not Fractinal distance Metric as in Aggarwal et al.
            # COMMENTED THE TWO LINES BELOW OUT JUST FOR A PERFORMANCE EVALUATION
            if self.metric == 'minkowski' and self.p < 1:
                self.distance_matrix_ = np.power(self.distance_matrix_, self.p)
                
            ### By me: is nice I think, faster to load/compare
            # fylename = f"DELETE_metric_{self.metric}_p_{self.p}_n_neighbors_{self.n_neighbors}_seed_{self.seed}_adjusted_{self.adjusted}_device_{self.device}"
            # np.savez_compressed(os.path.join(self.output, fylename),
            #                    distance_matrix_ = self.distance_matrix_,
            #                    neighbor_metadata_ = self.neighbor_metadata_,
            #                    y_metadata_ = self.y_metadata_)


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
            #print("Using gpu for cosine")
            normalized_input_a = torch.nn.functional.normalize(Xtrain,dim=1)  
            normalized_input_b = torch.nn.functional.normalize(Xval,dim=1)
            res = torch.mm(normalized_input_a, normalized_input_b.T)
            res *= -1 # 1-res without copy
            res += 1

        else:
            #print("Not using gpu for cosine")
            normalized_input_a =  np.multiply(Xtrain,np.linalg.norm(Xtrain,axis=1)[:, np.newaxis]**-1)
            normalized_input_b = np.multiply(Xval,np.linalg.norm(Xval,axis=1)[:, np.newaxis]**-1)
            res = np.matmul(normalized_input_a,normalized_input_b.T)
            res *= -1 # 1-res without copy
            res += 1

        return res

    def _neighbors(self,SNP,dataloader,X_phenotype):
        line_SNP = np.zeros((SNP.shape[0],self.x_len))
        line_metadata = np.zeros((3,SNP.shape[0],self.x_len))
        start_idx = 0
        for x_batch in tqdm(dataloader,desc='X for y progress',leave=False):
            x_SNP,x_pheno,x_Sex,x_Age = x_batch

            # if torch.cuda.is_available():
            #     x_SNP = x_SNP.cuda()
            # else:
            x_SNP = x_SNP.numpy()
            x_pheno = x_pheno.numpy()
            x_Sex = x_Sex.numpy()
            x_Age = x_Age.numpy()
            
            if self.w is not None:
                x_SNP = np.multiply(x_SNP,self.w)
                
            ### Check GPU instruction and availability
            if torch.cuda.is_available() and self.device == 'gpu':

                SNP_t = torch.from_numpy(SNP).cuda().float()
                x_SNP_t = torch.from_numpy(x_SNP).cuda().float()
                # COMMENTED OUT 4 LINES BELOW: WROTE LINE 5 LINES BELOW TO TEST PERFORMANCE
                #if self.metric == 'minkowski':
                #    distmatrix = torch.cdist(SNP_t, x_SNP_t, p = self.p)
                #if self.metric == 'cosine':
                #    distmatrix = self._own_cosine(SNP_t, x_SNP_t)
                distmatrix = self.dist_fun(SNP_t, x_SNP_t,self.p)
                
                line_SNP[:,start_idx:start_idx+x_SNP.shape[0]] = distmatrix.cpu().numpy()
                
                # del SNP_t,x_SNP_t,distmatrix
                # tor = torch.cdist(SNP_t, x_SNP_t, p = self.p).cpu().numpy()
                # sci = cdist(SNP,x_SNP,metric=self.metric,p=self.p)
                # print(f'np.allclose of torch.cdist and scipy.cidist: {np.allclose(tor,sci)}') # TRUE for small toy example
            elif not torch.cuda.is_available() and self.device == 'gpu':
                print('device == "gpu": no gpu available')
            else:     
                if self.metric == 'minkowski':
                    line_SNP[:,start_idx:start_idx+x_SNP.shape[0]] = cdist(SNP,x_SNP,metric=self.metric,p=self.p)
                elif self.metric == 'hamming':
                    line_SNP[:,start_idx:start_idx+x_SNP.shape[0]] = cdist(SNP,x_SNP,metric=self.metric)
                ##################################################################
                ### New by me
                # elif self.metric == 'fractional':
                #    print(self.p)
                #    line_SNP[:,start_idx:start_idx+x_SNP.shape[0]] = self._own_minkowski(SNP, x_SNP, self.p)
                #    # line_SNP[:,start_idx:start_idx+x_SNP.shape[0]] = cdist(SNP,x_SNP,metric=self.metric, p=self.p)
                elif self.metric == "cosine":
                    line_SNP[:,start_idx:start_idx+x_SNP.shape[0]] = self._own_cosine(SNP, x_SNP,self.p)
                ##################################################################
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
            neighbor_idxs = np.argsort(line_SNP,axis=1)[:,:self.n_neighbors]
            # neighbor_idxs = np.argpartition(line_SNP,self.n_neighbors,axis=1)[:,:self.n_neighbors]
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
