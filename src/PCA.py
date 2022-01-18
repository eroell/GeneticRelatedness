#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import h5py as h5
import numpy as np
import pandas as pd
from tqdm import tqdm
import time as t

from sklearn.base import BaseEstimator
# from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
# from scipy.spatial.distance import cdist
from torch.utils.data import Dataset,DataLoader,random_split
import torch
from torch import Generator

from src.utils import Height_Statistics,Height_Statistics_big#, hdf5Dataset

class bigPCA(BaseEstimator):
    '''
    A scikit-learn class for PCA computation.
    Input:
    - n_components: Number of components to keep. if n_components is not set all components are kept
    - whiten[not implemented]: When True (False by default) the components_ vectors are multiplied by the square root
      of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit
      component-wise variances.
    - n_jobs: the number or parallel jobs
    - batch_size: the batch size for the data loading
    - device: cpu or gpu
    - output: where to store the output
    '''
    def __init__(self,n_components=None, whiten = False, n_jobs=1,batch_size=500,device='cpu',output=""):
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.device = device
        self.output = output
        
        
    def fit(self,X_dataset):
        '''
        Fit the model with X_dataset.
        Input:
        train_dataset: A hdf5Dataset containing the training points (hdf5Dataset implements torch.utils.data.Dataset)
        
        Fits:
         - the principal components. sorted by explained variance
         - The feature means
         - The covariance matrix
         - The explained variance
         - The explained covariance ratio
         - The metadata containing phenotype, sex and age of the nearest neighbours, an (3 x Ny x n_neighbours) matrix. Stored in self.neighbor_metadata_
        '''
        
        self.x_len = len(X_dataset)
        dataloader = DataLoader(X_dataset,self.batch_size,num_workers=self.n_jobs)
        self.dim = next(iter(dataloader))[0].shape[1]       

        # Compute the mean of all features
        self._compute_means(dataloader)        

        # Compute covariance matrix
        self._compute_covariance_matrix(dataloader)

        # Compute spectral decomposition of covariance matrix
        self._compute_spectral_decomposition()

        return 0

    def _compute_means(self, dataloader):
        self.means_ = torch.zeros(self.dim) # get dimension from data
        if self.device == "gpu":
            self.means_ = self.means_.cuda().float()
            
        for batch in tqdm(dataloader,desc='Mean progress',leave=True):
            SNP,true,sex,age = batch
            # Put to GPU if settings are so (will throw an error if no gpu available)
            if self.device == "gpu":
                SNP = SNP.cuda().float()
            self.means_ = torch.add(self.means_, torch.sum(SNP,axis=0))
            
        self.means_ = torch.div(self.means_, self.x_len)
        
        return 0
    
    def _compute_covariance_matrix(self, dataloader):
        self.covariance_matrix_ = torch.zeros((self.dim,self.dim), dtype = torch.float64) # get dimension from data
        if self.device == "gpu":
            self.covariance_matrix_ = self.covariance_matrix_.cuda().float()
                    
        for batch in tqdm(dataloader,desc='Covariance progress',leave=True):
            SNP,true,sex,age = batch
            
            # Put to GPU if settings are so (will throw an error if no gpu available)
            if self.device == "gpu":
                SNP = SNP.cuda().float()
            SNP = torch.subtract(SNP, self.means_)
            
            self.covariance_matrix_ = torch.add(self.covariance_matrix_, torch.matmul(SNP.T,SNP))
        
        self.covariance_matrix_ = torch.div(self.covariance_matrix_, self.x_len - 1) ## np.cov also divides by number of samples
        
        return 0
        
    def _compute_spectral_decomposition(self):
        #print("Device: (-1 if on CPU)",self.covariance_matrix_.get_device())
        self.explained_variance_, self.components_ = torch.linalg.eig(self.covariance_matrix_) #use eigh instead
        self.explained_variance_ = torch.real(self.explained_variance_)
        self.components_ = torch.real(self.components_)
        
        return 0
        
    def transform(self,X_dataset):
        '''
        Transform X_dataset to lower dimension.
        Input:
        train_dataset: A hdf5Dataset containing the data points (hdf5Dataset implements torch.utils.data.Dataset)
        Output:
            torch.tensor of shape nxd
        Transforms:
         - nxd dataset to nx(n_components) dataset
        '''
        
        dataloader = DataLoader(X_dataset,self.batch_size,num_workers=self.n_jobs)
        
        dataset_x_len = len(X_dataset)        
        transformed_data = torch.zeros((dataset_x_len,self.n_components), dtype = torch.float64)
        
        ### Batchwise data loading and transformation
        idx = 0
        for batch in tqdm(dataloader,desc='Transform progress',leave=True):
            SNP,true,sex,age = batch
            
            # Put to GPU if settings are so (will throw an error if no gpu available)
            if self.device == "gpu":
                SNP = SNP.cuda().float()
            SNP = torch.subtract(SNP, self.means_)
            batch_len = SNP.shape[0]
            
            transformed_batch = torch.matmul(self.components_[:,:self.n_components].float().T, SNP.T).T
            
            transformed_data[idx:idx+batch_len,:] = transformed_batch
            idx += batch_len
            # self.covariance_matrix_ = torch.add(self.covariance_matrix_, torch.matmul(SNP.T,SNP))
        
        return transformed_data