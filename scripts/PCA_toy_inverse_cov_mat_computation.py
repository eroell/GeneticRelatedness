#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:39:58 2021

@author: eljas
"""

import numpy as np
import h5py as h5
import os
import argparse
import pathlib
import logging
import torch
import sys

from src.utils import hdf5Dataset
from src.PCA import  bigPCA
'''
Test the written files.

Compare e.g. the variables (in for example Spyder's variable explorer') with PCA_small_test.py
'''
def get_SNPs(data_path,file):
    hf = h5.File(os.path.join(data_path,file),'r')
    return hf['X/SNP'][:,:]

if __name__ == '__main__':
    
    intermediates_path = pathlib.Path(__file__).resolve().parents[1] / 'intermediates'
    
    
    # cov_file = np.load(os.path.join(intermediates_path, "cov_mat_small_device_cpu_batchsize_400.npz"))
    # cov_loaded = cov_file['cov_bigPCA']
    # print(sys.getsizeof(cov_loaded))
    # inv_cov_file = np.load(os.path.join(intermediates_path, "inv_cov_mat_small_device_cpu_batchsize_400.npz"))
    # inv_cov_loaded = inv_cov_file['inv_cov_bigPCA']
    # expl_var_file = np.load(os.path.join(intermediates_path, "expl_var_small_device_cpu_batchsize_400.npz"))
    # expl_var_loaded = expl_var_file['expl_var_bigPCA']
    # comp_file = np.load(os.path.join(intermediates_path, "components_small_device_cpu_batchsize_400.npz"))
    # components_loaded = comp_file['components_bigPCA']
    
    with h5.File(os.path.join(intermediates_path, "pca_intermediates_cvsplit_0_dataset_small_random_state_47_device_cpu.hdf5"), 'r') as hf:
        # hf = h5.File(os.path.join(intermediates_path, "cov_mat_small_device_cpu_batchsize_400.hdf5"), 'r')
        # means_loaded = np.array(hf['/means'])
        cov_loaded = np.array(hf['/covariance_matrix'])
        # inv_cov_loaded = np.array(hf['/inverted_covariance_matrix'])
        expl_var_loaded = np.array(hf['/explained_variance'])
        components_loaded = np.array(hf['/components'])
        hf.close()
        
    cov_inv = np.linalg.inv(cov_loaded)


