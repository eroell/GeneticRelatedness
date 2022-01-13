#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:52:23 2021

@author: eljas
"""

'''
Generate synthetic dataset 1.
3 Clusters in 2 dimensions.
'''


import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import os
import pathlib
from sklearn.datasets import make_blobs

##############################################################################
### Synthetic Dataset 1

# Datapoints and clusters
CLUSTER = 3
N_CLUSTER = 30
N_TEST = 10
centers = [[-10,-10],[1,2],[13,5]]

# Create Data
X, y = make_blobs(n_samples=CLUSTER*N_CLUSTER, centers=centers,n_features=2,random_state=0)
X_test, y_test = make_blobs(n_samples=CLUSTER*N_TEST, centers=centers,n_features=2,random_state=1)

# Construct height, age and sex for training data
height_1 = np.zeros(CLUSTER*N_CLUSTER)
height_1[y==0] = 5
height_1[y==1] = 10
height_1[y==2] = 15
sex_1 = np.ones(N_CLUSTER*CLUSTER) == 1
age_1 = np.ones(N_CLUSTER*CLUSTER) * 1968

# Construct height, age and sex for test data
height_test_1 = np.zeros(CLUSTER*N_TEST)
height_test_1[y_test==0] = 5
height_test_1[y_test==1] = 10
height_test_1[y_test==2] = 15
sex_test_1 = np.ones(N_TEST*CLUSTER) == 1
age_test_1 = np.ones(N_TEST*CLUSTER) * 1968

# Safe Datasets as hdf5 dataset
dataset_path = pathlib.Path(__file__).resolve().parents[0]
with h5.File(os.path.join(dataset_path, "synthetic_1_train.hdf5"), 'w') as f:
    f.create_dataset("X/SNP",data=X)
    f.create_dataset("X/Sex",data=sex_1)
    f.create_dataset("X/Age",data=age_1)
    f.create_dataset("y/Height", data=height_1)#, dtype='float64')
    

with h5.File(os.path.join(dataset_path, "synthetic_1_val.hdf5"), 'w') as f:
    f.create_dataset("X/SNP",data=X_test)
    f.create_dataset("X/Sex",data=sex_test_1)
    f.create_dataset("X/Age",data=age_test_1)
    f.create_dataset("y/Height", data=height_test_1)#, dtype='float64')


# Visualize training and test data
fig, ax = plt.subplots()
ax.set_title("Synthetic Dataset 1")
colors = ["#4EACC5", "#FF9C34","#4E9A06"]
colors_test = ['b','m','r']
for k,center in enumerate(centers):
    cluster_data = y==k
    ax.scatter(X[cluster_data,0],X[cluster_data,1],c=colors[k],label = f"C{k}")

for k,center in enumerate(centers):
    cluster_data = y_test==k
    ax.scatter(X_test[cluster_data,0],X_test[cluster_data,1],c=colors_test[k],label = f"test_C{k}")
ax.legend()