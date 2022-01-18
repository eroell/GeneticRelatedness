#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 15:24:01 2021

@author: eljas
"""
##############################################################################
##############################################################################
### Benchmarking for different distance calculations
##############################################################################
##############################################################################

import numpy as np
import time
from scipy.spatial.distance import cdist
from tqdm import tqdm
from fastdist import fastdist
import torch

import matplotlib.pyplot as plt

### Generate nxd data matrix (simulating training data of KNN)
def gen_Xtrain(n,d,seed):
    np.random.seed(seed)
    return 2*np.random.rand(n,d)

### Generate mxd data matrix (simulating validation data of KNN)
def gen_Xval(m,d):
    return 2*np.random.rand(m,d)

# def brute_minkowski(Xtrain,Xval,p):
#     ds_mink = np.zeros((Xtrain.shape[0], Xval.shape[0]))
#     for i in range(Xtrain.shape[0]):
#         for j in range(Xval.shape[0]):
#             ds_ij = 0
#             for k in range(Xtrain.shape[1]):
#                 ds_ij += (np.abs(Xtrain[i,k] - Xval[j,k]))**p
#             ds_mink[i,j] = ds_ij**(1/p)
#     return ds_mink

# def improved_minkowski(Xtrain,Xval,p):
#     ds_mink = np.zeros((Xtrain.shape[0], Xval.shape[0]))
#     for i in range(Xtrain.shape[0]):
#         for j in range(Xval.shape[0]):
#             ds_mink[i,j] = sum(np.abs((Xtrain[i,:]-Xval[j,:]))**p)**(1/p)
#             # ds_ij = 0
#             # for k in range(Xtrain.shape[1]):
#                 # ds_ij += (np.abs(Xtrain[i,k] - Xval[j,k]))**p
#             # ds_mink[i,j] = ds_ij**(1/p)
#     return ds_mink

### Own Minkowski implementation: I am not sure whether more vectorization is possible.
### Probably it is.
def own_minkowski(Xtrain,Xval,p):
    ds_mink = np.zeros((Xtrain.shape[0], Xval.shape[0]))
    for j in range(Xval.shape[0]):
        ds_mink[:,j] = np.sum(np.abs((Xtrain[:,:]-Xval[j,:]))**p, axis = 1)**(1/p)

    return ds_mink


def fastdist_distmatrix(Xtrain, Xval, p):
    ds_fastdist = np.zeros((Xtrain.shape[0], Xval.shape[0]))
    for i in range(Xval.shape[0]):
        for j in range(Xtrain.shape[0]):
            ds_fastdist[i,j] = fastdist.minkowski(Xtrain[i,:],Xval[j,:],p=p)
    return ds_fastdist

def run_benchmarking(distance_funs, n_vec, d, repeats = 1):
    '''
    Run different distanc-matrix-calculating-functions.
    Note I here set n=m for "easy" visualization
    
    
    Parameters
    ----------
    distance_funs : Nested list of dimension #functions x 2
        Example: [['function_description_string', function parameters], ...]

    n_vec : List of ints
        Simulating number of samples.
        
    d : int
        Simulating dimensionality
        
    repeats : int, optional
        Number of repeats of the experiment. The default is 1.

    Returns
    -------
    exec_means: Float numpy array of dimension #functions x len(n_vec)
        Mean across the repeats
    exec_sd: Float numpy array of dimension #functions x len(n_vec)
        Standard deviation across the repeats

    '''
    seed = 42
    # exec_time = {}
    
    exec_times = np.zeros((repeats, len(n_vec), len(distance_funs)) )
    
    # qdm(n_m_vec, desc="n progress", leave=False)
    for r in tqdm(range(repeats), desc = "repeat progress", leave = False):

        for n,n_val in tqdm(enumerate(n_vec), desc = "n progress", leave = False):
            Xtrain = gen_Xtrain(n_val,d,seed)
            Xval = gen_Xval(n_val,d)            
            
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            Xtrain_t = torch.from_numpy(Xtrain)
            Xval_t = torch.from_numpy(Xval)
            
            for i,dist in enumerate(distance_funs):
                
                if dist[0] == 'cdist_euclidean':
                    ### cdist euclidean
                    st = time.time()
                    ds_cdist_euclidean = cdist(Xtrain, Xval, metric = 'euclidean')
                    t = time.time() - st
                    exec_times[r,n,i] = t
                    
                elif dist[0] == 'cdist_minkowski':
                    ### cdist minkowski
                    st = time.time()
                    ds_cdist_mink = cdist(Xtrain, Xval, metric = 'minkowski', p = dist[1])
                    t = time.time() - st
                    exec_times[r,n,i] = t

                elif dist[0] == 'own_minkowski':
                    ### own_minkowski
                    st = time.time()
                    ds_own_mink = own_minkowski(Xtrain, Xval,dist[1])
                    t = time.time() - st
                    exec_times[r,n,i] = t

                elif dist[0] == 'fastdist':
                    ### fastdist
                    st = time.time()
                    ds_fastdist_euclidean = fastdist_distmatrix(Xtrain, Xval, p = dist[1])
                    t = time.time() - st
                    exec_times[r,n,i] = t
                    
                elif dist[0] == 'cdist_torch':
                    ### torch
                    st = time.time()
                    ds_cdist_torch = torch.cdist(Xtrain_t, Xval_t, p = dist[1])
                    t = time.time() - st
                    exec_times[r,n,i] = t
                   
                    
                else:
                    print(f"Not valid distance function: {dist[0]}")
                    return 0
                
    exec_means = np.mean(exec_times, axis = 0).T
    exec_sd = np.std(exec_times, axis = 0).T
    return exec_means, exec_sd

    

### Plot the execution times for differen values of n for a run_benchmarking run
def plot_benchmark(n_lys, exec_means, exec_sd, label_lys, repeats):
    for i in range(len(exec_means)):
        # plt.plot(n_lys, exec_means[i], label = label_lys[i], linestyle = "-.")
        plt.errorbar(n_lys, exec_means[i], yerr = exec_sd[i], label = label_lys[i], linestyle = "-.")
    plt.ylabel("time (s)")
    plt.xlabel("n (number of vectors)")
    plt.title(f"Time for pairwise vector distance calculations of two nxd matrices (d fixed to 10'000).\nMean of {repeats} runs. Errorbars are standard deviation. \n(Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz)")
    plt.legend()
    plt.savefig('Distance_Calc_Benchmarking_tests.png')
    plt.show()

if __name__ == '__main__':
    
    # set repeats
    # repeats = 3
    
    # # set n (m will automatically be set to n inside run_benchmarking)

    # n_lys = [100,200,500,1000]
    # # n_lys = [100,200,400]

    # # set d
    # d = 10000
    
    # '''
    # Define distance functions of interest for run_benchmarking
    # '''
    # distance_funs = [['cdist_euclidean', None],
    #               ['cdist_minkowski', 2],
    #                ['cdist_minkowski', 1/2],
    #                ['own_minkowski', 2],
    #                ['own_minkowski', 1/2],
    #               # ['fastdist', 2],
    #               # ['fastdist', 1/2],
    #               ['cdist_torch',2],
    #               ['cdist_torch',1/2]]
    
    # '''
    # Define distance functions of interest for plotting
    # '''
    # label_lys = ['cdist_euclidean',
    #               'cdist_minkowski p=2',
    #               'cdist_minkowski p=1/2',
    #                'own_minkowski p=2',
    #                'own_minkowski p=1/2',
    #               # 'fastdist_distmatrix p=2',
    #               # 'fastdist_distmatrix p=1/2',
    #               'cdist_torch p=2',
    #               'cdist_torch p=1/2']
    
    # # Benchmarking run
    # exec_means, exec_sd = run_benchmarking(distance_funs, n_lys, d, repeats)
    # print(exec_means)
    # # print(exec_sd)

    # # Plot the results
    # plot_benchmark(n_lys, exec_means, exec_sd, label_lys, repeats)
    
    ##########################################################################
    ##########################################################################
    ##########################################################################
    ### Testing correctness of different distance functions
    n_val = 5
    d = 100
    seed = 42
    
    Xtrain = gen_Xtrain(n_val,d,seed)
    Xval = gen_Xval(n_val,d) 
    
    Xtrain_t = torch.from_numpy(Xtrain)
    Xval_t = torch.from_numpy(Xval)
    
    ds_cdist_euclidean = cdist(Xtrain, Xval, metric = 'euclidean')
    ds_cdist_mink2 = cdist(Xtrain, Xval, metric = 'minkowski', p = 2)
    ds_own_mink2 = own_minkowski(Xtrain, Xval,2)
    ds_fastdist_mink2 = fastdist_distmatrix(Xtrain, Xval, 2)
    ds_fastdist_matmat_mink2 = fastdist.matrix_to_matrix_distance(Xtrain, Xval, fastdist.euclidean,'euclidean')
    ds_cdist_torch2 = torch.cdist(Xtrain_t, Xval_t, p = 2)
    
    print("cdist_euclidean || cdist_mink2", np.allclose(ds_cdist_euclidean, ds_cdist_mink2))
    print("cdist_euclidean || own_mink2", np.allclose(ds_cdist_euclidean, ds_own_mink2))
    print("cdist_euclidean || fastdist_mink2", np.allclose(ds_cdist_euclidean, ds_fastdist_mink2))
    print("cdist_euclidean || fastdist_matmat_mink2", np.allclose(ds_cdist_euclidean, ds_fastdist_matmat_mink2))
    print("cdist_euclidean || ds_cdist_torch2", np.allclose(ds_cdist_euclidean, ds_cdist_torch2))

    ds_cdist_mink05 = cdist(Xtrain, Xval, metric = 'minkowski', p = 0.5)
    ds_own_mink05 = own_minkowski(Xtrain, Xval,0.5)
    ds_fastdist_mink05 = fastdist_distmatrix(Xtrain, Xval, 0.5)
    # ds_fastdist_matmat_mink05 = fastdist.matrix_to_matrix_distance(Xtrain, Xval, fastdist.minkowski, 'minkowski')
    ds_cdist_torch05 = torch.cdist(Xtrain_t, Xval_t, p = 0.5)
    
    print("cdist_mink05 || own_mink05", np.allclose(ds_cdist_mink05, ds_own_mink05))
    print(ds_cdist_mink05)
    print(ds_own_mink05)
    print("cdist_mink05 || fastdist_mink05", np.allclose(ds_cdist_mink05, ds_fastdist_mink05))
    print("cdist_mink05 || ds_cdist_torch05", np.allclose(ds_cdist_mink05, ds_cdist_torch05))
    
    
    
        