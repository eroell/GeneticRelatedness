#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 18:42:55 2021

@author: eljas
"""

'''
Test bigPCA here
'''


from sklearn.decomposition import PCA
import numpy as np
import h5py as h5
import os
import argparse

from src.utils import hdf5Dataset
from src.PCA import  bigPCA


'''
Test bigPCA on the small toy dataset; compare the PCA's results using
np.allclose versus sklearn.decomposition's PCA (used as gold-standard)

'''

def get_SNPs(data_path,file):
    hf = h5.File(os.path.join(data_path,file),'r')
    return hf['X/SNP'][:,:]

def test_eigenvectors(components_PCA, components_bigPCA, n_vectors=50):
    '''
    Test whethere eigenvectors from different implementations are np.allclose():
    Opposite sign is allowed.
    
    Parameters
    ----------
    components_pca : dxd numpy.ndarray
        eigenvectors are coloumns
    components_bigPCA : dxd numpy.ndarray
        eigenvectors are coloumns
    n_vectors : int
        How many of the first n_vectors are compared. The default is 80.
        Since the last eigenvectors should belong to the smallest eigenvalues,
        they are unlikely to be used in further computations and we accept
        numerical unstability there.

    Returns
    -------
    bool
        Wether the first n_vectors eigenvectors are np.allclose (up to sign)

    '''
    assert components_pca.shape == components_bigPCA.shape
    
    
    # Test each pair of eigenvectors for np.allclose-ness up to sign
    allclose = True
    for i in range(min(components_PCA.shape[1],n_vectors)):
        bool_same_sign = np.allclose(components_PCA[:,i],components_bigPCA[:,i], rtol=1e-4, atol = 1e-5)
        bool_opp_sign = np.allclose(-components_PCA[:,i],components_bigPCA[:,i], rtol=1e-4, atol = 1e-5)
        
        if not bool_same_sign and not bool_opp_sign:
            allclose=False
            break
    
    return allclose


def get_PCs_signs(components_PCA, components_bigPCA):
    '''
    For each PC of components_PCA and components_bigPCA, check if they are of same or of oppisite sign.
    Return a np.ndarray of +1s and -1s (+1 if same sign, -1 if opposite sign)
    
    Parameters
    ----------
    components_pca : dxn_components numpy.ndarray
        eigenvectors are coloumns
    components_bigPCA : dxn_components numpy.ndarray
        eigenvectors are coloumns

    Returns
    -------
    signs
        np.ndarray filled with +1 and -1
    '''
    assert components_PCA.shape == components_bigPCA.shape
    
    # Test each pair of eigenvectors for np.allclose-ness up to sign
    signs = np.zeros(components_PCA.shape[1])
    for i in range(components_PCA.shape[1]):
        bool_same_sign = np.allclose(components_PCA[:,i],components_bigPCA[:,i], rtol=1e-3, atol = 1e-3)
        bool_opp_sign = np.allclose(-components_PCA[:,i],components_bigPCA[:,i], rtol=1e-3, atol = 1e-3)
        
        if bool_same_sign:
            signs[i] = 1
        elif bool_opp_sign:
            signs[i] = -1
        # else:
            ## should not happen. could throw exception if want to
            
    return signs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--device',
      type=str,
      default='cpu',
      help='cpu or gpu',
      required=False,
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
    
    ### Set data paths
    if args.locality:
        data_path = '/home/eljas/ownCloud/SLR/HeightPrediction/eljas_2/toydata'
    else:
        data_path = '/home/roelline/HeightPrediction/eljas_2/toydata'

            
    X_file = 'Xy_small_toy_train.hdf5'
    
    
    ### Load dataset
    train_dataset = hdf5Dataset(data_path,X_file)
    X = get_SNPs(data_path, X_file)
     
    ### Instantiate and fit sklearn.decomposition.PCA
    pca = PCA()
    pca.fit(X)
    # pca.transform(X)    
    # X_means = np.mean(X,axis=0)
    # print(model.means_[:10])
    # print(X_means[:10])
    # print(np.allclose(model.means_, X_means))
        
    
    ### Instantiate and fit bigPCA
    model = bigPCA()
    model.fit(train_dataset)
    
    ### Test means
    means = np.mean(X,axis=0)
    means_bigPCA = model.means_.numpy()
    print(f"\nMeans allclose: {np.allclose(means, means_bigPCA, rtol=1e-5, atol=1e-8)}")
       
    ### Test covariance matrix
    cov = np.cov(X, rowvar=False)
    cov_bigPCA = model.covariance_matrix_.numpy()
    # print(cov[:5,:5])
    # print(cov_bigPCA[:5,:5])
    print(f"\nCovariance matrix allclose: {np.allclose(cov, cov_bigPCA, rtol=1e-3, atol=1e-8)}")
    
    ### Test eigenvalues
    explained_var_pca = pca.explained_variance_
    explained_var_bigPCA = model.explained_variance_.numpy()
    print(f"Eigenvalues allclose(rtol=1e-5, atol = 1e-8) (only first 80/100 considered): {np.allclose(explained_var_pca[:80],explained_var_bigPCA[:80], rtol=1e-5, atol = 1e-8)}")
    
    ### Test eigenvectors
    components_pca = pca.components_.T
    components_bigPCA = model.components_.numpy()
    print(f"Eigenvectors allclose(rtol=1e-4, atol = 1e-5) (only first 40/100 considered): {test_eigenvectors(components_pca,components_bigPCA,n_vectors=40)}")

    ### Test Transformation
    N_COMPONENTS = 5
    # sklearn.decomposition.PCA
    pca_trafo = PCA(n_components = N_COMPONENTS)
    pca_trafo.fit(X)
    X_pca_trafo = pca_trafo.transform(X)
    
    # bigPCA
    model_trafo = bigPCA(n_components=N_COMPONENTS, device = args.device)
    
    model_trafo.fit(train_dataset)
    # Make first 5 PC's of both method to be signed equally
    components_signs = get_PCs_signs(pca_trafo.components_.T, model_trafo.components_[:,:N_COMPONENTS])
    for i in range(len(components_signs)):
        model_trafo.components_[:,i] *= components_signs[i]
        print(components_signs[i])
    
    X_bigPCA_trafo = model_trafo.transform(train_dataset).numpy()
    
    print(X_pca_trafo[:5,:5])
    print(X_bigPCA_trafo[:5,:5])    
    
    print(f"\n5 Major PC's Transformed X-data np.allclose(rtol=1e-3, atol=1e-3)): {np.allclose(X_pca_trafo, X_bigPCA_trafo, rtol=1e-3, atol=1e-3)}")
    





