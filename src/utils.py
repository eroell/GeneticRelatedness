# -------------------------------------------------------------------
# UKBiobank - height prediction project
#
# Code for generating the dataframes for prediction of adult height based on SNPs selected from preprocessing and weight at birth
#
# L. Bourguignon
# 05.01.2021
# -------------------------------------------------------------------

# ---- Load packages ---- #
import numpy as np
import pandas as pd
import h5py as h5
import torch
import csv
import os
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

# ---- Declare paths --- #
# home_data_path = #to be changed
data_path = '/local0/scratch/madamer/height-prediction/'
hdf5_path = '/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data/'
hdf5_path_50k = '/links/groups/borgwardt/Data/UKBiobank/genotype_500k/EGAD00010001497/hdf5/'
hdf5_path_big = '/links/groups/borgwardt/Data/UKBiobank/genotype_500k/EGAD00010001497/hdf5/'
csv_path = '/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data/'

# ---- Helper classes ---- #

class Height_Statistics():
    def __init__(self):
        self.means = {'male':175.67252960367327,'female':162.49780160116387}
        self.stds = {'male':6.843439632761728,'female':6.303442466347971}
        self.age_reg_coeffs = {'b_1':0.021869934349245713, 'b_0':-42.680572362398614}

        # Old (calculated on train AND val set)
        # self.means = {'male':175.61737227110672,'female':162.4423303666864}
        # self.stds = {'male':6.847034853772383,'female':6.312158412031903}
        # self.age_reg_coeffs = {'b_1':0.02185025362375086, 'b_0':-42.64027880830227}

    def calculate_height(self,height,sex,age):
        correction_height = age*self.age_reg_coeffs['b_1'] + self.age_reg_coeffs['b_0']
        height = height + correction_height
        female_idx = torch.where(sex==0)
        male_idx = torch.where(sex==1)
        height[female_idx] = height[female_idx]*self.stds['female'] + self.means['female']
        height[male_idx] = height[male_idx]*self.stds['male'] + self.means['male']
        return height

class Height_Statistics_np():
    def __init__(self):
        self.means = {'male':175.67252960367327,'female':162.49780160116387}
        self.stds = {'male':6.843439632761728,'female':6.303442466347971}
        self.age_reg_coeffs = {'b_1':0.021869934349245713, 'b_0':-42.680572362398614}

        # Old (calculated on train AND val set)
        # self.means = {'male':175.61737227110672,'female':162.4423303666864}
        # self.stds = {'male':6.847034853772383,'female':6.312158412031903}
        # self.age_reg_coeffs = {'b_1':0.02185025362375086, 'b_0':-42.64027880830227}

    def calculate_height(self,height,sex,age):
        correction_height = age*self.age_reg_coeffs['b_1'] + self.age_reg_coeffs['b_0']
        height = height + correction_height
        female_idx = np.where(sex==0)[0]
        male_idx = np.where(sex==1)[0]
        height[female_idx] = height[female_idx]*self.stds['female'] + self.means['female']
        height[male_idx] = height[male_idx]*self.stds['male'] + self.means['male']
        return height

class Height_Score():
    def __init__(self):
        self.means = {'male':175.67252960367327,'female':162.49780160116387}
        self.stds = {'male':6.843439632761728,'female':6.303442466347971}
        self.age_reg_coeffs = {'b_1':0.021869934349245713, 'b_0':-42.680572362398614}

    def calculate_score(self,df):
        male_df = df[df['"31-0.0"']==1].copy()
        female_df = df[df['"31-0.0"']==0].copy()
        male_df.loc[:,'score'] = (male_df['"50-0.0"']-self.means['male'])/self.stds['male']
        female_df.loc[:,'score'] = (female_df['"50-0.0"']-self.means['female'])/self.stds['female']
        df = pd.concat([male_df,female_df],axis=0).sort_values('id')
        df.loc[:,'score'] = df['score']-(df['"34-0.0"']*self.age_reg_coeffs['b_1'] + self.age_reg_coeffs['b_0'])
        return df.loc[:,['id','score']]

class Height_Statistics_big():
    def __init__(self):
        self.means = {'male':175.6724565726849,'female':162.4978568575439}
        self.stds = {'male':6.843617921273466,'female':6.303270989924073}
        self.age_reg_coeffs = {'b_1':0.021870142331929222, 'b_0':-42.68098588591243}

    def calculate_height(self,height,sex,age):
        correction_height = age*self.age_reg_coeffs['b_1'] + self.age_reg_coeffs['b_0']
        height = height + correction_height
        female_idx = torch.where(sex==0)
        male_idx = torch.where(sex==1)
        height[female_idx] = height[female_idx]*self.stds['female'] + self.means['female']
        height[male_idx] = height[male_idx]*self.stds['male'] + self.means['male']
        return height

class Height_Score_big():
    def __init__(self):
        self.means = {'male':175.6724565726849,'female':162.4978568575439}
        self.stds = {'male':6.843617921273466,'female':6.303270989924073}
        self.age_reg_coeffs = {'b_1':0.021870142331929222, 'b_0':-42.68098588591243}

    def calculate_score(self,df):
        male_df = df[df['"31-0.0"']==1].copy()
        female_df = df[df['"31-0.0"']==0].copy()
        male_df.loc[:,'score'] = (male_df['"50-0.0"']-self.means['male'])/self.stds['male']
        female_df.loc[:,'score'] = (female_df['"50-0.0"']-self.means['female'])/self.stds['female']
        df = pd.concat([male_df,female_df],axis=0).sort_values('id')
        df.loc[:,'score'] = df['score']-(df['"34-0.0"']*self.age_reg_coeffs['b_1'] + self.age_reg_coeffs['b_0'])
        return df.loc[:,['id','score']]
    
    
class hdf5Dataset(Dataset):
    def __init__(self,data_path='data',f_name=None, phenotype=None):
        super().__init__()
        self.hf = h5.File(os.path.join(data_path,f_name),'r')
        self.X = self.hf['X/SNP']
        if phenotype is None:
            self.Height = self.hf['y/Height']
        else:
            self.Height = phenotype
        self.Sex = self.hf['X/Sex']
        self.Age = self.hf['X/Age']
    def __getitem__(self,idx):
        return (self.X[idx].astype(np.float32),self.Height[idx],self.Sex[idx],self.Age[idx])
    def __len__(self):
        return self.X.shape[0]


def dataset_split(dataset,n_splits=3,random_state=47):
    '''
    Creates a list of dictionaries of the datasets splits:
    [dict_1,...,dict_n_splits], where dict_1={train_idxs: [12,17,...], test_idxs:[2,22,...]}
    Input:
    - dataset: a hdf5File
    - n_splits: n_splits-Fold crossvalidation
    '''
    # Create reproducible random number generator
    rng = np.random.RandomState(random_state)
    # Create reproducible random permuation of indices of dataset
    indices = rng.permutation(len(dataset))
    # Create data container for dictionaries
    split_lys = []
    # Create one dictionary for each of the n_splits splits
    for i in range(n_splits):
        left_idx = int(len(dataset)/n_splits * i)
        right_idx = int(len(dataset)/n_splits * (i+1))
        train_idxs = np.concatenate((indices[:left_idx],indices[right_idx:]))
        test_idxs = indices[left_idx:right_idx]
        split_lys.append({'train_idxs': train_idxs, 'test_idxs':test_idxs})
    
    return split_lys

def cross_val_score(model,dataset,n_splits=3,random_state=47):#,store_dist_mat=False,outdir=None,metric=None,p=None,n_neighbors=None,seed=None,device=None):
    '''
    similar to to scikit-learn method to perform crossvalidation of K-nearest-neighbours estimator for the Biobank data and height.
    Input:
    - model: A bigKNN object
    - dataset: a hdf5File
    - n_splits: n_splits-Fold crossvalidation
    '''
    # for each of the n_splits splits, create the train-indices and test-indices 
    split_lys = dataset_split(dataset,n_splits,random_state)
    # create list to store the evaluation of each of the n_splits splits
    evaluate_metrics_lys = []
    # run n_splits-crossvalidation
    for i,split in enumerate(split_lys):
        train_dataset = Subset(dataset, split['train_idxs'])
        test_dataset = Subset(dataset, split['test_idxs'])
    
        model.fit(train_dataset,test_dataset)
        model.predict()
        evaluate_metrics_lys.append(model.evaluate_metrics())
        
        # # store data
        # if store_dist_mat:
        #     fylename = f"cv_split_{i}_metric_{metric}_p_{p}_n_neighbors_{n_neighbors}_seed_{random_state}_adjusted_device_{device}"
        #     np.savez_compressed(os.path.join(outdir, fylename),
        #                         distance_matrix_ = model.distance_matrix_,
        #                         neighbor_metadata_ = model.neighbor_metadata_,
        #                         y_metadata_ = model.y_metadata_)
        
    return evaluate_metrics_lys
    
    