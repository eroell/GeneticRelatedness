import argparse
import logging
import os
import pathlib
import json
import h5py as h5
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.KNN import bigKNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from src.utils import Height_Statistics_np, hdf5Dataset, dataset_split, cross_val_score


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

def get_df_from_cv_results(crossvalscore):
    # Writes convenient CVs from the crossvalidation results
    # Is a bit hacky, but does not need to change interfaces of the established
    # functions
    
    df = pd.DataFrame({'Split':[],'MSE':[],'MAE':[],'r': [],'r2': []})
    split_nr = 1
    for dic in crossvalscore:
        mse_mae_r_r2 = [str(split_nr)]
        split_nr += 1
        for key in dic:
            mse_mae_r_r2.append(dic[key])
        df.loc[len(df.index)] = mse_mae_r_r2
        
    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--dataset',
      type=str,
      default='toy',
      help='Which dataset to use',
      required=False,
    )

    parser.add_argument(
        '-nn', '--n_neighbors',
        type=int,
        default=1000,
        help='The number of nearest neighbours'
    )

    parser.add_argument(
        '--weights',
        type=str,
        default='uniform',
        help='Use uniform or distance weighted KNN'
    )

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
        default=4,
        help='The number of parallel processes to load the data. Default is 4.'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='cpu or gpu to use. make sure gpu is available if wanting one.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=400,
        help='Batch size for data loader. For 10000 SNPs, 400 should not be exceeded.'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=pathlib.Path(__file__).resolve().parents[1] / 'output',
                                                               #/ name,
        type=str,
        help='Output path for storing the results.'
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

    logging.info(f'Dataset: {args.dataset}')
    logging.info(f'Metric: {args.metric}')
    logging.info(f'Adjusted: {args.unadjusted}')
    logging.info(f'p: {args.p}')
    logging.info(f'Batch size for DataLoader: {args.batch_size}')
    logging.info(f'Device: {args.device}')

    ### Modified to run on my machine
    # data_path = '/home/michael/ETH/data/HeightPrediction'
    if args.locality:
        data_path = '/home/eljas/ownCloud/SLR/HeightPrediction/eljas/toydata'
    else:
    #    data_path = '/home/roelline/HeightPrediction/eljas/toydata'
    #    data_path = '/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data'
        data_path = '/local0/scratch/roelline/files'
        
    
    filename = f'cv_{args.dataset}_n_neighbours_{args.n_neighbors}_weights_{args.weights}_metric_{args.metric}_adjusted_{args.unadjusted}_njobs_{args.n_jobs}_batchsize_{args.batch_size}_device_{args.device}.csv'
    output_filename = os.path.join(args.output, filename)

    X_file = 'Xy_toy_train.hdf5'
    y_file = 'Xy_toy_val.hdf5'

    if args.locality:
        w = np.ones(100)
    else:
        df = pd.read_csv(os.path.join(data_path,'GWAS_p_vals.csv'),index_col='SNP')
        top100 = np.loadtxt(os.path.join(data_path,'top100snps.txt'),dtype=str)
        w = df[df.index.isin(top100)]['BETA'].values
        # w = (1./w)/np.sum(1./w)
        w = np.abs(w)/np.sum(np.abs(w))
    
        #w = np.ones(100)
    
    model = bigKNN(n_neighbors=args.n_neighbors,
                weights=args.weights,
                metric=args.metric,
                n_jobs=args.n_jobs,
                adjusted=args.unadjusted,
                scorer=Height_Statistics_np(),
                data_path=data_path,
                batch_size=args.batch_size,
                w=w,
                device=args.device,
                output=args.output,
                p=args.p)

    ### Load dataset
    train_dataset = hdf5Dataset(data_path,X_file)

    ### Doing n-fold Crossvalidation
    crossvalscore = cross_val_score(model, train_dataset,n_splits = 5)

    ### Print crossvalscore
    logging.info(crossvalscore)

    #### Writing the file
    output = {
                'dataset': args.dataset,
                'neighbors': args.n_neighbors,
                'weights':args.weights,
                'metric':args.metric,
                'adjusted':args.unadjusted,
                'batch_size':args.batch_size,
                'device':args.device,
                }
                
    #f = open(output_filename,"w")
    #json.dump(output, f, indent=4)
    #json.dump(crossvalscore,f,indent=4)
    #f.close()

    # Write csv
    df = get_df_from_cv_results(crossvalscore)
    
    with open(output_filename, 'w') as file:
        file.write(f'dataset_{args.dataset}_n_neighbours_{args.n_neighbors}_weights_{args.weights}_metric_{args.metric}_adjusted_{args.unadjusted}_njobs_{args.n_jobs}_batchsize_{args.batch_size}_device_{args.device}\n')
        df.to_csv(file,index=False)

    file.close()

