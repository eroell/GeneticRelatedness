import argparse
import logging
import os
import pathlib
import h5py as h5
import numpy as np
import pandas as pd

from src.KNN_PCA_enabled import bigKNN_PCA_enabled
from torch.utils.data import Subset

from src.utils import Height_Statistics_np, hdf5Dataset, dataset_split, cross_val_score

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


def cross_val_score_pca(model,dataset,n_splits=5,random_state=47,
                        store_dist_mat=False,n_components=None,data_set_name=None,
                        n_neighbors=None,w_state=None,metric=None,p=None,
                        adjusted=None, device=None, intermediate_dir=None, distance_matrix_dir=None,
                        ):
    '''
    perform crossvalidation of K-nearest-neighbours estimator for the Biobank data and height.
    Input:
    - model: A bigKNN object
    - dataset: a hdf5File
    - n_splits: n_splits-Fold crossvalidation
    -random_state: the random state used for the split generation
    -rest: arguments to store the results in uniquely named filenames
    '''
    # for each of the n_splits splits, create the train-indices and test-indices 
    split_lys = dataset_split(dataset,n_splits,random_state)
    # create list to store the evaluation of each of the n_splits splits
    evaluate_metrics_lys = []
    # run n_splits-crossvalidation
    for i,split in enumerate(split_lys):
        
        ### Load corresponding hdf5 file:
        pca_fyle = 'pca_intermediates_cvsplit_'+str(i)+'_dataset_toy_random_state_47_device_cpu.hdf5'

        with h5.File(os.path.join(intermediate_dir, pca_fyle), 'r') as hf:
        #    # hf = h5.File(os.path.join(intermediates_path, "cov_mat_10k_bellot_device_cpu_batchsize_400.hdf5"), 'r')
        #    # cov_loaded = np.array(hf['/covariance_matrix'])
            inv_cov_loaded = np.array(hf['/inverted_covariance_matrix'])
        #    # expl_var_loaded = np.array(hf['/explained_variance'])
        #    components = np.array(hf['/components'])[:,:n_components]
            hf.close()

        # not a clean fix/solution but does the trick
        model.VI = inv_cov_loaded
        train_dataset = Subset(dataset, split['train_idxs'])
        test_dataset = Subset(dataset, split['test_idxs'])
    
        model.fit(train_dataset,test_dataset)#,components=components)
        model.predict()
        evaluate_metrics_lys.append(model.evaluate_metrics())

        # store data
        if store_dist_mat:
            fylename = f"distance_matrix_cv_split_{i}_n_components_{n_components}_dataset_{data_set_name}_n_neighbors_{n_neighbors}_wstate_{w_state}_metric_{metric}_p_{p}_adjusted_{adjusted}_device_{device}"
                
            np.savez_compressed(os.path.join(distance_matrix_dir, fylename),
                                distance_matrix_ = model.distance_matrix_,
                                neighbor_metadata_ = model.neighbor_metadata_,
                                y_metadata_ = model.y_metadata_)
        
    return evaluate_metrics_lys

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_components',
        type=int,
        default=0,
        help='number of principal components to be considered'
    )

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
        '--w_state',
        type=str,
        default='BETA',
        help='Use to weight the positions'
    )

    parser.add_argument(
        '--metric',
        type=str,
        default='euclidean',
        help='The metric used to find the KNN'
    )

    parser.add_argument(
        '--p',
        type=float,
        default=2,
        help='Parameter p used for fractional distance measure'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default='uniform',
        help='Use uniform or distance weighted KNN'
    )
    
    parser.add_argument(
        '--unadjusted',
        action='store_false',
        help='Average the adjusted or raw phenotypes'
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
        '-nj', '--n_jobs',
        type=int,
        default=4,
        help='The number of parallel processes to load the data. Default is 4.'
    )
    
    
    parser.add_argument(
        '--intermediate_dir',
        default=pathlib.Path(__file__).resolve().parents[1] / 'intermediates',
        type=str,
        help='path to precomputed pca intermediates (PCs)'
    )
    
    parser.add_argument(
        '--distance_matrix_dir',
        default=pathlib.Path(__file__).resolve().parents[1] / 'distance_matrices',
        type=str,
        help='where to store the computed distance matrices'
    )

    parser.add_argument(
        '-o', '--output',
        default=pathlib.Path(__file__).resolve().parents[1] / 'output',
                                                               #/ name,
        type=str,
        help='Output path for storing the evaluation metrics.'
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

    logging.info(f'n_components: {args.n_components}')
    logging.info(f'Dataset: {args.dataset}')
    logging.info(f'n_neighbors: {args.n_neighbors}')
    logging.info(f'w_state: {args.w_state} (will be set to None internally if metric==mahalanobis')
    logging.info(f'Metric: {args.metric}')
    logging.info(f'p: {args.p}')
    logging.info(f'weights: {args.weights}')
    logging.info(f'Adjusted: {args.unadjusted}')
    logging.info(f'Device: {args.device}')
    logging.info(f'Batch size for DataLoader: {args.batch_size}')
    logging.info(f'n_jobs: {args.n_jobs}')

    ### Modified to run on my machine
    if args.locality:
        data_path = '/home/eljas/ownCloud/SLR/HeightPrediction/eljas/toydata'
    else:
    #    data_path = '/home/roelline/HeightPrediction/eljas/toydata'
    #    data_path = '/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data'
        data_path = '/local0/scratch/roelline/files'      
    
    filename = f'evaluate_metrics_{args.dataset}_n_components_{args.n_components}_n_neighbours_{args.n_neighbors}_wstate_{args.w_state}_weights_{args.weights}_metric_{args.metric}_p_{args.p}_adjusted_{args.unadjusted}_njobs_{args.n_jobs}_batchsize_{args.batch_size}_device_{args.device}.csv'
                
    output_filename = os.path.join(args.output, filename)

    #X_file = f'Xy_train_{args.dataset}.hdf5'
    #y_file = f'Xy_val_{args.dataset}.hdf5'
    
    X_file = 'Xy_toy_train.hdf5'
    y_file = 'Xy_toy_val.hdf5'

    if args.locality:
        # w = np.ones(10000)
        w = np.ones(100)
    else:
        df = pd.read_csv(os.path.join(data_path,'GWAS_p_vals.csv'),index_col='SNP')
        snp_ids = np.loadtxt(os.path.join(data_path,'top100snps.txt'),dtype=str)
        if args.w_state == "BETA":
            w = df[df.index.isin(snp_ids)]['BETA'].values
        elif args.w_state == "p":
            w = df[df.index.isin(snp_ids)]['p'].values
        else: # w_state == None
            w = np.ones(100)
        
    w = np.abs(w)/np.sum(np.abs(w))
    
    model = bigKNN_PCA_enabled(n_neighbors=args.n_neighbors,
                metric=args.metric,
                p=args.p,
                weights=args.weights,
                adjusted=args.unadjusted,
                data_path=data_path,
                batch_size=args.batch_size,
                device=args.device,
                n_jobs=args.n_jobs,
                scorer=Height_Statistics_np(),
                w=w)

    ### Load dataset
    train_dataset = hdf5Dataset(data_path,X_file)

    ### Doing n-fold Crossvalidation
    crossvalscore = cross_val_score_pca(model,
                                    train_dataset,
                                    n_splits = 5,
                                    store_dist_mat = True,
                                    # n_components=args.n_components,
                                    data_set_name=args.dataset,
                                    n_neighbors=args.n_neighbors,
                                    w_state=args.w_state,
                                    metric=args.metric,
                                    p = args.p,
                                    adjusted = args.unadjusted,
                                    device = args.device,
                                    intermediate_dir = args.intermediate_dir,
                                    distance_matrix_dir=args.distance_matrix_dir,
                                    )
    
    ### Print crossvalscore
    logging.info(crossvalscore)    

    # Write csv
    df = get_df_from_cv_results(crossvalscore)
    
    with open(output_filename, 'w') as file:
        file.write('cv_{args.dataset}_n_components_{args.n_components}_n_neighbours_{args.n_neighbors}_weights_{args.weights}_metric_{args.metric}_p_{args.p}_adjusted_{args.unadjusted}_njobs_{args.n_jobs}_batchsize_{args.batch_size}_device_{args.device}\n')
                   
        df.to_csv(file,index=False)

    file.close()