import argparse
import logging
import os
import pathlib
import json
import h5py as h5
import numpy as np
import pandas as pd

from src.KNN import bigKNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from src.utils import Height_Statistics_np, hdf5Dataset

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
        default=10,
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
        default=2,
        help='Parameter p used for fractional distance measure'
    )

    parser.add_argument(
        '--unadjusted',
        action='store_false',
        help='Average the adjusted or raw phenotypes'
    )

    parser.add_argument(
        '-o', '--output',
        default=pathlib.Path(__file__).resolve().parents[1] / 'output',
                                                               #/ name,
        type=str,
        help='Output path for storing the results.'
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

    # Create the output directory for storing all results of the
    # individual combinations.
    os.makedirs(args.output, exist_ok=True)

    # Basic log configuration to ensure that we see where the process
    # spends most of its time.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )

    logging.info(f'Dataset: {args.dataset}')
    logging.info(f'Neighbors: {args.n_neighbors}')
    logging.info(f'Weight: {args.weights}')
    logging.info(f'Metric: {args.metric}')
    logging.info(f'Adjusted: {args.unadjusted}')
    logging.info(f'Batch size for DataLoader: {args.batch_size}')
    logging.info(f'Device: {args.device}')
    
    output = {
        'dataset': args.dataset,
        'neighbors': args.n_neighbors,
        'weights':args.weights,
        'metric':args.metric,
        'adjusted':args.unadjusted,
        'batch_size':args.batch_size,
        'device':args.device
        }
        
    filename = f'test_wrap_{args.dataset}_n_neighbours_{args.n_neighbors}_weights_{args.weights}_metric_{args.metric}_adjusted_{args.unadjusted}_njobs_{args.n_jobs}_batchsize_{args.batch_size}_device_{args.device}.txt'
    output_filename = os.path.join(args.output, filename)

    ### Modified to run on my machine
    # data_path = '/home/michael/ETH/data/HeightPrediction'
    if args.locality:
        data_path = '/home/eljas/ownCloud/SLR/HeightPrediction/eljas/toydata'
    else:
    #    data_path = '/home/roelline/HeightPrediction/eljas/toydata'
        data_path = '/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data'
    #    data_path = '/local0/scratch/roelline/files'
        
    if not os.path.exists(output_filename) or args.force:
        
        X_file = 'Xy_small_toy_train.hdf5'
        y_file = 'Xy_small_toy_val.hdf5'
        y_metadata = get_metadata(data_path,y_file)
        y_true = get_phenotype(data_path,y_file)
        X = get_SNPs(data_path,X_file)
        y = get_SNPs(data_path,y_file)
        X_phenotype = get_phenotype(data_path,X_file)
        X_metadata = get_metadata(data_path,X_file)
        scorer = Height_Statistics_np()

        model_linear = LinearRegression()
        model_linear.fit(X,X_phenotype)
        predictions = model_linear.predict(y)
        if args.unadjusted:
            predictions = scorer.calculate_height(predictions,y_metadata[0],y_metadata[1])

        print("Linear Regression Results: ", evaluate_metrics(y_true,predictions,scorer=scorer,y_metadata=y_metadata))


        # model_base = KNeighborsRegressor(n_neighbors=args.n_neighbors,
        #                                  weights=args.weights,
        #                                  metric=args.metric)
        # model_base.fit(X,X_phenotype)
        # predictions = model_base.predict(y)
        # if args.unadjusted:
        #     predictions = scorer.calculate_height(predictions,y_metadata[0],y_metadata[1])
        #
        # print(bigKNN(scorer=scorer).evaluate_metrics(y_true,predictions,y_metadata=y_metadata))

        del X,y,X_phenotype,X_metadata,y_metadata,y_true,scorer
        
        if args.locality:
            w = np.ones(100)
        else:
            df = pd.read_csv(os.path.join(data_path,'GWAS_p_vals.csv'),index_col='SNP')
            top100 = np.loadtxt(os.path.join(data_path,'top100snps.txt'),dtype=str)
            w = df[df.index.isin(top100)]['BETA'].values
            # w = (1./w)/np.sum(1./w)
            w = np.abs(w)/np.sum(np.abs(w))
    
            del df, top100
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
                    p=args.p)

        train_dataset = hdf5Dataset(data_path,X_file)
        test_dataset = hdf5Dataset(data_path,y_file)
        
        model.fit(train_dataset,test_dataset)
        model.predict()
        results = model.evaluate_metrics()

        print(f'Results:{results}')
        output.update(results)

        logging.info(f'Saving {os.path.basename(output_filename)}')

        with open(output_filename, 'w') as f:
            json.dump(output, f, indent=4)
    else:
        logging.warning(
            f'Skipping {output_filename} because it already exists.'
        )
