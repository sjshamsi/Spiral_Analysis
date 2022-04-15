'''This script will just run through a list of paths (in this case, we use `final_sample_paths` in
../Selecting_Sample/) and call H-alpha and H-beta covariance matrives for each path, saving it to
disk. This makes sure the matrices are available locally in future calls so they go much faster.'''

import numpy as np
import sys
sys.path.append('../')
from spiral_resources import return_dict, return_df
from sfr_methods import ret_cov_matrices

# Please do change this if you need to save resources for other paths
file_paths = np.load('../Selecting_Sample/final_sample_paths.npy', allow_pickle=True) 

for path in file_paths:
    dic = return_dict(path)
    df = return_df(dic)
    ha_cov = ret_cov_matrices(df, dic, mode='Ha')
    hb_cov = ret_cov_matrices(df, dic, mode='Hb')
