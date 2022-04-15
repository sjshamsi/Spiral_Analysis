'''This script will just run through a list of paths (in this case, we use `final_sample_paths` in
../Selecting_Sample/) and call the information dictionary and the spaxel DF for each path, saving it to
disk. This makes sure the dictionaries and DFs are available locally in future calls.'''

import numpy as np
import sys
sys.path.append('../')
from spiral_resources import return_dict, return_df

# Please do change this if you need to save resources for other paths
file_paths = np.load('../Selecting_Sample/final_sample_paths.npy', allow_pickle=True) 

for path in file_paths:
    dic = return_dict(path)
    df = return_df(dic)