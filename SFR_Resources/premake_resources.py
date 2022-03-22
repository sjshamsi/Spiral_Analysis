#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import numpy as np
import logging

import sys
sys.path.append('../')
from sa_utils import save_dict, load_dict
from spiral_resources import return_df, return_dict


# In[2]:


logging.basicConfig(filename='resource_run.log', encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)


# In[3]:


file_paths = np.load('../Selecting_Sample/manga_gz3d_spirals.npy', allow_pickle=True) # Please do change this if you need to save resources for other paths


# In[5]:


file_paths_len = len(file_paths)

for idx, path in enumerate(file_paths):
    filename = path.split('/')[-1]
    dict_file = Path("Dicts/" + filename + '.pkl')
    df_file = Path("DFs/" + filename + '.pkl')
    
    if not dict_file.exists():
        dic = return_dict(str(dict_file.resolve()))
        save_dict(dic, str(dict_file.resolve()))
    else:
        dic = load_dict(str(dict_file.resolve()))
        
    if not df_file.exists():
        df = return_df(dic)
        df.to_pickle(str(df_file.resolve()))
        
    if (idx+1) % 25 == 0: #just to keep track of processing
        print((file_paths_len - idx + 1), 'galaxies left')


# We'll turn all the dictionaries into a comprehensive DataFrame as this'll enable us to use it faster.
