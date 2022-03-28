#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import numpy as np
import pandas as pd
from skimage.draw import disk
from scipy.sparse import csc_matrix, save_npz, load_npz
from spiral_resources import return_dict, return_df


# In[2]:


class Galaxy:
    def __init__(self, path):
        self.dic = return_dict(path)
        self.df = return_df(self.dic)
        
        
    def flux2sfr(self, ha_flux, ha_stdv, hb_flux, hb_stdv, avg=False):
        '''Take an H-alpha and H-beta flux, and then make an SFR measurement out of them.'''
        ha_flux = ha_flux * 1E-13
        hb_flux = hb_flux * 1E-13
        ha_stdv = ha_stdv * 1E-13
        hb_stdv = hb_stdv * 1E-13
        
        sfr = (self.dic['delta'] * (ha_flux**3.36) * (hb_flux**-2.36))
        sfr_stdv = np.sqrt((3.36 * self.dic['delta'] * (ha_flux**2.36) * (hb_flux**-2.36) * ha_stdv)**2 +
                           (-2.36 * self.dic['delta'] * (ha_flux**3.36) * (hb_flux**-3.36) * hb_stdv)**2)
        if avg:
            return sfr / self.dic['spax_area'], sfr_stdv / self.dic['spax_area']
        return sfr, sfr_stdv
    
    
    def sparse_arr_maker(self, mode=None):
        cache = {}
        sparse_rows = np.array([], dtype=int)
        sparse_cols = np.array([], dtype=int)
        sparse_vals = np.array([])
        
        map_shape = self.dic['map_shape'][0]
        err_array = self.df['sig_' + mode].to_numpy()
        valid_indices = self.df.dropna().index.to_numpy()
        
        for ind1 in valid_indices:
            err1 = err_array[ind1]
            i, j = ind1 // map_shape, ind1 % map_shape
            indices = disk((i, j), 6.4, shape=(map_shape, map_shape))
            cache[ind1] = {}
            
            for k, l in zip(indices[0], indices[1]):
                ind2 = k * map_shape + l
                if ind1 == ind2:
                    val = err1**2
                    sparse_rows = np.append(sparse_rows, ind1)
                    sparse_cols = np.append(sparse_cols, ind2)
                    sparse_vals = np.append(sparse_vals, val)
                    continue
                    
                err2 = err_array[ind2]
                if np.isnan(err2):
                    continue
                if (ind2 in cache) and (ind1 in cache[ind2]):
                    continue
                    
                dist = np.sqrt((i - k)**2 + (j - l)**2)
                val = np.exp(-0.5 * (dist / 1.9)**2) * err1 * err2
                
                sparse_rows = np.append(sparse_rows, [ind1, ind2])
                sparse_cols = np.append(sparse_cols, [ind2, ind1])
                sparse_vals = np.append(sparse_vals, [val, val])
                cache[ind1][ind2] = val
        return sparse_rows, sparse_cols, sparse_vals
    
    
    def ret_cov_matrices(self, mode=None, save_matrix=True):
        '''This method loads and returns the H-a/H-b covariance matrix. If not available,
        it calculates it, which is resource intensive.'''
        if mode == None:
            raise ValueError('Argument "mode" must be set to "Ha" or "Hb".')
            
        cov_file = Path('/home/sshamsi/galaxyzoo/Spiral_Analysis/Matrices/' + self.dic['filename'] + '.' + mode + '.npz')
        if cov_file.exists():
            cov_matrix = load_npz(str(cov_file.resolve()))
        else:
            print(mode, 'covariance file does not exist. Calculating...')
            cov_matrix_shape = (self.dic['map_shape'][0]**2, self.dic['map_shape'][0]**2)
            rows, cols, data = self.sparse_arr_maker(mode=mode)
            cov_matrix = csc_matrix((data, (rows, cols)), shape=cov_matrix_shape)
        if save_matrix:
            save_npz('/home/sshamsi/galaxyzoo/Spiral_Analysis/Matrices/' + self.dic['filename'] + '.' +
                     mode, cov_matrix)    
        return cov_matrix


    def get_sfr(self, spax_bin, avg=False):
        '''Return the SFR for a bin of spaxels.'''
        if len(spax_bin) == 0:
            raise ValueError('The spax_bin array must not be empty.')
            
        ha_flux, ha_stdv = get_emission(spax_bin, mode='Ha', avg=avg)
        hb_flux, hb_stdv = get_emission(spax_bin, mode='Hb', avg=avg)
        
        return self.flux2sfr(ha_flux, ha_stdv, hb_flux, hb_stdv, avg=avg)
    
    
    def get_emission(self, spax_bin, mode=None, avg=False):
        '''Return the H-a or H-b flux.'''
        if mode not in ['Ha', 'Hb']:
            raise ValueError('Argument "mode" must be set to "Ha" or "Hb".')
        if len(spax_bin) == 0:
            raise ValueError('The spaxel_bin array is empty.')
            
        summ = self.df.loc[spax_bin.tolist(), mode].sum()
        
        w_vals = np.ones(len(spax_bin))
        w_rows = np.zeros(len(spax_bin))
        w_vec = csc_matrix((w_vals, (w_rows, spax_bin)), shape=(1, len(df)))
        cov_mat = self.ret_cov_matrices(mode=mode)
        var = w_vec.dot(cov_mat.dot(w_vec.transpose()))[0, 0]
        
        if avg:
            n = len(spax_bin)
            return summ / n, np.sqrt(var / (n**2))
        return summ, np.sqrt(var)
