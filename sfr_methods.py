#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import numpy as np
import pandas as pd


# In[2]:


def flux2sfr(ha_flux, ha_stdv, hb_flux, hb_stdv, galdict, avg=False):
    '''Take an H-alpha and H-beta flux, and then make an SFR measurement out of them.'''
    
    ha_flux = ha_flux * 1E-13
    hb_flux = hb_flux * 1E-13

    ha_stdv = ha_stdv * 1E-13
    hb_stdv = hb_stdv * 1E-13
    
    sfr = (galdict['delta'] * (ha_flux**3.36) * (hb_flux**-2.36))
    sfr_stdv = np.sqrt((3.36 * galdict['delta'] * (ha_flux**2.36) * (hb_flux**-2.36) * ha_stdv)**2 +
                       (-2.36 * galdict['delta'] * (ha_flux**3.36) * (hb_flux**-3.36) * hb_stdv)**2)
    
    if avg:
        return sfr / galdict['spax_area'], sfr_stdv / galdict['spax_area']
    
    return sfr, sfr_stdv


# In[3]:


def cov_matrix_maker(mapshape, err_series):
    '''Calculates the covaraince matrix for our galaxy. That is rather intensive.'''
    
    corr_matrix = np.load('/home/sshamsi/galaxyzoo/Spiral_Analysis/Matrices/corr_matrices/corr_matrix_' + str(mapshape[0]) + '.npy')
    
    r = mapshape[0]**2
    cov_mat = np.zeros((r, r))
    
    for item, frame in err_series.iteritems():
        if pd.isnull(frame):
            k = 0
        else:
            k = frame
            
        cov_mat[item] = corr_matrix[item] * k
        cov_mat[:, item] = corr_matrix[:, item] * k
        
    return cov_mat


# In[4]:


def ret_cov_matrices(df, galdict, mode=None):
    '''This method loads and returns the H-a/H-b covariance matrix. If not available,
    it calculates it, which is resource intensive.'''
    
    if mode == None:
        raise ValueError('Argument "mode" must be set to "ha" or "hb".')
        
    elif mode == 'ha':
        hafile = pathlib.Path("/home/sshamsi/galaxyzoo/Spiral_Analysis/Matrices/cov_matrices/" + galdict['filename'] + '.ha.npy')
        
        if hafile.exists():
            ha_cov = np.load(hafile, allow_pickle=True )
            return ha_cov
        
        else:
            print("H-a covariance file does not exist. Calculating...")
            ha_cov = cov_matrix_maker(galdict['map_shape'], df.sig_Ha)
            return ha_cov
        
    elif mode == 'hb':
        hbfile = pathlib.Path("/home/sshamsi/galaxyzoo/Spiral_Analysis/Matrices/cov_matrices/" + galdict['filename'] + '.hb.npy')
        
        if hbfile.exists():
            hb_cov = np.load(hbfile)
            return hb_cov
        
        else:
            print("H-b covariance file does not exist. Calculating...")
            hb_cov = cov_matrix_maker(galdict['map_shape'], df.sig_Hb)
            return hb_cov


# In[ ]:


def get_sfr(bin_index, df, galdict, avg=False):
    '''Return the SFR for a bin of spxels.'''
    
    ha_flux, ha_stdv = get_emission(bin_index, df, galdict, mode='ha', avg=avg)
    hb_flux, hb_stdv = get_emission(bin_index, df, galdict, mode='hb', avg=avg)
    
    if ha_flux == 0 or hb_flux == 0:
        print("Returning SFR = 0 as bin results in Ha or Hb == 0. MaNGA ID:", galdict['mangaid'])
        return 0, 0
        
    return flux2sfr(ha_flux, ha_stdv, hb_flux, hb_stdv, galdict, avg=avg)


# In[ ]:


def get_emission(bin_index, df, galdict, mode=None, avg=False):
    '''Return the H-a or H-b flux.'''
    
    if len(bin_index) == 0:
        print('Returning', mode, 'emission = 0 as no spaxels in bin. MaNGA ID:', galdict['mangaid'])
        return 0, 0
    
    set_index = set(bin_index)
    tot_index = list(df.index)
    
    w_vec = np.array([[x in set_index for x in tot_index]]) * 1
    
    if mode == None:
        raise ValueError('Argument "mode" must be "ha", or "hb".')
    elif mode == 'ha':
        summ = df.loc[bin_index.tolist(), 'Ha'].sum()
        
        ha_cov = ret_cov_matrices(df, galdict, mode=mode)
        cov_mat = ha_cov
    elif mode == 'hb':
        summ = df.loc[bin_index.tolist(), 'Hb'].sum()
        
        hb_cov = ret_cov_matrices(df, galdict, mode=mode)
        cov_mat = hb_cov
        
    var = np.linalg.multi_dot([w_vec, cov_mat, w_vec.T])[0][0]

    if avg:
        n = len(bin_index)
        return summ / n, np.sqrt(var / (n**2))
    
    return summ, np.sqrt(var)


# In[ ]:




