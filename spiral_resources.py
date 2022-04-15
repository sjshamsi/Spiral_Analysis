#!/usr/bin/env python
# coding: utf-8

# ### In this notebook, we create two objects necessary for spiral galaxy analysis. These are:
# 
# * The dictionary with details for the spiral galaxy.
# * The PANDAS dataframe with details for all spaxels within a galaxy
# 
# The dictionary is being made to replace the `SpiralGalaxy` object. This is because the `SpiralGalaxy` object consumes more memory than necessary, and it is easy to deliver galaxy details with a normal dictionary. These two separate objects are, unfortunately, necessary.
# 
# We cannot incorporate the galaxy details within the PANDAS dataframe as processing the dataframe can get rid of these details.

# In[1]:


#First we'll import all the module we need
from marvin.tools.maps import Maps
from pathlib import Path
from sa_utils import save_dict, load_dict
import pandas as pd
import numpy as np
import sys

sys.path.append('/home/sshamsi/galaxyzoo/GZ3D_production/') #this might need changing if working across platforms
import gz3d_fits


# In[2]:


def return_dict(filepath, save=True):
    filename = filepath.split('/')[-1]
    dict_file = Path('/home/sshamsi/galaxyzoo/Spiral_Analysis/SFR_Resources/Dicts/' + filename + '.pkl')
    
    if dict_file.exists():
        dic = load_dict(str(dict_file.resolve()))
        return dic
    
    galdict = {
        'filepath': filepath
    }
    
    galdict['filename'] = filename
    galdict['mangaid'] = galdict['filename'].split('_')[0]
    
    maps = Maps(galdict['mangaid'])
    
    galdict['z'] = maps.nsa['z']
    galdict['d_mpc'] = (299792.458 * galdict['z']) / 70 #Mpc
    galdict['d_kpc'] = galdict['d_mpc'] * 1E3 #Kpc
    galdict['d_m'] = galdict['d_mpc'] * 3.085677581E+22 # m
    galdict['delta'] = (4 * np.pi * (galdict['d_m']**2)) / ((2.8**2.36) * (10**41.1))
    galdict['spax_area'] = (0.0000024240684055477 * galdict['d_kpc'])**2 # Kpc^2
    galdict['map_shape'] = maps.emline_gflux_ha_6564.shape
    
    galdict['eff_rad'] = maps.nsa['elpetro_th50_r'] * 2 #maps.nsa['elpetro_th50_r'] is in " units, so we multiply by 2 to get effective radius in units of spaxels
    galdict['mass'] = maps.nsa['sersic_mass']
    galdict['theta'] = np.radians(maps.nsa['elpetro_phi'] - 90.0)
    galdict['elpetro_ba'] = maps.nsa['elpetro_ba']
    
    if save:
        save_dict(galdict, str(dict_file.resolve()))
    return galdict


# In[3]:


def return_df(galdict, spiral_threshold=3, other_threshold=3, save=True):
    filename = galdict['filename']
    df_file = Path('/home/sshamsi/galaxyzoo/Spiral_Analysis/SFR_Resources/DFs/' + filename + '.pkl')
    
    if df_file.exists():
        df = pd.read_pickle(str(df_file.resolve()))
        return df
    
    df = form_global_df(galdict, spiral_threshold=spiral_threshold,
                        other_threshold=other_threshold)
    if save:
        df.to_pickle(str(df_file.resolve()))
    return df


# In[4]:


spaxel_health = lambda x: x & 1073741824 == 0


# In[5]:


def btp_masks(maps):
    '''Simple make arrays of indicating the BPT classification of each spxel.
    Then we append this to the global DF later.'''
    
    bpt_masks = maps.get_bpt(return_figure=False, show_plot=False)
    
    comp = bpt_masks['comp']['global']
    agn = bpt_masks['agn']['global']
    seyfert = bpt_masks['seyfert']['global']
    liner = bpt_masks['liner']['global']
    
    return comp, agn, seyfert, liner


# In[6]:


def make_r_array(map_shape, theta, elpetro_ba):
    '''Goes through all spaxels and creates a global array of spaxel distances from the map centre.
    Distances are in units of spaxels.'''
    r_array = np.array([])

    a, b = map_shape
    k, h = (a - 1) / 2.0, (b - 1) / 2.0 #map centre

    for y, x in [(y, x) for y in range(a) for x in range(b)]:
        j, i = (-1 * (y - k), x - h) #vector from centre

        spax_angle = (np.arctan(j / i)) - theta
        vec_len = (j**2.0 + i**2.0)**0.5
        r = vec_len * ((np.cos(spax_angle))**2.0 + ((np.sin(spax_angle))/elpetro_ba)**2.0)**0.5
        
        r_array = np.append(r_array, r)
    
    return r_array


# In[7]:


def form_global_df(galdict, spiral_threshold=3, other_threshold=3):
    '''Make a global DF.'''
    #load the maps we need
    maps = Maps(galdict['mangaid'])
    hamap = maps.emline_gflux_ha_6564
    hbmap = maps.emline_gflux_hb_4862
    
    r_array =  make_r_array(galdict['map_shape'], galdict['theta'], galdict['elpetro_ba'])
    
    #Forming the H-a and H-b values, errors, and SNR arrays
    ha_array = hamap.value.flatten()
    sig_ha_array = hamap.error.value.flatten()
    ha_snr = hamap.snr.flatten()
    
    hb_array = hbmap.value.flatten()
    sig_hb_array = hbmap.error.value.flatten()
    hb_snr = hbmap.snr.flatten()
    
    #Forming the the BPT label arrays
    comp, agn, seyfert, liner = btp_masks(maps)
    
    comp_array = comp.flatten()
    agn_array = agn.flatten()
    seyfert_array = seyfert.flatten()
    liner_array = liner.flatten()
    
    ha_health_array = spaxel_health(hamap.mask.flatten())
    hb_health_array = spaxel_health(hbmap.mask.flatten())
    
    data_array = np.array([r_array, ha_array, sig_ha_array, ha_snr, hb_array, sig_hb_array, hb_snr, comp_array,
                           agn_array, seyfert_array, liner_array, ha_health_array, hb_health_array]).transpose()
    
    df = pd.DataFrame(data=data_array, columns=['radius', 'Ha', 'sig_Ha', 'sn_Ha', 'Hb', 'sig_Hb', 'sn_Hb', 'comp',
                                                'agn', 'seyfert', 'liner', 'ha_spax_healthy', 'hb_spax_healthy'])
    df['r_re'] = df['radius'] / galdict['eff_rad']
    spiral_spaxel_bool, nonspiral_spaxel_bool = get_morph_masks(galdict['filepath'], galdict['map_shape'],
                                                                spiral_threshold=spiral_threshold, other_threshold=other_threshold)
    df['sp_{Tsp}{Tnsp}'.format(Tsp=spiral_threshold, Tnsp=other_threshold)] = spiral_spaxel_bool.flatten()
    df['nsp_{Tsp}{Tnsp}'.format(Tsp=spiral_threshold, Tnsp=other_threshold)] = nonspiral_spaxel_bool.flatten()
    df['mangaid'] = galdict['mangaid']
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df


# In[8]:


def get_morph_masks(file_path, map_shape, spiral_threshold=3, other_threshold=3):
    '''This function returns boolean spiral arm and non-spiral arm masks'''
    data = gz3d_fits.gz3d_fits(file_path)
    data.make_all_spaxel_masks(grid_size=map_shape)
    
    center_mask_spaxel_bool = data.center_mask_spaxel > other_threshold
    star_mask_spaxel_bool = data.star_mask_spaxel > other_threshold
    bar_mask_spaxel_bool = data.bar_mask_spaxel > other_threshold
    spiral_mask_spaxel_bool = data.spiral_mask_spaxel > spiral_threshold
    
    combined_mask = center_mask_spaxel_bool | star_mask_spaxel_bool | bar_mask_spaxel_bool
    
    spiral_spaxel_bool = spiral_mask_spaxel_bool & (~combined_mask)
    nonspiral_spaxel_bool = (~spiral_mask_spaxel_bool) & (~combined_mask)
    
    return spiral_spaxel_bool, nonspiral_spaxel_bool


# In[9]:


def update_spirals(file_path, map_shape, thresholds=(np.array([3], dtype=int), np.array([3], dtype=int))):
    '''Do you want to change the parametres for what does/doesn't count as a spaxel or a
    non-spaxel? Use this function to simply add a column to the galaxy DF'''
    filename = file_path.split('/')[-1]
    df_file = Path('/home/sshamsi/galaxyzoo/Spiral_Analysis/SFR_Resources/DFs/' + filename + '.pkl')
    if not df_file.exists():
        raise ValueError("The galaxy provided does not have a DF saved. Use `return_df` to make and save the DF.")
    df = pd.read_pickle(str(df_file.resolve()))
    change = False
    
    for spiral_threshold, other_threshold in zip(thresholds[0], thresholds[1]):
        sp_colname = 'sp_{Tsp}{Tnsp}'.format(Tsp=spiral_threshold, Tnsp=other_threshold)
        nsp_colname = 'nsp_{Tsp}{Tnsp}'.format(Tsp=spiral_threshold, Tnsp=other_threshold)
        
        if (all(x in df.columns for x in [sp_colname, nsp_colname])):
            continue
        else:
            spiral_spaxel_bool, nonspiral_spaxel_bool = get_morph_masks(file_path, map_shape, spiral_threshold=spiral_threshold,
                                                                        other_threshold=other_threshold)
            df[sp_colname] = spiral_spaxel_bool.flatten()
            df[nsp_colname] = nonspiral_spaxel_bool.flatten()
            change = True
    if change:
        df.to_pickle(str(df_file.resolve()))
    return df
