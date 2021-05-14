
# coding: utf-8

# ### This object will return a global DF with all the necessary information for each spaxel

# In[1]:


#First we'll import all the module we need
from marvin.tools.maps import Maps

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '/homes/sshamsi/galaxy_zoo/GZ3D_production/') #this might need changing if working across platforms

import gz3d_fits


# In[2]:


def return_df(file_path):
    return form_global_df(file_path, spiral_threshold=3, other_threshold=3)


# In[3]:


def make_emmasks(hamask, hbmask):
    '''Takes masks from the MaNGA maps and if a spaxel is flagges as
    "DO NOT USE (2**30) then it marks it as "0". Creates global maek objects
    from these.'''
    
    ha_mask_array = hamask.flatten()
    hb_mask_array = hbmask.flatten()

    for i in range(len(ha_mask_array)):
        if ha_mask_array[i] & 1073741824 == 0:
            ha_mask_array[i] = 0
            hb_mask_array[i] = 0
        else:
            ha_mask_array[i] = 1
            
            if hb_mask_array[i] & 1073741824 == 0:
                hb_mask_array[i] = 0
            else:
                hb_mask_array[i] = 1

    ha_mask_array = np.array(ha_mask_array, dtype=bool)
    hb_mask_array = np.array(hb_mask_array, dtype=bool)
    
    return ha_mask_array, hb_mask_array


# In[4]:


def btp_masks(maps):
    '''Simple make arrays of indicating the BPT classification of each spxel.
    Then we append this to the global DF later.'''
    
    bpt_masks = maps.get_bpt(return_figure=False, show_plot=False)

    comp = bpt_masks['comp']['global']
    agn = bpt_masks['agn']['global']
    seyfert = bpt_masks['seyfert']['global']
    liner = bpt_masks['liner']['global']
    
    return comp, agn, seyfert, liner


# In[5]:


def make_r_array(map_shape, theta, elpetro_ba):
    '''Goes through all spaxels and creates a global array of spaxel distances from the map centre'''
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


# In[6]:


def form_global_df(file_path, spiral_threshold=3, other_threshold=3):
    '''Make a global DF.'''
    
    #Load up the maps we need
    mangaid = file_path.split('/')[-1].split('_')[0]
    
    maps = Maps(mangaid)
    hamap = maps.emline_gflux_ha_6564
    hbmap = maps.emline_gflux_hb_4862
    
    eff_rad = maps.nsa['elpetro_th50_r'] * 2
    redshift = maps.nsa['z']
    mass = maps.nsa['sersic_mass']
    
    theta = np.radians(maps.nsa['elpetro_phi'] - 90.0)
    elpetro_ba = maps.nsa['elpetro_ba']
    
    r_array =  make_r_array(hamap.shape, theta, elpetro_ba)
    ha_mask_array, hb_mask_array = make_emmasks(hamap.mask, hbmap.mask)
    
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
    
    data_array = np.array([r_array, ha_array, sig_ha_array, ha_snr, hb_array, sig_hb_array, hb_snr,
                           comp_array, agn_array, seyfert_array, liner_array]).transpose()

    df = pd.DataFrame(data=data_array, columns=['Radius', '$H_{\\alpha}$', '$\sigma H_{\\alpha}$',
                                                'S/N $H_{\\alpha}$', '$H_{\\beta}$', '$\sigma H_{\\beta}$',
                                                'S/N $H_{\\beta}$', 'Comp', 'AGN', 'Seyfert', 'Liner'])

    df['$r/r_e$'] = df['Radius'] / eff_rad
    
    df.iloc[ha_mask_array, df.columns.get_loc('$H_{\\alpha}$')] = np.nan
    df.iloc[ha_mask_array, df.columns.get_loc('$\sigma H_{\\alpha}$')] = np.nan
    df.iloc[ha_mask_array, df.columns.get_loc('$H_{\\beta}$')] = np.nan
    df.iloc[ha_mask_array, df.columns.get_loc('$\sigma H_{\\beta}$')] = np.nan
    
    df.iloc[hb_mask_array, df.columns.get_loc('$H_{\\beta}$')] = np.nan
    df.iloc[hb_mask_array, df.columns.get_loc('$\sigma H_{\\beta}$')] = np.nan
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    df = update_spirals(df, file_path, hamap.shape, spiral_threshold=spiral_threshold, other_threshold=other_threshold)
    
    df['MaNGA ID'] = mangaid
    df['Mass'] = mass
    
    return df


# In[7]:


def update_spirals(df, file_path, map_shape, spiral_threshold=3, other_threshold=3, ret_spiral_bool=False):
    '''Do you want to change the parametres for what does/doesn't count as a spaxel or a
    non-spaxel? Use this function to simply update the "Spiral" column of the global DF'''
    
    data = gz3d_fits.gz3d_fits(file_path)
    data.make_all_spaxel_masks(grid_size = map_shape)
    
    center_mask_spaxel_bool = data.center_mask_spaxel > other_threshold
    star_mask_spaxel_bool = data.star_mask_spaxel > other_threshold
    bar_mask_spaxel_bool = data.bar_mask_spaxel > other_threshold
    spiral_mask_spaxel_bool = data.spiral_mask_spaxel > spiral_threshold
    
    combined_mask = center_mask_spaxel_bool | star_mask_spaxel_bool | bar_mask_spaxel_bool
    
    spiral_spaxel_bool = spiral_mask_spaxel_bool & (~combined_mask)
    
    if ret_spiral_bool:
        return spiral_spaxel_bool
    
    df['Spiral Arm'] = spiral_spaxel_bool.flatten()
    return df