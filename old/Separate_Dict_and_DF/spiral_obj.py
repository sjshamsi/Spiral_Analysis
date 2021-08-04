
# coding: utf-8

# In[1]:


#First we'll import all the module we need
import pathlib
import numpy as np
from numpy.linalg import multi_dot
from astropy.io import fits


# In[2]:


hdu_drp = fits.open('/raid5/homes/sshamsi/sas/mangawork/manga/spectro/redux/v3_0_1/drpall-v3_0_1.fits')
mangaid_array = hdu_drp[1].data['mangaid']
redshift_array = hdu_drp[1].data['z']


# In[8]:


# The SpiralObj class is defined below
class SpiralObj(object):
    def __init__(self, filepath):
        self.filename = filepath.split('/')[-1]
        self.mangaid = self.filename.split('_')[0]
        self.redshift = redshift_array[np.where(mangaid_array == self.mangaid)][0]
        
        self.d_mpc = (299792.458 * self.redshift) / 70 #Mpc
        self.d_kpc = self.d_mpc * 1E3
        self.d_m = self.d_mpc * 3.085677581E+22 # m
        self.delta = (4 * np.pi * (self.d_m**2)) / ((2.8**2.36) * (10**41.1))
        self.spax_area = (0.0000024240684055477 * self.d_kpc)**2
        
        
    def __repr__(self):
        return 'MaNGA ID {}'.format(self.mangaid)
    
    
    def flux2sfr(self, ha_flux, ha_stdv, hb_flux, hb_stdv, avg=False):
        '''Take an H-alpha and H-beta flux, and then make an SFR measurement out of them.'''
        
        ha_flux = ha_flux * 1E-13
        hb_flux = hb_flux * 1E-13
        
        ha_stdv = ha_stdv * 1E-13
        hb_stdv = hb_stdv * 1E-13
        
        sfr = self.delta * (ha_flux**3.36) * (hb_flux**-2.36)
        sfr_stdv = np.sqrt((3.36 * self.delta * (ha_flux**2.36) * (hb_flux**-2.36) * ha_stdv)**2 +
                           (-2.36 * self.delta * (ha_flux**3.36) * (hb_flux**-3.36) * hb_stdv)**2)
        
        if avg:
            return sfr / self.spax_area, sfr_stdv / self.spax_area
        
        return sfr, sfr_stdv
            
            
    def cov_matrix_maker(self, err_series):
        '''Calculates the covaraince matrix for our galaxy.That is rather intensive.'''
        
        corr_matrix = np.load('/raid5/homes/sshamsi/galaxy_zoo/GZ3D_spiral_analysis/Matrices/corr_matrices/corr_matrix' + str(self.map_shape[0]) + '.npy')
        
        r = self.hamap.size
        cov_mat = np.zeros((r, r))
        
        for item, frame in err_series.iteritems():
            if pd.isnull(frame):
                k = 0
            else:
                k = frame
                
            cov_mat[item] = corr_matrix[item] * k
            cov_mat[:, item] = corr_matrix[:, item] * k
        
        return cov_mat
    
    
    def ret_cov_matrices(self, df, mode=None):
        '''This method loads and returns the H-a/H-b covariance matrix. If not available,
        it calculates it, which is resource intensive.'''
        
        if mode == None:
            raise ValueError('Argument "mode" must be set to "ha" or "hb".')
            
        elif mode == 'ha':
            hafile = pathlib.Path("/raid5/homes/sshamsi/galaxy_zoo/GZ3D_spiral_analysis/Matrices/cov_matrices/" + self.filename.split('.')[0] + 'ha.npy')
            
            if hafile.exists():
                ha_cov = np.load(hafile)
                return ha_cov
            
            else:
                print ("H-a covariance file does not exist. Calculating...")
                ha_cov = self.cov_matrix_maker(df['$\sigma H_{\\alpha}$'])
                return ha_cov
        
        elif mode == 'hb':
            hbfile = pathlib.Path("/raid5/homes/sshamsi/galaxy_zoo/GZ3D_spiral_analysis/Matrices/cov_matrices/" + self.filename.split('.')[0] + 'hb.npy')
            
            if hbfile.exists():
                hb_cov = np.load(hbfile)
                return hb_cov
            
            else:
                print ("H-b covariance file does not exist. Calculating...")
                hb_cov = self.cov_matrix_maker(df['$\sigma H_{\\beta}$'])
                return hb_cov
            
                            
    def get_sfr(self, bin_index, df, avg=False):
        '''Return the SFR for a bin of spxels.'''
        
        ha_flux, ha_stdv = self.get_emission(bin_index, df, mode='ha', avg=avg)
        hb_flux, hb_stdv = self.get_emission(bin_index, df, mode='hb', avg=avg)
        
        return self.flux2sfr(ha_flux, ha_stdv, hb_flux, hb_stdv, avg=avg)
    
    
    def get_emission(self, bin_index, df, mode=None, avg=False):
        '''Return the H-a or H-b flux.'''
                
        set_index = set(bin_index)
        tot_index = list(df.index)
        
        w_vec = np.array([[x in set_index for x in tot_index]]) * 1
        
        if mode == None:
            raise ValueError('Argument "mode" must be "ha", or "hb".')
        elif mode == 'ha':
            summ = df.loc[bin_index.tolist(), '$H_{\\alpha}$'].sum()
            
            ha_cov = self.ret_cov_matrices(df, mode=mode)
            cov_mat = ha_cov
        elif mode == 'hb':
            summ = df.loc[bin_index.tolist(), '$H_{\\beta}$'].sum()
            
            hb_cov = self.ret_cov_matrices(df, mode=mode)
            cov_mat = hb_cov
            
        var = np.linalg.multi_dot([w_vec, cov_mat, w_vec.T])[0][0]
        
        if avg:
            n = len(bin_index)
            return summ / n, np.sqrt(var / (n**2))
        
        return summ, np.sqrt(var)