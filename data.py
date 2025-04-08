# -*- coding: utf-8 -*-
""" Load data from various sources and prepare it for training.

    See main SlicesFeed class for the main data store and feeding class.
""" 

import torch, numpy as np, pandas as pd
import jax, jax.numpy as jnp
import scipy.io as sio
from glob import glob
from natsort import natsorted
from pydicom import dcmread 
import os
import xarray as xr
import collections
import pyDOE


gamma = 42.58 * 2*np.pi * 1e6   # gamma' = 42.58 MHz/T
B0_base_DEF = 7.  # TESLA
wc_ppm_DEF = 0    # MT  
wb_ppm_DEF = -3.5  # rNOE
T2b_ms_DEF = 5 # for rNOE
# MT: 10us w Superlorentzian shape approximated by 40us Lorentzian (Zaiss2022, Perlman2022)
T2c_ms_DEF = 0.04  # for MT

def_ranges4LHS = collections.OrderedDict({\
    'T1a_ms': [500, 3500],  # vol7stats
    'T2a_ms': [15,  1000],  # vol7stats
    'B0_shift_ppm_map': [-1.2, 1.2],  # vol7stats
    'B1_fix_factor_map': [0.7, 1.3],  # vol7stats
    'fb_gt_T': [0,  0.01],
    'kb_gt_T': [0,  400],
    'fc_gt_T': [0,  0.3],
    'kc_gt_T': [0,  100]
})    


def_grids4cartesian = collections.OrderedDict({
    'T1a_ms': np.arange(700, 3000+100, 100),
    'T2a_ms': np.concatenate((np.arange(30, 150+10, 10), np.arange(200, 1000+100, 100))),
    'B0_shift_ppm_map': [0.],  
    'B1_fix_factor_map': [1.], 
    'fb_gt_T': [0.],
    'kb_gt_T': [0.],
    'fc_gt_T': np.arange(0.01, 0.3+.01, 0.01),
    'kc_gt_T': np.arange(4, 100+4, 4)
})               

# The 79M dict
perlman2022_cartesian = collections.OrderedDict({
    'T1a_ms': np.concatenate((np.arange(800, 2000+200, 200), np.array([2400, 3000]))),    
    'T2a_ms': np.arange(50, 150+10, 10),
    'B0_shift_ppm_map': np.arange(-0.3, 0.3+0.1, 0.1),
    'B1_fix_factor_map': np.array([1.,]),
    'fb_gt_T': np.arange(100, 1000+50, 50)/110e3,  # mM->fraction
    'kb_gt_T': np.arange(5, 100+5, 5),
    'fc_gt_T': np.arange(0.02, 0.3+.02, 0.02),
    'kc_gt_T': np.arange(5, 100+5, 5)
})           
# https://static-content.springer.com/esm/art%3A10.1038%2Fs41551-021-00809-7/MediaObjects/41551_2021_809_MOESM1_ESM.pdf


def get_w1_wrf(B1, wrf_diff_Hz, B0_base=B0_base_DEF, verbose=False):    
    w1 = B1 * gamma    
    w0_base = gamma * B0_base    
    wrf_rad = w0_base - 2*np.pi*wrf_diff_Hz  # PPM is "to left" by convention. Use minus for MT/NOE
    wrf_ppm = (wrf_rad-w0_base)/w0_base*1e6    
    if verbose:
        print(f'{wrf_ppm:.2f} ppm') 
    return w1, wrf_rad


def get_w_abc(B0_base=B0_base_DEF, B0_shift_ppm=0, wb_ppm=wb_ppm_DEF, wc_ppm=wc_ppm_DEF):     
    B0_corrected = B0_base * (1 + 1e-6 * B0_shift_ppm)  # negligible effect..?
    wa = w0_cli = gamma * B0_corrected         
    wb = wa * (1 - wb_ppm*1e-6)  ## ppm is "to left" by convention
    wc = wa * (1 - wc_ppm*1e-6) 

    return wa, wb, wc


def get_seq(mt_seq_txt_fname = 'data/52_MT_with_M0_3T_.txt', 
            larg_seq_txt_fname = 'data/51_L_arg_with_M0_3T.txt',
            drop_first=False 
    ):
    colnames = ['TR_ms', 'B1_uT', 'dwRF_Hz', 'FA', 'Tsat_ms']    
    seq_df_MT = pd.read_csv(mt_seq_txt_fname, skiprows=1, sep='\s+', names=colnames)
    seq_df_MT['dwRF_Hz'] *= -1  # "right" is "negative ppm" by convention

    seq_df_LARG = pd.read_csv(larg_seq_txt_fname, skiprows=1, sep='\s+', names=colnames) 
    seq_df_AMIDE = pd.read_csv(larg_seq_txt_fname, skiprows=1, sep='\s+', names=colnames)    
    LARG_TO_AMIDE = 3.5/3
    seq_df_AMIDE['dwRF_Hz'] *= LARG_TO_AMIDE  # !!!
        
    if drop_first:        
        return {'mt': seq_df_MT[1:], 'amide': seq_df_AMIDE[1:], 'larg': seq_df_LARG[1:]}
    else:
        # fix the degenerate 1st iteration: no-saturation reference scan
        for seq in [seq_df_AMIDE, seq_df_MT, seq_df_LARG]:
            seq.loc[0, 'Tsat_ms'] = 0
            seq.loc[0, 'dwRF_Hz'] = 10000
            seq.loc[0, 'B1_uT'] = 0.001
        return {'mt': seq_df_MT, 'amide': seq_df_AMIDE, 'larg': seq_df_LARG}    


def get_lh_sample(var2range_od, shape):
    """Latin Hypercube Sampling for a given shape and variable ranges.
    """
    n_samples = np.product(shape)
    n_dim = len(var2range_od)
    lhd01 = pyDOE.lhs(n_dim, samples=n_samples)
    lhd_d = {}
    for ii, (pname, prange) in enumerate(var2range_od.items()):
        if len(prange) == 2:  # interpret as min-max
            lhd_d[pname] = lhd01[:, ii] * (prange[1] - prange[0]) + prange[0]
        else:                 # interpret as array of possible values
            lhd_d[pname] = prange[ np.int32(np.floor(lhd01[:, ii] * len(prange))) ]
        lhd_d[pname] = np.reshape(lhd_d[pname], shape)
    return lhd_d


def decode_data_entry(data_entry, detorch=True):
    ''' fixing the rogue 1st dimension added by torch.data.dataset
    '''
    roi_mask_nans_T, measured_normed_T, w_dict, R_dict, gt_dict = data_entry        
    roi_mask_nans_T = roi_mask_nans_T[0].detach().numpy() if detorch else roi_mask_nans_T[0]
    measured_normed_T = measured_normed_T[0].detach().numpy() if detorch else measured_normed_T[0]
    for di in [w_dict, R_dict, gt_dict]:
        di.update({k: (v[0].detach().numpy()*roi_mask_nans_T if detorch else v[0]) for k, v in di.items()})        
        
    return roi_mask_nans_T, measured_normed_T, w_dict, R_dict, gt_dict
    
    
class SlicesFeed(torch.utils.data.Dataset):

    norm_type = 'max'  # alternatively: "first", 'l2'
    # ! the above is specifically referred to as class attribute so should be set as SlicesFeed.norm_type
    # all the below are used as object attributes - consider moving to init. 
    add_noise_to_signal = 0
    ds = 1             # downsampling ratio
    slw = 10           # slab width            
    downsample_or_slab = True  # otherwise, will cut slabs instead of downsampling. 
    random_ds = True           # otherwise, will scan slab/downsample offsets sequentially.
    xo_state = 0
    yo_state = 0

    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def from_args(cls, shape, T1a_ms, T2a_ms, mt_or_amide=None, seq_df=None,
            kb_gt_T=300, fb_gt_T=2e-3, kc_gt_T=70, fc_gt_T=0.2,  # WM
            b_ppm=wb_ppm_DEF, T2b_ms=T2b_ms_DEF, c_ppm=wc_ppm_DEF, T2c_ms=T2c_ms_DEF, same_T1=True, 
            B0=B0_base_DEF, B0_shift_ppm_map=0, B1_fix_factor_map=1,
            roi_mask_nans=1, signal=np.nan, drop_first=False):
        """
            All tissue-parameter arguments can be maps shaped as <shape> arg, or scalars.
            The "ground truth" parameters are only used in pure forward-simulation flows (no fitting/training), e.g., synthetic data generation
        """
        data4nbmf = cls()        
        data4nbmf.shape = shape
        data4nbmf.slw = min(data4nbmf.slw, data4nbmf.shape[1])    
        if type(seq_df) == type(None):
            seq_df = get_seq(drop_first=drop_first)[mt_or_amide]
        data4nbmf.seq_df = seq_df
        data4nbmf.seq_len = seq_len = len(data4nbmf.seq_df['B1_uT'])  # signal.shape[0]        
        
        data4nbmf.kb_gt_T = kb_gt_T * np.ones(shape) 
        data4nbmf.fb_gt_T = fb_gt_T * np.ones(shape)
        data4nbmf.kc_gt_T = kc_gt_T * np.ones(shape) 
        data4nbmf.fc_gt_T = fc_gt_T * np.ones(shape)            
        data4nbmf.B0_shift_ppm_map = B0_shift_ppm_map * np.ones(shape)
        data4nbmf.B1_fix_factor_map = B1_fix_factor_map * np.ones(shape)        
        
        data4nbmf.R1a_V = 1 / (1e-3 * T1a_ms + 1e-6) * np.ones(shape)        
        if same_T1:
            # Same spin-lattice relaxation for all pools
            data4nbmf.R1c_V = data4nbmf.R1b_V = data4nbmf.R1a_V 
        else:
            assert 0, "Not implemented"	
        data4nbmf.R2a_V = 1 / (1e-3 * T2a_ms + 1e-6) * np.ones(shape)
        data4nbmf.R2b_V = 1 / (1e-3 * T2b_ms + 1e-6) * np.ones(shape)
        data4nbmf.R2c_V = 1 / (1e-3 * T2c_ms + 1e-6) * np.ones(shape)

        w1_seq, wrf_seq = np.zeros(seq_len), np.zeros(seq_len)         
        for ii in range(seq_len):
            w1_seq[ii], wrf_seq[ii] = get_w1_wrf(\
                data4nbmf.seq_df['B1_uT'][ii] * 1e-6, 
                data4nbmf.seq_df['dwRF_Hz'][ii], B0_base=B0
                ) #, verbose=True)

        w_a, w_b, w_c = get_w_abc(B0_base=B0, B0_shift_ppm=B0_shift_ppm_map*np.ones(shape), wb_ppm=b_ppm, wc_ppm=c_ppm)       
        data4nbmf.wa_T = w_a[None, ...]
        data4nbmf.wb_T = w_b[None, ...]
        data4nbmf.wc_T = w_c[None, ...]

        data4nbmf.w1_T = w1_seq[:, None, None, None]
        data4nbmf.w1_T = data4nbmf.w1_T * B1_fix_factor_map * np.ones(shape)
        data4nbmf.w1_a0_mean = np.nanmean(data4nbmf.w1_T[0])  # ..to recover b1 map later..
        data4nbmf.wrf_T = wrf_seq[:, None, None, None] 
        data4nbmf.roi_mask_nans = roi_mask_nans * np.ones(shape)

        data4nbmf.measured_normed_T = \
            data4nbmf.normalize(signal * np.ones((seq_len, *shape)))

        return data4nbmf
    
    @classmethod
    def make_lh_sample(
        cls, 
        mt_or_amide,
        tissue_parameter_ranges_od=None,
        shape=[100,25,100],        
        **kwargs
        ):           
        
        if type(tissue_parameter_ranges_od) == type(None): 
            tissue_parameter_ranges_od = def_ranges4LHS           
        tp_d = get_lh_sample(tissue_parameter_ranges_od, shape)
        return cls.from_args(mt_or_amide=mt_or_amide, shape=shape, **tp_d, **kwargs)
    
    @classmethod
    def make_cartesian_sample(
        cls, 
        mt_or_amide,
        parameter_values_od=None,
        shape=None, # [100,25,100],    
        slices=[0,500], # max cutoff
        **kwargs
        ):                           
        if type(parameter_values_od) == type(None): 
            shape = [99, 40, 100]
            parameter_values_od = def_grids4cartesian
        grids = np.meshgrid(*parameter_values_od.values(), indexing='ij')
        print(grids[0].shape)

        tp_d = {name: grid.ravel().reshape([shape[0], -1, shape[2]])[:, slices[0]:slices[1],:]    # [:, :shape[1],:] 
                for name, grid in zip(parameter_values_od.keys(), grids)
                }
        shard_shape = list(tp_d.values())[0].shape
        return cls.from_args(mt_or_amide=mt_or_amide, shape=shard_shape, **tp_d, **kwargs)        
        
    @classmethod
    def from_nv_dict(cls, nvdict_fname='mtd.npz', shape=(88, 50, 90)):
        """Loading dictionary created using Vladimirov2024's code
        """
        nvdict = np.load('mtd.npz', fix_imports=True, allow_pickle=True)['arr_0'].item()
        bsf_lhsynth_mt = cls.from_args(
            mt_or_amide='mt',
            shape=shape, 
            T1a_ms=nvdict['t1w'].reshape(shape)*1000,
            T2a_ms=nvdict['t2w'].reshape(shape)*1000,
            fb_gt_T=0, kb_gt_T=0,
            fc_gt_T=nvdict['fs_0'].reshape(shape),
            kc_gt_T=nvdict['ksw_0'].reshape(shape),
            signal=nvdict['sig'].T.reshape(31,*shape),
        )
        return bsf_lhsynth_mt
    
    @classmethod
    def from_xarray(cls, brxarray, mt_or_amide=None, seq_df=None, **kwargs):
        """ Load data from a xarray dataset aggregating registered maps
        """                
        if type(seq_df) != type(None):
            signal = brxarray['measured_normed_T'].values
        else: 
            signal = brxarray['MT_data'].values if mt_or_amide == 'mt' else brxarray['AMIDE_data'].values
        return cls.from_args(
            mt_or_amide=mt_or_amide,
            shape=brxarray['T1ms'].values.shape,
            seq_df=seq_df,
            #signal=brxarray['measured_normed_T'].values,
            signal=signal, #brxarray['MT_data'].values if mt_or_amide == 'mt' else brxarray['AMIDE_data'].values,
            roi_mask_nans=brxarray['roi_mask_nans'].values,
            T1a_ms=brxarray['T1ms'].values,
            T2a_ms=brxarray['T2ms'].values,
            B0_shift_ppm_map=brxarray['B0_shift_ppm_map'].values,
            B1_fix_factor_map=brxarray['B1_fix_factor_map'].values,
            **kwargs
        )        
    
    @classmethod
    def normalize(cls, signal, norm_type=None):        
        norm_type = norm_type or cls.norm_type
        if norm_type == 'none':
            signal_pixelwise_norm = 1
        elif norm_type == 'l2': 
            signal_pixelwise_norm = np.linalg.norm(signal, axis=0) 
        elif norm_type == 'first':
            signal_pixelwise_norm = signal[0]
        elif norm_type == 'max':
            signal_pixelwise_norm = np.nanmax(signal, axis=0)        
        else:
            assert 0
        signal_normed = signal / (signal_pixelwise_norm + 1e-6)
        return signal_normed
    
    @classmethod
    def normalize_jax(cls, signal, norm_type=None):
        norm_type = norm_type or cls.norm_type
        if norm_type == 'none':
            signal_pixelwise_norm = 1
        elif norm_type == 'l2': 
            signal_pixelwise_norm = jnp.linalg.norm(signal, axis=0) 
        elif norm_type == 'first':
            signal_pixelwise_norm = signal[0]
        elif norm_type == 'max':
            signal_pixelwise_norm = jnp.nanmax(signal, axis=0)        
        else:
            assert 0
        signal_normed = signal / (signal_pixelwise_norm + 1e-6)
        return signal_normed
    
    def __len__(self):
        return self.R1a_V.shape[1]
    
    def __getitem__(self, idx):
        """ get a SLAB of <self.slw> slices starting at slice #idx, 
            downsampled (by sampling or cutout) by stride <self.ds> in each of X, Y
        """
        z0 = idx
        slw = self.slw
        ds = self.ds
        shape = self.R1a_V.shape      
        if self.random_ds:  
            xo, yo, zo = np.random.choice(ds, 3)          
        else:            
            self.xo_state += 1            
            if self.xo_state == ds:
                self.xo_state = 0
                self.yo_state = (self.yo_state + 1) % ds
            xo, yo = self.xo_state, self.yo_state
        
        if self.downsample_or_slab:  # downsample
            xe, ye = self.shape[0], self.shape[2]
        else:                        # cutout                            
            x_slabsize = (self.shape[0] // self.ds) 
            y_slabsize = (self.shape[2] // self.ds) 
            xo, xe = xo*x_slabsize, (xo+1)*x_slabsize
            yo, ye = yo*y_slabsize, (yo+1)*y_slabsize
            ds = 1 

        R_dict = {
            'R1a_T': self.R1a_V[xo:xe:ds, z0:z0+slw:, yo:ye:ds] + 0.0,
            'R2a_T': self.R2a_V[xo:xe:ds, z0:z0+slw:, yo:ye:ds] + 0.0,
            'R2b_T': self.R2b_V[xo:xe:ds, z0:z0+slw:, yo:ye:ds] + 0.0,
            'R2c_T': self.R2c_V[xo:xe:ds, z0:z0+slw:, yo:ye:ds] + 0.0
            }
        w_dict = {
            'wa_T': self.wa_T[:, xo:xe:ds, z0:z0+slw:, yo:ye:ds],
            'wb_T': self.wb_T[:, xo:xe:ds, z0:z0+slw:, yo:ye:ds],
            'wc_T': self.wc_T[:, xo:xe:ds, z0:z0+slw:, yo:ye:ds],
            'w1_T': self.w1_T[:, xo:xe:ds, z0:z0+slw:, yo:ye:ds]
            }
        measured_normed_T = self.measured_normed_T[:, xo:xe:ds, z0:z0+slw:, yo:ye:ds]
        
        if self.add_noise_to_signal != 0:            
            measured_normed_T = measured_normed_T + self.add_noise_to_signal * np.random.randn(*measured_normed_T.shape)            
            measured_normed_T = self.normalize(measured_normed_T)
            
        roi_mask_nans_T = self.roi_mask_nans[xo:xe:ds, z0:z0+slw:ds, yo:ye:ds]
        
        gt_dict = {
            'kb_gt_T': self.kb_gt_T[xo:xe:ds, z0:z0+slw:, yo:ye:ds],
            'fb_gt_T': self.fb_gt_T[xo:xe:ds, z0:z0+slw:, yo:ye:ds],
            'kc_gt_T': self.kc_gt_T[xo:xe:ds, z0:z0+slw:, yo:ye:ds],
            'fc_gt_T': self.fc_gt_T[xo:xe:ds, z0:z0+slw:, yo:ye:ds]
            }
        
        return (roi_mask_nans_T, measured_normed_T, w_dict, R_dict, gt_dict, (xo,yo,z0))

    @property
    def batch_shape(self):
        shape = np.array(self.shape)
        shape[1] = self.slw
        return shape//self.ds
    
    
