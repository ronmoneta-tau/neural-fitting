""" Jax-accelerated dictionary matching
    A.Finkel. Dec2023
"""

import scipy, numpy as np
import jax, jax.numpy as jnp    
import matplotlib.pyplot as plt
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
        

#@jax.jit
@partial(jax.jit, static_argnames=['constrain_T1T2', 'constrain_B0B1'])
def mrf_dict_resolve_1D(normed_signal_dict_to_use_T, signal, 
                        constrain_T1T2=False,
                        T1_dict=np.array(0), T2_dict=np.array(0), T1_row=np.array(0), T2_row=np.array(0),
                        constrain_B0B1=False,
                        B0_dict=np.array(0), B1_dict=np.array(0), B0_row=np.array(0), B1_row=np.array(0),
                        ):    
    ''' HW-accelerated resolution for a ROW of voxels.
        Note optional restriction to externally-specified T1, T2 (same can be done for B0,B1,etc.)
    '''        
    dot_prod = jnp.matmul(normed_signal_dict_to_use_T, signal)
    # e.g., 0.2 to constrain to 20% change w.r.t independently estimated 
    # (a "slab" in the grid around the auxiliary value)
    threshold = 0.2  
    if constrain_T1T2: # T1 is not None:
        T1_constraint_mask = (jnp.abs(T1_dict[:, None] / T1_row[None, :] - 1) < threshold).astype(jnp.int32)   
        # print(T1_constraint_mask.shape, dot_prod.shape)
        dot_prod = dot_prod * T1_constraint_mask
        T2_constraint_mask = (jnp.abs(T2_dict[:, None] / T2_row[None, :] - 1) < threshold).astype(jnp.int32)    
        dot_prod = dot_prod * T2_constraint_mask
        
    if constrain_B0B1: # T1 is not None:
        B0th = 0.1  # ppm 
        B0_constraint_mask = (jnp.abs(B0_dict[:, None] - B0_row[None, :]) < B0th).astype(jnp.int32)           
        dot_prod = dot_prod * B0_constraint_mask
        B1th = 0.1  # ratio 
        B1_constraint_mask = (jnp.abs(B1_dict[:, None] - B1_row[None, :]) < B1th).astype(jnp.int32)           
        dot_prod = dot_prod * B1_constraint_mask
        
    best_inds = jnp.argmax(dot_prod, axis=0)
    best_values = jnp.max(dot_prod, axis=0)
    if False:
        print(best_inds.shape)
        print(dot_prod.shape)
        jax.debug.print("{}", jnp.max(inds)) 
    
    return best_inds, best_values

    
class DictMatcher:

    def __init__(
            self, normed_signal_dict_to_use, 
            T1_dict=None, T2_dict=None,
            B0_dict=None, B1_dict=None
            ):
        self.normed_signal_dict_to_use = normed_signal_dict_to_use
        self.T1_dict = T1_dict
        self.T2_dict = T2_dict
        self.B0_dict = B0_dict
        self.B1_dict = B1_dict
        
    #@jax.jit   # - crashes the kernel..?
    def mrf_dict_resolve_2D(self, data, T1=None, T2=None, B0=None, B1=None):             
        ''' Solution for a SLICE of voxels.
            Note optional restriction to externally-specified T1, T2 (same can be done for B0,B1,etc.)
        '''
        for rowind in range(data.shape[1]):            
            constrain_T1T2 = (T1 is not None) and (T2 is not None)
            constrain_B0B1 = (B0 is not None) and (B1 is not None)
            new_inds, new_values = mrf_dict_resolve_1D(
                self.normed_signal_dict_to_use.T, 
                data[:,rowind,:], 
                constrain_T1T2 = constrain_T1T2,
                constrain_B0B1 = constrain_B0B1,
                T1_dict=self.T1_dict, 
                T2_dict=self.T2_dict, 
                T1_row=T1[rowind,:] if T1 is not None else None, 
                T2_row=T2[rowind,:] if T2 is not None else None,
                B0_dict=self.B0_dict, 
                B1_dict=self.B1_dict, 
                B0_row=B0[rowind,:] if B0 is not None else None, 
                B1_row=B1[rowind,:] if B1 is not None else None,                                                              
                )

            inds = jnp.concatenate( (inds, new_inds[None, ...]), axis=0 ) if rowind>0 else new_inds[None, ...]
            values = jnp.concatenate( (values, new_values[None, ...]), axis=0 ) if rowind>0 else new_values[None, ...]

        return inds, values

    def match(
            self, normed_signal_measured, fs_grid, ks_grid, 
            T1constraint=None, T2constraint=None, B0constraint=None, B1constraint=None
            ):
        ''' Matching a VOLUME of voxels
        '''
        inds = np.zeros(normed_signal_measured.shape[1:], dtype=np.int32)  
        values = np.zeros(normed_signal_measured.shape[1:])

        for slind in range(normed_signal_measured.shape[2]):
            print(slind)    
            _inds, _values = self.mrf_dict_resolve_2D(
                normed_signal_measured[:,:,slind,:],
                T1constraint[:,slind,:] if T1constraint is not None else None,
                T2constraint[:,slind,:] if T2constraint is not None else None,
                B0constraint[:,slind,:] if B0constraint is not None else None,
                B1constraint[:,slind,:] if B1constraint is not None else None
                )
            inds[:,slind,:] = _inds
            values[:,slind,:] = _values

        fspred = fs_grid.flatten()[inds] 
        kspred = ks_grid.flatten()[inds] 

        return fspred, kspred, inds


fs2mM = 110e3 / 3  # this is L-arg specific, need to expose for config..

def visualize_uncertainty(
        ks_grid, fs_grid, normed_signal_dict_to_use, d_shape, 
        normed_signal_measured, xy_points, Larg_mM_GT, PH_GT, sl=5, predmaps=[]
        ):
    u_ks = np.unique(ks_grid)
    u_fs = np.unique(fs_grid)
    fig = plt.figure(figsize=(15, 10))

    positions = [
        [0.05, 0.55, 0.24, 0.35], [0.38, 0.55, 0.24, 0.35], [0.71, 0.55, 0.24, 0.35],
        [0.05, 0.1, 0.24, 0.35], [0.38, 0.1, 0.24, 0.35], [0.71, 0.1, 0.24, 0.35]
    ]
    plt.rcParams.update({        
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'legend.fontsize': 10
    })   
    # see private for old viz options..
    for jj, pos in enumerate(positions):
        ax = fig.add_axes(pos)
        X_, Y_ = xy_points[jj]
        normed_signal = normed_signal_measured[:, X_, Y_]        

        dot_prod = jnp.matmul(normed_signal_dict_to_use.T, normed_signal)    
        dot_prod = dot_prod.reshape(d_shape)
        best_pos = np.unravel_index(dot_prod.argmax(), dot_prod.shape)    
        # Fixate the best B1, T1, T2, to remain with ks-fs 2D matrix
        mymap = np.array(dot_prod[best_pos[0], best_pos[1], best_pos[2]]) # .T
        mymap[mymap<0.999] = np.nan        
        nrmse = np.sqrt(1-mymap)
        kticks = np.arange(0, mymap.shape[1], 5)
        fticks = np.arange(3, mymap.shape[0], 5)
        ktick_labels = np.int32(u_ks[kticks])
        ftick_labels = np.int32(u_fs[fticks] * fs2mM)  # 110e3/3
        nrmse = nrmse.T        
        plt.grid()    
        im = plt.imshow(nrmse, aspect='auto', clim=[0.009, 0.033])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="7%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)    
        cbar.set_label('NRMSE', labelpad=-55, y=.5, rotation=90, fontweight='bold')
        ax.set_yticks(kticks, ktick_labels)
        ax.set_xticks(fticks, ftick_labels)
        ax.set_ylabel('k$_{sw}$ (s$^{-1}$)')
        ax.set_xlabel('[L-arg] (mM)')
        imgmin = np.nanmin(nrmse)
        levels = imgmin*np.array([1.1, 1.2, 1.4, 1.8])        
        c=ax.contour(nrmse, levels=levels, colors=['y', 'r', 'orange', 'brown'])        
        ax.set_title(f'[L-arg]={Larg_mM_GT[jj]}mM, pH={PH_GT[jj]}')

        fs_coo = (mymap.shape[0]-1) * (Larg_mM_GT[jj]-np.min(u_fs)*fs2mM) / (np.max(u_fs)*fs2mM-np.min(u_fs)*fs2mM)  
        line, = ax.plot([fs_coo, fs_coo], [0, mymap.shape[1]-1], '--', color="gray", linewidth=2.5, label='[L-arg] as prepared')
        handles = [line]
        for marker, (_mM, _ksw, _name) in zip(('rX', 'ro', 'rs'), predmaps):
            fs_coo = (mymap.shape[0]-1) * (_mM[X_,Y_]/fs2mM - np.min(u_fs)) / (np.max(u_fs)-np.min(u_fs))
            ksw_coo = (mymap.shape[1]-1) * (_ksw[X_,Y_] - np.min(u_ks)) / (np.max(u_ks)-np.min(u_ks))
            line, = ax.plot(
                [fs_coo, fs_coo], [ksw_coo, ksw_coo], marker,
                label=_name, markeredgecolor='k', markeredgewidth=1.5, markersize=8,
                )   
            handles.append(line)            
        if True: 
            ax.legend(handles=handles, loc='upper left' if jj<3 else 'lower right') 
        if jj==0:  # Contour legend
            legend_handles = [
                mpatches.Patch(color='y', label='nrmse = 1.1 x min(nrmse)'),
                mpatches.Patch(color='r', label='nrmse = 1.25 x min(nrmse)'),
                mpatches.Patch(color='orange', label='nrmse = 1.5 x min(nrmse)'),
                mpatches.Patch(color='brown', label='nrmse = 1.75 x min(nrmse)')
            ]
            ax.legend(
                handles=legend_handles, 
                loc='upper left'
            ) 
