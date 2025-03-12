import gc
import time
import numpy as np
import torch
import jax, jax.numpy as jnp 
from dataclasses import dataclass
import tqdm 

import data
import simulation


@dataclass
class InferConfig:        
    
    # Interpret net outputs as k_ab, k_ba ->f=ka/kb)
    predict_k_k: bool = True       
        
    # Auxiliary inputs to the NN predictor    
    use_b1_aux_input: bool = True
    use_b0_aux_input: bool = False
    
    # Give the NN predictor an auxiliary input of MT "ground truth" tissue params
    # (typically set to the estimates from MT protocol in previous stage )
    # (!) set to True for CEST
    use_cfsskss_inp: bool = False  
        
    # Fix auxiliary params using outputs from the NN predictor 
    # (experimental, not tested since original POC)
    use_pred_B1_fix: bool = False
    use_pred_R2c_fix: bool = False
    use_pred_T2_fix: bool = False
    
    # Parameter ranges for the "b" (aka CEST) pool
    # Set to L-arg values as default (phantom)
    # (!) change for Amide, NOE, etc.
    fb_scale_fact: float = 1e-3  
    kb_scale_fact: float = 500

    # Parameter ranges for the "c" (aka MT aka semisolid/ss) pool
    fc_scale_fact: float = 0.3
    kc_scale_fact: float = 100
    
infer_config = InferConfig()


def nn_predict_tissue_params(nn_predict_func, measured_normed_T, w_dict, R_dict, forward_config=None, subkey=None):
    """ Note side-effect on w_dict, R_dict in case the "fixes" are enabled.        
    """
    forward_config = forward_config or infer_config
    nn_input = measured_normed_T + 0.0 
    
    aux_inputs = jnp.concatenate((R_dict['R1a_T'][None, ...], R_dict['R2a_T'][None, ...]))
    if infer_config.use_b0_aux_input:
        b0ppm = (w_dict['wa_T'][0] / jnp.nanmean(w_dict['wa_T'][0]) - 1) * 1e6  # recover local ppm        
        aux_inputs = jnp.concatenate((aux_inputs, b0ppm[None, ...]))
    if infer_config.use_b1_aux_input:
        b1fac = w_dict['w1_T'][0] / jnp.nanmean(w_dict['w1_T'][0])              # recover local factor           
        aux_inputs = jnp.concatenate((aux_inputs, b1fac[None, ...]))

    nn_input = jnp.concatenate((aux_inputs, nn_input), axis=0)  

    nn_input = jnp.transpose(nn_input, [1, 2, 3, 0]) # channel last for jax
    nn_input = nn_input[None, ...]  # 5D      
    nn_input = jnp.nan_to_num(nn_input)
    
    # (!) Apply the actual neural network
    output, misc_nn_updates = nn_predict_func(nn_input)

    if forward_config.predict_k_k:
        # Output encodes k_ab and k_ba thus indirectly encoding f_b as the ratio
        k_ba = output[0,:,:,:,1] * forward_config.kb_scale_fact
        k_ab = output[0,:,:,:,0] * forward_config.fb_scale_fact * forward_config.kb_scale_fact
        k_b_ispos = (jnp.sign(k_ba)+1)/2  # 1 if pos else 0
        f_b = k_ab / (k_ba * k_b_ispos + 0.1)                

        k_ca = output[0,:,:,:,3] * forward_config.kc_scale_fact
        k_ac = output[0,:,:,:,2] * forward_config.fc_scale_fact * forward_config.kc_scale_fact
        k_c_ispos = (jnp.sign(k_ca)+1)/2
        f_c = k_ac / (k_ca * k_c_ispos + 0.1)
    else:
        k_ba = output[0,:,:,:,1] * forward_config.kb_scale_fact
        f_b = output[0,:,:,:,0] * forward_config.fb_scale_fact
        k_ca = output[0,:,:,:,3] * forward_config.kc_scale_fact
        f_c = output[0,:,:,:,2] * forward_config.fc_scale_fact

    pred_tissue_params = {}

    # -------- Extracting the "learned noise covariance" from net predictions --------
    cov_attenuation_fac = 1.   
    theta = jnp.pi * (output[0,:,:,:,7] - 0.5)  # sigmoid [0,1] --> [-pi/2, +pi/2]
    s12 = output[0,:,:,:,8:10] * cov_attenuation_fac            
    brshape = output.shape[1:-1]
    
    # Rotation matrix with theta
    u = jnp.concatenate([
        jnp.cos(theta)[..., None], jnp.sin(theta)[..., None],
        -jnp.sin(theta)[..., None], jnp.cos(theta)[..., None]
    ], axis=-1).reshape([*theta.shape, 2, 2])
    
    # Diagonal matrix with a, b
    s = jnp.concatenate([
        s12[..., :1], jnp.zeros_like(s12[..., :1]), 
        jnp.zeros_like(s12[..., :1]), s12[..., -1:]
    ], axis=-1).reshape([*theta.shape, 2, 2])
    
    # The trio theta,a,b parameterize a 2D covariance matrix:
    cov = jnp.matrix_transpose(u) @ s @ u

    pred_tissue_params.update({'cov': cov.reshape([*theta.shape, 4])})
    pred_tissue_params.update({'ucov': u.reshape([*theta.shape, 4])})
    pred_tissue_params.update({'scov': s.reshape([*theta.shape, 4])})
    
    if subkey != None:        
        wgn2 = jax.random.normal(subkey, shape=[*brshape, 2, 1])
        
        # (!) Create the diff vector by taking a N(0,1) vectors, 
        #     then scaling and rotating them with the NN-predicted parameters above
        dfk = u @ jnp.sqrt(s) @ wgn2

        if False:  # alternative way is to use multivariate. Not used - we prefer direct.
            dfk = jax.random.multivariate_normal(subkey, mean=jnp.zeros(cov.shape[:-1]), cov=cov) # , (5000,)).T       

        pred_tissue_params['df_b'] = dfk[..., 0, 0] * forward_config.fb_scale_fact 
        pred_tissue_params['dk_ba'] = dfk[..., 1, 0] * forward_config.kb_scale_fact 
        pred_tissue_params['df_c'] = dfk[..., 0, 0] * forward_config.fc_scale_fact 
        pred_tissue_params['dk_ca'] = dfk[..., 1, 0] * forward_config.kc_scale_fact         
        # Note: when upgrading to predict b,c simulataneously, will need to separate noise generation
    # ---------------

    pred_tissue_params.update({'kb_T': k_ba, 'fb_T': f_b}) 
    pred_tissue_params.update({'kc_T': k_ca, 'fc_T': f_c}) 
    
    # --- Experimental:  "back-fixes" for B1, R2c, T2a ---
    if infer_config.use_pred_B1_fix:
        pred_tissue_params['B1_fix'] =  jnp.clip(1 + 0.1*output[0,:,:,:,4], 0.8, 1.2)
    if infer_config.use_pred_R2c_fix:
        pred_tissue_params['R2c_fix'] = jnp.clip(1 + 0.1*output[0,:,:,:,5], 0.3, 3)
    if infer_config.use_pred_T2_fix:    
        t2a_fix_min, t2a_fix_max = 0.5, 1.5
        pred_tissue_params['R2a_fix'] = 1 / (t2a_fix_min + jax.nn.sigmoid(100*output[0,:,:,:,6]) * (t2a_fix_max-t2a_fix_min))
    # -----------------------------------------------

    return pred_tissue_params, misc_nn_updates


# Note: for massive applications of forward inference alone (e.g., large dictionary generation),
# consider to JIT this for further acceleration. For training, covered by JIT of training_step.
# @partial(jax.jit, static_argnames=['simulation_mode']) 
def physical_model(
        tissue_params, w_dict, wrf_T, tr_tsat, simulation_mode='isar2_b', 
        measured_normed_T=None, seq_mode='sequential',                   
        normalizer=None,
        ):
    ''' simulation_mode:  isar2_(b|c), expm_bmmat, eigen_bmmat(deprecated)
        seq_mode:  sequential vs. parallel (using measurements, only used in training)
        Note: parallel mode assumes normalization by M0
    '''        
    tr_tsat = np.array(tr_tsat).reshape(-1, 2)
    if simulation_mode.startswith('isar2'):
        # ISAR2 top eigenvalues-based approximation (Roeloffs et. al., 2015)
        if simulation_mode == 'isar2_c':     #  (!!) the 2-pool MT case
            R2_pool2, f_pool2, k_pool2 = (tissue_params[x] for x in ('R2c_T','fc_T','kc_T'))
            w_pool2 = w_dict['wc_T']
        elif simulation_mode == 'isar2_b':   #  (!!) the 2-pool CEST case:  (phantoms)
            R2_pool2, f_pool2, k_pool2 = (tissue_params[x] for x in ('R2b_T','fb_T','kb_T'))
            w_pool2 = w_dict['wb_T']
        else:
            assert 0
        predicted_signal_T = \
                simulation.forward_mrf_isar2(
                    f_pool2, k_pool2, 
                    w_dict['wa_T'], w_pool2,  # w_pool2: b/c
                    tissue_params['R1a_T'], 
                    tissue_params['R2a_T'],
                    tissue_params['R1a_T'], # ! assuming same R1 for all pools                                                       
                    R2_pool2, 
                    w_dict['w1_T'], wrf_T, 
                    TRs=tr_tsat[:,0], TSATs=tr_tsat[:,1],
                    mode=seq_mode, Z_acq_meas=measured_normed_T)
    else: 
        # Full BM matrix usage - via exponentiation OR explicit eigen-decomposition (deprecated)
        predicted_signal_T = \
            simulation.forward_mrf(
                tissue_params, w_dict, wrf_T, 
                TRs=tr_tsat[:,0], TSATs=tr_tsat[:,1],
                mode=seq_mode, Z_acq_meas=measured_normed_T,
            )     
    normalizer = normalizer or data.SlicesFeed.normalize_jax
    predicted_signal_normed_T = normalizer(predicted_signal_T)
    return predicted_signal_normed_T


def infer(brain_ds, forward_config=None, do_forward=True, nn_predictor=None,
            simulation_mode='expm_bmmat', pool2predict='b', seq_mode='sequential'):     
    """ 
        With default arguments, runs the physical measurement model on stored tissue params.
        Pass nnmodel to infer tissue params (f_ss, k_ssw or f_s, k_sw) from stored measurements.
        if using nnmodel, passing do_forward=False will stop short of running forward physical model to get it faster,
        otherwise will also produce a round-trip predicted signal that can be compared to the measured one.                 
    """
    forward_config = forward_config or infer_config
    tr_tsat = np.array(brain_ds.seq_df[['TR_ms', 'Tsat_ms']] / 1000)
    tr_tsat = tuple(tr_tsat.flatten())
    tissue_params = {}    
    predicted_signal_normed_np = np.zeros((brain_ds.seq_len, *brain_ds.shape))

    b0_dl = torch.utils.data.DataLoader(brain_ds, sampler=range(0, brain_ds.R1a_V.shape[1], brain_ds.slw))
    tissue_params_pred_full = {}    
    
    # We want to infer all voxels so no downsampling; 
    # if single-slice still doesn't fit in memory, need to use part-slice, see some stubs in data.py
    assert brain_ds.ds == 1
    
    with tqdm.tqdm(total=len(b0_dl)) as pbar:
        for sli, data_entry in enumerate(b0_dl):                
            roi_mask_nans_T, measured_normed_T, w_dict, R_dict, gt_dict = data.decode_data_entry(data_entry[:-1])            
            tissue_params = {}
            if nn_predictor is not None:
                if forward_config.use_cfsskss_inp and pool2predict == 'b': 
                    data2input = jnp.concatenate((measured_normed_T,  gt_dict['fc_gt_T'][None,:,:,:], gt_dict['fc_gt_T'][None,:,:,:]), axis=0)                          
                else:
                    data2input = measured_normed_T
                tissue_params, _ = nn_predict_tissue_params(nn_predictor, data2input, w_dict, R_dict)            
                
            if 'b' not in pool2predict or nn_predictor is None:
                tissue_params.update({
                    'fb_T': gt_dict['fb_gt_T'],
                    'kb_T': gt_dict['kb_gt_T']
                    }
                )
            if 'c' not in pool2predict or nn_predictor is None:
                tissue_params.update({
                    'fc_T': gt_dict['fc_gt_T'],
                    'kc_T': gt_dict['kc_gt_T']
                    }
                )
            tissue_params.update(R_dict)            

            # --- Experimental:  "fixes" for B1, R2c, T2a ---
            if forward_config.use_pred_T2_fix:                
                tissue_params['R2a_T'] = tissue_params['R2a_T'] * tissue_params['R2a_fix']
            if forward_config.use_pred_R2c_fix:        
                tissue_params['R2c_T'] = tissue_params['R2c_T'] * tissue_params['R2c_fix']
            if forward_config.use_pred_B1_fix:        
                w_dict['w1_T'] = w_dict['w1_T'] * tissue_params['B1_fix']    
            # ----------------------------------------------
            
            tissue_params = {k: v * (
                    roi_mask_nans_T if (v.shape == roi_mask_nans_T.shape) 
                    else roi_mask_nans_T[..., None]
                    ) for k, v in tissue_params.items() 
                } 
            
            # ..accumulate over slabs..
            if tissue_params_pred_full == {}:
                tissue_params_pred_full = {k: np.array(v) for k, v in tissue_params.items()}
            else:                    
                tissue_params_pred_full = {k: np.concatenate((tissue_params_pred_full[k], v), axis=1) for k, v in tissue_params.items()}                    

            if do_forward:
                wrf_T = brain_ds.wrf_T
                predicted_signal_normed_slab = physical_model(
                    tissue_params, w_dict, wrf_T, tr_tsat=tr_tsat, 
                    simulation_mode=simulation_mode, seq_mode=seq_mode,
                    measured_normed_T=measured_normed_T
                ) 
                # ..accumulate over slabs..
                predicted_signal_normed_np[:, :, sli*brain_ds.slw: (1+sli)*brain_ds.slw, :] = predicted_signal_normed_slab
                
            pbar.update(1)    

    return tissue_params_pred_full, predicted_signal_normed_np


def construct_large_dictionary():
    """    
        Demonstration of constructing a 79M dictionary (Perlman 22) by running the forward model ,
        on a large collection of tissue parameter values combinations (Cartesian grid).
        somewhere between 6M and 8M blows out the 32GB of RAM, so we shard it into ~17 pieces.
    """
    print("Constructing large dictionary demo running...")
    data.SlicesFeed.norm_type = 'l2'  # (!) works better in supervised.
    total_slices = 8778  # 79000200 / 9000
    #shard_size = 1100    # 10M shards (x8) - crashes after saving?
    shard_size = 550    # 5M shards (x16) 
    for shard in range(int(np.ceil(total_slices/shard_size))):  
        t0 = time.time()
        bsf_lhsynth_mt = data.SlicesFeed.make_cartesian_sample(
            mt_or_amide='amide',
            parameter_values_od=data.perlman2022_cartesian,
            #shape=(100, 600, 90)
            shape=(100, 8778, 90),
            slices=[shard*shard_size, (shard+1)*shard_size]             
        ) 
        bsf_lhsynth_mt.slw = 50
        _, signal_3p = infer(bsf_lhsynth_mt)
        signal_3p = bsf_lhsynth_mt.normalize(signal_3p)
        np.savez_compressed(f'opdict/{shard}', signal_3p)
        print(f"Shard {shard} done in {time.time()-t0} seconds")
        del bsf_lhsynth_mt
        gc.collect()


if __name__ == "__main__":
    construct_large_dictionary()