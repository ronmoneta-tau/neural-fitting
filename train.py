import numpy as np
from typing import Any
from functools import partial
import jax, jax.numpy as jnp, optax
from jax import custom_vjp
from flax.training import train_state
import torch, tqdm
from dataclasses import dataclass
import logging

import net   
import data
import infer


@dataclass
class TrainConfig:

    # Simulation - sequential/parallel
    sim_seq_mode: str = 'parallel'
    
    # Data feed
    auto_reduce_batch: bool = True  # on 

    # Network
    hidden_layers:int = 2
    hidden_width: int = 256
    output_features: int = 20  # some extra for the learned-noise
    
    # Training process
    weight_decay: float = 1e-3
    k_force: bool = None
    reglosstype: str = 'L1'  
    # NOTE: this is not really L1 loss but rather abs averaging over voxels rather than root-mean-square;
    #       within the voxel it's anyways L2 over the sequence.
    use_shuffled_sampler: bool = False
    cosine_schedule: bool = True

    # Injection of noise in tissue-params
    tp_noise:bool = True
    std_up_fact: float = 0.8
    
    # Early stopping params
    min_delta:float = 0.1
    patience:float = 7    
    
    # debugging
    print_mean_TP: bool = False
    force_round_trip_eval_period: int = 300


train_config = TrainConfig()
    
# ----- morphed clip-gradient example for NAN_TO_NUM ----
@custom_vjp
def clip_gradient(lo, hi, x):
    return x  # identity function

def clip_gradient_fwd(lo, hi, x):
    return x, (lo, hi)  # save bounds as residuals

def clip_gradient_bwd(res, g):
    lo, hi = res
    return (None, None, jnp.clip(jnp.nan_to_num(g), lo, hi))  # use None to indicate zero cotangents for lo and hi

clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)
#  ------------------------------------------------------


def bmfit_activate_tp(tissue_params):
    ''' from trainable variables to values - invoke in comp.graph and again on eval(consistency)
    '''
    tissue_params['fb_T'] = jax.nn.sigmoid(tissue_params['fb_T']) * infer.infer_config.fb_scale_fact
    tissue_params['kb_T'] = jax.nn.sigmoid(tissue_params['kb_T']) * infer.infer_config.kb_scale_fact
    tissue_params['fc_T'] = jax.nn.sigmoid(tissue_params['fc_T']) * infer.infer_config.fc_scale_fact
    tissue_params['kc_T'] = jax.nn.sigmoid(tissue_params['kc_T']) * infer.infer_config.kc_scale_fact        

    
def bmfit_set_gt_from_model_state(brainds, model_state):
    """ ..some redunancy to the above, need to merge.."""
    brainds.fb_gt_T = np.array(jax.nn.sigmoid(np.array(model_state.params['fb_T']))) * infer.infer_config.fb_scale_fact
    brainds.kb_gt_T = np.array(jax.nn.sigmoid(np.array(model_state.params['kb_T']))) * infer.infer_config.kb_scale_fact
    brainds.fc_gt_T = np.array(jax.nn.sigmoid(np.array(model_state.params['fc_T']))) * infer.infer_config.fc_scale_fact
    brainds.kc_gt_T = np.array(jax.nn.sigmoid(np.array(model_state.params['kc_T']))) * infer.infer_config.kc_scale_fact

    
def infer_round_trip_calc_loss(params, nn_predictor, data_entry, wrf_T=None, 
                                pool2predict='b', simulation_mode='expm_bmmat', mode='self_supervised', 
                                reglosstype=train_config.reglosstype, tr_tsat=None, 
                                force_round_trip_eval=False, slab_shape=(None,None,None), subkey=None):
    """
        Apply the actual computational graph (either supervised or self_supervised),
        and calculate the loss for a single batch of data.
    """
    roi_mask_nans_T, measured_normed_T, w_dict, R_dict, gt_dict, slab_apex = data_entry
    
    if mode in ('bloch_fitting',):        
        slab_tissue_params = {k: jax.lax.dynamic_slice(v, slab_apex, slab_shape) for k, v in params.items()}
        # Note the potentially surprising behavior for the case where the requested slice overruns the bounds of the array; 
        # in this case the start index is adjusted to return a slice of the requested size:        
        tissue_params, misc_nn_updates = slab_tissue_params, {}
        bmfit_activate_tp(tissue_params)        
    else:
        if pool2predict == 'b' and infer.infer_config.use_cfsskss_inp: 
            data2input = jnp.concatenate((measured_normed_T,  gt_dict['fc_gt_T'][None,:,:,:], gt_dict['fc_gt_T'][None,:,:,:]), axis=0)                          
        else:
            data2input = measured_normed_T
        tissue_params, misc_nn_updates = infer.nn_predict_tissue_params(nn_predictor, data2input, w_dict, R_dict, subkey=subkey)
    
    tissue_params.update(R_dict)
    
    # if reviving this experimental feature, consider merging into infer.nn_predict_tissue_params
    if infer.infer_config.use_pred_T2_fix:      
        tissue_params['R2a_T'] = tissue_params['R2a_T'] * tissue_params['R2a_fix']
    if infer.infer_config.use_pred_R2c_fix:        
        tissue_params['R2c_T'] = tissue_params['R2c_T'] * tissue_params['R2c_fix']
    if infer.infer_config.use_pred_B1_fix:        
        w_dict['w1_T'] = w_dict['w1_T'] * tissue_params['B1_fix']    
    
    aux_noise_enhancing_loss = 0
    if train_config.tp_noise and subkey != None :
        # Apply the learnt noising to the tissue params and create a loss term
        if 'c' in pool2predict:
            tissue_params['fc_T'] = jnp.clip(tissue_params['fc_T'] + tissue_params['df_c'], 0, infer.infer_config.fc_scale_fact)
            tissue_params['kc_T'] = jnp.clip(tissue_params['kc_T'] + tissue_params['dk_ca'], 0, infer.infer_config.kc_scale_fact)
            aux_noise_enhancing_loss += \
                - jnp.abs(tissue_params['df_c'] / infer.infer_config.fc_scale_fact) \
                - jnp.abs(tissue_params['dk_ca'] / infer.infer_config.kc_scale_fact)
        if 'b' in pool2predict:
            tissue_params['fb_T'] = jnp.clip(tissue_params['fb_T'] + tissue_params['df_b'], 0, infer.infer_config.fb_scale_fact)
            tissue_params['kb_T'] = jnp.clip(tissue_params['kb_T'] + tissue_params['dk_ba'], 0, infer.infer_config.kb_scale_fact)
            aux_noise_enhancing_loss += \
                - jnp.abs(tissue_params['df_b'] / infer.infer_config.fb_scale_fact) \
                - jnp.abs(tissue_params['dk_ba'] / infer.infer_config.kb_scale_fact)     
        aux_noise_enhancing_loss = jnp.nanmean(aux_noise_enhancing_loss) * train_config.std_up_fact
        
    est_error = np.nan   
    fk_super_loss = 0
    
    if 'b' not in pool2predict:
        tissue_params.update({'fb_T': gt_dict['fb_gt_T'], 'kb_T': gt_dict['kb_gt_T']})
    else: 
        if mode == 'reference_supervised':
            if reglosstype=='L1':
                fk_super_loss += jnp.nanmean(jnp.abs(tissue_params['fb_T'] - gt_dict['fb_gt_T'])) / infer.infer_config.fb_scale_fact
                fk_super_loss += jnp.nanmean(jnp.abs(tissue_params['kb_T'] - gt_dict['kb_gt_T'])) / infer.infer_config.kb_scale_fact
            elif reglosstype=='L2':
                f_mse = jnp.nanmean((tissue_params['fb_T'] - gt_dict['fb_gt_T'])**2) / infer.infer_config.fb_scale_fact**2
                k_mse = jnp.nanmean((tissue_params['kb_T'] - gt_dict['kb_gt_T'])**2) / infer.infer_config.kb_scale_fact**2
                fk_super_loss += jnp.sqrt(f_mse**2 + k_mse**2)
            else:     
                raise NotImplementedError()               
    if 'c' not in pool2predict:
        tissue_params.update({'fc_T': gt_dict['fc_gt_T'], 'kc_T': gt_dict['kc_gt_T']})
    else: 
        if mode == 'reference_supervised':
            if reglosstype=='L1':
                fk_super_loss += jnp.nanmean(jnp.abs(tissue_params['fc_T'] - gt_dict['fc_gt_T'])) / infer.infer_config.fc_scale_fact
                fk_super_loss += jnp.nanmean(jnp.abs(tissue_params['kc_T'] - gt_dict['kc_gt_T'])) / infer.infer_config.kc_scale_fact
            elif reglosstype=='L2':
                fk_super_loss += jnp.sqrt(jnp.nanmean((tissue_params['fc_T'] - gt_dict['fc_gt_T'])**2) / infer.infer_config.fc_scale_fact**2)
                fk_super_loss += jnp.sqrt(jnp.nanmean((tissue_params['kc_T'] - gt_dict['kc_gt_T'])**2) / infer.infer_config.kc_scale_fact**2)
            else:     
                raise NotImplementedError()       

    if mode in ('bloch_fitting', 'self_supervised') or (mode=='reference_supervised' and force_round_trip_eval) :   
        ## !! forward model here !! ##
        predicted_normed_T = infer.physical_model(
            tissue_params, w_dict, wrf_T, 
            measured_normed_T=measured_normed_T, seq_mode=train_config.sim_seq_mode,
            tr_tsat=tr_tsat, simulation_mode=simulation_mode
        ) * roi_mask_nans_T

        signal_est_diffnorm_map = jnp.linalg.norm(predicted_normed_T - measured_normed_T, axis=0) * roi_mask_nans_T
        signal_est_diffnorm_map /= jnp.linalg.norm(measured_normed_T, axis=0)  # ! for /first normalization
        
        if reglosstype=='L1':
            est_error = jnp.nanmean(jnp.abs(signal_est_diffnorm_map))
        elif reglosstype=='L2':
            est_error = jnp.sqrt(jnp.nanmean(signal_est_diffnorm_map**2))
        else:
            assert 'loss not supported'
                    
    if mode in ('self_supervised', 'bloch_fitting'):
        loss_total = 100 * est_error + aux_noise_enhancing_loss
    elif mode == 'reference_supervised':         
        loss_total = 100 * fk_super_loss
    else:
        assert 0    
    
    return loss_total, (est_error, tissue_params, misc_nn_updates)

                       
def infer_calc_loss_wrapped(model_state, params, batch, train=True, **kwargs):
    def nn_predictor(_batch): # perhaps can be simplified & pushed inside infer_round_trip_calc_loss?
        outs = model_state.apply_fn({'params': params, 'batch_stats': model_state.batch_stats},
                                    _batch, train=train, mutable=['batch_stats'] if train else False
                                    )        
        res, new_model_state = (clip_gradient(-1e3, 1e3, outs[0]), outs[1]) if train else \
                               (clip_gradient(-1e3, 1e3, outs), None)
        return res, new_model_state
    
    return infer_round_trip_calc_loss(params, nn_predictor, batch, **kwargs)


# Jit the function for efficiency (disable for debug)
@partial(jax.jit, static_argnames=['pool2predict', 'simulation_mode', 'mode', 'force_round_trip_eval', 'slab_shape', 'tr_tsat']) 
def train_step(state, batch, **kwargs):
    '''https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html#Creating-an-efficient-training-and-validation-step
    '''
    grad_fn = jax.value_and_grad(\
        infer_calc_loss_wrapped,  # Function to calculate the loss
        argnums=1,  # Parameters are second argument of the function
        has_aux=True  # Function has additional outputs,
    )    
    
    # Determine gradients for current model, parameters and batch
    (loss, (est_error, tissue_params, misc_nn_updates)), grads = grad_fn(state, state.params, batch, **kwargs)    

    # Perform parameter update with gradients and optimizer ( + update batch stats?)    
    state = state.apply_gradients(grads=grads, batch_stats=misc_nn_updates.get('batch_stats'))
        
    # Return state and any other value we might want
    return state, loss, est_error, tissue_params


class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    batch_stats: Any

    
class MyEarlyStopping:
    """ improving on flax.training.early_stopping.EarlyStopping """
    def __init__(self, min_delta=0.1, patience=5):
        self.min_delta = min_delta
        self.patience = patience
        self.best_value = 1e6
        self.counter = 0
        self.should_stop=False
        
    def update(self, metric):        
        self.new_best = (metric < self.best_value)                    
        diff_vs_best = metric - self.best_value
        if diff_vs_best < self.min_delta and diff_vs_best > -0.01:
            # ! looking for multiple epochs around (above or very slightly below) the best value..
            self.counter += 1
        else:
            self.counter = 0
        if self.counter > self.patience:
            self.should_stop=True
            
        self.best_value = np.min([self.best_value, metric])
        
        return None, self
            
        
def train(
    brain_ds, model_state=None, mode='self_supervised', steps=1000, lr=3e-3, 
    simulation_mode='expm_bmmat', pool2predict='b', force_sampler=None, logger=None
    ):
    
    logger = logger or logging.getLogger(__name__)
    tr_tsat = np.array(brain_ds.seq_df[['TR_ms', 'Tsat_ms']] / 1000)
    tr_tsat = tuple(tr_tsat.flatten())
    
    if train_config.auto_reduce_batch and not simulation_mode.startswith('isar2') and not mode == 'reference_supervised':
        # respect GPU mem lims' --> X40 less voxels per step 
        #  (vs. ds=1, slw=10, our standard for 2-pool isar2)
        brain_ds.ds = 2  
        brain_ds.slw = 1  
        # consider exposing in train_config (currently handled as config at the higher-level pipelines module)

    if mode == 'bloch_fitting':
        apply_fn = batch_stats = None       
        # NOTE: initializing to  mid-range, consider random and/or externally exposed.
        fb_var = 0*jnp.ones(brain_ds.shape) 
        kb_var = 0*jnp.ones(brain_ds.shape) # or cheat: brain_ds.kb_gt_T.numpy())
        fc_var = 0*jnp.ones(brain_ds.shape) 
        kc_var = 0*jnp.ones(brain_ds.shape) 
        params = {'fb_T': fb_var, 'kb_T': kb_var, 'fc_T': fc_var, 'kc_T': kc_var}
        brain_ds.downsample_or_slab = False  # slab not downsample
        net_kwargs = {}
    else:
        T1_aux_input = T2_aux_input = 1
        extra_inputs = T1_aux_input + T2_aux_input + infer.infer_config.use_b0_aux_input + infer.infer_config.use_b1_aux_input
        if (pool2predict=='b' and infer.infer_config.use_cfsskss_inp):
            extra_inputs += 2  #  fc,kc as estimated are also fed as auxiliary inputs
        net_kwargs = {
            'input_shape': list(brain_ds.batch_shape), 
            'mrf_len': brain_ds.seq_len,
            'hidden_width': train_config.hidden_width, 
            'hidden_layers': train_config.hidden_layers, 
            'output_features': train_config.output_features, 
            'extra_inputs': extra_inputs
        }
        if model_state == None:                    
            nnmodel, nnparams = net.get_net(**net_kwargs)
            apply_fn = nnmodel.apply
            batch_stats = nnparams['batch_stats']
            params = nnparams['params']
        else:
            apply_fn = model_state.apply_fn
            params = model_state.params
            batch_stats = model_state.batch_stats
    
    steps_per_epoch = max(1, len(brain_ds) // brain_ds.slw)
    epochs = steps // steps_per_epoch
    print(f"epochs = {epochs}")
    print(f"effective epochs = {epochs // (brain_ds.ds**2)}")
            
    epochs_per_decay = 10
    periods = epochs//epochs_per_decay
    cosine_restarts_lr_sched = optax.sgdr_schedule(
        [{"init_value":0., "peak_value":lr * (1-period/periods), 
          "decay_steps": epochs_per_decay*steps_per_epoch,
          "warmup_steps":1, "end_value":1e-6}  # "warmup_steps":10
          for period in range(periods)]
    )    
    if train_config.cosine_schedule:
        lr2use = cosine_restarts_lr_sched
    else:
        lr2use = lr  # just flat; can also consider simple decay or reduce-on-plateau
    # not clear if weight decay is in fact active..    
    optimizer = optax.adamw(learning_rate=lr2use, weight_decay=train_config.weight_decay)  
    early_stop = MyEarlyStopping(min_delta=train_config.min_delta, patience=train_config.patience) 
        
    model_state = TrainState.create(apply_fn=apply_fn,
                                    params=params,
                                    batch_stats=batch_stats,
                                    tx=optimizer,
                                    )
    slcrop = 0  # little hook to ignore top/bottom "edges" if needed

    def get_sampler(): 
        if train_config.use_shuffled_sampler:        
            sampler = np.random.choice(range(slcrop, len(brain_ds)-slcrop-brain_ds.slw+1), steps_per_epoch) 
        else:
            simplerange = np.arange(slcrop, len(brain_ds)-slcrop, brain_ds.slw)                    
            sampler = simplerange    
        sampler = np.clip(sampler, slcrop, len(brain_ds)-slcrop-brain_ds.slw)  # overlap prev instead of cut
        return sampler            
        
    rngkey = jax.random.key(0)
    est_error_np_MA = np.nan
    last_est_error_np_perc = np.nan
    est_loss_history = []
    mean_tp_hist = []
    
    with tqdm.tqdm(total=epochs*steps_per_epoch) as pbar:
        try:
            for epoch in range(epochs):
                sampler = get_sampler() if force_sampler is None else force_sampler
                b0_dl = torch.utils.data.DataLoader(brain_ds, sampler = sampler)
                for step, data_entry in enumerate(b0_dl):                     
                    data_entry_t = data.decode_data_entry(data_entry[:-1])
                    if mode == 'bloch_fitting':  
                        # adding the extra piece of data - the slab location - to enable trainable tissue params mode.
                        (x0, y0, z0) = data_entry[-1]                        
                        slab_shape = (brain_ds.shape[0] // brain_ds.ds, brain_ds.slw, brain_ds.shape[2] // brain_ds.ds)
                        slab_apex = [a.numpy()[0] for a in (x0, z0, y0)] # ! note z is axial, middle dim
                        data_entry_t = list(data_entry_t) + [slab_apex] 
                    else:
                        data_entry_t = list(data_entry_t) + [None]
                        slab_shape = None

                    rngkey, subkey = jax.random.split(rngkey)
                    if step < 100 and epoch==0:
                        subkey = None  # Start the noise only after we're in the ballpark..
                    total_step = epoch * steps_per_epoch + step
                    model_state, loss_total, est_error, tissue_params = train_step(\
                                model_state, data_entry_t,
                                wrf_T=brain_ds.wrf_T,
                                simulation_mode=simulation_mode, 
                                mode=mode,
                                pool2predict=pool2predict,
                                slab_shape=slab_shape,
                                tr_tsat=tr_tsat,
                                subkey=subkey,
                                force_round_trip_eval=(total_step % train_config.force_round_trip_eval_period == 1)
                                )
                    pbar.update(1)
                    tot_loss_np = loss_total
                    last_est_error_np_perc = 100*est_error if not np.isnan(est_error) else last_est_error_np_perc
                    aa = 0.98
                    est_error_np_MA = \
                        (aa*est_error_np_MA + (1-aa)*last_est_error_np_perc) \
                        if not (np.isnan(est_error_np_MA)) \
                        else last_est_error_np_perc
                    pbar.set_description(
                        f"loss_total = {tot_loss_np:.2f},"+\
                        f"signal reconstruction error (%) = {last_est_error_np_perc:.2f} (MA: {est_error_np_MA:.2f})"
                        )   
                    est_loss_history += [est_error_np_MA]
                    
                    if train_config.print_mean_TP:
                        mean_tp_hist.append({k: np.mean(tissue_params[k]) for k in ['fb_T', 'kb_T']})

                if epoch in range(0, epochs, int(np.ceil(epochs/20))): 
                    logger.info(str(pbar))
                
                _, early_stop = early_stop.update(est_error_np_MA)
                if early_stop.should_stop:
                    print(f'Met early stopping criteria, breaking at epoch {epoch}')
                    break
                if early_stop.new_best:                    
                    pass
                    
        except KeyboardInterrupt:
            print("Training cut short by user request")
            pass
    
    return \
        model_state, \
        {'est_loss_history': est_loss_history, 'mean_tp_hist': mean_tp_hist}, \
        net_kwargs


