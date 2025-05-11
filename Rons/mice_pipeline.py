import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
import jax.numpy as jnp
from inputs import  Inputs
from pathlib import Path
import matplotlib.colors as mcolors
sys.path.append(str(os.getcwd()))
import data, pipelines, utils, net, infer, simulation
import analyze_uncertainty as au


def cutout_dataset(dataset, cutout_height, cutout_width):
    data_xa_cutout = dataset.isel(height=cutout_height,width=cutout_width)

    plt.figure(figsize=(4,4))
    plt.imshow((data_xa_cutout['T1ms'].data * data_xa_cutout['roi_mask_nans']).squeeze())
    plt.title('T1ms')
    plt.colorbar()
    plt.show()

    return data_xa_cutout

def train_pipeline(data_feed_mt, data_feed_amide):

    pipelines.pipeline_config.train_config.std_up_fact = 0.2
    pipelines.pipeline_config.mt_lr = 0.01
    pipelines.pipeline_config.mt_steps = 500
    pipelines.pipeline_config.infer_config.kc_scale_fact = 80
    pipelines.pipeline_config.train_config.auto_reduce_batch = False
    pipelines.pipeline_config.train_config.tpnoise_augmentation_burn_in = 50
    net.MyMLP.sigmoid_scale_fac = 10  # priority for center of range
    data_feed_mt.ds = 1
    data_feed_mt.slw = 1
    data_feed_amide.ds = 1
    data_feed_amide.slw = 1


    # pipelines.pipeline_config.train_config.tp_noise = False # try without the noise augmentation
    # This should stay commented as it made the std later extremely large. seems like without it The network doesn't learn to predict uncertainty properly. and it defaults to predicting almost constant, large uncertainty values for all pixels

    predictor_mt, predictor_amide = pipelines.run_train(
        brain2train_mt=data_feed_mt,
        brain2train_amide=data_feed_amide,
        ckpt_folder=os.path.abspath(f'./goo'),
        #do_amide=False,
        mt_sim_mode = 'expm_bmmat'
    )

    return predictor_mt, predictor_amide

def inference_pipeline(data_feed, nn_predictor, pool2predict):
    data_feed.ds = 1
    tissue_params_est, pred_signal_normed_np = infer.infer(data_feed, nn_predictor=nn_predictor,do_forward=True, pool2predict=pool2predict)

    return tissue_params_est, pred_signal_normed_np

def plot_tissue_params_and_error(data_feed, pred_signal_normed_np,tissue_params_est, solute_name):
    if solute_name == 'MT':
        f_values = tissue_params_est.get('fc_T', None)
        k_values = tissue_params_est.get('kc_T', None)
    elif solute_name == 'rNOE':
        f_values = tissue_params_est.get('fb_T', None)
        k_values = tissue_params_est.get('kb_T', None)
    else:
        raise ValueError("Unknown solute name. Expected 'MT' or 'rNOE'.")

    plt.figure(figsize=(7, 3))
    plt.suptitle(f'{solute_name} params')
    plt.subplot(1,3,1)
    plt.imshow(f_values.squeeze() * 100)
    plt.colorbar()
    plt.title(f'{solute_name} fs values')

    plt.subplot(1,3,2)
    plt.imshow(k_values.squeeze(), cmap='magma')
    plt.colorbar()
    plt.title(f'{solute_name} ks values')

    plt.subplot(1,3,3)
    
    err, norm = calculate_error_map(data_feed, pred_signal_normed_np)

    cmap_nrmse = plt.cm.get_cmap("YlOrRd").copy()
    cmap_nrmse.set_bad('1.0')
    img0 = plt.imshow(err[:,0,:], norm=norm, cmap=cmap_nrmse) #'hot_r'); # plt.colorbar(img0)  # vmin=0, vmax=0.1,
    cbar = plt.colorbar(img0)

    plt.show()

def calculate_error_map(data_feed, pred_signal_normed_np):
    # Visualize the error maps in a slightly more informative way
    err = np.linalg.norm(pred_signal_normed_np - data_feed.measured_normed_T, axis=0) / np.linalg.norm(data_feed.measured_normed_T, axis=0)
    log_bins = np.logspace(np.log10(1.), np.log10(20), num=12)/100  # logarithmic
    norm = mcolors.BoundaryNorm(log_bins, ncolors=plt.get_cmap('hot_r').N, clip=True)

    return err, norm
    
def create_bound_maps(data_feed, tissue_params_est, points, solute_name, explore_amide_uncertainty=False):    

    if solute_name == 'MT':
        f_val = 100*tissue_params_est.get('fc_T', None)
        k_val = tissue_params_est.get('kc_T', None)
        f_scale_fact = infer.infer_config.fc_scale_fact
        k_scale_fact = infer.infer_config.kc_scale_fact
    elif solute_name == 'rNOE':
        f_val = 100*tissue_params_est.get('fb_T', None)
        k_val = tissue_params_est.get('kb_T', None)
        f_scale_fact = infer.infer_config.fb_scale_fact
        k_scale_fact = infer.infer_config.kb_scale_fact
    else:
        raise ValueError("Unknown solute name. Expected 'MT' or 'rNOE'.")

    u = tissue_params_est['ucov'].reshape((*data_feed.shape, 2, 2))
    s = tissue_params_est['scov'].reshape((*data_feed.shape, 2, 2))
    # cov = mt_tissue_param_est['cov'].reshape((*data_feed_mt.shape, 2, 2))

    dfk = np.array([1, 0])[None, None, None, :, None]
    dfk = u @ jnp.sqrt(s) @ dfk
    df_c_0 = dfk[..., 0, 0] * f_scale_fact
    dk_ca_0 = dfk[..., 1, 0] * k_scale_fact

    dfk = np.array([0, 1])[None, None, None, :, None]
    dfk = u @ jnp.sqrt(s) @ dfk
    df_c_1 = dfk[..., 0, 0] * f_scale_fact
    dk_ca_1 = dfk[..., 1, 0] * k_scale_fact

    f_total = np.sqrt(df_c_0**2 + df_c_1**2)
    k_total = np.sqrt(dk_ca_0**2 + dk_ca_1**2)

    f_sigma = 100*f_total
    k_sigma = k_total

    print("f_val stats (min, mean, max):", np.nanmin(f_val), np.nanmean(f_val), np.nanmax(f_val))
    print("f_sigma stats (min, mean, max):", np.nanmin(f_sigma), np.nanmean(f_sigma), np.nanmax(f_sigma))
    print("k_val stats (min, mean, max):", np.nanmin(k_val), np.nanmean(k_val), np.nanmax(k_val))
    print("k_sigma stats (min, mean, max):", np.nanmin(k_sigma), np.nanmean(k_sigma), np.nanmax(k_sigma))
    


    sli = 0
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    fig.suptitle(f'{solute_name} params and uncertainty, with points: {points}', fontsize=16)
    explore_amide_uncertainty = explore_amide_uncertainty # TODO: probably remove variable

    map = f_val[:,sli,:] - 2*f_sigma[:,sli,:]
    img = axes[0, 0].imshow(map, vmin=0, vmax=30 if not explore_amide_uncertainty else 3, cmap='viridis');
    cbar = au.cbarhist(img, map,axes[0, 0], pad=0.2)
    cbar.set_label(r'$\hat{f}_{ss} - 2\hat{\sigma}_f$  (%)'.replace('ss', 's' if explore_amide_uncertainty else 'ss'), fontsize=14)
    cbar.ax.yaxis.set_ticks_position('left')

    map = f_val[:,sli,:]
    img = axes[0, 1].imshow(map, vmin=0, vmax=30 if not explore_amide_uncertainty else 3, cmap='viridis');
    cbar = au.cbarhist(img, map,axes[0, 1], pad=0.2)
    cbar.set_label(r'$\hat{f}_{ss}$      (%)'.replace('ss', 's' if explore_amide_uncertainty else 'ss'), fontsize=14)
    cbar.ax.yaxis.set_ticks_position('left')

    map = f_val[:,sli,:] + 2*f_sigma[:,sli,:]
    img = axes[0, 2].imshow(map, vmin=0, vmax=30 if not explore_amide_uncertainty else 3, cmap='viridis');
    cbar = au.cbarhist(img, map, axes[0, 2], pad=0.2)
    cbar.set_label(r'$\hat{f}_{ss} + 2\hat{\sigma}_f$  (%)'.replace('ss', 's' if explore_amide_uncertainty else 'ss'), fontsize=14)
    #cbar.set_label(r'f$_{ss}$ (%) :  $\mu+2\sigma$', fontsize=14)
    cbar.ax.yaxis.set_ticks_position('left')

    map = k_val[:,sli,:] - 2*k_sigma[:,sli,:]
    img = axes[1, 0].imshow(map, vmin=0, vmax=60 if not explore_amide_uncertainty else 100, cmap='magma');
    cbar = au.cbarhist(img, map,axes[1, 0], pad=0.2)
    cbar.set_label(r'$\hat{k}_{ssw}$ - 2$\hat{\sigma}_k$   (s$^{-1}$)'.replace('ss', 's' if explore_amide_uncertainty else 'ss'), fontsize=14)
    cbar.ax.yaxis.set_ticks_position('left')

    map = k_val[:,sli,:]
    img = axes[1, 1].imshow(map, vmin=0, vmax=60 if not explore_amide_uncertainty else 100, cmap='magma');
    cbar = au.cbarhist(img, map,axes[1, 1], pad=0.2)
    cbar.set_label(r'$\hat{k}_{ssw}$     (s$^{-1}$)'.replace('ss', 's' if explore_amide_uncertainty else 'ss'), fontsize=14)
    cbar.ax.yaxis.set_ticks_position('left')

    map = k_val[:,sli,:] + 2*k_sigma[:,sli,:]
    img = axes[1, 2].imshow(map, vmin=0, vmax=60 if not explore_amide_uncertainty else 100, cmap='magma');
    cbar = au.cbarhist(img, map, axes[1, 2], pad=0.2)
    cbar.set_label(r'$\hat{k}_{ssw}$ + 2$\hat{\sigma}_k$   (s$^{-1}$)'.replace('ss', 's' if explore_amide_uncertainty else 'ss'), fontsize=14)
    cbar.ax.yaxis.set_ticks_position('left')

    for ax in axes.flatten():
        utils.remove_spines(ax)


    for point in points:
        circle = patches.Circle((point[1], point[0]), 1, facecolor='none', edgecolor='red', linewidth=2)
        axes[0, 1].add_patch(circle)
        #axes[0, 1].set_ylim(map.shape[0], 0)
    
    plt.show()
    
    return u, s, f_val, k_val, df_c_0, dk_ca_0, df_c_1, dk_ca_1

def create_uncertainty_maps(data_feed_mt, mt_tissue_param_est, u, s, f_val, k_val, df_c_0, dk_ca_0, df_c_1, dk_ca_1, points, mt_params_path, rnoe_params_path, data_feed_amide=None, amide=False, explore_amide_uncertainty=False):
    _cov = u @ s @ np.transpose(u, [0,1,2,4,3])
    _z = 0

    for jj, (_x, _y) in enumerate(points): 
        f_best, k_best, nrmse, _df, _dk = au.plot_empirical_nrmse_blob(
            None, _x, _y, _z,
            data_feed_mt, mt_tissue_param_est,
            data_feed_amide=None, amide=False,
            mt_sim_mode='expm_bmmat', do_plot=False,
            mt_seq_txt_fname=mt_params_path, larg_seq_txt_fname=rnoe_params_path
        )
        maha1, maha2, posterior_cov, CR_area, CIk_x_CIf = au.viz_posteriors(f_val, k_val, _x, _y, _z,_cov, df_c_0, dk_ca_0, df_c_1, dk_ca_1,f_best, k_best, nrmse, _df, _dk,
            explore_amide_uncertainty=explore_amide_uncertainty, fontsize=14, # figsize=(6, 4),  # good w.o. marginals
            do_marginals=True, figsize=[7, 5], show_text=False, show_NN=True)
        plt.title(f'Point: ({_x},{_y}). CR_area: {CR_area:.2f}', loc='center', pad=15)  # Increase the pad value for more spacing

        plt.show
    

def main():
    data.B0_base_DEF = 7
    simulation.num_flip_pulses = 1
    simulation.flip_angle = 90 * jnp.pi / 180

    simulation.tpulse_DEF = 2.5
    simulation.tdelay_DEF = 0.1
    simulation.n_pulses_DEF = 1
    simulation.DO_SL = False

    inpt = Inputs(Path('/home/ron/pediatric-tumor-mice/Pediatric tumor model_Nov2024/20241120_134917_OrPerlman_ped_tumor_immuno_C3_2R_5_1_3'), "C3_2R_2024-11-20")
    data_xa = inpt.dataset
    data.SlicesFeed.norm_type = 'l2'
    data_xa_cutout = cutout_dataset(data_xa, cutout_height=slice(18, 42), cutout_width=slice(17, 50))
    data_feed_mt = data.SlicesFeed.from_xarray(data_xa_cutout, mt_or_amide='mt',mt_seq_txt_fname=inpt.mt_params_path, larg_seq_txt_fname=inpt.rnoe_params_path)
    data_feed_amide = data.SlicesFeed.from_xarray(data_xa_cutout, mt_or_amide='amide',mt_seq_txt_fname=inpt.mt_params_path, larg_seq_txt_fname=inpt.rnoe_params_path)

    predictor_mt, predictor_amide = train_pipeline(data_feed_mt, data_feed_amide)
    # mt_tissue_params_est, mt_pred_signal_normed_np = inference_pipeline(data_feed_mt, predictor_mt, pool2predict='c')
    amide_tissue_params_est, amide_pred_signal_normed_np = inference_pipeline(data_feed_amide, predictor_amide, pool2predict='b')
    # plot_tissue_params_and_error(data_feed_mt, mt_pred_signal_normed_np, mt_tissue_params_est, 'MT')
    # plot_tissue_params_and_error(data_feed_amide, amide_pred_signal_normed_np, amide_tissue_params_est, 'rNOE')

    points = [[6, 6], [10, 8], [5, 25], [10, 25]]
    # mt_u, mt_s, mt_f_val, mt_k_val, mt_df_c_0, mt_dk_ca_0, mt_df_c_1, mt_dk_ca_1 = create_bound_maps(data_feed_mt, mt_tissue_params_est, points, "MT" ,explore_amide_uncertainty=False)
    amide_u, amide_s, amide_f_val, amide_k_val, amide_df_c_0, amide_dk_ca_0, amide_df_c_1, amide_dk_ca_1 = create_bound_maps(data_feed_amide, amide_tissue_params_est, points, "rNOE", explore_amide_uncertainty=True)
    # create_uncertainty_maps(data_feed_mt, mt_tissue_params_est, mt_u, mt_s, mt_f_val, mt_k_val, mt_df_c_0, mt_dk_ca_0, mt_df_c_1, mt_dk_ca_1, points, inpt.mt_params_path, inpt.rnoe_params_path, explore_amide_uncertainty=False)
    # create_uncertainty_maps(data_feed_amide, amide_tissue_params_est, amide_u, amide_s, amide_f_val, amide_k_val, amide_df_c_0, amide_dk_ca_0, amide_df_c_1, amide_dk_ca_1, points, inpt.mt_params_path, inpt.rnoe_params_path, data_feed_amide=data_feed_amide, amide=True, explore_amide_uncertainty=True)


if __name__ == "__main__":
    main()
