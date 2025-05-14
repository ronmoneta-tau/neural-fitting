import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, Patch 
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
    pipelines.pipeline_config.train_config.tp_noise_augmentation_burn_in = 50
    pipelines.pipeline_config.amide_patience = 100
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

    plt.savefig(f'./{solute_name}_error_map.png', dpi=200)
    plt.show()

def calculate_error_map(data_feed, pred_signal_normed_np):
    # Visualize the error maps in a slightly more informative way
    err = np.linalg.norm(pred_signal_normed_np - data_feed.measured_normed_T, axis=0) / np.linalg.norm(data_feed.measured_normed_T, axis=0)
    log_bins = np.logspace(np.log10(1.), np.log10(20), num=12)/100  # logarithmic
    norm = mcolors.BoundaryNorm(log_bins, ncolors=plt.get_cmap('hot_r').N, clip=True)

    return err, norm
    
def create_bound_maps(data_feed, tissue_params_est, solute_name, points_tumor, points_contralateral, sli=0):    

    if solute_name == 'MT':
        f_val = 100*tissue_params_est.get('fc_T', None)
        k_val = tissue_params_est.get('kc_T', None)
        is_amide=False
    elif solute_name == 'rNOE':
        f_val = 100*tissue_params_est.get('fb_T', None)
        k_val = tissue_params_est.get('kb_T', None)
        is_amide=True
    else:
        raise ValueError("Unknown solute name. Expected 'MT' or 'rNOE'.")
    
    _, _, cov_nnpred_scaled, f_sigma, k_sigma, height, width, angle = \
    au.get_post_estimates(tissue_params_est, data_feed.shape, is_amide=is_amide)

    axes = au.plot_CI_maps(f_val[:,sli,:], f_sigma[:,sli,:], k_val[:,sli,:], k_sigma[:,sli,:], is_mt=not is_amide)
    
    points = points_contralateral + points_tumor
    labels = [0]*len(points_contralateral) + [1]*len(points_tumor)
    for jj, point in enumerate(points):
        circle = patches.Circle((point[1], point[0]), 1, facecolor='none', edgecolor=['cyan','red'][labels[jj]], linewidth=2)
        axes[0, 1].add_patch(circle)


    print("f_val stats (min, mean, max):", np.nanmin(f_val), np.nanmean(f_val), np.nanmax(f_val))
    print("f_sigma stats (min, mean, max):", np.nanmin(f_sigma), np.nanmean(f_sigma), np.nanmax(f_sigma))
    print("k_val stats (min, mean, max):", np.nanmin(k_val), np.nanmean(k_val), np.nanmax(k_val))
    print("k_sigma stats (min, mean, max):", np.nanmin(k_sigma), np.nanmean(k_sigma), np.nanmax(k_sigma))
    
    plt.savefig(f'./{solute_name}_CI_maps.png', dpi=200)
    plt.show()
    
    return f_val, k_val, height, width, angle, labels, cov_nnpred_scaled

def create_ROIs_uncertainty_maps(points, f_est, k_est, width, height, angle, labels, soulte_name, sli = 0) -> None:
    ds = 1
    is_mt = True
    fig, ax = plt.subplots(figsize=(7, 3))
    plt.xlim(0, 30)
    plt.ylim(0, 150)
    plt.xlabel(r'$\hat{f}_{ss}$ (%)' if is_mt else r'$\hat{f}_{s}   (%)$')
    plt.ylabel(r'$\hat{k}_{ss}\ (s^{-1})$' if is_mt else r'$\hat{k}_{s}\ (s^{-1})$')
        
    for mu_f, mu_k, ew, eh, eangle, label in zip(
        f_est[[p[0] for p in points], sli, [p[1] for p in points]].flatten()[::ds],
        k_est[[p[0] for p in points], sli, [p[1] for p in points]].flatten()[::ds], 
        width[[p[0] for p in points], sli, [p[1] for p in points]].flatten()[::ds], 
        height[[p[0] for p in points], sli, [p[1] for p in points]].flatten()[::ds], 
        angle[[p[0] for p in points], sli, [p[1] for p in points]].flatten()[::ds],
        labels
    ):
        # (!) angle=0 is vertical (hence "height"), but atan(y,x)=atan(y/x) is w horizontal
        ellipse = Ellipse(
            xy=(mu_f, mu_k), width=ew, height=eh, angle=eangle-90, edgecolor=['c','r'][label], 
            facecolor='none', linewidth=.5, zorder=0, alpha=0.5
        )
        ax.add_patch(ellipse)
        ellipse = Ellipse(
            xy=(mu_f, mu_k), width=ew/2, height=eh/2, angle=eangle-90, edgecolor='none', 
            facecolor=['c','r'][label], alpha=0.2
        )
        ax.add_patch(ellipse)
         # Create legend handles for the two categories
        contralateral_patch = Patch(facecolor='cyan', alpha=0.2, edgecolor='cyan', label='Contralateral')
        tumor_patch = Patch(facecolor='red', alpha=0.2, edgecolor='red', label='Tumor')

        # Add the legend to the plot
        ax.legend(handles=[contralateral_patch, tumor_patch], loc='upper right')
    
    plt.title(f'{soulte_name} Uncertainty ellipses for tumor and contralateral points')
    plt.savefig(f'./{soulte_name}_regional_uncertainty_ellipses.png', dpi=200)
    plt.show()


def create_uncertainty_maps(data_feed_mt, mt_tissue_param_est, f_est, k_est, height, width, angle, labels, cov_nnpred_scaled, points, mt_params_path, rnoe_params_path, solute_name, data_feed_amide=None, amide=False, sli=0):
    _z = 0

    for jj, (_x, _y) in enumerate(points):   
        f_best_dotprod, k_best_dotprod, nrmse, _df, _dk = au.get_nrmse_grid(
            None, _x, _y, _z, 
            data_feed_mt, mt_tissue_param_est,
            data_feed_amide=data_feed_amide, amide=amide,
            mt_sim_mode='expm_bmmat', do_plot=False, mt_seq_txt_fname=mt_params_path, larg_seq_txt_fname=rnoe_params_path
        )          
        maha1, maha2, posterior_cov, CR_area, CIk_x_CIf = au.viz_posteriors(
            f_est[_x, sli, _y], k_est[_x, sli, _y], cov_nnpred_scaled[_x, sli, _y], 
            width[_x, sli, _y], height[_x, sli, _y], angle[_x, sli, _y],
            f_best_dotprod, k_best_dotprod, nrmse, _df, _dk,        
            is_amide= amide, fontsize=14, # figsize=(6, 4),  # good w.o. marginals
            do_marginals=True, figsize=[7, 5], show_text=False, show_NN=True
            )    
        plt.title(f'Point: ({_x},{_y}). CR_area: {CR_area:.2f}', 
              loc='center', 
              pad=15)  # Increase the pad value for more spacing
        
        plt.suptitle(f'{solute_name} posterior distribution', fontsize=16)
        # plt.tight_layout()
        plt.savefig(f'./{solute_name}_posterior_pics_ind{jj}_{_x}_{_z}_{_y}.png', dpi=200)
        plt.show()
    

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
    mt_tissue_param_est, mt_pred_signal_normed_np = inference_pipeline(data_feed_mt, predictor_mt, pool2predict='c')

    data_feed_amide.fc_gt_T = mt_tissue_param_est['fc_T']
    data_feed_amide.kc_gt_T = mt_tissue_param_est['kc_T']
    amide_tissue_param_est, amide_pred_signal_normed_np = inference_pipeline(data_feed_amide, predictor_amide, pool2predict='b')
    plot_tissue_params_and_error(data_feed_mt, mt_pred_signal_normed_np, mt_tissue_param_est, 'MT')
    plot_tissue_params_and_error(data_feed_amide, amide_pred_signal_normed_np, amide_tissue_param_est, 'rNOE')

    points_tumor = [[6, 6],[8, 7],[10, 7], [12, 8]]
    points_contralateral = [[5, 25], [7, 26], [9, 25], [11, 25]]

    mt_f_est, mt_k_est, mt_height, mt_width, mt_angle, mt_labels, mt_cov_nnpred_scaled = create_bound_maps(data_feed_mt, mt_tissue_param_est, "MT" ,points_tumor = [[6, 6],[8, 7],[10, 7], [12, 8]], points_contralateral = [[5, 25], [7, 26], [9, 25], [11, 25]])
    rnoe_f_est, rnoe_k_est, rnoe_height, rnoe_width, rnoe_angle, rnoe_labels, rnoe_cov_nnpred_scaled = create_bound_maps(data_feed_amide, amide_tissue_param_est, "rNOE", points_tumor = [[6, 6],[8, 7],[10, 7], [12, 8]], points_contralateral = [[5, 25], [7, 26], [9, 25], [11, 25]])
    create_ROIs_uncertainty_maps(points_contralateral + points_tumor, mt_f_est, mt_k_est, mt_width, mt_height, mt_angle, mt_labels, "MT")
    create_ROIs_uncertainty_maps(points_contralateral + points_tumor, rnoe_f_est, rnoe_k_est, rnoe_width, rnoe_height, rnoe_angle, rnoe_labels, "rNOE")

    create_uncertainty_maps(data_feed_mt, mt_tissue_param_est, mt_f_est, mt_k_est, mt_height, mt_width, mt_angle, mt_labels, mt_cov_nnpred_scaled, points_contralateral + points_tumor, inpt.mt_params_path, inpt.rnoe_params_path, "MT", data_feed_amide=None, amide=False)
    create_uncertainty_maps(data_feed_mt, mt_tissue_param_est, rnoe_f_est, rnoe_k_est, rnoe_height, rnoe_width, rnoe_angle, rnoe_labels, rnoe_cov_nnpred_scaled, points_contralateral + points_tumor, inpt.mt_params_path, inpt.rnoe_params_path, "rNOE", data_feed_amide=data_feed_amide, amide=True)


if __name__ == "__main__":
    main()
