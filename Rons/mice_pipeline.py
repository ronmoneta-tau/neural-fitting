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
from config import SimulationConfig, DataConfig, TrainingConfig, ROIConfig

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

def train_pipeline(mt_data, solute_data):
    training_config = TrainingConfig()
    training_config.apply(mt_data, solute_data)

    predictor_mt, predictor_amide = pipelines.run_train(
        brain2train_mt=mt_data.data_feed,
        brain2train_amide=solute_data.data_feed,
        ckpt_folder=os.path.abspath(f'./goo'),
        #do_amide=False,
        mt_sim_mode = 'expm_bmmat'
    )

    return predictor_mt, predictor_amide

def inference_pipeline(solute_data:DataConfig):
    solute_data.data_feed.ds = 1
    tissue_param_est, pred_signal_normed_np = infer.infer(solute_data.data_feed, nn_predictor=solute_data.predictor,do_forward=True, pool2predict=solute_data.pool)

    solute_data.f_values = tissue_param_est.get(f'f{solute_data.pool}_T', None)
    solute_data.k_values = tissue_param_est.get(f'k{solute_data.pool}_T', None)

    return tissue_param_est, pred_signal_normed_np

def plot_tissue_params_and_error(solute_data:DataConfig):
    plt.figure(figsize=(7, 3))
    plt.suptitle(f'{solute_data.name} params')
    plt.subplot(1,3,1)
    plt.imshow(solute_data.f_values.squeeze() * 100)
    plt.colorbar()
    plt.title(f'{solute_data.name} fs values')

    plt.subplot(1,3,2)
    plt.imshow(solute_data.k_values.squeeze(), cmap='magma')
    plt.colorbar()
    plt.title(f'{solute_data.name} ks values')

    plt.subplot(1,3,3)

    err, norm = calculate_error_map(solute_data.data_feed, solute_data.pred_signal_normed_np)

    cmap_nrmse = plt.cm.get_cmap("YlOrRd").copy()
    cmap_nrmse.set_bad('1.0')
    img0 = plt.imshow(err[:,0,:], norm=norm, cmap=cmap_nrmse) #'hot_r'); # plt.colorbar(img0)  # vmin=0, vmax=0.1,
    cbar = plt.colorbar(img0)

    plt.savefig(f'./{solute_data.name}_error_map.png', dpi=200)
    plt.show()

def calculate_error_map(data_feed, pred_signal_normed_np):
    # Visualize the error maps in a slightly more informative way
    err = np.linalg.norm(pred_signal_normed_np - data_feed.measured_normed_T, axis=0) / np.linalg.norm(data_feed.measured_normed_T, axis=0)
    log_bins = np.logspace(np.log10(1.), np.log10(20), num=12)/100  # logarithmic
    norm = mcolors.BoundaryNorm(log_bins, ncolors=plt.get_cmap('hot_r').N, clip=True)

    return err, norm

def create_bound_maps(solute_data, roi_config, sli=0):
    _, _, cov_nnpred_scaled, f_sigma, k_sigma, height, width, angle = \
    au.get_post_estimates(solute_data.tissue_param_est, solute_data.data_feed.shape, is_amide= (solute_data.name != 'MT'))

    axes = au.plot_CI_maps(solute_data.f_values[:,sli,:] * 100, f_sigma[:,sli,:], solute_data.k_values[:,sli,:], k_sigma[:,sli,:], is_mt=(solute_data.name == 'MT'))

    labels = [0]*len(roi_config.points_contralateral) + [1]*len(roi_config.points_tumor)
    for jj, point in enumerate(roi_config.points):
        circle = patches.Circle((point[1], point[0]), 1, facecolor='none', edgecolor=['cyan','red'][labels[jj]], linewidth=2)
        axes[0, 1].add_patch(circle)


    print("f_val stats (min, mean, max):", np.nanmin(solute_data.f_values), np.nanmean(solute_data.f_values), np.nanmax(solute_data.f_values))
    print("f_sigma stats (min, mean, max):", np.nanmin(f_sigma), np.nanmean(f_sigma), np.nanmax(f_sigma))
    print("k_val stats (min, mean, max):", np.nanmin(solute_data.k_values), np.nanmean(solute_data.k_values), np.nanmax(solute_data.k_values))
    print("k_sigma stats (min, mean, max):", np.nanmin(k_sigma), np.nanmean(k_sigma), np.nanmax(k_sigma))

    plt.savefig(f'./{solute_data.name}_CI_maps.png', dpi=200)
    plt.show()
    solute_data.height, solute_data.width, solute_data.angle, solute_data.labels, solute_data.cov_nnpred_scaled = height, width, angle, labels, cov_nnpred_scaled

def create_ROIs_uncertainty_maps(solute_data, roi_config, sli = 0) -> None:
    ds = 1
    is_mt = (solute_data.name == 'MT')

    fig, ax = plt.subplots(figsize=(7, 3))
    plt.xlim(*solute_data.f_lims)
    plt.ylim(*solute_data.k_lims)
    plt.xlabel(r'$\hat{f}_{ss}$ (%)' if is_mt else r'$\hat{f}_{s}$ (%)')
    plt.ylabel(r'$\hat{k}_{ss}\ (s^{-1})$' if is_mt else r'$\hat{k}_{s}\ (s^{-1})$')

    for mu_f, mu_k, ew, eh, eangle, label in zip(
        solute_data.f_values[[p[0] for p in roi_config.points], sli, [p[1] for p in roi_config.points]].flatten()[::ds],
        solute_data.k_values[[p[0] for p in roi_config.points], sli, [p[1] for p in roi_config.points]].flatten()[::ds],
        solute_data.width[[p[0] for p in roi_config.points], sli, [p[1] for p in roi_config.points]].flatten()[::ds],
        solute_data.height[[p[0] for p in roi_config.points], sli, [p[1] for p in roi_config.points]].flatten()[::ds],
        solute_data.angle[[p[0] for p in roi_config.points], sli, [p[1] for p in roi_config.points]].flatten()[::ds],
        solute_data.labels
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

    plt.title(f'{solute_data.name} Uncertainty ellipses for tumor and contralateral points')
    plt.savefig(f'./{solute_data.name}_regional_uncertainty_ellipses.png', dpi=200)
    plt.show()


def create_uncertainty_maps(solute_data, roi_config, inpt, auxiliary_mt_data = None, sli=0):
    _z = 0

    for jj, (_x, _y) in enumerate(roi_config.points):
        if auxiliary_mt_data is not None:
            f_best_dotprod, k_best_dotprod, nrmse, _df, _dk = au.get_nrmse_grid(
                None, _x, _y, _z,
                auxiliary_mt_data.data_feed, auxiliary_mt_data.tissue_param_est,
                data_feed_amide=solute_data.data_feed, amide=True,
                mt_sim_mode='expm_bmmat', do_plot=False, mt_seq_txt_fname=inpt.mt_params_path, larg_seq_txt_fname=inpt.rnoe_params_path
            )
        else:
            f_best_dotprod, k_best_dotprod, nrmse, _df, _dk = au.get_nrmse_grid(
                None, _x, _y, _z,
                solute_data.data_feed, solute_data.tissue_param_est,
                data_feed_amide=None, amide=False,mt_sim_mode='expm_bmmat', do_plot=False,
                mt_seq_txt_fname=inpt.mt_params_path, larg_seq_txt_fname=inpt.rnoe_params_path
            )

        maha1, maha2, posterior_cov, CR_area, CIk_x_CIf = au.viz_posteriors(
            solute_data.f_values[_x, sli, _y], solute_data.k_values[_x, sli, _y], solute_data.cov_nnpred_scaled[_x, sli, _y],
            solute_data.width[_x, sli, _y], solute_data.height[_x, sli, _y], solute_data.angle[_x, sli, _y],
            f_best_dotprod, k_best_dotprod, nrmse, _df, _dk,
            is_amide= (solute_data.name != 'MT'), fontsize=14, # figsize=(6, 4),  # good w.o. marginals
            do_marginals=True, figsize=[7, 5], show_text=False, show_NN=True
            )
        plt.title(f'Point: ({_x},{_y}). CR_area: {CR_area:.2f}',
              loc='center',
              pad=15)  # Increase the pad value for more spacing

        plt.suptitle(f'{solute_data.name} posterior distribution', fontsize=16)
        # plt.tight_layout()
        plt.savefig(f'./{solute_data.name}_posterior_pics_ind{jj}_{_x}_{_z}_{_y}.png', dpi=200)
        plt.show()


def main():
    simulation_config = SimulationConfig()
    simulation_config.apply()
    inpt = Inputs(Path('/home/ron/pediatric-tumor-mice/Pediatric tumor model_Nov2024/20241120_134917_OrPerlman_ped_tumor_immuno_C3_2R_5_1_3'), "C3_2R_2024-11-20")
    data_xa_cutout = cutout_dataset(inpt.dataset, cutout_height=slice(18, 42), cutout_width=slice(17, 50))

    mt_data = DataConfig('MT', data_xa_cutout, inpt)
    rNOE_data = DataConfig('rNOE', data_xa_cutout, inpt)

    mt_data.apply_data_config(rNOE_data)

    mt_data.predictor, rNOE_data.predictor = train_pipeline(mt_data, rNOE_data)

    mt_data.tissue_param_est, mt_data.pred_signal_normed_np = inference_pipeline(mt_data)
    rNOE_data.update_ground_truth(mt_data.tissue_param_est)
    rNOE_data.tissue_param_est, rNOE_data.pred_signal_normed_np = inference_pipeline(rNOE_data)

    plot_tissue_params_and_error(mt_data)
    plot_tissue_params_and_error(rNOE_data)

    roi_config = ROIConfig()

    create_bound_maps(mt_data, roi_config)
    create_bound_maps(rNOE_data, roi_config)

    create_ROIs_uncertainty_maps(mt_data, roi_config)
    create_ROIs_uncertainty_maps(rNOE_data, roi_config)

    create_uncertainty_maps(mt_data, roi_config, inpt, auxiliary_mt_data=None)
    create_uncertainty_maps(rNOE_data, roi_config, inpt, auxiliary_mt_data=mt_data)


if __name__ == "__main__":
    main()
