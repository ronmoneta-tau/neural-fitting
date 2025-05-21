import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
import pandas as pd
from matplotlib.patches import Ellipse, Patch
import net
from Rons.Experiment import Experiment
from micedata import MiceData
from pathlib import Path
import matplotlib.colors as mcolors
from xarray import Dataset
from Parsers import parse_arguments, parse_input_file

sys.path.append(str(os.getcwd()))
from config import SimulationConfig, DataConfig, TrainingConfig, ROIConfig
import data, pipelines, utils, infer
import analyze_uncertainty as au


def cutout_dataset(dataset: Dataset, cutout_height: slice, cutout_width: slice) -> Dataset:
    """
    Cut out a specific region from the dataset.
    Args:
        dataset (Dataset): The input dataset.
        cutout_height (slice): The slice object for the height dimension.
        cutout_width (slice): The slice object for the width dimension.
    Returns:
        Dataset: The cutout dataset.
    """
    data_xa_cutout = dataset.isel(height=cutout_height, width=cutout_width)

    plt.figure(figsize=(4, 4))
    plt.imshow((data_xa_cutout['T1ms'].data * data_xa_cutout['roi_mask_nans']).squeeze())
    plt.title('T1ms')
    plt.colorbar()
    plt.show()

    return data_xa_cutout


def train_pipeline(mt_data: DataConfig, solute_data: DataConfig, checkpoints_path: Path) -> tuple:
    """
    Train the model using the given data.
    Args:
        mt_data (DataConfig): The MT data configuration.
        solute_data (DataConfig): The solute data configuration.
        checkpoints_path (Path): The output directory for saving the model.
    Returns:
        tuple: The trained predictors for MT and solute data.    
    """
    training_config = TrainingConfig()
    training_config.apply(mt_data.data_feed, solute_data.data_feed)

    predictor_mt, predictor_amide = pipelines.run_train(
        brain2train_mt=mt_data.data_feed,
        brain2train_amide=solute_data.data_feed,
        ckpt_folder=checkpoints_path.absolute(),
        # do_amide=False,
        mt_sim_mode='expm_bmmat',
        solute_name = solute_data.name
    )

    return predictor_mt, predictor_amide


def inference_pipeline(solute_data: DataConfig) -> tuple:
    """
    Perform inference using the trained model.
    Args:
        solute_data (DataConfig): The solute data configuration.
    Returns:
        tuple: The tissue parameter estimates and the predicted normalized signal.
    """
    solute_data.data_feed.ds = 1
    tissue_param_est, pred_signal_normed_np = infer.infer(solute_data.data_feed, nn_predictor=solute_data.predictor,
                                                          do_forward=True, pool2predict=solute_data.pool)

    solute_data.f_values = tissue_param_est.get(f'f{solute_data.pool}_T', None)
    solute_data.k_values = tissue_param_est.get(f'k{solute_data.pool}_T', None)

    return tissue_param_est, pred_signal_normed_np


def plot_tissue_params_and_error(solute_data: DataConfig) -> None:
    """
    Plot the tissue parameters and error maps.
    Args:
        solute_data (DataConfig): The solute data configuration.
    """
    plt.figure(figsize=(7, 3))
    plt.suptitle(f'{solute_data.name} params')
    plt.subplot(1, 3, 1)
    plt.imshow(solute_data.f_values.squeeze() * 100)
    plt.colorbar()
    plt.title(f'{solute_data.name} fs values')

    plt.subplot(1, 3, 2)
    plt.imshow(solute_data.k_values.squeeze(), cmap='magma')
    plt.colorbar()
    plt.title(f'{solute_data.name} ks values')

    plt.subplot(1, 3, 3)

    err, norm = calculate_error_map(solute_data.data_feed, solute_data.pred_signal_normed_np)

    cmap_nrmse = plt.cm.get_cmap("YlOrRd").copy()
    cmap_nrmse.set_bad('1.0')
    img0 = plt.imshow(err[:, 0, :], norm=norm, cmap=cmap_nrmse)  # 'hot_r'); # plt.colorbar(img0)  # vmin=0, vmax=0.1,
    cbar = plt.colorbar(img0)

    plt.savefig(f'{solute_data.fig_path}/{solute_data.name}_error_map.png', dpi=200)
    plt.show()
    plt.close()

    utils.fancy_histogram(err, title=f'{solute_data.name} error histogram')


def calculate_error_map(data_feed: data.SlicesFeed, pred_signal_normed_np: np.ndarray) -> tuple:
    """
    Calculate the error map between the predicted and measured normalized signals.
    Args:
        data_feed (data.SlicesFeed): The data feed containing the measured normalized signal.
        pred_signal_normed_np (np.ndarray): The predicted normalized signal.
    Returns:
        tuple: The error map and the normalization for visualization.
    """
    # Visualize the error maps in a slightly more informative way
    err = np.linalg.norm(pred_signal_normed_np - data_feed.measured_normed_T, axis=0) / np.linalg.norm(
        data_feed.measured_normed_T, axis=0)
    log_bins = np.logspace(np.log10(1.), np.log10(20), num=12) / 100  # logarithmic
    norm = mcolors.BoundaryNorm(log_bins, ncolors=plt.get_cmap('hot_r').N, clip=True)

    return err, norm


def create_bound_maps(solute_data: DataConfig, roi_config: ROIConfig, sli: int = 0) -> None:
    """
    Create bound maps for the given solute data and ROI configuration.  
    Args:
        solute_data (DataConfig): The solute data configuration.
        roi_config (ROIConfig): The ROI configuration.
        sli (int): The slice index to use for the bound maps.
    """
    _, _, cov_nnpred_scaled, f_sigma, k_sigma, height, width, angle = \
        au.get_post_estimates(solute_data.tissue_param_est, solute_data.data_feed.shape,
                              is_amide=(solute_data.name != 'MT'))

    vmax_f = 20 if (solute_data.name == 'MT') else 1.5
    vmax_k = 90 if (solute_data.name == 'MT') else 350
    fig, axes = au.plot_CI_maps(solute_data.f_values[:, sli, :] * 100, f_sigma[:, sli, :],
                                solute_data.k_values[:, sli, :], k_sigma[:, sli, :], is_mt=(solute_data.name == 'MT'),
                                vmin_f=0, vmax_f=vmax_f, vmin_k=0, vmax_k=vmax_k)

    labels = [0] * len(roi_config.points_contralateral) + [1] * len(roi_config.points_tumor)
    for jj, point in enumerate(roi_config.points):
        circle = patches.Circle((point[1], point[0]), 1, facecolor='none', edgecolor=['cyan', 'red'][labels[jj]],
                                linewidth=2)
        axes[0, 1].add_patch(circle)

    print(f"{solute_data.name} f and k mean and std:")
    print("f_val stats (min, mean, max):", np.nanmin(solute_data.f_values) * 100,
          np.nanmean(solute_data.f_values) * 100, np.nanmax(solute_data.f_values) * 100)
    print("f_sigma stats (min, mean, max):", np.nanmin(f_sigma), np.nanmean(f_sigma), np.nanmax(f_sigma))
    print("k_val stats (min, mean, max):", np.nanmin(solute_data.k_values), np.nanmean(solute_data.k_values),
          np.nanmax(solute_data.k_values))
    print("k_sigma stats (min, mean, max):", np.nanmin(k_sigma), np.nanmean(k_sigma), np.nanmax(k_sigma))
    print("\n")

    fig.suptitle(f'{solute_data.name} CI maps', fontsize=20)
    plt.savefig(f'{solute_data.fig_path}/{solute_data.name}_CI_maps.png', dpi=200)
    plt.tight_layout()
    plt.show()
    solute_data.height, solute_data.width, solute_data.angle, solute_data.labels, solute_data.cov_nnpred_scaled = height, width, angle, labels, cov_nnpred_scaled


def create_ROIs_uncertainty_maps(solute_data: DataConfig, roi_config: ROIConfig, sli: int = 0) -> None:
    """
    Create uncertainty maps for the regions of interest (ROIs) in the given solute data.
    Args:
        solute_data (DataConfig): The solute data configuration.
        roi_config (ROIConfig): The ROI configuration.
        sli (int): The slice index to use for the uncertainty maps.
    """
    ds = 1
    is_mt = (solute_data.name == 'MT')

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.xlabel(r'$\hat{f}_{ss}$ (%)' if is_mt else r'$\hat{f}_{s}$ (%)')
    plt.ylabel(r'$\hat{k}_{ss}\ (s^{-1})$' if is_mt else r'$\hat{k}_{s}\ (s^{-1})$')

    ellipses = []

    for mu_f, mu_k, ew, eh, eangle, label in zip(
            solute_data.f_values[[p[0] for p in roi_config.points], sli, [p[1] for p in roi_config.points]].flatten()[
            ::ds],
            solute_data.k_values[[p[0] for p in roi_config.points], sli, [p[1] for p in roi_config.points]].flatten()[
            ::ds],
            solute_data.width[[p[0] for p in roi_config.points], sli, [p[1] for p in roi_config.points]].flatten()[
            ::ds],
            solute_data.height[[p[0] for p in roi_config.points], sli, [p[1] for p in roi_config.points]].flatten()[
            ::ds],
            solute_data.angle[[p[0] for p in roi_config.points], sli, [p[1] for p in roi_config.points]].flatten()[
            ::ds],
            solute_data.labels
    ):
        # (!) angle=0 is vertical (hence "height"), but atan(y,x)=atan(y/x) is w horizontal
        ellipse = Ellipse(
            xy=(mu_f * 100, mu_k), width=ew, height=eh, angle=eangle - 90, edgecolor=['c', 'r'][label],
            facecolor='none', linewidth=.5, zorder=0, alpha=0.5
        )
        ax.add_patch(ellipse)
        ellipses.append(ellipse)

        ellipse = Ellipse(
            xy=(mu_f * 100, mu_k), width=ew / 2, height=eh / 2, angle=eangle - 90, edgecolor='none',
            facecolor=['c', 'r'][label], alpha=0.2
        )
        ax.add_patch(ellipse)
        ellipses.append(ellipse)

        # Create legend handles for the two categories
        contralateral_patch = Patch(facecolor='cyan', alpha=0.2, edgecolor='cyan', label='Contralateral')
        tumor_patch = Patch(facecolor='red', alpha=0.2, edgecolor='red', label='Tumor')

        # Add the legend to the plot
        ax.legend(handles=[contralateral_patch, tumor_patch], loc='upper right')

    # compute and apply limits
    (xlim, ylim) = compute_limits_from_corners(ellipses, margin=0.1)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.title(f'{solute_data.name} Uncertainty ellipses for tumor and contralateral points')
    plt.tight_layout()
    plt.savefig(f'{solute_data.fig_path}/{solute_data.name}_regional_uncertainty_ellipses.png', dpi=200)
    plt.show()


def compute_limits_from_corners(ellipses: list, margin: float = 0.05) -> tuple:
    """
    Compute the limits for the plot based on the corners of the ellipses.
    Args:
        ellipses (list): List of Ellipse objects.
        margin (float): Margin to add to the limits.
    Returns:
        tuple: The x and y limits for the plot.
    """
    # Stack all 4 corners per ellipse into one big array
    all_corners = np.vstack([e.get_corners() for e in ellipses])
    xmin, ymin = all_corners.min(axis=0)
    xmax, ymax = all_corners.max(axis=0)

    # pad by a fraction of the span
    dx, dy = xmax - xmin, ymax - ymin
    xmin, xmax = xmin - dx * margin, xmax + dx * margin
    ymin, ymax = ymin - dy * margin, ymax + dy * margin

    return (xmin, xmax), (ymin, ymax)


def create_uncertainty_maps(solute_data: DataConfig, roi_config: ROIConfig, mice_data: MiceData,
                            auxiliary_mt_data: DataConfig = None, sli: int = 0) -> None:
    """
    Create uncertainty maps for the given solute data and ROI configuration.
    Args:
        solute_data (DataConfig): The solute data configuration. can be MT or Solute.
        roi_config (ROIConfig): The ROI configuration.
        mice_data (MiceData): The Mice data configuration.
        auxiliary_mt_data (DataConfig, optional): The auxiliary MT data configuration.Needed only for Solute usage. otherwise defaults to None.
        sli (int, optional): The slice index to use for the uncertainty maps. Defaults to 0.
    """
    _z = 0
    maps_path = Path(f'{solute_data.fig_path}/uncertainty_maps')
    maps_path.mkdir(parents=True, exist_ok=True)
    for jj, (_x, _y) in enumerate(roi_config.points):
        if auxiliary_mt_data is not None:
            f_best_dotprod, k_best_dotprod, nrmse, _df, _dk = au.get_nrmse_grid(
                None, _x, _y, _z,
                auxiliary_mt_data.data_feed, auxiliary_mt_data.tissue_param_est,
                data_feed_amide=solute_data.data_feed, amide=True,
                loc_dict_res=100, max_f_mt=.2, max_k_mt=140, max_f_amide=0.01, max_k_amide=400,
                mt_sim_mode='expm_bmmat', do_plot=False, mt_seq_txt_fname=mice_data.mt_params_path,
                larg_seq_txt_fname=mice_data.amide_params_path
            )
        else:
            f_best_dotprod, k_best_dotprod, nrmse, _df, _dk = au.get_nrmse_grid(
                None, _x, _y, _z,
                solute_data.data_feed, solute_data.tissue_param_est,
                data_feed_amide=None, amide=False, mt_sim_mode='expm_bmmat', do_plot=False,
                loc_dict_res=100, max_f_mt=.2, max_k_mt=140,
                mt_seq_txt_fname=mice_data.mt_params_path, larg_seq_txt_fname=mice_data.amide_params_path
            )

        maha1, maha2, posterior_cov, CR_area, CIk_x_CIf = au.viz_posteriors(
            solute_data.f_values[_x, sli, _y] * 100, solute_data.k_values[_x, sli, _y],
            solute_data.cov_nnpred_scaled[_x, sli, _y],
            solute_data.width[_x, sli, _y], solute_data.height[_x, sli, _y], solute_data.angle[_x, sli, _y],
            f_best_dotprod, k_best_dotprod, nrmse, _df, _dk,
            is_amide=(solute_data.name != 'MT'), fontsize=10,  # figsize=(6, 4),  # good w.o. marginals
            do_marginals=True, figsize=None, show_text=False, show_NN=True,
            loc_dict_res=100
        )

        roi_name = 'tumor' if [_x, _y] in roi_config.points_tumor else 'contralateral'
        plt.suptitle(
            f'{solute_data.name} posterior distribution\n{roi_name} Point: ({_x},{_y}). CR_area: {CR_area:.2f}',
            fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{maps_path}/{solute_data.name}_posterior_pics_ind{jj}_{_x}_{_z}_{_y}.png', dpi=200)
        plt.show()


# def statistical_analysis(experiments_list: [DataConfig], context: str, figs_path: Path) -> None:
#     """
#     Perform statistical analysis on the data for tumor and contralateral ROIs.
#     Args:
#         experiments_list ([DataConfig]): List of experiments.
#         context (str): Pipeline context.
#         figs_path (Path): Path to save the figures.
#     """
#     days_post = [5, 12, 19, 26]
#     stats = process_maps_for_statistics(experiments_list, context, figs_path, days_post)
#
#     control = pd.concat([stats[stats['cage'] == 'C1'], stats[stats['cage'] == 'C3']])
#     treatment = pd.concat([stats[stats['cage'] == 'C2'], stats[stats['cage'] == 'C4']])
#     control.drop('cage', axis='columns', inplace=True)
#     treatment.drop('cage', axis='columns', inplace=True)
#
#     create_boxplot(control, "Control", args.output_dir)
#     create_boxplot(treatment, "Treatment", args.output_dir)
#     create_joined_boxplot((control, "Control"), (treatment, "Treatment"), output_dir=args.output_dir)
#
#
# def process_maps_for_statistics(experiments_list: [DataConfig], context: str, output_dir: Path, days_post: list[int]) -> pd.DataFrame:
#     """
#     Process the inference maps for statistical analysis.
#     Produces a dataframe of rows of day, measurement, ROI, average value
#     :param experiments_list: List of experiments
#     :param context: Pipeline context
#     :param output_dir: Path to the output directory
#     :param days_post: List of days post tumor inoculation
#     :return: Dataframe of data_per_day
#     """
#     data_rows = []
#
#     for exp in experiments_list:
#         scan_day = int(exp.date.date().day)
#         day_post = min(days_post, key=lambda x: abs(x - scan_day))  # finds the closest day
#
#         maps = {
#             'fs': exp.f_values * 100,
#             'ksw': exp.k_values
#         }
#
#         for key, value_map in maps.items():
#
#             masked_tumor = value_map * exp.tumor_mask
#             masked_contralateral = value_map * exp.contralateral_mask
#
#             # Append data to rows (for DataFrame)
#             data_rows.append({
#                 'name': exp.name,
#                 'cage': exp.cage,
#                 'day': day_post,
#                 'measurement': key,
#                 'region': 'tumor',
#                 'value': np.mean(masked_tumor[masked_tumor > 0])
#             })
#             data_rows.append({
#                 'name': exp.name,
#                 'cage': exp.cage,
#                 'day': day_post,
#                 'measurement': key,
#                 'region': 'contralateral',
#                 'value': np.mean(masked_contralateral[masked_contralateral > 0])
#             })
#
#     # Convert to DataFrame
#     df = pd.DataFrame(data_rows)
#
#     return df

def learning_pipeline(args: argparse.Namespace, experiments: list[Experiment]) -> None:
    """
    Based on given flags, run training and/or inference for MT and the wanted solutes.
    """
    for exp in experiments:
        print(f"Running experiment: {exp.name}")
        local_output_dir = args.output_dir/exp.working_path
        local_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving results in {local_output_dir}\n")

        figs_path = Path(f'{local_output_dir}/figures')
        figs_path.mkdir(parents=True, exist_ok=True)

        checkpoints_path = Path(f'{local_output_dir}/checkpoints')
        checkpoints_path.mkdir(parents=True, exist_ok=True)


        data_xa_cutout = cutout_dataset(exp.data.dataset, cutout_height=slice(18, 42), cutout_width=slice(17, 50))
        mt_data = DataConfig('MT', data_xa_cutout, exp.data, figs_path)
        for solute in args.solutes:
            print(f"Running pipeline for {solute}..")
            solute_data = DataConfig(solute, data_xa_cutout, exp.data, figs_path)

            if args.train:
                print(f"Running training..")
                mt_data.predictor, solute_data.predictor = train_pipeline(mt_data, solute_data, checkpoints_path)

            if args.inference:
                print(f"Running inference..")
                if not args.train:
                    mt_data.predictor = net.load_ckpt(folder=checkpoints_path.absolute() / 'MT')[0]
                    solute_data.predictor = net.load_ckpt(folder=checkpoints_path.absolute() / solute)[0]
                    infer.infer_config = pipelines.pipeline_config.infer_config

                mt_data.tissue_param_est, mt_data.pred_signal_normed_np = inference_pipeline(mt_data)
                mt_data.save(checkpoints_path)
                plot_tissue_params_and_error(mt_data)

                solute_data.update_ground_truth(mt_data.tissue_param_est)
                solute_data.tissue_param_est, solute_data.pred_signal_normed_np = inference_pipeline(solute_data)
                solute_data.save(checkpoints_path)
                plot_tissue_params_and_error(solute_data)


def uncertainty_pipeline(args: argparse.Namespace, experiments: list[Experiment]) -> None:
    for exp in experiments:
        print(f"Running experiment: {exp.name}")
        local_output_dir = args.output_dir / exp.working_path
        checkpoints_path = Path(f'{local_output_dir}/checkpoints')
        training_config = TrainingConfig()
        roi_config = ROIConfig()

        mt_data = DataConfig.load(checkpoints_path / f"MT_data_config.pkl")
        training_config.apply(mt_data.data_feed)

        create_bound_maps(mt_data, roi_config)
        create_ROIs_uncertainty_maps(mt_data, roi_config)
        create_uncertainty_maps(mt_data, roi_config, exp.data, auxiliary_mt_data=None)

        for solute in args.solutes:
            solute_data = DataConfig.load(checkpoints_path / f"{solute}_data_config.pkl")
            training_config.apply(solute_data.data_feed)

            create_bound_maps(solute_data, roi_config)
            create_ROIs_uncertainty_maps(solute_data, roi_config)
            create_uncertainty_maps(solute_data, roi_config, exp.data, auxiliary_mt_data=mt_data)


def main():
    simulation_config = SimulationConfig()
    simulation_config.apply()

    args = parse_arguments()
    experiments = [Experiment(path, args.output_dir) for path in parse_input_file(args.input_file)]

    if args.train or args.inference:
        learning_pipeline(args, experiments)
    if args.uncertainty:
        uncertainty_pipeline(args, experiments)


if __name__ == "__main__":
    main()
