import numpy as np, matplotlib.pyplot as plt
import cv2, skimage, scipy
import seaborn as sns
import jax.numpy as jnp
import cmcrameri.cm as cmc


def myplot1(data, bcrop=0, tcrop=0, rotate=False, log_hist=False,
            lims=None, range_perc_crop=2., cmap='magma', units=''):
    data_crop_rot = data.T if rotate else data
    data_flat = data_crop_rot.flatten()
    lims = np.percentile(data_flat[~np.isnan(data_flat)],
                         [range_perc_crop, 100 - range_perc_crop]) if lims is None else lims
    data_crop_rot_clipped = np.clip(data_crop_rot, *lims)
    plt.imshow(data_crop_rot_clipped, vmin=lims[0], vmax=lims[1], cmap=cmap)  # rainbow'); plt.cm.get_cmap("magma")
    plt.gca().set_yticks([])
    plt.gca().set_xticks([])
    plt.colorbar(ax=plt.gca(), location='right', label=units)  # / np.max(data)
    ax_histx = plt.gca().inset_axes([0.91, 0, 0.15, 1])
    ax_histx.margins(0, 0)
    dataf = data.flatten()
    z = ax_histx.hist(data_crop_rot_clipped.flatten(), np.linspace(lims[0], lims[1], 32), orientation='horizontal',
                      alpha=0.5, log=log_hist)
    ax_histx.set_xticks([]);
    ax_histx.set_yticks([]);
    plt.xlim(bcrop, data_crop_rot.shape[1] - tcrop)
    return lims


def viz_seq(seq_df):
    # TODO also axis of ppm (along with Hz)
    B1_uT_seq = seq_df['B1_uT'].tolist()
    dwRF_Hz_seq = seq_df['dwRF_Hz'].tolist()
    plt.figure(figsize=(10, 2))
    plt.plot(B1_uT_seq, 'bo:')
    plt.grid()
    plt.ylabel('B1 (uT)', color='blue')
    plt.gca().tick_params(axis='y', colors='blue')
    plt.gca().twinx().plot(dwRF_Hz_seq, 'gs--')
    plt.ylabel('RF off-resonance (Hz)', color='green')
    plt.gca().tick_params(axis='y', colors='green')


def plot_estimation_scatter(fss_gt_np, kss_gt_np, pred_fss_np, pred_kss_np, roi_mask_zeros,
                            alpha=2e-3, poolname='ss', methodname='NBMF'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2))

    plt.sca(ax1)  # plt.subplot(1,3,1)
    plt.plot((fss_gt_np * roi_mask_zeros).flatten(), (pred_fss_np * roi_mask_zeros).flatten(), '.',
             alpha=alpha)  # [::ds,rs,::ds]
    plt.grid()
    # plt.plot([0, 0.25], [0, 0.25], 'b--'); plt.title('fss (predicted) vs. fss (oracle)')
    plt.plot([0, np.nanmax(fss_gt_np)], [0, np.nanmax(fss_gt_np)], 'b--');

    plt.title(f'$f_{{{poolname}}}$')
    plt.ylabel(f'{methodname}-estimate')
    plt.xlabel('reference (baseline method)')
    plt.xlim(0, np.nanmax(fss_gt_np))
    plt.ylim(0, np.nanmax(fss_gt_np))

    plt.sca(ax2)  # plt.subplot(1,3,2)
    plt.plot((kss_gt_np * roi_mask_zeros).flatten(), (pred_kss_np * roi_mask_zeros).flatten(), '.', alpha=alpha);
    plt.title(f'$k_{{{poolname}}}$')
    plt.ylabel(f'{methodname}-estimate')
    plt.xlabel('reference (baseline method)')
    # plt.plot([20, 70], [20, 70], 'b--')
    plt.grid()
    plt.plot([np.nanmin(kss_gt_np), np.nanmax(kss_gt_np)],
             [np.nanmin(kss_gt_np), np.nanmax(kss_gt_np)], 'b--')
    plt.xlim(np.nanmin(kss_gt_np), np.nanmax(kss_gt_np))
    plt.ylim(np.nanmin(kss_gt_np), np.nanmax(kss_gt_np))
    plt.suptitle(f'{methodname}-estimate vs. MRF-reference')
    return fig


def get_circles_mask(img):
    """ NV old hough-based circling """
    img = img / np.max(img)
    img = img * 128
    img = img.astype(np.uint8)
    dp = 2
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp, 12,
                               param1=250, param2=20, minRadius=0, maxRadius=6)
    # assert circles is not None and len(circles[0]) == 7, f'{len(circles[0])} circles find'
    rois = []
    mask = np.zeros(img.shape, dtype=int)
    for i in circles[0, :]:
        # i = np.array(i, dtype = int)
        i = np.int32(np.round(i))

        roi = np.zeros(img.shape[:2], np.uint8)
        radius = i[2] - 1
        x, y = i[0], i[1]
        roi = cv2.circle(roi, (x, y), radius, 1, cv2.FILLED)
        rois.append((roi, x, y, radius))
        mask += roi

    return mask, rois


def get_corr(gt, pred):
    """ get pearson's correlation across items (e.g. voxels) that are non-NaN
        in both arrays (e.g. predicted and measured)"""
    f1 = gt.flatten()
    f2 = pred.flatten()
    f1_ = f1[~np.isnan(f1) & ~np.isnan(f2)]  # & np.where(f2>0) & np.where(f1>0)]
    f2_ = f2[~np.isnan(f1) & ~np.isnan(f2)]  # & np.where(f2>0) & np.where(f1>0)]
    # print(np.corrcoef(f1_, f2_)[0, 1])
    return scipy.stats.pearsonr(f1_, f2_)[0]


def get_rmse(f1, f2, relative=True):
    """ get root-mean-square error, optionally relative (to RMS of signal)"""
    f1 = f1.flatten();
    f2 = f2.flatten();
    f1_ = f1[~np.isnan(f1) & ~np.isnan(f2) & (f1 != 0) & (f2 != 0)]
    f2_ = f2[~np.isnan(f1) & ~np.isnan(f2) & (f1 != 0) & (f2 != 0)]
    if relative:
        return np.linalg.norm(f1_ - f2_) / np.linalg.norm(f1_)
    else:
        return np.sqrt(np.nanmean((f1_ - f2_) ** 2))


def explore_brain_masks(brainmask, sl=70):
    """ Explore the GM/WM/CSF/.. segmentation mask created by the matlab pipeline"""
    plt.figure(figsize=(20, 3))
    plt.subplot(1, 6, 1)
    # plt.imshow(brainmask['c1_nii'][:, sl, :]); plt.colorbar()
    # brainmask['c1_nii'][:, sl, :][brainmask['c1_nii'][:, sl, :]==0] = np.nan
    myplot1(brainmask['c1_nii'][:, sl, :], )
    plt.subplot(1, 6, 2)
    # plt.imshow(brainmask['c2_nii'][:, sl, :])
    white_and_gray = brainmask['c2_nii'][:, sl, :] + brainmask['c1_nii'][:, sl, :]
    # brainmask['c2_nii'][:, sl, :][brainmask['c2_nii'][:, sl, :]==0] = np.nan
    # white_and_gray[:, :][white_and_gray[:, :]==0] = np.nan
    white_and_gray[:, :][white_and_gray[:, :] < 0.1] = np.nan
    myplot1(white_and_gray)
    plt.subplot(1, 6, 3)
    plt.imshow(brainmask['c3_nii'][:, sl, :])
    plt.subplot(1, 6, 4)
    plt.imshow(brainmask['c4_nii'][:, sl, :])
    plt.subplot(1, 6, 5)
    plt.imshow(brainmask['c5_nii'][:, sl, :])
    plt.subplot(1, 6, 6)
    plt.imshow(brainmask['c6_nii'][:, sl, :])


def phantom_get_geometry(data, centers, sl=5, radius=3, visualize=True):
    _img = np.copy(data[3, :, sl, :])
    """
       TODO non-integer centers and radius!! 
       TODO automate centers+radius choice. 
           - Hough transform not so good, but can try semi-auto, e.g. flood-fill from centers, also can morphologically remove boundary..
           - Use DNN, e.g. SAM! or fine tuned
       TODO consider upsampling, then better circle fit       
    """
    rois_nan = []
    mask = np.zeros(_img.shape)
    for (x, y) in centers:
        roi = np.zeros(_img.shape, np.uint8)
        roi = cv2.circle(roi, (y, x), radius, 1, cv2.FILLED)
        roi_nan = np.copy(np.float32(roi))
        roi_nan[roi_nan == 0] = np.nan
        rois_nan.append(roi_nan)
        mask += roi

    if visualize:
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 3))
        ax1.imshow(_img, cmap='gray')
        ax2.imshow(_img * mask, cmap='gray')
        ax3.imshow(_img * (1 - mask), cmap='gray')
        # plt.imshow()
        plt.sca(ax4)
        for (x, y) in centers:
            sig = data[:, x, sl, y]
            plt.plot(sig / np.linalg.norm(sig), 's-', alpha=0.5)
        plt.legend(range(6))

    roi_mask_nans = np.copy(np.float32(mask))
    roi_mask_nans[roi_mask_nans == 0] = np.nan

    return rois_nan, roi_mask_nans


H2O_mM = 110e3
Larg_spins = 3

cmap_vir = plt.cm.get_cmap("viridis").copy()
cmap_vir.set_bad('black', 1.)
cmap_mgm = plt.cm.get_cmap("magma").copy()
cmap_mgm.set_bad('black', 1.)


def phantom_viz(
        ax, centers, data, radius=3, texts=None, vmax=None,
        cmap=cmap_vir, dx=-5, dy=-2, label='[L-arg] (mM)', do_colorbar=True
):  # , rois=None):
    #  dx=-2.5, dy=-3.5
    # rois = rois or [cv2.circle(np.full_like(data, np.nan), center, radius, 1, -1) for center in centers]  # TODO test
    texts = texts or 'ABCDEF'  # ['A', 'B', 'C', 'D', 'E', 'F']
    image = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax or np.nanmax(data) * 1.2)
    ax.axis('off')
    if do_colorbar:
        cax = ax.inset_axes([1.05, 0, .05, 1])
        cbar = plt.colorbar(image, label=label, cax=cax, ax=ax)
        cbar.set_label(label, fontsize=14)
    for ci, center in enumerate(centers):
        y, x = center
        ax.text(x + dx, y + dy, texts[ci], color='white', fontsize=15, ha='left', va='center')


def swarm_my_violin(data, Larg_mM_GT, axes, swarm=True, fnts=16):
    if swarm:
        sns.violinplot(
            ax=axes[0], x="vial", y="fs", hue="method",
            palette=["m", "g"],
            data=data, inner=None)
        sns.swarmplot(
            ax=axes[0], x="vial", y="fs", hue="method", data=data,
            palette=["gray", "gray"],
            dodge=True, size=3)

    else:
        sns.violinplot(
            ax=axes[0], x="vial", y="fs", hue="method", data=data,
            palette=["m", "g"], split=True, gap=.1, inner='quart', density_norm='width'
        )  # cut=0,

    axes[0].set_ylabel('[L-arg] (mM)', fontsize=fnts)
    axes[0].set_xlabel('Vial', fontsize=fnts)
    axes[0].grid()
    axes[0].set_ylim(0, 120)
    GTmM_line = axes[0].plot(Larg_mM_GT, 'b ', marker='p', markeredgecolor='k', markersize=14)

    # import ipdb; ipdb.set_trace()
    axes[0].legend(
        [axes[0].patches[0], axes[0].patches[1], axes[0].lines[-1]],
        ['Dictionary matching', 'NBMF', 'Ground truth'],
        fontsize=fnts)  # axes=axes[0])
    for text in axes[0].get_legend().get_texts():
        text.set_fontsize(fnts)  # Set font size
        text.set_fontfamily('serif')

    if swarm:
        sns.swarmplot(
            ax=axes[1], x="vial", y="ks", hue="method", palette=["gray", "gray"],
            data=data, dodge=True, size=3
        )
        sns.violinplot(
            ax=axes[1], x="vial", y="ks", hue="method",
            palette=["m", "g"],
            data=data, inner=None)  # , density_norm='width') # 'point')
    else:
        sns.violinplot(
            ax=axes[1], x="vial", y="ks", hue="method", data=data,
            palette=["m", "g"], split=True, gap=.1, inner='quart', density_norm='width'
        )  # cut=0,

    axes[1].set_ylabel('k$_{sw}$ (s$^{-1}$)', fontsize=fnts)  # ('$K_s  (s^{-1})$'
    axes[1].set_xlabel('Vial', fontsize=fnts)
    axes[1].grid()
    axes[1].set_ylim(0, 700)
    plt.legend([axes[1].patches[0], axes[1].patches[1]],
               ['Dictionary\nmatching', 'NBMF'],  # title='reconstruction:', title_fontsize=fnts,
               fontsize=fnts, loc='lower left')
    for text in axes[1].get_legend().get_texts():
        text.set_fontsize(fnts)  # Set font size
        text.set_fontfamily('serif')


def larg_phantom_viz(
        nansmask,
        fspred,
        kspred,
        centers,
        rois_nan,
        sl=5,
        radius=3,
        Larg_mM_GT=[100, 75, 50, 50, 25, 50],
        PH=np.array([5, 5, 5, 4, 5, 4.5]),
        Ksw_Hz_GT_QUESP=None,
        Ksw_Hz_GT_QUESP_std=None,
        vial_labels='ABCDEF',  # ['E','F','G','H','I','J'],
        axes=None,
        figname_sfx='', figsize=(12, 10), dx=-5, dy=-2, do_draw_circle=False
):
    figs = []
    fig = plt.figure(figsize=figsize)
    figs.append(fig)
    ylabel_fontsize = 14
    plt.subplot(221)
    fspred_mM = fspred * H2O_mM / Larg_spins  # H22O concentration, protons per molecule

    cmap_vir = plt.cm.get_cmap("viridis").copy()
    cmap_vir.set_bad('black', 1.)
    plt.imshow(
        (fspred_mM * nansmask)[:, sl, :], cmap=cmap_vir,
        vmin=np.nanmin(fspred_mM * nansmask) * 0.8,
        vmax=np.nanmax(fspred_mM * nansmask) * 1.2
    )
    cax = plt.gca().inset_axes([1.05, 0, .05, 1])
    clabel = '[L-arg] (mM)'
    cbar = plt.colorbar(label=clabel, cax=cax)  # , NBMF reconstruction
    cbar.set_label(clabel, fontsize=ylabel_fontsize)
    fs_pred_mM_by_roi = []
    ks_pred_Hz_by_roi = []
    fs_pred_mM_means = []

    for vi, (roi, center) in enumerate(zip(rois_nan, centers)):
        y, x = center
        mM_data = fspred_mM * nansmask * roi[:, None, :].repeat(nansmask.shape[1], axis=1)
        mu = np.nanmean(mM_data)
        std = np.nanstd(mM_data)
        # print('fs mu', mu);  print('fs center', fspred_mM[y, x])
        _mM_data = mM_data[~np.isnan(mM_data)]
        fs_pred_mM_by_roi.append(mM_data)  # _mM_data
        fs_pred_mM_means.append(mu)
        if False:  # mu +- std:  removing for paper (Oct27)
            plt.text(x, y - 4, f'{mu:.0f}±{2 * std:.0f}', color='white', fontsize=10, ha='center', va='center')
        plt.text(x + dx, y + dy, vial_labels[vi], color='white', fontsize=15, ha='center', va='center')
        do_draw_circle = False
        if do_draw_circle:
            circle = plt.Circle((x, y), radius, color='w', fill=False)
            plt.gca().add_patch(circle)

    plt.axis('off')
    plt.subplot(223)
    cmap_mgm = plt.cm.get_cmap("magma").copy()
    cmap_mgm.set_bad('black', 1.)
    ksvmax = np.nanmax(kspred)  # 800
    if Ksw_Hz_GT_QUESP is not None:
        ksvmax = np.maximum(ksvmax, np.max(Ksw_Hz_GT_QUESP))
    plt.imshow((kspred * nansmask)[:, sl, :], cmap=cmap_mgm, vmin=0, vmax=ksvmax)
    cax = plt.gca().inset_axes([1.05, 0, .05, 1])
    clabel = 'k$_{sw}$ ($s^{-1}$)'
    cbar = plt.colorbar(label=clabel, cax=cax)  # , NBMF reconstruction
    cbar.set_label(clabel, fontsize=ylabel_fontsize)

    kpred_means = []
    kpred_stds = []
    for vi, (roi, center) in enumerate(zip(rois_nan, centers)):
        y, x = center
        ks_data = kspred * nansmask * roi[:, None, :].repeat(nansmask.shape[1], axis=1)
        _ks_data = ks_data[~np.isnan(ks_data)]  # ! auto-flattened
        ks_pred_Hz_by_roi.append(ks_data)  # _ks_data
        mu = np.nanmean(ks_data)
        std = np.nanstd(ks_data)
        kpred_means.append(mu)
        kpred_stds.append(std)
        if False:
            plt.text(x, y - 4, f'{mu:.0f}±{2 * std:.0f}', color='white', fontsize=10, ha='center', va='center')
        plt.text(x + dx, y + dy, vial_labels[vi], color='white', fontsize=15, ha='center', va='center')
    plt.axis('off')

    plt.subplot(222)
    _img = np.zeros(fspred[:, sl, :].shape)
    pred_mM_means = []
    # for (y,x), gt_mM in zip(centers, Larg_mM_GT):
    for vi, ((y, x), gt_mM) in enumerate(zip(centers, Larg_mM_GT)):
        # print('gt_mM', gt_mM)
        _zeros = np.zeros(fspred[:, sl, :].shape, np.uint8)
        _img = _img + np.float32(cv2.circle(_zeros, (x, y), radius, 1., cv2.FILLED)) * gt_mM
        plt.text(x, y, f'{gt_mM:.0f}\nmM', color='k', fontweight='bold', fontsize=9, ha='center', va='center')
        plt.text(x + dx, y + dy, vial_labels[vi], color='white', fontsize=15, ha='center', va='center')
    _img[_img == 0] = np.nan
    plt.imshow(_img, cmap=cmap_vir,
               vmin=np.nanmin(fspred_mM * nansmask) * 0.8,
               vmax=np.nanmax(fspred_mM * nansmask) * 1.2)
    cax = plt.gca().inset_axes([1.05, 0, .05, 1])
    clabel = '[L-arg] (mM)'  # , as prepared
    cbar = plt.colorbar(label=clabel, cax=cax)
    cbar.set_label(clabel, fontsize=ylabel_fontsize)
    plt.axis('off')

    # TODO why not do this too in 3D?
    mM_pred_slice_masked = (fspred_mM * nansmask)[:, sl, :]
    print('VOXELwise statistics (in central slide): ')
    print(f'fs corr: {get_corr(_img, mM_pred_slice_masked) :.2f}')
    print(f'fs relative RMSE (%): {100 * get_rmse(_img, mM_pred_slice_masked) :.1f}')
    print(f'fs RMSE (mM): {np.sqrt(np.nanmean((_img - mM_pred_slice_masked) ** 2)):.1f}')
    # MAPE - relative abs (L1) error voxelwise, then averaged?
    print(f'fs MAPE (%): {100 * (np.nanmean(np.abs((_img - mM_pred_slice_masked) / (_img + 1e-6)))):.1f}')
    print('VIALwise statistics: ')
    Larg_mM_GT = np.array(Larg_mM_GT)
    fs_pred_mM_means = np.array(fs_pred_mM_means)
    print(f'->fs RMSE (mM): {np.sqrt(np.nanmean((Larg_mM_GT - fs_pred_mM_means) ** 2)):.1f}')
    print(f'->fs MAPE (%): {100 * np.nanmean(np.abs(fs_pred_mM_means / Larg_mM_GT - 1)):.1f}')

    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c
        # TUB quest fit gave: 2.13908777 1.02665966 56.0028085
        # note b=1 --> good fit to K as a linear func of [H+]=exp(pH)
        # TODO: fit with b=1 forced?

    plt.subplot(224)
    # phexpfac = 3.1
    _img = np.zeros(fspred[:, sl, :].shape)
    Ksw_GT = Ksw_Hz_GT_QUESP if Ksw_Hz_GT_QUESP is not None else exp_func(PH, 2.1, 1., 56.)

    if Ksw_Hz_GT_QUESP is not None:
        plt.title('GT pH, color: $k_{sw}$ (QUESP)', fontsize=16)
        # for (y,x,r), gt_Hz, gt_Hz_std, gt_ph in zip(rois, Ksw_Hz_GT_QUESP, Ksw_Hz_GT_QUESP_std, PH):
        for vi, ((y, x), gt_ph, gt_Hz, gt_Hz_std) in enumerate(zip(centers, PH, Ksw_Hz_GT_QUESP, Ksw_Hz_GT_QUESP_std)):
            _zeros = np.zeros(fspred[:, sl, :].shape, np.uint8)
            _img = _img + np.float32(cv2.circle(_zeros, (x, y), radius, 1., cv2.FILLED)) * gt_Hz
            if False:
                plt.text(x, y - 5, f'{gt_Hz:.0f}±{gt_Hz_std:.0f}', color=[1, 1, 1], fontsize=10, ha='center',
                         va='center')
            plt.text(x, y, f'pH\n{gt_ph}', color='y', fontweight='bold', fontsize=10, ha='center', va='center')
            plt.text(x + dx, y + dy, vial_labels[vi], color='white', fontsize=15, ha='center', va='center')
    else:
        # plt.title('GT pH, color: $k_{sw}$ (a*exp(pH)+c)', fontsize=16)
        # plt.title('Ground truth (as titrated)', fontsize=16)
        Ksw_GT = exp_func(PH, 2.1, 1., 56.)
        for vi, ((y, x), gt_ph, gt_Hz) in enumerate(zip(centers, PH, Ksw_GT)):
            _zeros = np.zeros(fspred[:, sl, :].shape, np.uint8)
            _img = _img + np.float32(cv2.circle(_zeros, (x, y), radius, 1., cv2.FILLED)) * gt_ph  # gt_Hz
            plt.text(x, y, f'pH\n{gt_ph}', color='k', fontweight='bold', fontsize=10, ha='center', va='center')
            plt.text(x + dx, y + dy, vial_labels[vi], color='white', fontsize=15, ha='center', va='center')
    _img[_img == 0] = np.nan

    # plt.imshow(_img, cmap=cmap_mgm, vmin=0, vmax=ksvmax)
    import matplotlib as mpl
    plt.imshow(_img, cmap=cmap_mgm,
               norm=mpl.colors.BoundaryNorm(np.arange(3.3, 5.9, 0.2),  # [3, 3.75, 4.25, 4.75, 5.25, 6],
                                            ncolors=cmap_mgm.N))

    cax = plt.gca().inset_axes([1.05, 0, .05, 1])
    # plt.colorbar(label='1/sec', cax=cax)
    clabel = 'pH'  # , as titrated
    cbar = plt.colorbar(label=clabel, cax=cax)
    cbar.set_label(clabel, fontsize=ylabel_fontsize)
    plt.axis('off')
    ks_pred_slice_masked = (kspred * nansmask)[:, sl, :]
    print('\nVOXELwise statistics (in central slide): ')
    print(f'ks corr: {get_corr(_img, ks_pred_slice_masked):.2f}')
    print(f'ks RMSE (Hz): {np.sqrt(np.nanmean((_img - ks_pred_slice_masked) ** 2)):.1f}')
    print(f'ks MAPE (%): {100 * (np.nanmean(np.abs((_img - ks_pred_slice_masked) / (_img + 1e-6)))):.1f}')
    print('VIALwise statistics: ')
    Ksw_GT = np.array(Ksw_GT)
    kpred_means = np.array(kpred_means)
    print(f'->ks RMSE (Hz): {np.sqrt(np.nanmean((Ksw_GT - kpred_means) ** 2)):.1f}')
    print(f'->ks MAPE (%): {100 * np.nanmean(np.abs(kpred_means / Ksw_GT - 1)):.1f}')

    plt.savefig(f'figs/phantom_{figname_sfx}.png', bbox_inches='tight')
    # ================================================= #

    # restore matplotlib styling
    plt.style.use('default')

    return figs, fs_pred_mM_by_roi, ks_pred_Hz_by_roi


def upsample(_img, fname=None, factor=3, viz=True):
    """ TODO """
    _img = cv2.resize(_img, dsize=[_img.shape[1] * factor, _img.shape[0] * factor],
                      interpolation=cv2.INTER_CUBIC)  # LANCZOS4)
    if fname is not None:
        plt.imsave(fname, _img, cmap='gray')
    if viz:
        plt.figure(figsize=(10, 3))
        plt.imshow(_img, cmap='gray')


def print_latex_basic_stats_fk_gw(data, tp, mask_th=0.9):
    for tissue_type, mask in zip(['WM', 'GM'], [data.white, data.gray]):
        mask = np.float32(mask > mask_th)
        mask[mask == 0] = np.nan

        string = ''
        tmp = 100 * (mask * tp['fc_T']).flatten();
        tmp = tmp[~np.isnan(tmp)]
        string += f"f_{{ss}}={tmp.mean():.2f}\pm{tmp.std():.2f}\ (\%),\ "

        tmp = (mask * tp['kc_T']).flatten();
        tmp = tmp[~np.isnan(tmp)]
        string += f"k_{{ss}}={tmp.mean():.1f}\pm{tmp.std():.1f}\ (s^{{-1}}),\ "

        tmp = 100 * (mask * tp['fb_T']).flatten()
        tmp = tmp[~np.isnan(tmp)]
        string += f"f_{{s}}={tmp.mean():.2f}\pm{tmp.std():.2f}\ (\%),\ "

        tmp = (mask * tp['kb_T']).flatten();
        tmp = tmp[~np.isnan(tmp)]
        string += f"k_{{s}}={tmp.mean():.1f}\pm{tmp.std():.1f}\ (s^{{-1}})"
        print(tissue_type)
        print(string)


def boxplot_white_vs_gray(gray, white, f, k, pool='CEST',
                          lit_f_wm_gm=[[.19, .24], [.22, .25], [.31, .32], [.1, .17]],
                          lit_k_wm_gm=[[162, 365], [280, 280], [42.3, 35], [260, 130]],
                          lit_names=('Heo 2019', 'Liu 2013', 'Perlman 2022', 'Carradus 2023')):
    avg_in_slice = False
    if avg_in_slice:
        gray_f = np.nanmean((gray * f), axis=(0, 2)).flatten()
        white_f = np.nanmean((white * f), axis=(0, 2)).flatten()
        gray_k = np.nanmean((gray * k), axis=(0, 2)).flatten()
        white_k = np.nanmean((white * k), axis=(0, 2)).flatten()
    else:
        gray_f = (gray * f).flatten()
        white_f = (white * f).flatten()
        gray_k = (gray * k).flatten()
        white_k = (white * k).flatten()

    _gray_f = gray_f[~np.isnan(gray_f)];
    _white_f = white_f[~np.isnan(white_f)]
    _gray_k = gray_k[~np.isnan(gray_f)];
    _white_k = white_k[~np.isnan(white_f)]

    print(f'GM: f={_gray_f.mean():3f}+-{_gray_f.std():3f} , k={_gray_k.mean():3f}+-{_gray_k.std():3f} ')
    print(f'WM: f={_white_f.mean():3f}+-{_white_f.std():3f} , k={_white_k.mean():3f}+-{_white_k.std():3f} ')

    fig1 = plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    # plt.title(f'quantitative {pool} - nbmf vs. literature values\nVOLUME FRACTION (%)', fontsize=12)
    plt.gca().set_xticklabels(('WM', 'GM'), fontsize=15);
    plt.ylabel("$f_{ss}$ (%)" if pool == 'MT' else "$f_{s}$ (%)", fontsize=15);  # plt.ylim(0, 1)
    # styles = [':rp', ':go', ':bs', ':md', ':cx']  # TODO the connecting lines may be confusing
    styles = ['rp', 'go', 'bs', 'md', 'cx']
    # styles = ['r_', 'g_', 'b_', 'm_', 'c_']
    msize = 10
    for vals, style in zip(lit_f_wm_gm, styles):
        plt.plot([1, 2], vals, style, markersize=msize, alpha=0.5)  # , fillstyle='none')
    plt.legend(lit_names, loc='upper center')

    violinplot = False
    boxplot_95 = True
    if violinplot or not boxplot_95:
        import seaborn as sns
        # sns.set_theme(style="ticks", palette="pastel")
    if violinplot:
        aa = sns.violinplot((_white_f * 100, _gray_f * 100), palette=["m", "g"], fill=False)
    elif boxplot_95:
        aa = plt.boxplot((_white_f * 100, _gray_f * 100), whis=[5., 95.], sym="")
    else:
        aa = sns.boxplot((_white_f * 100, _gray_f * 100), flierprops=dict(marker='o', color='gray', alpha=0.5))
    plt.grid(axis='y')  # notch=True)

    print(f"-- concentration: -- \nwhite = {100 * np.mean(_white_f):.2f}+-{100 * np.std(_white_f):.2f}, "
          f"\ngray: {100 * np.mean(_gray_f):.2f}+-{100 * np.std(_gray_f):.2f}")
    # fig2 = plt.figure()
    plt.subplot(2, 1, 2)
    # plt.title(f'quantitative {pool} - nbmf vs. literature values\nEXCHANGE RATE', fontsize=12)
    plt.gca().set_xticklabels(('WM', 'GM'), fontsize=15);
    plt.ylabel("$k_{ssw}$ $(s^{-1})$" if pool == 'MT' else "$k_{sw}$ $(s^{-1})$", fontsize=15);  # plt.ylim(0, 1)
    for vals, style in zip(lit_k_wm_gm, styles):
        plt.plot([1, 2], vals, style, markersize=msize, alpha=0.5)  # , fillstyle='none')
    plt.legend(lit_names, loc='upper center')
    if violinplot:
        aa = sns.violinplot((_white_k, _gray_k))  # , whis=[5., 95.], sym="");
    elif boxplot_95:
        aa = plt.boxplot((_white_k, _gray_k), whis=[5., 95.], sym="");
    else:
        aa = sns.boxplot((_white_k, _gray_k), flierprops=dict(marker='o', color='gray', alpha=0.5))
    plt.grid(axis='y')

    print(f"-- exchange-rate: -- \nwhite = {np.mean(_white_k):.1f}+-{np.std(_white_k):.1f}, "
          f"\ngray: {np.mean(_gray_k):.1f}+-{np.std(_gray_k):.1f}")

    return [fig1]  # , fig2] #, fig3]


cmap_mgm = plt.cm.get_cmap("magma").copy();
cmap_mgm.set_bad('black', 1.)
cmap_vir = plt.cm.get_cmap("viridis").copy();
cmap_vir.set_bad('black', 1.)
# cmap_err = plt.cm.get_cmap("YlOrRd").copy(); cmap_err.set_bad('black', 0.)    # darks: # CMRmap  # seismic # twilight_shifted # bone
cmap_R = plt.cm.get_cmap("YlOrRd_r").copy();
cmap_R.set_bad('1.')


def slice_row_plot(fss_pred, kss_pred, err_3d, slices=None,
                   fss_lims=[0, 25], kss_lims=[0, 80], errlims=[.98, 1.], figsize=None, do_err=True,
                   texts=['semi-solid volume fraction (%)', '$f_{ss}$ (%)',
                          'semi-solid exchange rate (Hz)', '$k_{ssw}$ $(s^{-1})$',
                          'signal reconstruction', '$R^2_{fit}$'  # signal reconstruction RMSE (%)
                          ]):
    # TODO rename from fss/kss to generic
    slices = slices or np.int32(np.linspace(5, 40, 10))[3:-2]  # cutting best/central 5 out of 10 rep.
    figsize = figsize or (25, 10)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(wspace=0, hspace=0)
    cols = 10
    clrbr_dims = [1.05, 0.05, 0.1, .9]
    fss_pred = np.clip(fss_pred, *fss_lims)
    kss_pred = np.clip(kss_pred, *kss_lims)
    rows = 3 if do_err else 2
    for slind, slice2plot in enumerate(slices):
        ax = plt.subplot(rows, cols, 1 + slind)
        plt.gca().margins(0, 0);
        plt.gca().set_xticklabels([]);
        plt.gca().set_yticklabels([])
        imshow_res = plt.imshow(fss_pred[:, slice2plot, :], vmin=fss_lims[0], vmax=fss_lims[1], cmap=cmap_vir)
        if slice2plot == slices[-1]:
            cax = ax.inset_axes(clrbr_dims);
            fig.colorbar(imshow_res, cax=cax, orientation='vertical')  # , label=texts[0])
            ax.text(1.4, .3, texts[1], transform=ax.transAxes, fontsize=20, rotation=90)

        ax = plt.subplot(rows, cols, cols + 1 + slind)
        imshow_res = plt.imshow(kss_pred[:, slice2plot, :], vmin=kss_lims[0], vmax=kss_lims[1], cmap=cmap_mgm)
        plt.gca().margins(0, 0);
        plt.gca().set_xticklabels([]);
        plt.gca().set_yticklabels([])
        if slice2plot == slices[-1]:
            cax = ax.inset_axes(clrbr_dims);
            fig.colorbar(imshow_res, cax=cax, orientation='vertical')
            ax.text(1.4, .3, texts[3], transform=ax.transAxes, fontsize=20, rotation=90)

        if do_err:
            ax = plt.subplot(rows, cols, cols * 2 + 1 + slind)
            Rsq = 1 - err_3d[:, slice2plot, :] ** 2
            imshow_res = plt.imshow(Rsq, cmap=cmap_R, vmin=errlims[0], vmax=errlims[1])
            plt.gca().margins(0, 0);
            plt.gca().set_xticklabels([]);
            plt.gca().set_yticklabels([]);
            plt.axis('off')
            if slice2plot == slices[-1]:
                cax = ax.inset_axes(clrbr_dims);
                fig.colorbar(imshow_res, cax=cax, orientation='vertical')
                cax.set_yticklabels(['1.00' if tick == 1 else f"{tick:.3f}".lstrip('0') for tick in cax.get_yticks()])
                ax.text(1.5, .5, texts[5], transform=ax.transAxes, fontsize=20, rotation=90)


def get_similarity_metrics(vol_a, vol_b, mask_nans):
    vol_a_z = np.copy(vol_a)
    vol_b_z = np.copy(vol_b)
    vol_a_z[np.isnan(vol_a) | np.isnan(vol_b)] = 0
    vol_b_z[np.isnan(vol_b) | np.isnan(vol_b)] = 0
    ssim = [
        np.nanmean(skimage.metrics.structural_similarity(
            vol_a_z[:, jj, :], vol_b_z[:, jj, :],
            data_range=np.nanmax(vol_a),
            full=True
        )[1] * mask_nans[:, jj, :])
        for jj in range(vol_a.shape[1])
    ]

    mape = [
        100 * np.nanmean(np.abs((vol_a[:, jj, :] - vol_b[:, jj, :]) /
                                (np.maximum(vol_b[:, jj, :], vol_a[:, jj, :]) + 1e-6)
                                )) \
        for jj in range(vol_a.shape[1])
    ]
    nrmse = [
        np.sqrt(np.nanmean((vol_a[:, jj, :] - vol_b[:, jj, :]) ** 2)) /
        (np.nanmax(vol_b[:, jj, :]) - np.nanmin(vol_b[:, jj, :]) + 1e-6)
        for jj in range(vol_a.shape[1])
    ]
    # / np.nanmean( np.maximum( vol_b[:,jj,:]**2, vol_a[:,jj,:]**2) + 1e-6 ) \

    r_pearson = [
        get_corr(vol_b[:, jj, :], vol_a[:, jj, :]) \
        for jj in range(vol_a.shape[1])
    ]

    icc = [
        slicewise_icc(vol_b[:, jj, :], vol_a[:, jj, :])
        for jj in range(vol_a.shape[1])
    ]
    # print(f'r_p={r_pearson:.2f}, mssim={np.mean(ssim):.2f} nrmse={nrmse:.2f}, mape={mape:.1f}%')
    return {"Pearson's r": r_pearson, 'SSIM': ssim, 'NRMSE': nrmse, 'MAPE (%)': mape, "ICC(2,1)": icc}


import pandas as pd, seaborn as sns


def show_correspondence_stats_boxplots(
        tp_A, tp_B, mask,
        df=None, metrics2drop=[],  # ["Pearson's r", "SSIM"],
        xlims={
            "Pearson's r": {'lims': [0, 1], 'ticklabels': ['0', '0.25', '0.5', '0.75', '1']},
            "SSIM": {'lims': [0, 1], 'ticklabels': ['0', '0.25', '0.5', '0.75', '1']},
            "NRMSE": {'lims': [0, 1], 'ticklabels': ['0', '0.25', '0.5', '0.75', '1']},
            "MAPE (%)": {'lims': [0, 100]},
            "ICC(2,1)": {'lims': [0, 1], 'ticks': [0, 0.25, 0.5, 0.75, 1],
                         'ticklabels': ['0', '0.25', '0.5', '0.75', '1']},
        },
        figsize=(25, 3)
):
    if type(df) == type(None):
        df = pd.DataFrame({
            r'f$_{ss}$': get_similarity_metrics(tp_A['fc_T'], tp_B['fc_T'], mask),
            r'k$_{ssw}$': get_similarity_metrics(tp_A['kc_T'], tp_B['kc_T'], mask),
            r'f$_{s}$': get_similarity_metrics(tp_A['fb_T'], tp_B['fb_T'], mask),
            r'k$_{sw}$': get_similarity_metrics(tp_A['kb_T'], tp_B['kb_T'], mask)
        })

    df1 = pd.DataFrame(df.stack()).T
    df_exploded = df1.apply(lambda x: x.explode()).reset_index(drop=True)

    metrics = df_exploded.columns.levels[0]
    for metricname in metrics2drop:
        metrics = metrics.drop(metricname)  # 'NRMSE')  # WTF shouldn't be needed..

    fig = plt.figure(figsize=figsize)

    for jj, top_col in enumerate(metrics):
        ax = plt.subplot(1, len(metrics), jj + 1)
        # _=plt.boxplot(df_exploded[[top_col]])
        df_part = df_exploded[top_col].apply(pd.to_numeric)
        boxplot_95 = True
        if boxplot_95:
            sns.boxplot(data=df_part, orient='y', fill=False, whis=[5., 95.])  # , sym="")
        else:
            sns.boxplot(data=df_part, orient='y', fill=False)  # , hue="alive",  gap=.1)
        plt.xlim(*xlims[top_col]['lims'])
        plt.title(top_col, fontsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_color('gray')  # visible(False)
        ax.xaxis.grid(True)
        if jj > 0:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
        else:
            ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=16)
        if 'ticks' in xlims[top_col]:  # top_col=="Pearson's r":
            ax.set_xticks(xlims[top_col]['ticks'])
        if 'ticklabels' in xlims[top_col]:
            ax.set_xticklabels(xlims[top_col]['ticklabels'])

    return fig, df


def plot_basic_brain_stats(volunteers):
    volunteer_names = ["Volunteer 1", "Volunteer 2", "Volunteer 3", "Volunteer 4"]

    def preprocess_data(data_list, attr):
        # Flatten the data and remove NaNs
        return [getattr(data, attr).flatten()[~np.isnan(getattr(data, attr).flatten())] for data in data_list]

    parameters = ['T2ms', 'T1ms', 'dB0ppm', 'B1map']
    # titles = [r'T$_2$ (ms)', r'T$_1$ (ms)', r'B$_0$ deviation (ppm)', r'B$_1$ multiplicative error (x)']
    titles = [r'T$_2$', r'T$_1$', r'B$_0$ deviation (additive)', r'B$_1$ deviation (multiplicative)']
    units = ['ms', 'ms', 'ppm', 'x']

    fig, axes = plt.subplots(1, len(parameters), figsize=(16, 6),
                             sharey=True)  # , gridspec_kw={'width_ratios': [1, 1, 1]})
    plt.rcParams.update({'axes.labelsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14})

    for i, param in enumerate(parameters):
        # Flattened and filtered data for each volunteer
        data = preprocess_data(volunteers, param)
        if False:  # param=='T2ms':
            xlim2perc = 95
            data = [np.clip(d, 0, 300) for d in data]
        else:
            xlim2perc_L = 0.1
            xlim2perc_H = 99.9
        sns.violinplot(data=data, orient='h', ax=axes[i])
        axes[i].set_title(titles[i])
        axes[i].set_xlim(np.percentile(data[0], [xlim2perc_L, xlim2perc_H]))
        if i == 0:
            axes[i].set_yticks(range(len(volunteer_names)))
            axes[i].set_yticklabels(volunteer_names)
        axes[i].set_xlabel(units[i])

    fig.tight_layout()

    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if ax != axes[0]:  # Hide left spine except for the first (leftmost) plot
            ax.spines['left'].set_visible(False)
        # ax.grid()


def slice_row_plot_auxiliary(data, figsize=None, slices=None, t2ms_lims=None):
    slices = slices or np.int32(np.linspace(5, 40, 10))[3:-2]  # cutting best/central 5 out of 10 rep.
    figsize = figsize or (25, 10)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(wspace=0, hspace=0)
    cols = 10
    # cols = len(slices)
    clrbr_dims = [1.05, 0.05, 0.1, .9]
    T2ms = np.clip(data.T2ms, *t2ms_lims) if t2ms_lims != None else data.T2ms * data.roi_mask_nans
    T1ms = data.T1ms
    dB0ppm = data.dB0ppm
    B1fact = data.B1map

    from matplotlib.colors import LinearSegmentedColormap
    colors = [
        (0.0, "brown"),
        (0.03, "brown"),  # Start color
        (0.08, "orange"),  # Compressed range
        (0.3, "yellow"),  # Start of saturated region
        (1.0, "white")  # End color (saturated)
    ]
    T2cmap = LinearSegmentedColormap.from_list("compressed_cmap", colors)
    # cmap_R = plt.cm.get_cmap("YlOrRd_r").copy();  cmap_R.set_bad('1.')
    T2cmap.set_bad('1.')
    rows = 4
    for slind, slice2plot in enumerate(slices):
        ax = plt.subplot(rows, cols, 1 + slind)
        plt.gca().margins(0, 0);
        plt.gca().set_xticklabels([]);
        plt.gca().set_yticklabels([])
        imshow_res = plt.imshow(T2ms[:, slice2plot, :], cmap=T2cmap)
        if slice2plot == slices[-1]:
            cax = ax.inset_axes(clrbr_dims);
            fig.colorbar(imshow_res, cax=cax, orientation='vertical')
            cax.set_yticklabels(cax.get_yticklabels(), fontsize=13)
            ax.text(1.6, .3, 'T$_2$ (ms)', transform=ax.transAxes, fontsize=16, rotation=90)
        ax.spines[:].set_visible(False);
        ax.tick_params(left=False, right=False, top=False, bottom=False)
        ax = plt.subplot(rows, cols, cols + 1 + slind)
        plt.gca().margins(0, 0);
        plt.gca().set_xticklabels([]);
        plt.gca().set_yticklabels([])
        imshow_res = plt.imshow(T1ms[:, slice2plot, :], cmap='YlOrBr_r')
        if slice2plot == slices[-1]:
            cax = ax.inset_axes(clrbr_dims);
            fig.colorbar(imshow_res, cax=cax, orientation='vertical')
            cax.set_yticklabels(cax.get_yticklabels(), fontsize=13)
            ax.text(1.6, .3, 'T$_1$ (ms)', transform=ax.transAxes, fontsize=16, rotation=90)
        ax.spines[:].set_visible(False);
        ax.tick_params(left=False, right=False, top=False, bottom=False)
        ax = plt.subplot(rows, cols, 2 * cols + 1 + slind)
        plt.gca().margins(0, 0);
        plt.gca().set_xticklabels([]);
        plt.gca().set_yticklabels([])
        imshow_res = plt.imshow(dB0ppm[:, slice2plot, :], vmin=-0.3, vmax=0.3, cmap='bwr')
        if slice2plot == slices[-1]:
            cax = ax.inset_axes(clrbr_dims);
            fig.colorbar(imshow_res, cax=cax, orientation='vertical')
            cax.set_yticklabels(cax.get_yticklabels(), fontsize=13)
            ax.text(1.6, .1, 'B$_0$ shift (ppm)', transform=ax.transAxes, fontsize=16, rotation=90)
        ax.spines[:].set_visible(False);
        ax.tick_params(left=False, right=False, top=False, bottom=False)
        ax = plt.subplot(rows, cols, 3 * cols + 1 + slind)
        plt.gca().margins(0, 0);
        plt.gca().set_xticklabels([]);
        plt.gca().set_yticklabels([])
        imshow_res = plt.imshow(B1fact[:, slice2plot, :], vmin=0.6, vmax=1.4, cmap='bwr')
        if slice2plot == slices[-1]:
            cax = ax.inset_axes(clrbr_dims);
            fig.colorbar(imshow_res, cax=cax, orientation='vertical')
            cax.set_yticklabels(cax.get_yticklabels(), fontsize=13)
            ax.text(1.6, .1, 'B$_1$ factor (x)', transform=ax.transAxes, fontsize=16, rotation=90)
        ax.spines[:].set_visible(False);
        ax.tick_params(left=False, right=False, top=False, bottom=False)


def signal_fit_eval_viz(measured, reconstituted, ds=5, draw_ratio=False):
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 2, 1)
    reconstituted = reconstituted[:, ::ds, ::ds, ::ds]
    measured = measured[:, ::ds, ::ds, ::ds]
    ratio = reconstituted / measured
    ratio = ratio.reshape(31, -1)
    xaxis = np.arange(31)[:, None] + 0.1 * np.random.randn(*ratio.shape)
    plt.plot(xaxis + 0.5, measured.reshape(31, -1), 'g.', alpha=0.01)
    if draw_ratio:
        plt.plot(xaxis, ratio, 'b.', alpha=0.02)
        # plt.legend(['measured', 'reconstituted / measured'], loc='upper left')
    else:
        plt.plot(xaxis, reconstituted.reshape(31, -1), 'b.', alpha=0.02)
        a = reconstituted.reshape(31, -1)
        b = measured.reshape(31, -1)
        for mrfind in range(31):
            inds_sorted = np.argsort(a[mrfind])  # TODO replace by percentile?
            for ri in np.int32(np.linspace(0, a.shape[1] - 1, 10)):
                z = inds_sorted[ri]
                # z = np.random.choice(np.arange(a.shape[1]))
                x = xaxis[mrfind, z]
                plt.plot([np.round(x), np.round(x) + 0.5], [a[mrfind, z], b[mrfind, z]], 'k', linewidth=1)
        # plt.legend(['measured','reconstituted'], loc='upper left')
    # plt.ylabel('reconstituted / measured')
    plt.ylim(0, 1.2)  # plt.ylim(0.5, 1.5)
    plt.xlabel('MRF iteration')

    # plt.figure(figsize=(10,3))
    plt.subplot(1, 2, 2)
    signal_est_diffnorm_map = jnp.linalg.norm(reconstituted - measured, axis=0)
    signal_est_diffnorm_map /= jnp.linalg.norm(measured, axis=0)
    hvals, hbins, patches = plt.hist(100 * signal_est_diffnorm_map.flatten(), bins=100)
    plt.xlabel('signal decode-reconstitute loop regression loss (%)')
    plt.ylabel('voxels')
    plt.text(0.5, 0.6,
             f"mean={100 * jnp.nanmean(signal_est_diffnorm_map):.1f}%, "
             f"\nrms={100 * jnp.sqrt(jnp.nanmean(signal_est_diffnorm_map ** 2)):.1f}%",
             transform=plt.gca().transAxes, fontsize=20)

    plt.tight_layout()
    return hvals, hbins


def slicewise_icc(map1, map2, icctype='ICC2'):
    """ ICC3.1 measures "consistency" while ICC2.1 measures "absolute agreement".
    """
    import pingouin as pg
    # icc_results = []
    # for sl in range(map1.shape[1]):
    if True:
        map1 = map1.flatten()  # [:, sl, :].flatten()
        map2 = map2.flatten()  # [:, sl, :].flatten()
        slice1 = map1[~np.isnan(map1) & ~np.isnan(map2)]
        slice2 = map2[~np.isnan(map1) & ~np.isnan(map2)]
        data = pd.DataFrame({
            'targets': np.repeat(np.arange(len(slice1)), 2),
            'raters': np.tile(['slice1', 'slice2'], len(slice1)),
            'ratings': np.concatenate([slice1, slice2])
        })

        # Compute the ICC
        icc = pg.intraclass_corr(data=data, targets='targets', raters='raters', ratings='ratings', nan_policy='omit')

        # Filter for correct type (ICC(2,1) / ICC(2,1)) and store the result
        icc_x = icc[icc['Type'] == icctype]
        icc_result = icc_x['ICC'].values[0]
        # icc_results.append(icc_result)

    return icc_result
    # return np.array(icc_results)


def plot_XY_bland_altman(
        method1_arr, method2_arr, param_name, units, method1_name, method2_name,
        axes=None, figsize=(12, 6), alpha=0.01
):
    means = (method1_arr + method2_arr) / 2
    differences = method1_arr - method2_arr
    mean_diff = np.nanmean(differences)
    sd_diff = np.nanstd(differences)

    # Publication-quality plots (?)
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'legend.fontsize': 10
    })

    if type(axes) == type(None):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    sns.set_style("whitegrid")

    # 1. X-Y Plot
    ax1 = axes[0]
    sns.scatterplot(
        x=method1_arr, y=method2_arr,
        s=30, color="royalblue", ax=ax1, alpha=alpha,
        # label=f"{param_name} (scatter)"
    )
    minmin = np.min([np.nanmin(method1_arr), np.nanmax(method1_arr)])
    maxmax = np.max([np.nanmax(method1_arr), np.nanmax(method1_arr)])
    ax1.plot([minmin, maxmax], [minmin, maxmax], "k--", label="Ideal line $y=x$")
    reg = scipy.stats.linregress(
        method1_arr[~np.isnan(method1_arr) & ~np.isnan(method2_arr)],
        method2_arr[~np.isnan(method1_arr) & ~np.isnan(method2_arr)]
    )
    xgrid = np.linspace(np.nanmin(method1_arr), np.nanmax(method1_arr), 10)
    label = rf"Linear fit"  # , R$^2$={reg.rvalue ** 2:.2f}"
    pearsons = np.floor(reg.rvalue * 1000) / 1000
    if reg.pvalue < 1e-6:
        label += ",\n" + f"r>{pearsons:.3f}, " + r"p-value<10$^{-6}$"
    else:
        label += ",\n" + f"r>{pearsons:.3f}, " + r"p-value={reg.pvalue:.1e}"
    ax1.plot(
        xgrid, reg.slope * xgrid + reg.intercept,
        color="darkorange", linewidth=1,  # linestyle="--",
        label=label
    )
    ax1.set_xlim(minmin, maxmax)
    ax1.set_ylim(minmin, maxmax)
    ax1.set_xlabel(f"{param_name} ({units}), est. by {method1_name} ")
    ax1.set_ylabel(f"{param_name} ({units}), est. by {method2_name} ")
    ax1.legend(loc="upper left", title=f"{param_name} (scatter)", title_fontsize=14)
    plt.grid()

    # 2. Bland-Altman Plot
    ax2 = axes[1]
    sns.scatterplot(x=means, y=differences, s=30, color="teal", ax=ax2,
                    alpha=alpha)  # , label=f"{param_name} (Bland-Altman)")
    ax2.axhline(mean_diff, color="darkred", linestyle="--", label=f"Mean diff = {mean_diff:.2f}")
    ax2.axhline(mean_diff + 1.96 * sd_diff, color="grey", linestyle="--", label=r"+1.96 SD")
    ax2.axhline(mean_diff - 1.96 * sd_diff, color="grey", linestyle="--", label=r"-1.96 SD")
    ax2.set_xlabel(f"{param_name} ({units}), methods' mean")
    ax2.set_ylabel(f"{param_name} ({units}), methods' difference ")

    ax2.legend(loc="upper right", title=f"{param_name} (Bland-Altman)", title_fontsize=14)
    # plt.ylim(mean_diff - 5 * sd_diff, mean_diff + 5 * sd_diff)
    xlim = ax1.get_xlim();
    diff = xlim[1] - xlim[0]
    ax2.set_ylim(-(maxmax - minmin) / 2, (maxmax - minmin) / 2)  # mean_diff - 5 * sd_diff, mean_diff + 5 * sd_diff)
    ax2.set_xlim(minmin, maxmax)  # mean_diff - 5 * sd_diff, mean_diff + 5 * sd_diff)
    plt.grid()
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')


def robustness_boxplot_train_test(
        synth_data_list,
        brain_data_list,
        label_train='Dictionary entries (in-silico training set)',
        label_test='Brain voxels (in-vivo test set)'
):
    # Interleave synth and brain data
    interleaved_data = []
    labels = []
    colors = []
    for idx, (synth_data, brain_data) in enumerate(zip(synth_data_list, brain_data_list)):
        interleaved_data.append(synth_data)
        interleaved_data.append(brain_data)
        labels.append(f'run{idx + 1} (synth)')
        labels.append(f'run{idx + 1} (brain)')
        colors.append('g')
        colors.append('r')

    sns.set(style="whitegrid")
    sns.set_context("talk", font_scale=1.2)

    # Plot the boxplots with color per series
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=interleaved_data, palette=sns.color_palette(["g", "r"]), showfliers=False)
    plt.xticks(ticks=range(len(labels)), labels=[f'run {i // 2 + 1}' if i % 2 == 0 else '' for i in range(len(labels))],
               rotation=45)
    plt.ylabel('NRMSE')
    # plt.title('Multi-Histogram as Paired Boxplot (Color per Series)')
    plt.legend(handles=[plt.Line2D([0], [0], color='g', lw=4, label=label_train),
                        plt.Line2D([0], [0], color='r', lw=4, label=label_test)])
    plt.ylim(0, 0.3)
    plt.tight_layout()

    plt.show()


import numpy as np
from skimage.draw import disk


def pack_circles(image, centers, radius):
    """
    Extracts circular regions from the image based on the given centers and radius,
    and packs them into a smaller matrix for display.

    Parameters:
    - image: 2D numpy array representing the original image.
    - centers: List of tuples representing the (x, y) coordinates of the circle centers.
    - radius: Integer representing the radius of the circles.

    Returns:
    - packed_image: 2D numpy array containing the packed circular regions.
    """

    # Determine the size of the packed image
    num_circles = len(centers)
    circle_diameter = 2 * radius
    # packed_image_size = (circle_diameter, num_circles * circle_diameter)
    packed_image_size = int(radius * 3.7), (num_circles + 1) * radius
    packed_image = np.zeros(packed_image_size, dtype=image.dtype)
    packed_image[:, :] = np.nan

    new_centers = []
    new_rois_nan = []
    for i, (x_center, y_center) in enumerate(centers):
        # Extract the circular region and place it in the packed image
        x_start = max(0, x_center - radius)
        y_start = max(0, y_center - radius)
        x_end = min(image.shape[0], x_center + radius)
        y_end = min(image.shape[1], y_center + radius)

        # Trivial (square)
        circle_region_cropped = image[x_start:x_end, y_start:y_end]

        roi_nan = np.zeros_like(packed_image)
        roi_nan[:, :] = np.nan
        if (i % 2) == 1:
            packed_image[:circle_diameter, i * radius: (i + 2) * radius] = np.where(
                np.isnan(packed_image[:circle_diameter, i * radius: (i + 2) * radius]),
                circle_region_cropped,
                packed_image[:circle_diameter, i * radius: (i + 2) * radius]
            )
            new_centers.append((radius, (i + 1) * radius))
            roi_nan[:circle_diameter, i * radius: (i + 2) * radius] = packed_image[:circle_diameter,
                                                                      i * radius: (i + 2) * radius]
        else:
            # packed_image[-circle_diameter:, i * radius: (i + 2) * radius] = circle_region_cropped
            packed_image[-circle_diameter:, i * radius: (i + 2) * radius] = np.where(
                np.isnan(packed_image[-circle_diameter:, i * radius: (i + 2) * radius]),
                circle_region_cropped,
                packed_image[-circle_diameter:, i * radius: (i + 2) * radius]
            )
            new_centers.append((2 * radius + 2, (i + 1) * radius))
            roi_nan[:circle_diameter, i * radius: (i + 2) * radius] = packed_image[:circle_diameter,
                                                                      i * radius: (i + 2) * radius]

        new_rois_nan.append(roi_nan)

    return packed_image, new_centers, new_rois_nan


def phantom2pandas(fs_pred_mM_by_roi, ks_pred_Hz_by_roi, VIALS='ABCDEF'):
    slices = ks_pred_Hz_by_roi[0].shape[1]
    data = pd.concat([
        pd.DataFrame({
            'fs': [fs_pred_mM[:, si, :] for si in range(slices)],
            'ks': [ks_pred_Hz[:, si, :] for si in range(slices)],
            'slice': range(slices),
            'vial': VIALS[vi]
        }) for vi, (fs_pred_mM, ks_pred_Hz) in enumerate(zip(fs_pred_mM_by_roi, ks_pred_Hz_by_roi))
    ])

    data.fs = data.fs.apply(lambda x: x[~np.isnan(x)])
    data.ks = data.ks.apply(lambda x: x[~np.isnan(x)])

    data = data.reset_index().explode(['fs', 'ks'])  # , ignore_index=True)
    data['pixind'] = data.groupby(['slice', 'vial']).cumcount()
    data = data.drop('index', axis=1).reset_index().drop('index', axis=1)

    return data


def draw_vial_sample_signals(larg_ph_ds, centers, nn_pred_signal_normed_np):
    fig = plt.figure(figsize=(9, 2))
    for ii, center in enumerate(centers):  # , ax in zip(centers, axes):
        ax = fig.add_axes([0, ii * 0.16, 0.8, 0.5])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ii > 0:
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.patch.set_alpha(0)
            ax.set_xticks([]);
            ax.set_yticks([]);
            ax.set_xticklabels([]);
            ax.set_yticklabels([]);
        ax.plot(larg_ph_ds.measured_normed_T[:, center[0], 5, center[1]], ':ko', linewidth=1.5, markerfacecolor='none')
        ax.plot(nn_pred_signal_normed_np[:, center[0], 5, center[1]], '-rd', linewidth=0.5, markerfacecolor='none')


def GMWM_T1T2_boxplot(data_xa, figsize=(10, 3)):
    white_mask = data_xa['white_mask'].values.flatten()
    gray_mask = data_xa['gray_mask'].values.flatten()
    t1_wm_values = data_xa['T1ms'].values.flatten()
    th = 0.9
    t1_wm_values = t1_wm_values[white_mask > th]
    t1_wm_values = t1_wm_values[~np.isnan(t1_wm_values)]
    t1_gm_values = data_xa['T1ms'].values.flatten()
    t1_gm_values = t1_gm_values[gray_mask > th]
    t1_gm_values = t1_gm_values[~np.isnan(t1_gm_values)]

    t2_wm_values = data_xa['T2ms'].values.flatten()
    t2_wm_values = t2_wm_values[white_mask > th]
    t2_wm_values = t2_wm_values[~np.isnan(t2_wm_values)]
    t2_gm_values = data_xa['T2ms'].values.flatten()
    t2_gm_values = t2_gm_values[gray_mask > th]
    t2_gm_values = t2_gm_values[~np.isnan(t2_gm_values)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    # ax2 = ax1.twinx()

    # Plot the boxplot for white matter
    box1wm = ax1.boxplot(t1_wm_values, positions=[0.8], widths=0.4, patch_artist=True,
                         boxprops=dict(facecolor="lightblue"), whis=[2.5, 97.5])
    box1gm = ax1.boxplot(t1_gm_values, positions=[1.8], widths=0.4, patch_artist=True,
                         boxprops=dict(facecolor="lightblue"), whis=[2.5, 97.5])
    ax1.set_ylabel('T1 (ms) ', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    box2wm = ax2.boxplot(t2_wm_values, positions=[1.2], widths=0.4, patch_artist=True,
                         boxprops=dict(facecolor="lightgreen"), whis=[2.5, 97.5])
    box2gm = ax2.boxplot(t2_gm_values, positions=[2.2], widths=0.4, patch_artist=True,
                         boxprops=dict(facecolor="lightgreen"), whis=[2.5, 97.5])
    ax2.set_ylabel('T2 (ms) ', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    # ax2.xaxis.set_visible(False)
    ax1.set_ylim(300, 3000)
    ax2.set_ylim(15, 150)
    for ax in [ax1, ax2]:
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['WM', 'GM'])
    for box in [box1wm, box2wm, box1gm, box2gm]:
        for flier in box['fliers']:
            flier.set(marker='o', color='red', alpha=0.1)


def viz_t1t2b0b1_sample_slices(data_xa, data_feed, sample=True):
    t2_map_log = np.log1p(data_xa['T2ms'].values)
    min_t2log, max_t2log = np.nanmin(t2_map_log), np.nanmax(t2_map_log)
    t1_map_log = np.log1p(data_xa['T1ms'].values)
    min_t1log, max_t1log = np.nanmin(t1_map_log), np.minimum(np.nanmax(t1_map_log), np.log(4000))
    num_ticks = 5
    t1_tick_positions = np.linspace(min_t1log, max_t1log, num_ticks)
    t1_tick_labels = np.round(np.exp(t1_tick_positions) / 10) * 10  # Inverse transform to real values
    t2_tick_positions = np.linspace(min_t2log, max_t2log, num_ticks)
    t2_tick_labels = np.round((np.exp(t2_tick_positions) - 1) / 5) * 5  # Inverse transform to real values
    slices2plot = list(range(data_feed.shape[1])) if sample else list(range(10, 40, 5))
    fig, axes = plt.subplots(5, len(slices2plot), figsize=(len(slices2plot) * 2.5 + 2, 12))
    plt.subplots_adjust(wspace=0, hspace=0)

    for ii, sliceind in enumerate(slices2plot):  # range(data_feed.shape[1]):
        res = axes[0, ii].imshow(t1_map_log[10:-10, sliceind, 7:-7], vmin=min_t1log, vmax=max_t1log, cmap=cmc.lipari)
    cbar = plt.colorbar(res, shrink=0.8)
    cbar.set_ticks(t1_tick_positions)
    cbar.set_ticklabels([f"{val:.0f}" for val in t1_tick_labels])
    cbar.set_label(r'T$_1$ [ms]', fontsize=14)

    for ii, sliceind in enumerate(slices2plot):
        res = axes[1, ii].imshow(t2_map_log[10:-10, sliceind, 7:-7], vmin=min_t2log, vmax=max_t2log, cmap=cmc.navia)
    cbar = plt.colorbar(res, shrink=0.8)
    cbar.set_ticks(t2_tick_positions)
    cbar.set_ticklabels([f"{val:.0f}" for val in t2_tick_labels])
    cbar.set_label(r'T$_2$ [ms]', fontsize=14)

    for ii, sliceind in enumerate(slices2plot):
        res = axes[2, ii].imshow(data_xa['B1_fix_factor_map'].values[10:-10, sliceind, 7:-7], vmin=0.7, vmax=1.3,
                                 cmap='PiYG')
    cbar = plt.colorbar(res, shrink=0.8)
    cbar.set_label(r'B$_1$ deviation [x]', fontsize=14)

    for ii, sliceind in enumerate(slices2plot):
        res = axes[3, ii].imshow(data_xa['B0_shift_ppm_map'][10:-10, sliceind, 7:-7], vmin=-.5, vmax=.5,
                                 cmap='coolwarm')
    cbar = plt.colorbar(res, shrink=0.8)
    cbar.set_label(r'B$_0$ deviation [ppm]', fontsize=14)

    for ii, sliceind in enumerate(slices2plot):
        # res = axes[4, ii].imshow((np.float32(data_xa['white_mask']>0.9) - np.float32(data_xa['gray_mask']>0.9))[10:-10, sliceind, 7:-7], vmin=-.5, vmax=.5, cmap='coolwarm')
        if False:
            norm_prob_wm = data_xa['white_mask'] / (data_xa['gray_mask'] + data_xa['white_mask'])
            res = axes[4, ii].imshow(norm_prob_wm[10:-10, sliceind, 7:-7], cmap='jet')
            cbar = plt.colorbar(res, shrink=0.8)
            cbar.set_label(r'WM / GM via SPM', fontsize=14)
        else:
            white = np.nan_to_num(data_xa['white_mask'])
            gray = np.nan_to_num(data_xa['gray_mask'])
            white_red_gray_green = white[..., None] * np.array((1., 0., 0.)) + \
                                   gray[..., None] * np.array((0., 1., 0.))
            white_red_gray_green /= np.max(white_red_gray_green)
            res = axes[4, ii].imshow(white_red_gray_green[10:-10, sliceind, 7:-7, :])
            axes[4, ii].set_title('RED: white, GREEN: gray', fontsize=8)
        # res = axes[4, ii].imshow((np.float32(data_xa['white_mask']) + np.float32(data_xa['gray_mask']))[10:-10, sliceind, 7:-7], cmap='coolwarm')
        # res = axes[4, ii].imshow((np.clip(np.float32(data_xa['white_mask']), -1, 1) - np.clip(np.float32(data_xa['gray_mask']), -1, 1))[10:-10, sliceind, 7:-7], vmin=-.5, vmax=.5, cmap='coolwarm')

    for ax in axes.flatten():
        ax.axis('off')
        ax.margins(0, 0)
    # plt.tight_layout()


def fancy_histogram(err, figsize=(7, 3)):
    err[err == 0] = np.nan
    sns.set(style="whitegrid")
    sns.set_context("talk", font_scale=0.8)  # 1.2
    palette = sns.color_palette("pastel")
    plt.figure(figsize=figsize)  # (14, 6))
    errf = err.flatten()
    _ = sns.histplot(errf, bins=np.linspace(0, 0.3, 100))  # , stat='density')
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    plt.xlabel('Modeling NRMSE')
    plt.ylabel('Voxel count')
    median_val = np.nanmedian(errf)
    perc_95_val = np.nanpercentile(errf, 95)
    plt.axvline(median_val, color='b', linestyle='--', label=f'Median')  #: {median_val:.2f}')
    plt.axvline(perc_95_val, color='r', linestyle='--', label=f'95th Percentile')  #: {perc_95_val:.2f}')
    plt.text(median_val + 0.03, plt.ylim()[1] * 0.8, f'Median: {median_val:.3f}', color='b', ha='center')
    plt.text(perc_95_val + 0.044, plt.ylim()[1] * 0.6, f'95th Percentile: {perc_95_val:.3f}', color='r', ha='center')
    plt.legend()
    plt.tight_layout()
    plt.rcdefaults()


def compare_fk_preds(bsf, estimated_params, pool='c'):
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 2, 1)
    # _=plt.hist(100*(bsf.fc_gt_T - estimated_params['fc_T']).flatten(), bins=100)
    plt.plot(
        100 * (bsf.__dict__[f'f{pool}_gt_T']).flatten()[::5],
        100 * (estimated_params[f'f{pool}_T']).flatten()[::5],  # -bsf.fc_gt_T -
        '.', alpha=0.03
    )
    plt.xlabel('true');
    plt.ylabel('estimated')
    plt.title('f')
    plt.subplot(1, 2, 2)
    # _=plt.hist((bsf_lhsynth_mt.kc_gt_T - estimated_params['kc_T']).flatten(), bins=100)
    plt.plot(
        (bsf.__dict__[f'k{pool}_gt_T']).flatten()[::6],
        (estimated_params[f'k{pool}_T']).flatten()[::6],  # - bsf.kc_gt_T
        '.', alpha=0.03
    )
    plt.xlabel('true');
    plt.ylabel('estimated')
    plt.title('k')


def remove_spines(ax=None):
    ax = ax or plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.xticks([]);
    plt.yticks([])
    ax.axis('off')
    ax.margins(0, 0)


import time


class Plot_Updater():
    from IPython.display import display, clear_output
    slice_in_batch = 1

    def __init__(self, pool):
        self.fig, axes = plt.subplots(2, 3, figsize=(17, 10))
        self.axes = axes
        self.pool = pool

        self.f_imsh = axes[0, 0].imshow(np.zeros((116, 88)), cmap="viridis", vmin=0,
                                        vmax=30 if self.pool == 'c' else 1);
        cbar = plt.colorbar(self.f_imsh);
        cbar.set_label("f$_{ss}$ (%)");
        axes[0, 0].set_title('Volume fraction estimate')

        self.k_imsh = axes[0, 1].imshow(np.zeros((116, 88)), cmap="magma", vmin=0, vmax=100 if pool == 'c' else 500);
        cbar = plt.colorbar(self.k_imsh);
        cbar.set_label("k$_{ssw}$ (s$^{-1}$)")
        axes[0, 1].set_title('Exchange rate estimate')

        self.nrmse_map = 0.1 * np.ones((116, 88)) if pool == 'c' else 0.1 * np.ones(
            (116 // 2, 88 // 2))  # downsample..
        self.nrmse_imsh = axes[0, 2].imshow(self.nrmse_map, cmap="hot_r", vmin=0, vmax=10)
        cbar = plt.colorbar(self.nrmse_imsh);
        cbar.set_label("100 * nrmse (%)")
        self.title_nrmse_02 = axes[0, 2].set_title('Signal modeling error', fontsize=16)

        self.f_imsh_clean = axes[1, 0].imshow(np.zeros((116, 88)), cmap="viridis", vmin=0,
                                              vmax=30 if self.pool == 'c' else 1)
        cbar = plt.colorbar(self.f_imsh_clean);
        cbar.set_label("f$_{ss}$ (%)")
        axes[1, 0].set_title('Noise-augmented $\widehat{f}_{ss}$')

        self.k_imsh_clean = axes[1, 1].imshow(np.zeros((116, 88)), cmap="magma", vmin=0,
                                              vmax=100 if pool == 'c' else 500)
        cbar = plt.colorbar(self.k_imsh_clean);
        cbar.set_label("k$_{ssw}$ (s$^{-1}$)")
        axes[1, 1].set_title('Noise-augmented $\widehat{k}_{ssw}$')

        self.nrmse_imsh_clean = axes[1, 2].imshow(self.nrmse_map, cmap="hot_r", vmin=0, vmax=10)
        self.title_nrmse_12 = axes[1, 2].set_title('NRMSE w. augmented estimates', fontsize=16)
        cbar = plt.colorbar(self.nrmse_imsh_clean);
        cbar.set_label("100 * nrmse (%)")

        self.nrmse_map_hist = self.nrmse_map[None, ...]
        self.t0 = time.time()
        self.f_key = 'fb_T' if pool == 'b' else 'fc_T'
        self.k_key = 'kb_T' if pool == 'b' else 'kc_T'

        self.artists_lists_by_frame = []
        for ax in axes.flatten():
            remove_spines(ax)
        self.txt1, self.txt2, self.txt3 = [None, None, None]

    def update_plot(self, tissue_params, step):
        frq = 4
        self.nrmse_map_hist = np.concatenate(
            (self.nrmse_map_hist[-(frq - 1):, :, :], tissue_params['signal_est_diffnorm_map'][None, :, jj, :]), axis=0)
        if step % frq != 0:  # 1:
            return
        self.nrmse_map = np.nanmin(self.nrmse_map_hist, axis=0)
        self.redraw_plot(tissue_params, step)

        # self.fig.suptitle(f"Step: {step}, mean NRMSE={100*np.nanmean(tissue_params['signal_est_diffnorm_map']):.2f}%", fontsize=16)
        # , Time: {time.time()-self.t0:.1f}sec
        # fig.canvas.draw_idle()   # plt.draw()
        # plt.pause(0.01)  # Small pause for the update to render
        clear_output(wait=True)  # Clears old output without flickering
        display(self.fig)

    def redraw_plot(self, tissue_params, step):
        # if step > 0:
        # for txt in self.axes[0,2].texts + self.axes[1,2].texts:
        #     if txt is not None:
        #         txt.remove()
        artists = []
        self.f_imsh = self.axes[1, 0].imshow(tissue_params[self.f_key][:, self.slice_in_batch, :]._value * 100,
                                             cmap="viridis", vmin=0, vmax=30 if self.pool == 'c' else 1);
        self.k_imsh = self.axes[1, 1].imshow(tissue_params[self.k_key][:, self.slice_in_batch, :]._value, cmap="magma",
                                             vmin=0, vmax=100 if self.pool == 'c' else 500)
        self.nrmse_imsh = self.axes[1, 2].imshow(
            100 * tissue_params['signal_est_diffnorm_map'][:, self.slice_in_batch, :]._value, cmap="hot_r", vmin=0,
            vmax=10)  # 0.1)
        artists += [self.f_imsh, self.k_imsh, self.nrmse_imsh]

        self.f_imsh_clean = self.axes[0, 0].imshow(
            tissue_params[self.f_key.replace('_T', '_clean')][:, self.slice_in_batch, :]._value * 100, cmap="viridis",
            vmin=0, vmax=30 if self.pool == 'c' else 1);
        self.k_imsh_clean = self.axes[0, 1].imshow(
            tissue_params[self.k_key.replace('_T', '_clean')][:, self.slice_in_batch, :]._value, cmap="magma", vmin=0,
            vmax=100 if self.pool == 'c' else 500)
        self.nrmse_imsh_clean = self.axes[0, 2].imshow(
            100 * tissue_params['clean_signal_est_diffnorm_map'][:, self.slice_in_batch, :]._value, cmap="hot_r",
            vmin=0, vmax=10)  # 0.1);
        artists += [self.f_imsh_clean, self.k_imsh_clean, self.nrmse_imsh_clean]

        if self.txt1 is not None:
            self.txt1.set_visible(False)  # alpha(0.0)
            self.txt2.set_visible(False)  # alpha(0.0)
            self.txt3.set_visible(False)  # alpha(0.0)

        txt1 = f"Step: {step}"
        self.txt1 = self.axes[0, 2].text(
            -0.05, 0.98, txt1,
            transform=self.axes[0, 2].transAxes, bbox=dict(facecolor='gray', alpha=0.5),
            verticalalignment='top', horizontalalignment='left', fontsize=16, color='red',
        )

        txt2 = f"<NRMSE>={100 * np.nanmean(tissue_params['signal_est_diffnorm_map']):.2f}%"
        self.txt2 = self.axes[1, 2].text(
            -0.05, 0.05, txt2,
            transform=self.axes[1, 2].transAxes, verticalalignment='top', horizontalalignment='left', fontsize=16,
            color='blue'
        )

        txt3 = f"<NRMSE>={100 * np.nanmean(tissue_params['clean_signal_est_diffnorm_map']):.2f}%"
        self.txt3 = self.axes[0, 2].text(
            -0.05, 0.05, txt3,
            transform=self.axes[0, 2].transAxes, verticalalignment='top', horizontalalignment='left', fontsize=16,
            color='blue'
        )

        artists += [self.txt1, self.txt2, self.txt3]  # self.title_nrmse_12, self.title_nrmse_02, self.txt]
        self.artists_lists_by_frame += [artists]
