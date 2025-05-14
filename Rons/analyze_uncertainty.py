import matplotlib.pyplot as plt, numpy as np, collections
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import pairwise_distances_argmin_min
from matplotlib.patches import Ellipse, Arrow
import matplotlib.patches as mpatches
from pathlib import Path
from dictionary_methods import dictbased_runner

import infer, utils
from matplotlib import gridspec
import matplotlib.colors as mcolors


def get_post_estimates(tissue_param_est, shape, is_amide):
    """ 
        Assuming results of NBMF inference included the covatiance matrix components,
        this function extracts the info - for all voxels, and returns the results below,
        All scaled to the "visualization space" of k, 100*f 
        
        Returns: 
        1,2 - estimates of f and k, their standard deviations, 
        3 - the covariance matrix of the estimates -
        4,5,6 - The gaussian 95% confidence regions (CRs) parameterized as uncertainty ellipses (width, height, angle) for visualization.
    """
    f_est = 100*tissue_param_est['fc_T']
    k_est = tissue_param_est['kc_T']

    u = tissue_param_est['ucov'].reshape((*shape, 2, 2))
    s = tissue_param_est['scov'].reshape((*shape, 2, 2))
    
    scales = np.array((100 * infer.infer_config.fb_scale_fact, infer.infer_config.kb_scale_fact)) if is_amide \
        else np.array((100 * infer.infer_config.fc_scale_fact, infer.infer_config.kc_scale_fact))     

    _cov = u @ s @ np.transpose(u, [0,1,2,4,3]) # if len(shape)==3 else [0,1,3,2]) 
    cov_scaled = scales[None, :] * _cov * scales[:, None]
    
    f_sigma = np.sqrt(cov_scaled[..., 0, 0])
    k_sigma = np.sqrt(cov_scaled[..., 1, 1])
    
    s2_scaled, u_scaled = np.linalg.eigh(cov_scaled)
    s_scaled = np.sqrt(s2_scaled)
    d0 = s_scaled[...,0,None] * u_scaled[..., 0]
    d1 = s_scaled[...,1,None] * u_scaled[..., 1]

    angle = np.degrees(np.arctan2(d1[..., 1], d1[..., 0]))  # ! d1 long axis, as eigenvals come in ascending order
    chisq = 2.45  # chi-square(2, 0.95)
    # CR of +-2*sigma hence 2*2 for total width
    height = np.linalg.norm(d1, axis=-1) * 2 * chisq 
    width = np.linalg.norm(d0, axis=-1) * 2 * chisq 
    # cov_scaled, d0, d1
    return f_est, k_est, cov_scaled, f_sigma, k_sigma, height, width, angle


def cbarhist(_img, _data, _ax, pad=0.05):
    divider = make_axes_locatable(_ax)
    cax = divider.append_axes("right", size="15%", pad=pad)
    cbar=plt.colorbar(_img, cax=cax);  
    hdata = _data.flatten(); hdata = hdata[~np.isnan(hdata)] 
    hist, bins = np.histogram(hdata, bins=100, density=True)
    hist = hist / np.max(hist)  # np.log10(10 * hist / np.max(hist))
    cbar.ax.plot(hist, bins[1:], color='black', linewidth=2)
    return cbar

def plot_map(ax, map, vmin, vmax, cmap, label):    
    img = ax.imshow(map, vmin=vmin, vmax=vmax, cmap=cmap);  
    cbar = cbarhist(img, map, ax, pad=0.2) 
    cbar.set_label(label, fontsize=14)
    cbar.ax.yaxis.set_ticks_position('left')
    return img

def plot_CI_maps(
        f_est, f_sigma, k_est, k_sigma, is_mt,
        vmin_f=0, vmax_f=None, vmin_k=0, vmax_k=None, 
        cmap_f = 'viridis',
        cmap_k = 'magma'
    ):
    """ Visualize the univariate confidence intervals (CI) of f and k estimates for a whole slice, 
        by plotting the mean and +/- 2 sigma of each estimate.
    """
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    f_label_base = r'$\hat{f}_{ss}$' if is_mt else r'$\hat{f}_{s}$'
    k_label_base = r'$\hat{k}_{ss}$' if is_mt else r'$\hat{k}_{s}$'
    vmax_f = vmax_f or (30 if is_mt else 2)
    vmax_k = vmax_k or (50 if is_mt else 150)

    plot_map(axes[0, 0], f_est - 2*f_sigma, vmin_f, vmax_f, cmap_f, f_label_base +r'$ - 2\hat{\sigma}_f$  (%)')
    plot_map(axes[0, 1], f_est, vmin_f, vmax_f, cmap_f, f_label_base + '       (%)')
    plot_map(axes[0, 2], f_est + 2*f_sigma, vmin_f, vmax_f, cmap_f, f_label_base +r' + 2$\hat{\sigma}_f$  (%)')
    plot_map(axes[1, 0], k_est - 2*k_sigma, vmin_k, vmax_k, cmap_k, k_label_base +r' - 2$\hat{\sigma}_k  (s^{-1})$')
    plot_map(axes[1, 1], k_est, vmin_k, vmax_k, cmap_k, k_label_base +r' +     (s^{-1})')
    plot_map(axes[1, 2], k_est + 2*k_sigma, vmin_k, vmax_k, cmap_k, k_label_base +r' + 2$\hat{\sigma}_k  (s^{-1})$')
    for ax in axes.flatten():
        utils.remove_spines(ax)    
    return axes


def get_nrmse_grid(
    ax, _x, _y, sli, 
    data_feed_mt, mt_tissue_param_est,
    data_feed_amide=None, amide=False, mt_sim_mode='expm_bmmat',
    do_plot=True, loc_dict_res=100,
    max_f_mt=.03, max_k_mt=100,
    max_f_amide=0.02, max_k_amide=100, **kwargs
    ):
    """
        Implements our per-voxel likelihood-mapping method, by creating a local specialized dictionary, 
        using the auxiliary parameters for this voxel (T1, T2, and f_c, k_c of MT if we do amide here),
        and computing the dot-product of the local dictionary with the measured signal (post normalization) - 
        equivalently, Euclidian distance and eventually map of Normalized Root-Mean Square (NRMSE) across a full 2D grid of f-k hypotheses.

        Args:
        data_feed_mt (_type_): The data feed for MT
        mt_tissue_param_est (_type_): The MT tissue parameter estimates
        data_feed_amide (_type_, optional): The data feed for amide. Defaults to None.
        amide (bool, optional): Use the (b) pool. Defaults to False.
        mt_sim_mode (str, optional): The mode of MT simulation. Defaults to 'expm_bmmat', use 'isar2_c' in human for speedup.
        do_plot (bool, optional): Whether to plot the results. Defaults to True.
        
        Returns: 
        1,2 - Volume fraction (f), Exchange rate (k) best estimates (lowest nrmse, highest dot-product)
        3 - The nrmse map
        4,5 - _df, _dk
        
    """
    if amide:
        signal = data_feed_amide.measured_normed_T[:, _x, sli, _y]  
        
        _df, _dk = max_f_amide/loc_dict_res, max_k_amide/loc_dict_res 
        local_dict_feed = dictbased_runner.get_dict(
            use_cartesian=True, mt_or_amide='amide',
            parameter_values_od=collections.OrderedDict({
                'T1a_ms': np.array(1000/data_feed_mt.R1a_V[_x, sli, _y]),
                'T2a_ms': np.array(1000/data_feed_mt.R2a_V[_x, sli, _y]),
                'B0_shift_ppm_map': np.array(data_feed_mt.B0_shift_ppm_map[_x, sli, _y]),
                'B1_fix_factor_map': np.array(data_feed_mt.B1_fix_factor_map[_x, sli, _y]),
                # NOTE: using MT estimated from the MT-MRF sequence previously, using NBMF
                'fc_gt_T': mt_tissue_param_est['fc_T'][_x, sli, _y],
                'kc_gt_T': mt_tissue_param_est['kc_T'][_x, sli, _y],
                'fb_gt_T': np.arange(_df, max_f_amide+_df, _df),
                'kb_gt_T': np.arange(_dk, max_k_amide+_dk, _dk)
                }),
            shape=[int(max_f_amide/_df), 1, int(max_k_amide/_dk)], mt_seq_txt_fname=kwargs['mt_seq_txt_fname'], larg_seq_txt_fname=kwargs['larg_seq_txt_fname']
            )
    else:
        signal = data_feed_mt.measured_normed_T[:, _x, sli, _y]                  
        _df, _dk = max_f_mt/loc_dict_res, max_k_mt/loc_dict_res
        local_dict_feed = dictbased_runner.get_dict(
            use_cartesian=True, mt_or_amide='mt', mt_sim_mode=mt_sim_mode,
            parameter_values_od=collections.OrderedDict({
                'T1a_ms': np.array(1000/data_feed_mt.R1a_V[_x, sli, _y]),
                'T2a_ms': np.array(1000/data_feed_mt.R2a_V[_x, sli, _y]),
                'B0_shift_ppm_map': np.array(data_feed_mt.B0_shift_ppm_map[_x, sli, _y]),
                'B1_fix_factor_map': np.array(data_feed_mt.B1_fix_factor_map[_x, sli, _y]),
                'fb_gt_T': [0.],
                'kb_gt_T': [0.],
                'fc_gt_T': np.arange(_df, max_f_mt+_df, _df),
                'kc_gt_T': np.arange(_dk, max_k_mt+_dk, _dk)
                }),
            shape=[int(max_f_mt/_df), 1, int(max_k_mt/_dk)], mt_seq_txt_fname=kwargs['mt_seq_txt_fname'], larg_seq_txt_fname=kwargs['larg_seq_txt_fname']
            )
    
    dict_signal = local_dict_feed.normalize(local_dict_feed.measured_normed_T, 'l2')[:,:,0,:]
    signal = local_dict_feed.normalize(signal, 'l2')

    dot_prod = np.transpose(dict_signal, [1,2,0]) @ signal  
    
    # -- !! For normalized vectors, the Dot-product diff from 1 is ~ half the Euclidian distance squared,
    #       aka sum(error^2) = n*MSE (MSE: mean square error), aka squared NRMSE of pre-normalized vectors
    #       (NRMSE: Normalized-Root-Mean-Square)
    # Details:
    #   Assume normalized vectors u, v (|u|=|v|=1), the dot product is ~cos(a) (alpha=a : angle between u and v)
    #   Hence, the NRMSE is ~sin(a) ~ 2*sin(a/2) ~ 2*sqrt((1-cos(a))/2) = sqrt(2*(1-dot_prod))
    #   Or in vector notation:  nrmse^2 = nmse = <v-u, v-u> = 2-2<v,u> = 2*(1-dot_prod)
    #   This assumes definition of NRMSE with "normalization" (MSE-->NMSE) as dividing by the norm of the signal 
    #    (in contrast to normalization by N*std or N*range which is also common)
    nrmse = np.sqrt(2*(1-dot_prod))             
    # ----------
    f_best = np.nanargmin(nrmse) // nrmse.shape[1] * _df*100
    k_best = np.nanargmin(nrmse) %  nrmse.shape[1] * _dk        
    
    # Optional visualization (see BCK for more obsoleted options), for perhaps debugging
    #   current visualization is in the viz_posteriors function which processes the NRMSE map further into posterior PDF and CRs
    if do_plot:
        extent = [0, max_f_mt*100, 0, max_k_mt]  
        area_fill = False 
        if area_fill:
            nrmse[np.logical_and(nrmse > 1.5*np.min(nrmse), nrmse>0.025)] = np.nan
            img = ax.imshow(nrmse, origin='lower', aspect='auto', extent=
                            extent, vmin=0.0, vmax=0.075)
            cbar = plt.colorbar(img, ax=ax, fraction=0.3, pad=0.01)
            cbar.set_label('NRMSE', labelpad=-45, y=.5, rotation=90, fontweight='bold')            
        nrmse_perc = nrmse*100
        levels = [np.nanmin(nrmse_perc)*1.037, np.nanmin(nrmse_perc)*1.1, np.nanmin(nrmse_perc)*1.3]
        contours=ax.contour(
            nrmse_perc, 
            levels=levels, 
            colors=['orange', 'orange', 'orange'], 
            linewidths=[1, 2, 1],
            extent=extent
        )
        ax.set_xlim(0, max_f_mt*100)
        ax.set_ylim(0, max_k_mt) # nrmse.shape[0]*_dk)
        labels = plt.clabel(
            contours, fontsize=8, fmt="%.2f", #inline=True, 
        )
        for txt in labels:
            txt.set_backgroundcolor('none')  # remove white box
            txt.set_bbox(dict(facecolor='none', edgecolor='none'))  # more robust
        ax.plot([f_best], [k_best], 'b.', markersize=10, zorder=3)
        plt.tight_layout()
    
    return f_best, k_best, nrmse, _df, _dk


def viz_posteriors(
    f_est_nn, k_est_nn, cov_nnpred_scaled, width, height, angle,
    f_best_dotprod, k_best_dotprod, nrmse, _df, _dk, n_meas=30,
    is_amide=False, 
    ax=None, figsize=(6, 4), fontsize=14, 
    do_marginals=True, show_text=True, show_NN=True,
):
    """
        Compute and visualize the reference posterior distribution's Point Spread Function (PDF), 
        the 95% Confidence Region (CR), marginal univariate distributions of f,k . 
        Compare to the NN-predicted posterior distribution (if show_NN=True).

    Args:
        f_est_nn (_type_): The volume fraction estimate by neural network (in %)
        k_est_nn (_type_): The exchange rate estimate by neural network
        cov_nnpred_scaled (_type_): The NN-predicted covariance (in %,1/s visualization units)
        width (_type_): of NN-predicted uncertainty ellipse (95% CR)
        height (_type_): of NN-predicted uncertainty ellipse (95% CR)
        angle (_type_): of NN-predicted uncertainty ellipse (95% CR)
        f_best_dotprod (_type_): The volume fraction estimate by dot-product with a local dictionary
        k_best_dotprod (_type_): The exchange-rate estimate by dot-product with a local dictionary
        nrmse (_type_): The map of the measured/simulated Normalized Mean Square Error (nrmse) across the local dictionary
        _df (_type_): local dict step size in volume fraction
        _dk (_type_): local dict step size in exchange rate
        is_amide (bool, optional):  use the (b) pool
        ax (_type_, optional):   draw main plot in these axes if given. 
        figsize (tuple, optional): _description_. Defaults to (6, 4).
        fontsize (int, optional): _description_. Defaults to 14.
        do_marginals (bool, optional): _description_. Defaults to True.
        show_text (bool, optional): Print the Mahalanobises. Defaults to True.
        show_NN (bool, optional): Show the Neural Net results alongside Likelihood-Heatmap. Defaults to True.

    Returns:
        1,2 - Mahalanobis distances of NN-estimated posterior to best local-dict estimate and vice versa, 
        3 - the covariance emergent from the likelihood-heatmap, 
        4,5 - The confidence region area - (a) in const-pdf contour enclosing 95% of the heatmap (b) as product of univariate 95% CIs
        
    """    
    diff = np.array((f_est_nn - f_best_dotprod, k_est_nn - k_best_dotprod)) # / scales
    #mahalanobis_NNdist_FGbestMAP = np.sqrt( (diff/scales).T @ np.linalg.inv(_cov.T) @ (diff/scales) )
    mahalanobis_NNdist_FGbestMAP = np.sqrt( diff.T @ np.linalg.inv(cov_nnpred_scaled.T) @ diff )
    
    # !! assuming model describes signal up to AWGN, i.e. the residuals are i.i.d gaussian 
    #    (AWGN: additive white gaussian noise),    
    sigma_est = np.min(nrmse)
    prob = np.exp(- n_meas * nrmse**2 / sigma_est**2 / 2)
    prob /= np.sum(prob)
    
    if ax is None:
        if not do_marginals:
            fig, ax = plt.subplots(1, 1, figsize=figsize)        
        else:
            fig = plt.figure(figsize=figsize)                    
            gs = gridspec.GridSpec(19, 20)
            # Main 2D density plot
            ax = fig.add_subplot(gs[2:, 2:-1])
            # Marginal for X (bottom)
            ax_marg_x = fig.add_subplot(gs[:2, 2:-1], sharex=ax)
            # Marginal for Y (left)
            ax_marg_y = fig.add_subplot(gs[2:, :2], sharey=ax)
    
    log_likelihood = - n_meas * (nrmse**2 - np.min(nrmse)**2 ) / (2 * sigma_est**2)
    # ! not normalized in any case so no real meaning to the  - np.min(nrmse)**2, TODO remove 2nd term
    
    # (!) Now we compute the posterior distribution as point-spread function (PDF), 
    #     using the Bayesian relation to likelihood and assuming a flat prior
    prior = 1
    pdf = np.exp(log_likelihood) * prior
    pdf /= np.sum(pdf)
    if False: # debug the route to PDF if needed
        plt.imshow(nrmse, origin='lower', aspect='auto', extent=[0, _df*100*100, 0, _dk*100], vmin=np.min(nrmse), vmax=1.5*np.min(nrmse), cmap='hot'); plt.colorbar()
        plt.imshow( np.clip(log_likelihood, -4, 0), origin='lower', aspect='auto', extent=[0, _df*100*100, 0, _dk*100], cmap='hot'); plt.colorbar()
    # transpose so that f is the x-axis and k is the y-axis
    imsh = ax.imshow(pdf.T/np.max(pdf), origin='lower', aspect='auto', extent=[0, _df*100*100, 0, _dk*100], cmap='hot'); 
    if not do_marginals:
        cbar = plt.colorbar(); 
    else:
        ax_cbar = fig.add_subplot(gs[2:, 19])
        cbar = fig.colorbar(imsh, cax=ax_cbar)        
    cbar.set_label('Full-grid posterior PDF (a.u.)', fontsize=12)

    if show_NN:
        # ! the API want angle w.r.t y-axis (hence "height" term for the long axis),
        #   but we have it computed as standard atan(y/x) so need to reduce 90
        ellipse = Ellipse(
            xy=(f_est_nn, k_est_nn), width=width, height=height, angle=angle-90,
            edgecolor='c', facecolor='none', linewidth=2) # .5, zorder=0, alpha=0.5)
        ax.add_patch(ellipse)
        nn_point, = ax.plot([f_est_nn, f_est_nn], 
                            [k_est_nn, k_est_nn], 'X', color='c', markersize=8)    
    
    ax.set_xlabel('f$_{s}$ (%)' if is_amide else 'f$_{ss}$ (%)', fontsize=fontsize)
    ax.set_ylabel('k$_{sw}$ (s$^{-1}$)' if is_amide else 'k$_{ssw}$ (s$^{-1}$)', fontsize=fontsize)
        
    ref_point, = ax.plot([f_best_dotprod, f_best_dotprod], [k_best_dotprod, k_best_dotprod], 'X', color='k', markersize=8)    

    nrmse_over_min_thresholds = np.arange(1.02, 1.2, 0.005)
    posterior_cdf = []
    for th in nrmse_over_min_thresholds: # np.arange(1.02, 1.25, 0.005):   # [1.04, 1.09, 1.2]:
        posterior_cdf.append(np.sum(pdf[nrmse < th*np.min(nrmse)]))

    nrmse_95cr_th_over_min= nrmse_over_min_thresholds[np.argmin(abs(np.array(posterior_cdf) - 0.95))]
    nrmse_95cr_th = np.min(nrmse) * nrmse_95cr_th_over_min    
    cont = ax.contour(
            nrmse.T,  # transpose so that f at axis=0 goes to horizontal x-axis
            levels=[nrmse_95cr_th], 
            colors=['magenta'], 
            linewidths=[2],
            extent=[0, _df*100*100, 0, _dk*100]
        )
    CR_area = np.sum(nrmse < nrmse_95cr_th)  # in samples 
    
    if show_NN:        
        ax.legend(
            [ref_point, mpatches.Patch(color='m'), nn_point, ellipse], 
            [
                'Full-grid posterior: mode (MAP)',
                'Full-grid posterior: 95% CR',
                r'NN-inference: MAP estimate ($\hat{\mu}_{\theta}$)',  # \leftarrow
                'NN-inference:  95% CR estimate '+r'(from $\boldsymbol{\hat{\Sigma}_{\theta}}$)'      
            ],
            loc='upper right', fontsize=10
        )
    else:
        ax.legend(
            [ref_point, mpatches.Patch(color='m')], 
            [
                'Full-grid posterior: mode (MAP)',
                'Full-grid posterior: 95% CR',
            ],
            loc='upper right', fontsize=10
        )
        
    df_grid = 100*_df*np.arange(0, 100)[:   , None]
    dk_grid =     _dk*np.arange(0, 100)[None, :   ]

    if do_marginals:
        marg_f = np.sum(pdf, axis=1)
        marg_f_cdf = np.cumsum(marg_f)
        marg_k = np.sum(pdf, axis=0)
        marg_k_cdf = np.cumsum(marg_k)
        ax_marg_x.fill_between(x=100*_df*np.arange(0, 100), y1=[0]*100, y2=marg_f, color='gray', alpha=0.5)                        
        ax_marg_y.fill_betweenx(y=_dk*np.arange(0, 100), x2=marg_k, x1=[0]*100, color='gray', alpha=0.5)                
        ax_marg_y.invert_xaxis()                
        utils.remove_spines(ax_marg_y)
        utils.remove_spines(ax_marg_x)        
        f_975 = _df*100*np.searchsorted(marg_f_cdf, 97.5/100)
        f_025 = _df*100*np.searchsorted(marg_f_cdf, 2.5/100)                                     
        ax.plot((f_025, f_025), (0, _dk*100), 'w--', alpha=0.2)
        ax.plot((f_975, f_975), (0, _dk*100), 'w--', alpha=0.2)
        k_975 = _dk*np.searchsorted(marg_k_cdf, 97.5/100)
        k_025 = _dk*np.searchsorted(marg_k_cdf, 2.5/100)              
        ax.plot((0, _df*100*100), (k_025, k_025), 'w--', alpha=0.2)
        ax.plot((0, _df*100*100), (k_975, k_975), 'w--', alpha=0.2)        
        CIk_x_CIf = (k_975 - k_025) * (f_975 - f_025) / (100 * _df * _dk)        
        
    # === FG: Grid --> Gauss ===    
    f_posterior_mu, k_posterior_mu = np.sum(pdf * df_grid), np.sum(pdf * dk_grid) # _dk*np.arange(0, 100)[:, None])
    # full-grid
    delta_vec_grid = np.stack((df_grid.repeat(100, axis=1) - f_posterior_mu, dk_grid.repeat(100, axis=0) - k_posterior_mu), axis=-1) #  [...,None]
    posterior_cov = (pdf[..., None, None] * delta_vec_grid[..., None] @ delta_vec_grid[..., None, :]).sum(axis=0).sum(axis=0)
    mahalanobis_NNmu_FGdist = np.sqrt( diff.T @ np.linalg.inv(posterior_cov) @ diff )
    
    if show_text:    
        ax.text(.1, 5, 'Mahalanobis distances: \n' + \
            r'$M(\hat{\theta}_{NN}, P_{FG})$=' + f'{mahalanobis_NNmu_FGdist:.1f}'+ \
            r'; $M(\hat{\theta}_{FG}, P_{NN})$=' + f'{mahalanobis_NNdist_FGbestMAP:.1f}',
            color='w'
        )
    return [mahalanobis_NNmu_FGdist, mahalanobis_NNdist_FGbestMAP, posterior_cov, CR_area, CIk_x_CIf]    


def simple_error_map(
        err_mt, err_amide, do_amide=False,
    ):
    '''Visualize the error maps in a slightly more informative way than the R^2 maps in the NBMF paper
    '''
    fig, axes = plt.subplots(1, 2, figsize=(9,4))
    sli = 20
    log_bins = np.logspace(np.log10(1.), np.log10(14), num=12)/100  # logarithmic
    # log_bins = np.arange(0., 0.14, 0.01)  # Similar to R^2 coloring in paper
    norm = mcolors.BoundaryNorm(log_bins, ncolors=plt.get_cmap('hot_r').N, clip=True)

    cmap_nrmse = plt.cm.get_cmap("YlOrRd").copy();  cmap_nrmse.set_bad('1.0')
    axes[0].set_title('MT')
    axes[1].set_title('AMIDE')
    img0 = axes[0].imshow(err_mt[:,sli,:], norm=norm, cmap=cmap_nrmse) #'hot_r'); # plt.colorbar(img0)  # vmin=0, vmax=0.1,
    if do_amide:
        img1 = axes[1].imshow(err_amide[:,sli,:], norm=norm, cmap=cmap_nrmse)
    else:
        img1 = axes[1].imshow(np.zeros_like(err_mt[:,sli,:]), norm=norm, cmap=cmap_nrmse)
    cbar = plt.colorbar(img1)
    cbar.set_ticks(log_bins)
    cbar.set_ticklabels([f"{v*100:.1f}%" for v in log_bins])
    cbar.set_label('NRMSE (%)')
            
            
# ================= OLD =========================

def viz_cov(
        f_sigma_slice, f_val_slice, k_sigma_slice, k_val_slice,
        df_c_0_slice, dk_ca_0_slice, df_c_1_slice, dk_ca_1_slice,
        f_total, k_total
    ):
    fig, axes = plt.subplots(2, 5, figsize=(18, 6))

    img = axes[0,0].imshow(f_sigma_slice/(1e-6+f_val_slice), cmap='hot_r', vmin=0, vmax=0.25) # if not explore_amide_uncertainty else 1); 
    cbar = cbarhist(img, f_sigma_slice/(1e-6+f_val_slice), axes[0,0])
    cbar.set_label(r'$\sigma / \mu\ [$f$_{ss}$]', fontsize=14)  
    img = axes[1,0].imshow(k_sigma_slice/k_val_slice, cmap='hot_r', vmin=0, vmax=0.25) # if not explore_amide_uncertainty else 1, cmap='hot'); 
    cbar = cbarhist(img, k_sigma_slice/k_val_slice, axes[1,0]) # cbar=plt.colorbar(img); 
    cbar.set_label(r'$\sigma / \mu\ [$k$_{ssw}]$', fontsize=14)

    img = axes[0,1].imshow(f_val_slice, vmin=0); 
    cbar = cbarhist(img, f_val_slice, axes[0, 1]) 
    cbar.set_label(r'$\mu\ [$f$_{ss}$] (%)', fontsize=14)
    img = axes[1,1].imshow(k_val_slice, vmin=0, cmap='magma'); 
    cbar = cbarhist(img, k_val_slice, axes[1, 1]) 
    cbar.set_label(r'$\mu\ [$k$_{ssw}]$ (s$^{-1})$', fontsize=14)

    img = axes[0,2].imshow(f_sigma_slice, vmin=0, vmax=np.nanmax(f_val_slice)/6); 
    cbar = cbarhist(img, f_sigma_slice, axes[0, 2]) 
    cbar.set_label(r'$\sigma\ [$f$_{ss}$] (%)', fontsize=14)
    img = axes[1,2].imshow(k_sigma_slice, vmin=0, vmax=np.nanmax(k_val_slice)/6, cmap='magma'); 
    cbar = cbarhist(img, k_sigma_slice, axes[1, 2]) 
    cbar.set_label(r'$\sigma\ [$k$_{ssw}]$ (s$^{-1})$', fontsize=14)

    img = axes[0,3].imshow(100*df_c_0_slice, vmin=-np.nanmax(100*f_total), vmax=np.nanmax(100*f_total), cmap='bwr');   
    cbar = cbarhist(img, 100*df_c_0_slice, axes[0, 3]) 
    cbar.set_label(r'$\sigma_{PC0}\ [$f$_{ss}$] (%)', fontsize=14)
    img = axes[1,3].imshow(dk_ca_0_slice, vmin=-np.nanmax(k_total), vmax=np.nanmax(k_total), cmap='bwr');  
    cbar = cbarhist(img, dk_ca_0_slice, axes[1, 3]) 
    cbar.set_label(r'$\sigma_{PC0}\ [$k$_{ssw}$] (%)', fontsize=14)

    img = axes[0, 4].imshow(100*df_c_1_slice, vmin=-np.nanmax(100*f_total), vmax=np.nanmax(100*f_total), cmap='bwr');   
    cbar = cbarhist(img, 100*df_c_1_slice, axes[0, 4]) 
    cbar.set_label(r'$\sigma_{PC1}\ [$f$_{ss}$] (%)', fontsize=14)
    img = axes[1, 4].imshow(dk_ca_1_slice, vmin=-np.nanmax(k_total), vmax=np.nanmax(k_total), cmap='bwr');  
    cbar = cbarhist(img, dk_ca_1_slice, axes[1, 4]) 
    cbar.set_label(r'$\sigma_{PC1}\ [$k$_{ssw}$] (%)', fontsize=14)

    #img = axes[0,4].imshow(100*df_c_1[:,sli,:], vmin=-np.nanmax(100*f_total), vmax=np.nanmax(100*f_total), cmap='bwr');   cbar=plt.colorbar(img); cbar.set_label(r'$\Delta$f$_{ss}$')
    #img = axes[1,4].imshow(dk_ca_1[:,sli,:], vmin=-np.nanmax(k_total), vmax=np.nanmax(k_total), cmap='bwr');  cbar=plt.colorbar(img); cbar.set_label(r'$\Delta$k$_{ssw}$')
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        
        
def farthest_point_sampling(f_val, k_val, num_samples=40):
    # Flatten the arrays and stack them as coordinate pairs
    coords = np.column_stack((f_val.flatten(), k_val.flatten()))

    # Remove NaN values
    valid_coords = coords[~np.isnan(coords).any(axis=1)]

    # Initialize the list of selected points with a random point
    selected_points = [valid_coords[np.random.choice(len(valid_coords))]]

    # Farthest Point Sampling
    for _ in range(num_samples - 1):
        _, min_distances = pairwise_distances_argmin_min(valid_coords, np.array(selected_points))
        next_point = valid_coords[np.argmax(min_distances)]
        selected_points.append(next_point)

    # Convert selected points back to indices
    selected_indices = [(np.where((f_val == point[0]) & (k_val == point[1]))[0][0], 
                        np.where((f_val == point[0]) & (k_val == point[1]))[1][0]) 
                        for point in selected_points]
    return selected_indices

    # # === NN: Gauss --> Grid ===
    # f_mu_NNest, k_mu_NNest = f_est_nn[_x, sli, _y], k_est_nn[_x, sli, _y]   # f_val[_x, sli, _y]/100
    # delta_vec_grid = np.stack((df_grid.repeat(100, axis=0) - f_mu_NNest, dk_grid.repeat(100, axis=1) - k_mu_NNest), axis=-1) #  [...,None]
    # #NN_cov_scaled = scales[None, :] * _cov[_x, sli, _y] * scales[:, None]
    # NN_cov_scaled = np.array(scales[None, :] * _cov[_x, sli, _y] * scales[:, None])

    # #print(NN_cov_scaled.shape, delta_vec_grid.shape)
    # nn_maha2 = delta_vec_grid[..., None, :] @ np.linalg.inv(NN_cov_scaled)[None, None, ...] @ delta_vec_grid[..., None]
    # nn_pdf = np.exp(-0.5 * nn_maha2.squeeze())
    # nn_pdf /= np.sum(nn_pdf)

    # if False:
    #     plt.figure()
    #     plt.imshow(nn_pdf, origin='lower', aspect='auto', extent=[0, _df*100*100, 0, _dk*100], cmap='hot')
        
            
def plot_ellipse(
        ax, _x, _y, sli, f_val, k_val, df_c_0, dk_ca_0, df_c_1, dk_ca_1, 
        amide=False, color='r', do_minor_ellipses=False, fontsize=14
    ): #), u, s):
    pd0 = np.array([100*df_c_0[_x, sli, _y], dk_ca_0[_x, sli, _y]])
    pd1 = np.array([100*df_c_1[_x, sli, _y], dk_ca_1[_x, sli, _y]])
    if np.linalg.norm(pd0) > np.linalg.norm(pd1):
        pd0, pd1 = pd1, pd0
    # (!) taking pd1 (2nd eigenvector) as the "main axis" (TODO do we need to sort by size?)
    angle = np.degrees(np.arctan2(pd1[1], pd1[0]))
    # CR of +-2*sigma hence 2*2 for total width
    height = np.linalg.norm(pd1) * 2 * 2 
    # (!) Need to re-orthogonilize after scaling
    width = np.linalg.norm(pd0 - (pd0.T@pd1)*pd1/np.linalg.norm(pd1)**2) * 2 * 2.45  # +- chisquare(2, 0.95) (=2.45)
    # print(angle, height, width, pd0, pd1)    
    # dfk = np.array([0, 1])[None, None, None, :, None] 
    # dfk = u @ jnp.sqrt(s) @ dfk
    _mean = [f_val[_x, sli, _y ], k_val[_x, sli, _y ]]
    # (!) angle=0 is vertical (hence "height"), but atan(y,x)=atan(y/x) is w horizontal
    
    maj_ellipse = ax.add_patch(Ellipse(xy=_mean, width=width, height=height, angle=angle-90, edgecolor=color, facecolor='none', linewidth=2, zorder=2))
    if do_minor_ellipses:  # sigma and 3*sigma in addition to 2*sigma ("95% CR")
        ax.add_patch(Ellipse(xy=_mean, width=width/2, height=height/2, angle=angle-90, edgecolor=color, facecolor='none',  linewidth=0.5, zorder=2)) #  alpha=0.5,
        ax.add_patch(Ellipse(xy=_mean, width=width*3/2, height=height*3/2, angle=angle-90, edgecolor=color, facecolor='none',  linewidth=0.5, zorder=2)) #  alpha=0.5,
    do_arrow = False
    if do_arrow:
        for vec in (pd0, pd1):
            arrow = Arrow(x=f_val[_x, sli, _y] , y=k_val[_x, sli, _y], dx=vec[0], dy=vec[1], width=.1, color=color, alpha=0.5)
            arrow_start = Arrow(x=f_val[_x, sli, _y], y=k_val[_x, sli, _y], dx=-vec[0], dy=-vec[1], width=.1, color=color, alpha=0.5)
            ax.add_patch(arrow_start)
            ax.add_patch(arrow)    
    ax.set_xlim(0, 30 if not amide else 1.2)
    ax.set_ylim(0, 100 if not amide else 600)
    ax.set_xlabel('f$_{s}$ (%)' if amide else 'f$_{ss}$ (%)', fontsize=fontsize)
    ax.set_ylabel('k$_{sw}$ (s$^{-1}$)' if amide else 'k$_{ssw}$ (s$^{-1}$)', fontsize=fontsize)
    return maj_ellipse
