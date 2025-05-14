import numpy as np, jax
import matplotlib.pyplot as plt
import copy
from dataclasses import asdict, dataclass
import logging
import time
import collections
import sys, os, gc

import utils
import data
import train
import net
import infer


@dataclass
class PipelineConfig:
    drop_first = False
#TODO: Ron change when you do config
    add_noise_to_signal = 1e-3
    # amide_mindelta = .15    
    amide_patience = 1
    mt_patience = 40

    mt_steps = 2000
    mt_lr = 1e-3           # Sep30 (B) exp (was 3e-3.. and before even 1e-2)
    amide_steps = 2000
    amide_lr = 1e-3
    
    # batch sizes (set according to available GPU memory)   
    mt_train_slw = 5    
    mt_test_slw = 5  
    amide_test_slw = 2

    infer_config = infer.InferConfig(
        kb_scale_fact = 102,
        fb_scale_fact = 2 / 100,
        use_cfsskss_inp = True
    )
    
    train_config = train.TrainConfig(
        use_shuffled_sampler = True,
        weight_decay = 1e-2,
        hidden_width = 256,   
        std_up_fact = 0.8,
        hidden_layers = 2,
    )
    
pipeline_config = PipelineConfig()


def run_train(
    brain2train_mt, brain2train_amide=None, train_config=None, infer_config=None,  # drop_first=False,
    mt_sim_mode='isar2_c', do_amide=True, mode='self_supervised',
    ckpt_folder='ckpts', ckptsfx='', logger=None
):
    """ Main training function
    """
    logger = logger or logging.getLogger(__name__)
    old_config = copy.copy(train.train_config)
    
    data.SlicesFeed.add_noise_to_signal = pipeline_config.add_noise_to_signal
    brain2train_mt.slw = min(pipeline_config.mt_train_slw, brain2train_mt.shape[1])
    
    train.train_config = train_config = train_config or pipeline_config.train_config
    infer.infer_config = infer_config = infer_config or pipeline_config.infer_config

    train.train_config.patience = pipeline_config.mt_patience
    t0 = time.time()
    logger.info("..Starting MT training..")
    model_state, loss_trend_mt, net_kwargs = \
        train.train(brain2train_mt, model_state=None, pool2predict='c', logger=logger,
                    mode=mode, simulation_mode=mt_sim_mode,
                    steps=pipeline_config.mt_steps, lr=pipeline_config.mt_lr)
    t1 = time.time()
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    np.save(f'{ckpt_folder}/mt{ckptsfx}_loss_trend', np.array(loss_trend_mt))
    predictor_mt = net.state2predictor(model_state)
    tc_dict = asdict(train.train_config)
    tc_dict = {k: v for k, v in tc_dict.items() if type(v) != str}  # WA jax bug
    # Note: ideally, infer_config should be also stored alongside checkpoint & used to ensure train/test consistency
    # in pre/post-processing (i.e. network's outputs' interpretation..)
    net.save_ckpt(
        model_state,
        config={'train_cfg': tc_dict, 'net_cfg': net_kwargs},
        folder=f'{ckpt_folder}/mt{ckptsfx}/',
        step=666)  # just a stub, need to upgrade by extracting correct step number

    if do_amide:
        brain2train_mt.ds = 1        
        brain2train_mt.slw = min(pipeline_config.mt_test_slw, brain2train_mt.shape[1])
        mt_tissue_params_est, nn_pred_signal_normed_np = infer.infer(
            brain2train_mt, nn_predictor=predictor_mt, 
            do_forward=False, pool2predict='c'
            )
        brain2train_amide.fc_gt_T = mt_tissue_params_est['fc_T']
        brain2train_amide.kc_gt_T = mt_tissue_params_est['kc_T']

        # Work around GPU memory limitation: ds=2, slw=1 is the max 
        # x40 less voxels than with slw=10, ds=1.  
        brain2train_amide.ds = 2
        brain2train_amide.slw = 1
        # Less patience to speed up the process        
        train.train_config.patience = pipeline_config.amide_patience

        t2 = time.time()
        logger.info("..Starting AMIDE training for..")
        model_state, loss_trend_amide, net_kwargs = \
            train.train(brain2train_amide, model_state=None, pool2predict='b',
                        mode=mode, simulation_mode='expm_bmmat', logger=logger,
                        steps=pipeline_config.amide_steps, lr=pipeline_config.amide_lr)
        t3 = time.time()
        np.save(f'{ckpt_folder}/amide{ckptsfx}_loss_trend',
                np.array(loss_trend_amide))
        tc_dict = asdict(train.train_config)
        tc_dict = {k: v for k, v in tc_dict.items() if type(v) != str}  # WA jax bug
        net.save_ckpt(
            model_state,
            config={'train_cfg': tc_dict, 'net_cfg': net_kwargs},
            folder=f'{ckpt_folder}/amide{ckptsfx}/',
            step=666 
        )
        logger.info(
            f"Timings: t(MTtrain)={t1-t0}sec, t(AMIDEtrain)={t3-t2}sec, t(total-train)={t3-t2+t1-t0}sec, gross-total: {t3-t0}sec")
        predictor_amide = net.state2predictor(model_state)
    else:
        predictor_amide = None

    train.train_config = old_config

    return predictor_mt, predictor_amide


def finetune(data2train, ckpt_path):
    """ NOTE: a stub, not tested since initial POC 
    """
    train.train_config = pipeline_config.train_config()
    infer.infer_config = pipeline_config.infer_config()

    data.SlicesFeed.add_noise_to_signal = pipeline_config.add_noise_to_signal
    brain2train_mt = data.SlicesFeed(
        **data.get_brain_kwargs(data2train, mt_or_amide='mt', drop_first=False))
    mngr = net.get_ckpt_mngr(ckpt_path)
    restored = mngr.restore(mngr.latest_step())
    model_state_d = restored['model_state']
    config = restored['config']
    get_net_kwargs = config['net_cfg']  # ..what about rest of config?
    nnmodel, nnparams = net.get_net(**get_net_kwargs)
    model_state_d['apply_fn'] = nnmodel.apply
    ModelState = collections.namedtuple(
        'ModelState', ['apply_fn', 'params', 'batch_stats'])
    model_state = ModelState(
        nnmodel.apply, model_state_d['params'], model_state_d['batch_stats'])

    model_state, loss_trend_mt, net_kwargs = \
        train.train(brain2train_mt, model_state=model_state,
                    pool2predict='c', mode='self_supervised',
                    simulation_mode='isar2_c', steps=2000, lr=1e-2)
    predictor_mt = train.state2predictor(model_state)
    return predictor_mt


def errstr(err):
    return  \
        f"L2(err): {np.linalg.norm(err)/np.sqrt(err.size):.3f} " + \
        f"L1(err): {np.linalg.norm(err.flatten(), ord=1)/err.size:.3f}" \
        if type(err) != type(None) else ''


def transfer_and_plot(
    brain2test_mt, brain2test_amide,
    predictor_mt, predictor_amide, 
    do_forward=True, slices=None, 
    do_boxplots=True, mask_gray=None, mask_white=None,
    figsfolder='figs', figsfx='', logger=None
):
    """ 
    main testing and plotting func
    """
    logger = logger or logging.getLogger(__name__)
    transfer_res = transfer(brain2test_mt, brain2test_amide, mt_reconstructor=predictor_mt,
                            amide_reconstructor=predictor_amide, do_forward=do_forward)
    mt_tissue_param_est, amide_tissue_param_est = transfer_res[1], transfer_res[4]
    if not os.path.exists(figsfolder):
        os.makedirs(figsfolder)
    err_mt, err_amide = plot_slice_rows_wrapper(transfer_res, slices=slices,
                                                mt_fig_name=f'{figsfolder}/MTslices{figsfx}.png',
                                                amide_fig_name=f'{figsfolder}/AMIDEslices{figsfx}.png')
    if do_boxplots:
        boxplots_wrapper(
            mask_gray, mask_white,
            mt_tissue_param_est,
            amide_tissue_param_est,
            f'{figsfolder}/mt_box{figsfx}.png',
            f'{figsfolder}/amide_box{figsfx}.png', mask_th=0.9
        )

    logger.info(f'MT error analysis: {errstr(err_mt)}')
    logger.info(f'AMIDE error analysis: {errstr(err_amide)}')
    return transfer_res, err_mt, err_amide


def transfer(
    brain2test_mt, brain2test_amide, 
    mt_reconstructor, amide_reconstructor=None, 
    infer_config=None, do_forward=True, simulation_mode='expm_bmmat'
    ):
    infer.infer_config = infer_config = infer_config or pipeline_config.infer_config 
    brain2test_mt.ds = 1
    brain2test_mt.slw = pipeline_config.mt_test_slw

    mt_tissue_param_est, _mt_reconstructed_signal = \
        infer.infer(
            brain2test_mt, pool2predict='c',
            nn_predictor=mt_reconstructor, simulation_mode=simulation_mode,
            do_forward=do_forward
            )
    if amide_reconstructor is None:
        amide_tissue_param_est = _amide_reconstructed_signal = None
    else:
        brain2test_amide.ds = 1
        brain2test_amide.slw = pipeline_config.amide_test_slw  

        # ! Set the parameters of the MT pool to our estimation from fitting the MT protocol:
        brain2test_amide.fc_gt_T = mt_tissue_param_est['fc_T']
        brain2test_amide.kc_gt_T = mt_tissue_param_est['kc_T']

        amide_tissue_param_est, _amide_reconstructed_signal = \
            infer.infer(brain2test_amide, pool2predict='b',
                        nn_predictor=amide_reconstructor, simulation_mode='expm_bmmat',
                        do_forward=do_forward)

    return (brain2test_mt, mt_tissue_param_est, _mt_reconstructed_signal,
            brain2test_amide, amide_tissue_param_est, _amide_reconstructed_signal)    

def plot_slice_rows_wrapper(transfer_res,
                            mt_fig_name, amide_fig_name='1',                            
                            fss_lims=[4, 11], kss_lims=[40, 60], # TODO: change limits
                            fs_lims=[0.2, 0.7], ks_lims=[80, 100], # TODO: change limits
                            figsize=None, slices=None, do_err=True):

    brain2test_mt, mt_tissue_param_est, mt_reconstructed_signal, \
        brain2test_amide, amide_tissue_param_est, amide_reconstructed_signal = transfer_res

    fss_pred = mt_tissue_param_est[f'fc_T'] * brain2test_mt.roi_mask_nans * 100
    # print(f"fss_pred mean and std: {np.nanmean(fss_pred):.3f} std: {np.nanstd(fss_pred):.3f}")
    kss_pred = mt_tissue_param_est[f'kc_T'] * brain2test_mt.roi_mask_nans
    # print(f"kss_pred mean and std: {np.nanmean(kss_pred):.3f} std: {np.nanstd(kss_pred):.3f}")
    err_3d = np.linalg.norm(mt_reconstructed_signal - brain2test_mt.measured_normed_T,
                            axis=0, ord=2) * brain2test_mt.roi_mask_nans
    # ! nontrivial if signal is normed-by-first
    err_3d /= np.linalg.norm(brain2test_mt.measured_normed_T, axis=0, ord=2)
    err_3d[np.isnan(err_3d)] = 0

    utils.slice_row_plot(
        fss_pred, kss_pred, err_3d,
        fss_lims=fss_lims, kss_lims=kss_lims,
        figsize=figsize, slices=slices, do_err=do_err
        )

    plt.savefig(mt_fig_name, bbox_inches='tight')
    err_3d_mt = err_3d

    if amide_tissue_param_est is not None:
        # amide_tissue_param_est, amide_reconstructed_signal, _1, _2 = amide_res
        fss_pred = amide_tissue_param_est[f'fb_T'] * \
            brain2test_amide.roi_mask_nans * 100
        # print(f"fs_pred mean and std: {np.nanmean(fss_pred):.3f} std: {np.nanstd(fss_pred):.3f}")
        kss_pred = amide_tissue_param_est[f'kb_T'] * \
            brain2test_amide.roi_mask_nans
        # print(f"ksw_pred mean and std: {np.nanmean(kss_pred):.3f} std: {np.nanstd(kss_pred):.3f}")
        err_3d = np.linalg.norm(amide_reconstructed_signal - brain2test_amide.measured_normed_T,
                                axis=0, ord=2) * brain2test_amide.roi_mask_nans
        # ! nontrivial if signal is normed-by-first
        err_3d /= np.linalg.norm(
            brain2test_amide.measured_normed_T,
            axis=0, ord=2
            )
        err_3d[np.isnan(err_3d)] = 0
        full_texts = [
            'Amide volume fraction (%)', '$f_{s}$ (%)',
            'Amide exchange rate (Hz)', '$k_{sw}$ $(s^{-1})$',
            'signal reconstruction fidelity', '$R^2_{fit}$',
        ]
        simple_texts = ['', '$f_{s}$ (%)',
                        '', '$k_{sw}$ $(s^{-1})$',
                        '', '$R^2_{fit}$',
                        ]
        utils.slice_row_plot(
            fss_pred, kss_pred, err_3d, fss_lims=fs_lims, kss_lims=ks_lims,
            figsize=figsize, slices=slices, do_err=do_err, texts=simple_texts
        )  # texts=full_texts
        plt.savefig(amide_fig_name, bbox_inches='tight')
        # texts=)
        err_3d_amide = err_3d
    else:
        err_3d_amide = None

    return err_3d_mt, err_3d_amide,


def boxplots_wrapper(
    mask_gray, mask_white, mt_tissue_param_est, amide_tissue_param_est,
    mt_fig_name, amide_fig_name, mask_th=0.9
):
    """ Draw the boxplot comparing fitting results to literature
    """    
    mask_gray[mask_gray == 0] = np.nan
    mask_white[mask_white == 0] = np.nan

    boxplot_mt = utils.boxplot_white_vs_gray(
        mask_gray, mask_white,
        mt_tissue_param_est['fc_T'],
        mt_tissue_param_est['kc_T'],
        pool='MT',
        lit_f_wm_gm=[[13.9, 5], [6.2, 3.4], [
            11.2, 6.3], [9.4, 4.2], [18.7, 12.4]],
        lit_k_wm_gm=[[23.0, 40.0], [67.5, 63.5], [
            29, 40], [14, 35.1], [33.9, 49.1]],
        lit_names=('Stanitz 2005', 'Liu 2013', 'Heo 2019', 'Perlman 2022', 'Weigand-Whittier 2022'))[0]
    plt.savefig(mt_fig_name, bbox_inches='tight')

    if amide_tissue_param_est is not None:
        boxplot_amide = utils.boxplot_white_vs_gray(
            mask_gray, mask_white,
            amide_tissue_param_est['fb_T'],
            amide_tissue_param_est['kb_T'],
            pool='AMIDE',
            lit_f_wm_gm=[[.19, .24], [.22, .25], [.31, .32], [.1, .17]],
            lit_k_wm_gm=[[162, 365], [280, 280], [42.3, 35], [260, 130]],
            lit_names=('Heo 2019', 'Liu 2013', 'Perlman 2022', 'Carradus 2023')
            )[0]
        plt.savefig(amide_fig_name, bbox_inches='tight')

def get_logger(fname):    
    logging.basicConfig(        
        level=logging.INFO,        
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',  # Date format
        handlers=[
            logging.FileHandler(fname),  # Log to a file
            logging.StreamHandler()  # Log to the console
        ]
    )
    return logging.getLogger(__name__)    


def simple_infer_wrap(ds, nn_predictor, simulation_mode='expm_bmmat', visualize=False, sloi=5):    
    tissue_params, nn_pred_signal_normed_np = infer.infer(
        ds, nn_predictor=nn_predictor, 
        do_forward=True, simulation_mode=simulation_mode
        )
    nsignal_error = np.linalg.norm(
        nn_pred_signal_normed_np - 
        ds.measured_normed_T, axis=0, ord=2
        ) * ds.roi_mask_nans
    print(f'100*np.nanmean(err_2d): {100*np.nanmean(nsignal_error):.3f}')
    print(f'100*L2(err_2d): {100 * np.sqrt(np.nanmean(nsignal_error**2)):.3f}')
    if visualize:
        plt.figure(figsize=(10,2))
        plt.suptitle("signal pred/meas RRMSE (%) - MAP  /  HISTOGRAM")
        plt.subplot(1,2,1)
        plt.imshow(nsignal_error[:,sloi,:]*100)
        plt.colorbar()        
        plt.subplot(1,2,2)
        plt.hist(nsignal_error.flatten()*100, 32)        
        plt.xlabel('signal pred/meas RRMSE (%)')
    
    return tissue_params, nn_pred_signal_normed_np, nsignal_error

    
def VBMF_MT_run(brain2train_mt, dirname='vbmf', mt_steps=None, slices2plot=None):
    run_dirname = os.path.join(dirname, time.strftime('%B%d_%H%M')) 
    if not os.path.exists(run_dirname):
        os.makedirs(run_dirname)
    logger = get_logger(f'{dirname}/log.log')        
        
    infer.infer_config = pipeline_config.infer_config
    train.train_config = pipeline_config.train_config
    train.train_config.tp_noise = False  
    train.train_config.use_shuffled_sampler = False
    train.train_config.patience = pipeline_config.mt_patience
    mt_steps = mt_steps or pipeline_config.mt_steps
    
    if False:
        # Solved some issue when trying to VBMF the CEST (still didn't get good results);
        #  not needed for MT but leaving for the case one want to try for CEST again.
        jax.config.update("jax_enable_x64", True)
    
    mt_lr = 1e-2
    t0 = time.time()        
    # brain2train_mt.slw = np.min([pipeline_config.mt_train_slw, brain2train_mt.shape[1]])
    model_state, loss_trend_mt, net_kwargs = \
        train.train(brain2train_mt, model_state=None, pool2predict='c', logger=None,                                
                    mode='bloch_fitting', simulation_mode='isar2_c', steps=mt_steps, lr=mt_lr)
    logger.info(f"T(VBMF)={time.time()-t0}sec")
    # To evaluate, we set the fitted values as "ground truth" on a copy dataset
    brain2test_mt = copy.deepcopy(brain2train_mt)    
    train.bmfit_set_gt_from_model_state(brain2test_mt, model_state)  
    mt_tissue_param_est, _mt_reconstructed_signal = infer.infer(
        brain2test_mt, pool2predict='bc', 
        nn_predictor=None, simulation_mode='isar2_c', do_forward=True
        )  

    brain2test_amide = None; amide_tissue_param_est = _amide_reconstructed_signal = None
    transfer_res = (brain2test_mt, mt_tissue_param_est, _mt_reconstructed_signal, \
                    brain2test_amide, amide_tissue_param_est, _amide_reconstructed_signal)
        
    err_mt, err_amide = plot_slice_rows_wrapper(
        transfer_res, slices=slices2plot, mt_fig_name=run_dirname+'/mt_VBMF.png'
        )
    np.savez_compressed(run_dirname+'/err', err_mt_train=err_mt, mt_tissue_param_est=mt_tissue_param_est)
    return mt_tissue_param_est, err_mt

if __name__ == "__main__":
    VBMF_MT_run()
    # mt_repeated_run()  
