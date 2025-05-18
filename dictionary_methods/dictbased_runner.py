
import numpy as np, matplotlib.pyplot as plt, os, time, logging, shutil
import data, train, pipelines, infer, utils, net
import dictionary_methods.dict_matcher as dict_matcher

def get_dict(
    mt_or_amide='mt', mt_sim_mode='isar2_c', use_cartesian = False, lhs_size_x10k=40,
    import_NV=False, nv_mt_fname='mtd.npz', nv_amide_fname='aptd.npz', **kwargs
    ):
    # NOTE: lately tested mainly with MT, ideally should extend to amide, l-arg and test
    if not import_NV:
        if use_cartesian:
            bsf_lhsynth_mt = data.SlicesFeed.make_cartesian_sample(
                mt_or_amide=mt_or_amide, **kwargs
                ) 
        else:  # Latin Hypercube Sampling
            slices = lhs_size_x10k # if mt_or_amide=='mt' else lhs_size_x10k
            bsf_lhsynth_mt = data.SlicesFeed.make_lh_sample( 
                mt_or_amide=mt_or_amide, 
                shape=(100, slices, 100), **kwargs
            )   
        # Synthesize the signals:
        bsf_lhsynth_mt.slw = 20
        _, signal_3p = infer.infer(bsf_lhsynth_mt, simulation_mode=mt_sim_mode if mt_or_amide=='mt' else 'expm_bmmat') 
        bsf_lhsynth_mt.measured_normed_T = bsf_lhsynth_mt.normalize(signal_3p)
    
    else: # import from dictionary created with Vladimirov2025 code
        """
        tested once by creating mtd.npz as follows, using [https://github.com/momentum-laboratory/molecular-mrf/]
        In the run python -m sequential_nn_example.mouse with some modifications.
        1. sequential_nn_example/mouse.py::write_seq_defs -
            add B1=0 reference prescan after which come 15s delay, set B0=3, set pulsed (13,0.1/0.1) 
        2. sequential_nn_example/configs.py  - change grid (don't use small fss, some issue there..)
        3: breakpoint in sequential_nn_example/mouse.py, line 58: to then run in debugger:
            np.savez_compressed('/hosthome/alexf/nbmf/mtd', dictionary)
        note no B0,B1 inhomogeneities right now in that route, need to figure out if reviving this
        """
        nvdict = np.load(
            nv_mt_fname if mt_or_amide=='mt' else nv_amide_fname, 
            fix_imports=True, allow_pickle=True
            )['arr_0'].item()
        #nvdict['sig'] = nvdict['sig'] / np.linalg.norm(nvdict['sig'], axis=1)[-1, None]        
        shape = (88, 50, 90) # nvdict['sig'].reshape(31, *shape).shape
        bsf_lhsynth_mt = data.SlicesFeed.from_args(
            mt_or_amide=mt_or_amide,
            shape=shape, 
            T1a_ms=nvdict['t1w'].reshape(shape)*1000,
            T2a_ms=nvdict['t2w'].reshape(shape)*1000,
            fb_gt_T=0, kb_gt_T=0,
            fc_gt_T=nvdict['fs_0'].reshape(shape),
            kc_gt_T=nvdict['ksw_0'].reshape(shape),
            signal=nvdict['sig'].T.reshape(31,*shape),
        )

    return bsf_lhsynth_mt


def train_on_dict(bsf_lhsynth, pool2predict='c', epochs=50):
    infer.infer_config = pipelines.pipeline_config.infer_config
    infer.infer_config.use_cfsskss_inp = False
    # infer.infer_config.kc_scale_fact = 100
    train.train_config = pipelines.pipeline_config.train_config
    train.train_config.patience=100  
    train.train_config.tp_noise = False
    train.train_config.use_shuffled_sampler = True
    train.train_config.reglosstype = 'L2'
    bsf_lhsynth.slw = 1     
    bsf_lhsynth.add_noise_to_signal = 5e-3
    # the amp of loss is de-facto x sqrt(30), so roughly 2.5% 

    logger = logging.getLogger('train')
    steps = bsf_lhsynth.shape[1]*epochs
    model_state, loss_trend_mt, net_kwargs = \
            train.train(bsf_lhsynth, model_state=None, 
                        pool2predict=pool2predict, logger=logger,
                        mode='reference_supervised', steps=steps, lr=3e-3)    

    predictor = net.state2predictor(model_state)
    return predictor
    

def match_to_dict(bsf_lhsynth_mt, brain2test_mt, constrain_T1T2=True, constrain_B0B1=False, quicky_slicer=False):    
        
    l2normed_d = bsf_lhsynth_mt.measured_normed_T.reshape(31, -1) + 0.0
    l2normed_d /= np.linalg.norm(l2normed_d, axis=0)

    afdm = dict_matcher.DictMatcher(
        l2normed_d, 
        T1_dict=1000/bsf_lhsynth_mt.R1a_V.flatten(), 
        T2_dict=1000/bsf_lhsynth_mt.R2a_V.flatten(),
        B0_dict=bsf_lhsynth_mt.B0_shift_ppm_map.flatten(),
        B1_dict=bsf_lhsynth_mt.B1_fix_factor_map.flatten(),
    ) 
    
    if quicky_slicer:
        demo_slices = np.int32(np.linspace(5,40,10))[3:-2]
        for k, v in brain2test_mt.__dict__.items():
            if type(v) == np.ndarray:
                print(k, v.shape, brain2test_mt.shape, brain2test_mt.shape==list(v.shape))
                if v.shape == tuple(brain2test_mt.shape):
                    brain2test_mt.__dict__[k] = v[:, demo_slices, :]
                elif v.shape[1:] == tuple(brain2test_mt.shape):
                    brain2test_mt.__dict__[k] = v[:, :, demo_slices, :]
        brain2test_mt.shape = [brain2test_mt.shape[0], 5, brain2test_mt.shape[2]]
    brain2test_mt.measured_normed_T.shape
    
    l2normed_signal = brain2test_mt.measured_normed_T + 0.
    l2normed_signal /= np.linalg.norm(l2normed_signal, axis=0)

    B0constraint = brain2test_mt.B0_shift_ppm_map if constrain_B0B1 else None
    B1constraint = brain2test_mt.B1_fix_factor_map if constrain_B0B1 else None
    
    afdm_match_res = afdm.match(
        l2normed_signal, 
        fs_grid=bsf_lhsynth_mt.fc_gt_T, 
        ks_grid=bsf_lhsynth_mt.kc_gt_T,
        T1constraint=1000/brain2test_mt.R1a_V if constrain_T1T2 else None,
        T2constraint=1000/brain2test_mt.R2a_V if constrain_T1T2 else None,
        B0constraint=B0constraint,
        B1constraint=B1constraint
    )
    
    fss_pred = afdm_match_res[0] * 100 * brain2test_mt.roi_mask_nans
    kss_pred = afdm_match_res[1] *       brain2test_mt.roi_mask_nans
    best_match = l2normed_d[:, afdm_match_res[2]] * brain2test_mt.roi_mask_nans
    best_match /= np.max(best_match, axis=0) 

    err_3d = np.linalg.norm(bsf_lhsynth_mt.normalize(best_match, norm_type='first') - 
                            bsf_lhsynth_mt.normalize(brain2test_mt.measured_normed_T, norm_type='first'), axis=0, ord=2)
    err_3d /= np.linalg.norm(bsf_lhsynth_mt.normalize(brain2test_mt.measured_normed_T, norm_type='first'), axis=0, ord=2)
    err_3d[np.isnan(err_3d)] = 0
    
    return fss_pred, kss_pred, best_match, err_3d


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