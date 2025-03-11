
"""
THREE-POOL SIMULATION OF RECTANGULAR-PULSE CEST/MT MRF SEQUENCES USING JAX 

===> Saturation protocol source:
MT:  https://github.com/kherz/pulseq-cest-library/tree/master/seq-library/MRF_CEST_MT_3T_003_13SL_DC50_2500ms
Larg:https://github.com/kherz/pulseq-cest-library/blob/master/seq-library/MRF_CEST_LArginine_3T_002_13SL_DC50_2500ms/MRF_CEST_LArginine_3T_002_13SL_DC50_2500ms.m

===> Acquisition protocol source (3D EPI) --- 
WriteEpiSeqForsimulation.m
% This is the script used to generate a simulation.seq file for the 3D EPI readout.
% The parameters relevant are TR, pulse duration and flip angle. The TR here, is the time between two shots.
nshots = 45;       !! 44 on actual scanner?
tr = 28.370*1e-3;  !! 26.25+2 ?
tp = 2e-3;
tpause = tr-tp;
fa = deg2rad(15);
seq = mr.Sequence;
flippulse = mr.makeBlockPulse(fa, 'Duration', tp);
relaxTime = mr.makeDelay(tpause);
for ii= 1:nshots
    seq.addBlock(flippulse);
    seq.addBlock(relaxTime);
    if ii == 1
        seq.addBlock(mr.makeAdc(1, 'Duration', 1e-3));
    end
end
seq.write('EPI_SIM.seq')
"""

import jax, jax.numpy as jnp

double_precision = False

## --- following WriteEpiSeqForsimulation.m
t_flip_period = 28.37e-3   # pulse + delay
flip_angle = 15 * jnp.pi / 180
flip_cos = jnp.cos(flip_angle)
num_flip_pulses = 45
## ---

tpulse_DEF = 0.1
tdelay_DEF = 0.1
n_pulses_DEF = 13
TRs_DEF = [3.5]*30
TSATs_DEF = [2.5]*30


def get_SL_rot(w1, dwa, sign=1):
    """ Spin Lock: simulating the on/off hard pulses as ideal rotations (good approximation for 1ms pulse).
        Assuming B1||X, rotate around the Y axis, with FA=theta=atan(Beff/B1),
        to get magnetization aligned with the effective field in the rotating system (B1+dB combined)

        Arguments: 
            w1 - the B1 precession angular freq (rad)
            dwa - the off-resonance freq (rad)
            sign - pos for "SL-on", neg for "SL-off"
        Assumptions:  (TODO constantly re-evaluate:)
            - hard pulse, affects all pools (a,b,c) despite PPM diff (even MT?!)
            - phase cycling not important, saturation pulse is phase matched,
                so the SL on/off is simply modeled as rotation with +- FA
    """
    cos_th = jnp.sqrt(1 / (1 + (w1/dwa)**2))
    sin_th = jnp.sqrt(1 / (1 + (dwa/w1)**2))    
    
    # ! The 180 phase if w(RF) higher/lower than w0 (ppm pos/neg) is essential
    sin_th *= jnp.sign(dwa)
    
    ZE = jnp.zeros(dwa.shape)
    ID = jnp.ones(dwa.shape)
    s_sin_th = sign*sin_th
    
    SL_rot_mat = [
            [cos_th,    ZE,      s_sin_th,   ZE,        ZE,  ZE,       ZE    ,    ZE,  ZE      ], 
            [ZE,        ID,      ZE,         ZE,        ZE,  ZE,       ZE    ,    ZE,  ZE      ], 
            [-s_sin_th, ZE,      cos_th,     ZE,        ZE,  ZE,       ZE    ,    ZE,  ZE      ], 
            [ZE,        ZE,      ZE,         cos_th,    ZE,  s_sin_th, ZE    ,    ZE,  ZE      ],
            [ZE,        ZE,      ZE,         ZE,        ID,  ZE,       ZE    ,    ZE,  ZE      ], 
            [ZE,        ZE,      ZE,         -s_sin_th, ZE,  cos_th,   ZE    ,    ZE,  ZE      ],          
            [ZE,        ZE,      ZE,         ZE,        ZE,  ZE,       cos_th,    ZE,  s_sin_th],
            [ZE,        ZE,      ZE,         ZE,        ZE,  ZE,       ZE,        ID,  ZE      ],
            [ZE,        ZE,      ZE,         ZE,        ZE,  ZE,       -s_sin_th, ZE,  cos_th  ],
    ]
    # Reshape: from 2D matrix of 3D-array entries, into 5D tensor (spatial + BM-2D)
    SL_rot_mat_T = jnp.concatenate([jnp.concatenate([entry[..., None, None] \
                                                    for entry in row], axis=3) \
                                    for row in SL_rot_mat], axis=2)

    return SL_rot_mat_T


def get_crusher_mat_T():
    crusher_mat = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ]

    return jnp.array(crusher_mat)    


# Deprecated:  7x7 BM matrix, using only Z for MT. TODO re-enable?
# Zaiss2015, Appendix A.  7x7, a(xyz), b(xyz), c(z). assuming R2c >> Rxx, Kxx 
# Rrfc = w1**2 * R2c / (R2c**2 + dwc**2) # (!) Lorentzian lineshape assumed for simplicity

def get_BMmat_N_pool(w1, dw_l, R1_l, R2_l, Ka2x_l, Kx2a_l):
    """ ! UNTESTED since the initial POC] 
        ! TODO test this with 3 pools, then with additional pools.        
        Get a full Bloch-McConnell Matrix for N pools.        
        assuming B1 along X
        Implicitly Taking lorentzian lineshape for (c) pool (semisolid MT). 
        note a - [0], c - [1]
    """
    ZE = jnp.zeros(R2_l[0].shape)  
    
    dwa, R1a, R2a = dw_l[0], R1_l[0], R2_l[0]
    dwc, R1c, R2c = dw_l[1], R1_l[1], R2_l[1]
    Kac, Kca = Ka2x_l[1], Kx2a_l[1]
    Ka2 = jnp.sum(Ka2x_l)
    
    BM_mat = [
        [-R2a-Ka2,  dwa,      ZE,       ZE,       ZE,       ZE      ], 
        [-dwa,      -R2a-Ka2, w1,       ZE,       ZE,       ZE      ],
        [ZE,        -w1,      -R1a-Ka2, ZE,       ZE,       ZE      ],
        [Kac,       ZE,       ZE,       -R2c-Kca, dwc,      ZE      ],
        [ZE,        Kac,      ZE,       -dwc,     -R2c-Kca, w1      ],
        [ZE,        ZE,       Kac,      ZE,       -w1,      -R1c-Kca]
    ]
    pools = 2
    for dwx, R1x, R2x, Kax, Kxa in zip(dw_l[2:], R1_l[2:], R2_l[2:], Ka2x_l[2:], Kx2a_l[2:]):
        pools += 1
        BM_mat[0] += [Kxa, ZE,  ZE]
        BM_mat[1] += [ZE,  Kxa, ZE]
        BM_mat[2] += [ZE, ZE,  Kxa]
        for row in BM_mat[3:]:
            row += [ZE, ZE, ZE]
        BM_mat += [[Kax, ZE, ZE] + [ZE, ZE, ZE]*(pools-2) + [-R2x-Kxa, dwx,      ZE      ]]
        BM_mat += [[ZE, Kax, ZE] + [ZE, ZE, ZE]*(pools-2) + [-dwx,     -R2x-Kxa, w1      ]]
        BM_mat += [[ZE, ZE, Kax] + [ZE, ZE, ZE]*(pools-2) + [ZE,       -w1,      -R1x-Kxa]]

    # Reshape: from 2D matrix of 3D-array entries, into 4D tensor (sequence + flat-spatial + BM-2D)
    BM_mat_T = jnp.concatenate([jnp.concatenate([entry[..., None, None] \
                                    for entry in row], axis=3) \
                                for row in BM_mat], axis=2)      

    return BM_mat_T


def matrix2tensor(mat_ll):
    """ convert from list(list()) form of matrix, to 4D jnp array,
        aligning shapes of the 2D elements by broadcast
    """
    mat_ll_flat = [el for row in mat_ll for el in row]
    max_shape = tuple(max(arr.shape[i] for arr in mat_ll_flat) for i in range(2))
    
    stacked_4d_array = jnp.stack(
        [jnp.stack([jnp.broadcast_to(el, max_shape) for el in row
                    ], axis=2
                    ) for row in mat_ll
        ], axis=3
    )
    return stacked_4d_array.transpose(0,1,3,2)
# !! axis 3 vs 2  # column_stack vs. stack


def get_BMmat_3pool(w1, dwa, dwb, dwc, R1a, R2a, R1b, R2b, R1c, R2c, Kab, Kba, Kac, Kca):
    """ The full Bloch-McConnell Matrix for 3 pools (Water, semisolid-MT, Amide).
        Implicitly taking lorentzian lineshape for (c) pool (semisolid MT). 
    """
    ZE = jnp.zeros(R2b.shape)          
    Kax = Kab+Kac
    
    # assuming B1 along X
    BM_mat = [
        [-R2a-Kax,  dwa,     ZE,        Kba,       ZE,       ZE,       Kca,      ZE,       ZE      ], 
        [-dwa,      -R2a-Kax, w1,       ZE,        Kba,      ZE,       ZE,       Kca,      ZE      ],
        [ZE,        -w1,      -R1a-Kax, ZE,        ZE,       Kba,      ZE,       ZE,       Kca     ],
        [Kab,       ZE,       ZE,       -R2b-Kba,  dwb,      ZE,       ZE,       ZE,       ZE      ],
        [ZE,        Kab,      ZE,       -dwb,      -R2b-Kba, w1,       ZE,       ZE,       ZE      ],
        [ZE,        ZE,       Kab,      ZE,        -w1,      -R1b-Kba, ZE,       ZE,       ZE      ],
        [Kac,       ZE,       ZE,       ZE,        ZE,       ZE,       -R2c-Kca, dwc,      ZE      ],
        [ZE,        Kac,      ZE,       ZE,        ZE,       ZE,       -dwc,     -R2c-Kca, w1      ],
        [ZE,        ZE,       Kac,      ZE,        ZE,       ZE,       ZE,       -w1,      -R1c-Kca]
    ]
    # Reshape: from 2D matrix of 3D-array entries, into 4D tensor (sequence + flat-spatial + BM-2D)
    return matrix2tensor(BM_mat)


def readout_block(Z_tmp, R1a):                         
    '''
    ASSUMPTIONS: 
    (A) Negligible impact of MT/CEST pools during this time (equilibrium quickly reached and maintained..)
    (B) Every flip pulse preceded by crusher/spoiler so no transverse magnetization back-flipping into Z...     
    '''    
    for ii in range(num_flip_pulses): 
        Z_tmp = Z_tmp * flip_cos  
        Z_tmp = 1 - (1 - Z_tmp) * jnp.exp( - R1a * t_flip_period)            

    return Z_tmp


def dicts2args(tissue_params, w_dict, wrf_T):

    Kba = tissue_params['kb_T'].reshape(1, -1)
    Kab = tissue_params['fb_T'].reshape(1, -1) * Kba
    
    Kca = tissue_params['kc_T'].reshape(1, -1)
    Kac = tissue_params['fc_T'].reshape(1, -1) * Kca
    
    R1a = tissue_params['R1a_T'].reshape(1, -1)
    R2a = tissue_params['R2a_T'].reshape(1, -1)
    
    R1b = (tissue_params.get('R1b_T') or R1a).reshape(1, -1)
    R2b = tissue_params['R2b_T'].reshape(1, -1)
        
    R1c = (tissue_params.get('R1c_T') or R1a).reshape(1, -1)
    R2c = tissue_params['R2c_T'].reshape(1, -1)
    
    wrf_T = wrf_T.reshape(-1, 1)  # TODO why not inside "w_dict"
        
    seq_len = wrf_T.shape[0]
    w1 =  w_dict['w1_T'].reshape(seq_len, -1)   

    dwa = w_dict['wa_T'].reshape(1, -1) - wrf_T.reshape(-1, 1)
    dwb = w_dict['wb_T'].reshape(1, -1) - wrf_T.reshape(-1, 1)
    dwc = w_dict['wc_T'].reshape(1, -1) - wrf_T.reshape(-1, 1)

    return seq_len, wrf_T, w1, dwa, dwb, dwc, R1a, R2a, R1b, R2b, R1c, R2c, Kab, Kba, Kac, Kca


def forward_mrf(
    tissue_params, w_dict, wrf_T, Z_acq_meas=None, TRs=TRs_DEF, mode='sequential',
    TSATs=TSATs_DEF, t_pulse=tpulse_DEF, t_delay=tdelay_DEF, num_pulses=n_pulses_DEF, **kwargs
    ):
    """ 
        Simulating CEST/MT dynamics by solving the Bloch-McConnell ODE - :
            dM(t)/dt = A*M + C ==> M(t) = exp(At)*(M0-Mss) + Mss ;  Mss = - A \\ C (linear system solution, i.e., Mss s.a. A*Mss=C)
        - within each saturation pulse (& pause), repeated across all pulses of saturation phase. 
        
        That's to be repeated across the sequence of {acquire+relax+saturate} experiments, 
        be it a Z-spectrum or CEST-MRF protocol. 
        
        Here, we parallelize it for gradient efficiency (in both computational and learnability aspects),
        leveraging the availability of each experiment's initial conditions from maps acquired in previous one.
        
        NOTE assumptions and simplifications:
        - Proper normalization by M0 is assumed. All magnetization values are "Z" values (in [0,1])
        - The acquisition+delay simulation is simplified, Mz only. 
    """    
    # NOTE dimensions:
    # 0: MRF-sequence  1: spatial (voxels, flattened) 
    # 2: BM axis (ax,ay,az, cx,cy,cz, ...) 3: 2nd BM axis (for matrices)

    seq_len, wrf_T, w1, dwa, dwb, dwc, R1a, R2a, R1b, R2b, R1c, R2c, Kab, Kba, Kac, Kca = \
        dicts2args(tissue_params, w_dict, wrf_T)
    num_voxels = R2b.shape[1]
    t_relax = (jnp.array(TRs) - jnp.array(TSATs)).reshape(-1, 1)

    ZE = jnp.zeros([seq_len, num_voxels])  
    MZ0a = jnp.ones([seq_len, num_voxels])
    
    # Assuming detailed balance:  M0b*Kba = M0a*Kab
    MZ0b = tissue_params['fb_T'].reshape(1, -1) * MZ0a
    MZ0c = tissue_params['fc_T'].reshape(1, -1) * MZ0a

    R1_l = [R1a, R1b, R1c]
    Mz0_l = [MZ0a, MZ0b, MZ0c]
    
    # The C constant in the ODE 
    # (3xNp vector of decay rates to non-zero thermal equilibrium at origin)
    BM_Cvec = sum([[ZE, ZE, R_ * Mz_] for R_, Mz_ in zip(R1_l, Mz0_l)], [])
    BM_Cvec = jnp.concatenate([e[..., None] for e in BM_Cvec], axis=2)     
        
    # Bloch-McConnell matrices during pulse/pause phases    
    BM_Amtx_pulse = get_BMmat_3pool(w1, dwa, dwb, dwc, R1a, R2a, R1b, R2b, R1c, R2c, Kab, Kba, Kac, Kca)
    BM_Amtx_delay = get_BMmat_3pool(ZE, dwa, dwb, dwc, R1a, R2a, R1b, R2b, R1c, R2c, Kab, Kba, Kac, Kca) 
    if double_precision:
        BM_Amtx_pulse = BM_Amtx_pulse.astype(jnp.float64)
        BM_Amtx_delay = BM_Amtx_delay.astype(jnp.float64) 

    expm_BM_Amtx_t_pulse = jax.scipy.linalg.expm(BM_Amtx_pulse * t_pulse * (jnp.array(TSATs)>0)[:, None, None, None])
    expm_BM_Amtx_t_delay = jax.scipy.linalg.expm(BM_Amtx_delay * t_delay * (jnp.array(TSATs)>0)[:, None, None, None])
    
    # NOTE: in theory possible to generalize to per-round tpulse/tpause, 
    #       or even a custom sequence of pulses and pauses (with a little upgrade),
    #       only #pulses is hard-constrained to be constant in current code
    # Here just enforcing no pulse/delay (no evolution) if tsat=0 (0th iteration w.o. saturation phase - M0/PD scan)    
    
    # Spin-lock in/out rotations matrices (assuming hard pulse, perfect phase handling, etc.)
    SL_rot_mat_T = get_SL_rot(w1, dwa, sign=1)
    SL_inv_rot_mat_T = get_SL_rot(w1, dwa, sign=-1)    
        
    # Get Mss = - A \ C  (solution of A @ x = - C)
    # (A) steady-state w. RF - should give SL-aligned "fully-saturated" result 
    Mss_pulse_vec = - jax.scipy.linalg.solve(BM_Amtx_pulse, BM_Cvec[..., None]).squeeze(-1) 
    Mss_pulse_mtx = Mss_pulse_vec[..., None]
    # (B) steady-state w.o. RF - should give (0, 0, 1, 0, 0, fb, 0, 0, fc)
    Mss_delay_vec = - jax.scipy.linalg.solve(BM_Amtx_delay, BM_Cvec[..., None]).squeeze(-1)  
    Mss_delay_mtx = Mss_delay_vec[..., None]
    
    if mode == 'sequential':
        MZa, MZb, MZc = MZ0a[0], MZ0b[0], MZ0c[0]
        ZE = ZE[0]
        R1a = R1a[0]
        Z_acq = []
        for scan_round in range(len(wrf_T)):         
            sr = scan_round   
            tr, tsat = TRs[sr], TSATs[sr]            
            t_rec = tr - tsat
            _Mss_pulse_mtx, _Mss_delay_mtx, _expm_BM_Amtx_t_pulse, _expm_BM_Amtx_t_delay, _SL_rot_mat_T, _SL_inv_rot_mat_T = \
                Mss_pulse_mtx[sr], Mss_delay_mtx[sr], expm_BM_Amtx_t_pulse[sr], expm_BM_Amtx_t_delay[sr], SL_rot_mat_T[sr], SL_inv_rot_mat_T[sr]
            
            Mvec = [ZE, ZE, MZa, ZE, ZE, MZb, ZE, ZE, MZc]        
            Mvec_T = jnp.concatenate([e[..., None] for e in Mvec], axis=1)
            M_mtx = Mvec_T[..., None]
            crusher_mat_T = get_crusher_mat_T()

            # --- SATURATION ---                
            for pulse in range(num_pulses):
                # PULSE: (a) SL on  (b) BM evolution  (c) SL off        
                M_mtx = jnp.matmul(_SL_rot_mat_T, M_mtx)          
                M_mtx = jnp.matmul(_expm_BM_Amtx_t_pulse, (M_mtx - _Mss_pulse_mtx)) + _Mss_pulse_mtx
                M_mtx = jnp.matmul(_SL_inv_rot_mat_T, M_mtx)
                
                # DELAY: BM evolution (no RF)
                if pulse < num_pulses-1:
                    M_mtx = jnp.matmul(_expm_BM_Amtx_t_delay, (M_mtx - _Mss_delay_mtx)) + _Mss_delay_mtx

            # Crusher (TODO is it functionally redundant as we just take the Z anyways, next?)
            Mvec_T = jnp.matmul(crusher_mat_T, M_mtx)[:, :, 0]
            
            # Extract the resultant Z magnetization of pool (a)
            Z_post_sat_T = Mvec_T[:, 2].reshape(tissue_params['R1a_T'].shape) 
            Z_acq.append(Z_post_sat_T[None, ...])
            
            # --- Readout + Recovery ---  
            #    NOTE: assuming equilibration: (K>>1, T1 similar) and spoiling/crushing so no Mxy flip into Z    
            Z_tmp = readout_block(Mvec_T[:, 2], R1a)   
            MZa = 1 - (1 - Z_tmp) * jnp.exp( - R1a * t_rec)       
            MZb = tissue_params['fb_T'].reshape(-1) * MZa   
            MZc = tissue_params['fc_T'].reshape(-1) * MZa   

        Z_acq = jnp.concatenate(Z_acq, axis=0)   
        return Z_acq  # !!
    
    else:  # parallel
        
        # -- Prologue: the initial values BEFORE a cycle of [acquire, relax, saturate], 
        #     using the acquired values themselves (shifted right), assuming a fully-relaxed initial state
        Z_acq = Z_acq_meas.reshape(Z_acq_meas.shape[0], -1)
        Z_prev = jnp.concatenate((jnp.ones((1, num_voxels)), Z_acq[:-1] ), axis=0)
        # Shift recovery time as belonging to the NEXT round + assume 15sec relaxation before the first round
        t_rec = jnp.concatenate((15*jnp.ones((1,1)), t_relax[:-1]), axis=0)
        
        # -- Step 1: Readout --
        #    NOTE: we assume the measured represents a snapshot of acquisition start,
        #          and the flip/decay changes while scanning k-space just reflect in PSF)
        Z_prev_postacq = readout_block(Z_prev, R1a)

        # -- Step 2: Recovery --    
        Z_prev_postrelax = 1 - (1 - Z_prev_postacq) * jnp.exp( - R1a * t_rec)
            
        # -- Step 3: Saturation --
        #    NOTE: assuming equilibration: (K>>1, T1 similar) and spoiling
        MZa = Z_prev_postrelax
        MZb = tissue_params['fb_T'].reshape(-1) * MZa   
        MZc = tissue_params['fc_T'].reshape(-1) * MZa   
        MZx_l = [MZa, MZb, MZc]
        
        M_l = sum([[ZE, ZE, Zx] for Zx in MZx_l], []) # MZa, ZE, ZE, MZb, ZE, ZE, MZc] 
        M_vec = jnp.concatenate([e[..., None] for e in M_l], axis=2)
        M_mtx = M_vec[..., None]

        # Actual propagation here:
        for pulse in range(num_pulses):
            # Pulse: (a) SL on  (b) BM evolution  (c) SL off        
            M_mtx = jnp.matmul(SL_rot_mat_T, M_mtx) # M[..., None])[:, :, 0]
            M_mtx = jnp.matmul(expm_BM_Amtx_t_pulse, (M_mtx - Mss_pulse_mtx)) + Mss_pulse_mtx  # [..., None])[:, :, 0] + Mss_pulse
            M_mtx = jnp.matmul(SL_inv_rot_mat_T, M_mtx) # M[..., None])[:, :, 0]
            
            # Delay: BM evolution (no RF)
            if pulse < num_pulses-1:
                M_mtx = jnp.matmul(expm_BM_Amtx_t_delay, (M_mtx - Mss_delay_mtx)) + Mss_delay_mtx # [..., None])[:, :, 0] + Mss_delay

        # -- Epilogue -- : 
        M_vec = M_mtx[:, :, :, 0]

        # Take Z component of water (a) pool which is what's actually acquired...
        Mza_sim_from_prev = M_vec[:, :, 2]    
        Z_acq_sim_from_prev = Mza_sim_from_prev.reshape(Z_acq_meas.shape)
        
        return Z_acq_sim_from_prev
    

###############################################################################
#####  Approximate top eigenvalue approach Two-pool (isar2, Roeloffs2015) #####
###############################################################################

def gen_dw_sq(wrf, wa):
    return jnp.square(wrf-wa)

def gen_dwb_sq(wrf, wb):
    return jnp.square(wrf-wb)
    
def gen_Reff(R1a, R2a, dw_sq, w1):
#    num =jnp.mul(jnp.neg(R1a),dw_sq) + jnp.mul(jnp.neg(R2a),jnp.power(w1,2))    
    num = R1a * dw_sq + R2a * jnp.square(w1)
    den = dw_sq + jnp.square(w1)
    return num / den

def gen_Quarter_Gamma_Square(kb, R2b, w1):
    a = (kb + R2b) / kb * jnp.square(w1)
    b = jnp.square(kb + R2b)
    Gamma_model = 2.0 * jnp.sqrt(a+b)
    return jnp.square(Gamma_model) * 0.25
    
def gen_Rex_max(fb, kb, w1, dw_sq, wa, wb, R2b, Quarter_Gamma_Square):
    num1 = fb * kb * jnp.square(w1) #fb*kb*(w1^2)
    
    num2a = jnp.square(wa-wb)
    num2b = (dw_sq + jnp.square(w1)) * R2b / kb
    num2c = R2b * (kb + R2b)
    num2 = num2a + num2b + num2c
    
    num = num1 * num2
    den = (dw_sq+jnp.square(w1)) * Quarter_Gamma_Square
    return num / den

def gen_Rex(Rex_max,Quarter_Gamma_Square,dwb_sq):
    # TODO - the numerator also changes whe dwb_sq is not zero, right?!?!
    num = Rex_max * Quarter_Gamma_Square
    den = Quarter_Gamma_Square+dwb_sq
    return num / den

def gen_R1rho(Reff, Rex):
    return Reff+Rex
    
def gen_R1rho_full(wa, wb, R1a, R2a, R2b, kb, fb, w1, wrf):
    dw_sq = gen_dw_sq(wrf, wa)
    dwb_sq = gen_dwb_sq(wrf, wb)
    Reff = gen_Reff(R1a,R2a, dw_sq, w1)    
    Quarter_Gamma_Square = gen_Quarter_Gamma_Square(kb, R2b, w1)            
    Rex_max = gen_Rex_max(fb, kb, w1, dw_sq, wa, wb, R2b, Quarter_Gamma_Square)    
    Rex = gen_Rex(Rex_max,Quarter_Gamma_Square,dwb_sq)    
    R1rho = gen_R1rho(Reff, Rex)
    return R1rho


def pulsed_evolution(Z_prev, Zss_cw, wa, wb, wrf, w1, R1a, R2a, R1b, R2b, kb, fb, tsat, tpulse, tdelay, is_spin_lock=False):
    '''
    '''
    Z0 = Z_prev
    dw_sq = (wa-wrf)**2   
    dwb_sq = (wb-wrf)**2  
    Reff = gen_Reff(R1a, R2a, dw_sq, w1)    
    Quarter_Gamma_Square = gen_Quarter_Gamma_Square(kb, R2b, w1)
    Rex_max = gen_Rex_max(fb, kb, w1, dw_sq, wa, wb, R2b, Quarter_Gamma_Square)
    Rex = gen_Rex(Rex_max,Quarter_Gamma_Square,dwb_sq)
    R1rho = gen_R1rho(Reff, Rex) 

    R1rho = gen_R1rho_full(wa, wb, R1a, R2a, R2b, kb, fb, w1, wrf)
        
    cos_sq = 1 / (1 + (w1/(wa-wrf))**2)
    
    PzeffPz = 1 if is_spin_lock else cos_sq
    
    ka = kb * fb  
    isar_lambda1 = -(R1a + fb*R1b) / (1 + fb)    
    isar_lambda2 = -(ka + kb + (R1b + fb*R1a) / (1 + fb))
    
    psi = fb - Rex / kb
    td = tdelay
    daa = -((isar_lambda2-R1a-ka)*jnp.exp(isar_lambda1*td) - (isar_lambda1-R1a-ka)*jnp.exp(isar_lambda2*td)) / (isar_lambda1 - isar_lambda2)
    dab = -kb*(jnp.exp(isar_lambda1*td) - jnp.exp(isar_lambda2*td)) / (isar_lambda1 - isar_lambda2)
    
    alpha = Zss_cw * (1 - jnp.exp(-R1rho*tpulse)) + PzeffPz * jnp.exp(-R1rho*tpulse) * (1 - (daa + fb*dab)) 
    
    beta =  (daa + dab*psi) * PzeffPz
    Zss_pulsed = alpha / (1 - beta * jnp.exp(-tpulse*R1rho))  # Good formula, eq [A5.1]

    n = jnp.ceil(tsat / (tpulse + tdelay) ) # !!
    Zn = (Z0 - Zss_pulsed) * jnp.exp(-R1rho*tpulse*n) * jnp.power(beta, n) + Zss_pulsed
    
    return Zn


def forward_mrf_isar2(f_T, k_T, wa_T, wb_T, R1a_T, R2a_T, R1b_T, R2b_T, w1_MT_T, wrf_MT_T, 
                      TRs=TRs_DEF, TSATs=TSATs_DEF, tpulse=tpulse_DEF, tdelay=tdelay_DEF,
                      simulate_acquisition=True, mode='sequential', Z_acq_meas=None):                           
    """ ISAR2
    """
    R1rho_T = gen_R1rho_full(wa_T, wb_T, R1a_T, R2a_T, R2b_T, k_T, f_T, w1_MT_T, wrf_MT_T)
    Zss_cw_T = R1a_T / R1rho_T  # maybe need cos(theta) ?!

    Z_pre_sat_T = jnp.ones_like(wa_T)
    Z_acq = []
        
    for scan_round in range(len(wrf_MT_T)):   
        sr = scan_round   
        tr = TRs[sr]
        tsat = TSATs[sr]
        t_acq_rlx = tr - tsat
        # jax.debug.print("tr: {}", tr)  # works, prints during actual iterations.        
        Z_post_sat_T = jax.lax.cond(tsat>0,
            true_fun=lambda z:  pulsed_evolution(z, Zss_cw_T[sr], 
                                                         wa_T, wb_T, wrf_MT_T[sr], w1_MT_T[sr], 
                                                         R1a_T, R2a_T, R1b_T, R2b_T, 
                                                         k_T, f_T, tsat, tpulse, tdelay, is_spin_lock=True),
            false_fun=lambda z: z,
            operand = Z_pre_sat_T)        
        Z_acq.append(Z_post_sat_T) 
        
        # ~~ EXPERIMENTAL: replacing by measured to "separate" experiments.
        if mode == 'parallel': 
            #import ipdb; ipdb.set_trace()
            Z_post_sat_T = Z_acq_meas[sr: (sr+1)] 
            # NOTE - this assumes normalization by equilibrated, no-saturated measurement (e.g. /= Z_acq_meas[0])
        # ~~

        if simulate_acquisition:   ## TODO how much improvement it gives now?>
        # TODO (!) what about other pools (e.g. ss) evolution during acquisition and exchange?!
            Z_tmp = Z_post_sat_T
            for ii in range(num_flip_pulses): 
                Z_tmp = Z_tmp * flip_cos  # ...assuming no transverse magnetisation flipping into Z ? 
                Z_tmp = 1 - (1 - Z_tmp) * jnp.exp(-R1a_T * t_flip_period)            
        else:
            Z_tmp = 0  #Z_pre_sat_T = 1 - torch.exp(-R1a_T * t_acq_rlx)

        Z_pre_sat_T = 1 - (1 - Z_tmp) * jnp.exp(-R1a_T * t_acq_rlx)
        
    Z_acq = jnp.concatenate(Z_acq, axis=0)    
    return Z_acq
