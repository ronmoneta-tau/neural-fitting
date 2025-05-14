import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import List
import data
import simulation
import pipelines
import net


@dataclass
class SimulationConfig:
    def __init__(self, B0_base: float = 7.0, num_flip_pulses: int = 1, flip_angle_deg: float = 90.0,
                 t_pulse: float = 2.5, t_delay: float = 0.1, n_pulses: int = 1, do_spin_lock: bool = False):
        self.B0_base = B0_base
        self.num_flip_pulses = num_flip_pulses
        self.flip_angle_deg = flip_angle_deg  # in degrees, will be converted to radians
        self.t_pulse = t_pulse
        self.t_delay = t_delay
        self.n_pulses = n_pulses
        self.do_spin_lock = do_spin_lock

    @property
    def flip_angle_rad(self) -> float:
        return self.flip_angle_deg * jnp.pi / 180.0

    def apply(self):
        data.B0_base_DEF = self.B0_base
        simulation.num_flip_pulses = self.num_flip_pulses
        simulation.flip_angle = self.flip_angle_rad
        simulation.tpulse_DEF = self.t_pulse
        simulation.tdelay_DEF = self.t_delay
        simulation.n_pulses_DEF = self.n_pulses
        simulation.DO_SL = self.do_spin_lock


@dataclass
class TrainingConfig:
    def __init__(self, std_up_fact: float = 0.2, mt_lr: float = 0.01, mt_steps: int = 500,
                 auto_reduce_batch: bool = False, tp_noise_augmentation_burn_in: int = 50, amide_patience: int = 100,
                 sigmoid_scale_fac: int = 10, tp_noise: bool = True):
        self.std_up_fact = std_up_fact
        self.mt_lr = mt_lr
        self.mt_steps = mt_steps
        self.auto_reduce_batch = auto_reduce_batch
        self.tp_noise_augmentation_burn_in = tp_noise_augmentation_burn_in
        self.amide_patience = amide_patience
        self.sigmoid_scale_fac = sigmoid_scale_fac
        self.tp_noise = tp_noise

    def apply(self):
        pipelines.pipeline_config.train_config.std_up_fact = self.std_up_fact
        pipelines.pipeline_config.mt_lr = self.mt_lr
        pipelines.pipeline_config.mt_steps = self.mt_steps
        pipelines.pipeline_config.train_config.auto_reduce_batch = self.auto_reduce_batch
        pipelines.pipeline_config.train_config.tp_noise_augmentation_burn_in = self.tp_noise_augmentation_burn_in
        pipelines.pipeline_config.amide_patience = self.amide_patience
        pipelines.pipeline_config.train_config.tp_noise = self.tp_noise
        net.MyMLP.sigmoid_scale_fac = self.sigmoid_scale_fac


@dataclass
class InferenceConfig:
    def __init__(self, scope: str):
        self.scope = scope
        self.pool = 'c' if scope == 'MT' else 'b'

    def apply(self, data_feed: data.SlicesFeed, mt_tissue_param_est: {}):
        if self.scope != 'MT':
            data_feed.fc_gt_T = mt_tissue_param_est['fc_T']
            data_feed.kc_gt_T = mt_tissue_param_est['kc_T']


@dataclass
class DataConfig:
    def __init__(self, scope: str, norm_type: str = 'l2', ds: int = 1, slw: int = 1,
                 cutout_height: slice = slice(18, 42), cutout_width: slice = slice(17, 50)):
        self.scope = scope
        self.norm_type = norm_type
        self.ds = ds
        self.slw = slw
        self.cutout_height = cutout_height
        self.cutout_width = cutout_width
        data.SlicesFeed.norm_type = self.norm_type

    def apply(self, data_feed: data.SlicesFeed):
        data_feed.ds = self.ds
        data_feed.slw = self.slw


@dataclass
class ROIConfig:
    points_tumor: List[List[int]] = field(default_factory=lambda: [[6, 6], [8, 7], [10, 7], [12, 8]])
    points_contralateral: List[List[int]] = field(default_factory=lambda: [[5, 25], [7, 26], [9, 25], [11, 25]])


# @dataclass
# class Config:
#     """Main configuration class for MT and rNOE experiments"""
#     simulation: SimulationConfig = field(default_factory=SimulationConfig)
#     training: TrainingConfig = field(default_factory=TrainingConfig)
#     inference: InferenceConfig = field(default_factory=InferenceConfig)
#     data: DataConfig = field(default_factory=DataConfig)
#     roi: ROIConfig = field(default_factory=ROIConfig)
#
#     def apply_all(self):
#         self.simulation.apply()
#         self.training.apply()
#         self.data.apply()

