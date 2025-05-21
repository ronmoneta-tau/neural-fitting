import jax.numpy as jnp
from dataclasses import dataclass
from micedata import MiceData
from xarray import Dataset
from pathlib import Path
import sys
import dill
sys.path.append(str(Path.cwd().parent))
import data, simulation, pipelines, net

@dataclass
class SimulationConfig:
    def __init__(self, B0_base: int = 7, num_flip_pulses: int = 1, flip_angle_deg: float = 90.0,
                 t_pulse: float = 2.5, t_delay: float = 0.1, n_pulses: int = 1, do_spin_lock: bool = False,
                 simulation_type: str = 'expm_bmmat', norm_type: str = 'l2'):
        self.B0_base = B0_base
        self.num_flip_pulses = num_flip_pulses
        self.flip_angle_deg = flip_angle_deg  # in degrees, will be converted to radians
        self.t_pulse = t_pulse
        self.t_delay = t_delay
        self.n_pulses = n_pulses
        self.do_spin_lock = do_spin_lock
        self.simulation_type = simulation_type
        self.norm_type = norm_type

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
        data.SlicesFeed.norm_type = self.norm_type


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

    def apply(self, *datafeeds: data.SlicesFeed) -> None:
        pipelines.pipeline_config.train_config.std_up_fact = self.std_up_fact
        pipelines.pipeline_config.mt_lr = self.mt_lr
        pipelines.pipeline_config.mt_steps = self.mt_steps
        pipelines.pipeline_config.train_config.auto_reduce_batch = self.auto_reduce_batch
        pipelines.pipeline_config.train_config.tp_noise_augmentation_burn_in = self.tp_noise_augmentation_burn_in
        pipelines.pipeline_config.amide_patience = self.amide_patience
        pipelines.pipeline_config.train_config.tp_noise = self.tp_noise
        net.MyMLP.sigmoid_scale_fac = self.sigmoid_scale_fac
        # pipelines.pipeline_config.train_config.tp_noise = False # try without the noise augmentation
        # This should stay commented as it made the std later extremely large. seems like without it The network
        # doesn't learn to predict uncertainty properly. and it defaults to predicting almost constant,
        # large uncertainty values for all pixels

        for data_feed in datafeeds:
            data_feed.ds = 1
            data_feed.slw = 1


@dataclass
class DataConfig:
    def __init__(self, name: str, data_cutout: Dataset, inpt: MiceData, figs_path: Path):
        self.name = name

        if self.name == 'MT':
            self.pool = 'c'
            self.f_scale_fact = 30 / 100
            self.k_scale_fact = 102
            self.scope = 'mt'
            self.ppm = -2.5
            self.T2 = 0.04
            pipelines.pipeline_config.infer_config.fc_scale_fact = self.f_scale_fact
            pipelines.pipeline_config.infer_config.kc_scale_fact = self.k_scale_fact
            data.wc_ppm_DEF = self.ppm
            data.T2c_ms_DEF = self.T2

        elif self.name == 'Amide':
            self.pool = 'b'
            self.f_scale_fact = 1.2 / 100
            self.k_scale_fact = 400
            self.scope = 'amide'
            self.ppm = 3.5
            self.T2 = 1
            pipelines.pipeline_config.infer_config.fb_scale_fact = self.f_scale_fact
            pipelines.pipeline_config.infer_config.kb_scale_fact = self.k_scale_fact
            data.wb_ppm_DEF = self.ppm
            data.T2b_ms_DEF = self.T2

        elif self.name == 'rNOE':
            self.pool = 'b'
            self.f_scale_fact = 2 / 100
            self.k_scale_fact = 25
            self.scope = 'amide'
            self.ppm = -3.5
            self.T2 = 5
            pipelines.pipeline_config.infer_config.fb_scale_fact = self.f_scale_fact
            pipelines.pipeline_config.infer_config.kb_scale_fact = self.k_scale_fact
            data.wb_ppm_DEF = self.ppm
            data.T2b_ms_DEF = self.T2

        self.data_feed = self.create_data_feed(data_cutout, inpt)
        self.f_lims = [0, self.f_scale_fact * 100]
        self.k_lims = [0, self.k_scale_fact]
        self.fig_path = figs_path / self.name
        self.fig_path.mkdir(parents=True, exist_ok=True)
        self.predictor = None
        self.tissue_param_est = None
        self.pred_signal_normed_np = None
        self.f_values = None
        self.k_values = None
        self.height = None
        self.width = None
        self.angle = None
        self.labels = None
        self.cov_nnpred_scaled = None

    def create_data_feed(self, data_cutout: Dataset, inpt: MiceData) -> data.SlicesFeed:
        return data.SlicesFeed.from_xarray(data_cutout, mt_or_amide=self.scope,
                                           mt_seq_txt_fname=inpt.mt_params_path,
                                           larg_seq_txt_fname=inpt.amide_params_path)

    def apply(self):
        if self.scope == 'mt':
            data.wc_ppm_DEF = self.ppm  # MT
            data.T2c_ms_DEF = self.T2  # for MT
            pipelines.pipeline_config.infer_config.fc_scale_fact = self.f_scale_fact
            pipelines.pipeline_config.infer_config.kc_scale_fact = self.k_scale_fact

        else:
            data.wb_ppm_DEF = self.ppm
            data.T2b_ms_DEF = self.T2
            pipelines.pipeline_config.infer_config.fb_scale_fact = self.f_scale_fact
            pipelines.pipeline_config.infer_config.kb_scale_fact = self.k_scale_fact

    def update_ground_truth(self, tissue_param_est: dict):
        self.data_feed.fc_gt_T = tissue_param_est['fc_T']
        self.data_feed.kc_gt_T = tissue_param_est['kc_T']

    def save(self, dir_path: Path) -> None:
        """
        Save the DataConfig object to a file.
        :param dir_path: Path to the folder where the object will be saved.
        """
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)

        file_path = dir_path / f"{self.name}_data_config.pkl"
        with open(file_path, 'wb') as f:
            dill.dump(self, f)

        print(f"DataConfig object saved to {file_path}\n")

    @staticmethod
    def load(file_path: Path) -> 'DataConfig':
        """
        Load a DataConfig object from a file.
        :param file_path: Path to the file from which the object will be loaded.
        :return: The loaded DataConfig object.
        """
        with open(file_path, 'rb') as f:
            obj = dill.load(f)
        obj.apply()
        print(f"DataConfig object loaded from {file_path}\n")
        return obj

@dataclass
class ROIConfig:

    def __init__(self):
        self.points_tumor = [[6, 6], [8, 7], [10, 7], [12, 8]]
        self.points_contralateral = [[5, 25], [7, 26], [9, 25], [11, 25]]
        self.points = self.points_contralateral + self.points_tumor