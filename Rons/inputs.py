import numpy as np
import xarray as xr
import pydicom
from pathlib import Path


class Inputs:
    def __init__(self, subject: Path):
        self.subject = subject
        self.t1_path = self.subject / ''  # TODO: does it need to be t1w, or qt1?
        self.t2_path = self.subject / ''  # TODO: does it need to be t2w, or qt2?
        self.b0_path = self.subject / ''
        self.b1_path = self.subject / ''  # TODO: folder for b1map exists, but sometimes empty - if empty assume perfect
        self.mt_path = self.subject / ''  # TODO: make sure its 31 (skip first one?)
        self.rnoe_path = self.subject / ''  # TODO: make sure its 31 (skip first one?)
        self.t1_wm_mask_path = self.subject / ''  # ignored for now
        self.t1_gm_mask_path = self.subject / ''  # ignored for now
        self.t1_map = self.load_auxiliary_data(self.t1_path, '3')  # TODO: Change to generic
        self.t2_map = self.load_auxiliary_data(self.t2_path, '3')  # TODO: Change to generic
        self.b0_map = self.load_auxiliary_data(self.b0_path, '02')  # TODO: Change to generic
        self.b1_map = self.load_auxiliary_data(self.b1_path, '2')  # TODO: Change to generic
        self.mt_map = self.load_mrf_data(self.mt_path)
        self.rnoe_map = self.load_mrf_data(self.rnoe_path)
        self.dataset = xr.Dataset(
            {'roi_mask_nans': self.t1_map, 'B1_fix_factor_map': self.b1_map, 'B0_shift_ppm_map': self.b0_map,
             'T2ms': self.t2_map, 'T1ms': self.t1_map, 'MT_data': self.mt_map, 'rNOE_data': self.rnoe_map})

    def load_auxiliary_data(self, map_path: Path, echo: str) -> xr.DataArray:
        """
        Loads an echo and inflates it to be of shape (height, 4, width)
        :param map_path: Path to the auxiliary maps
        :param echo: Index of the echo to load
        :return: DataArray of inflated echo of the auxiliary map
        """
        echo = pydicom.dcmread(map_path / f'MRIm{echo}.dcm').pixel_array
        echo = np.expand_dims(echo, axis=1)  # Add a new axis to the array
        echo = np.repeat(echo, 4, axis=1)  # Repeat the array along the new axis
        return xr.DataArray(echo, dims=("height", "slice", "width"))

    def load_mrf_data(self, mrf_path: Path) -> xr.DataArray:
        """
        Load the MRF data (31 images) and inflates it to be of shape (31, height, 4, width)
        :param mrf_path: Path to the MRF maps
        :return: DataArray of inflated MRF images
        """
        images = [pydicom.dcmread(im).pixel_array for im in mrf_path.glob('MRIm*.dcm') if im.name != 'MRIm01.dcm']
        mrf_data = np.stack(images, axis=0)
        mrf_data = np.expand_dims(mrf_data, axis=2)  # Add a new axis to the array
        mrf_data = np.repeat(mrf_data, 4, axis=2)  # Repeat the array along the new axis
        return xr.DataArray(mrf_data, dims=("MRF_cycles", "height", "slice", "width"))
