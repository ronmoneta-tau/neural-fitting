import numpy as np
import xarray as xr
import pydicom
from pathlib import Path


class Inputs:
    def __init__(self, subject: Path):
        self.subject = subject
        self.subject_metadata = self.parse_scan_doc()
        self.t1_path = self.subject / f'{self.subject_metadata["t1map_number"]}/pdata/4/dicom'  # TODO: does it need to be t1w, or qt1?
        self.t2_path = self.subject / f'{self.subject_metadata["t2map_number"]}/pdata/2/dicom'  # TODO: does it need to be t2w, or qt2?
        self.b0_path = self.subject / f'{self.subject_metadata["b0map_number"]}/pdata/1/dicom'
        self.b1_path = self.subject / f'{self.subject_metadata["b1map_number"]}/pdata/1/dicom'  # TODO: folder for b1map exists, but sometimes empty - if empty assume perfect
        self.mt_path = self.subject / f'{self.subject_metadata["mt_number"]}/pdata/1/dicom'  # TODO: make sure its 31 (skip first one?)
        self.rnoe_path = self.subject / f'{self.subject_metadata["rNOE_number"]}/pdata/1/dicom'  # TODO: make sure its 31 (skip first one?)
        self.t1_wm_mask_path = self.subject / ''  # ignored for now
        self.t1_gm_mask_path = self.subject / ''  # ignored for now
        self.t1_map = self.load_auxiliary_data(self.t1_path, '3')  # TODO: Change to generic
        self.t2_map = self.load_auxiliary_data(self.t2_path, '3')  # TODO: Change to generic
        self.b0_map = self.load_auxiliary_data(self.b0_path, '02')  # TODO: Change to generic
        self.b1_map = self.load_auxiliary_data(self.b1_path, '2')  # TODO: Change to generic
        self.mt_map = self.load_mrf_data(self.mt_path)
        self.rnoe_map = self.load_mrf_data(self.rnoe_path)
        self.dataset = xr.Dataset(  # TODO: Alex's code has AMIDE_data hardcoded, change when possible
            {'roi_mask_nans': self.t1_map, 'B1_fix_factor_map': self.b1_map, 'B0_shift_ppm_map': self.b0_map,
             'T2ms': self.t2_map, 'T1ms': self.t1_map, 'MT_data': self.mt_map, 'AMIDE_data': self.rnoe_map})

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

    def parse_scan_doc(self) -> {str: {str: str}}:
        """
        Given a subject path, find the scan_doc.csv file and parse it into a dictionary.
        specifically, create a dictionary with the following keys:
        'subject_path': {
            't1map_number': str,
            't2map_number': str,
            'b0map_number': str,
            'b1map_number': str,
            'mt_number': str,
            'rNOE_number': str
            }
        The scan_doc.csv file should have the following format:
        'export_idx','scan category','scan name','scan duration','scan dim'
        and the inner dictionary should be created by parsing the 'scan name' column in this manner:
        'scan name' == 't1_map' -> 'export_idx' == 't1map_number'
        'scan name' == 't2_map' -> 'export_idx' == 't2map_number'
        'scan name' == 'wasser' -> 'export_idx' == 'b0map_number'
        'scan name' == 'B1map' -> 'export_idx' == 'b1map_number'
        'scan name' == '52_MT' -> 'export_idx' == 'mt_number'
        'scan name' == '51_rnoe' -> 'export_idx' == 'rNOE_number'
        :return: dictionary of subject scan metadata
        """
        scan_doc_path = self.subject / 'scan_doc.csv'
        with open(scan_doc_path, 'r') as file:
            scan_doc = {line.split(',')[2]: line.split(',')[0] for line in file}
            subject_metadata = {
                't1map_number': scan_doc.get('t1_map', None),
                't2map_number': scan_doc.get('t2_map', None),
                'b0map_number': scan_doc.get('wasser', None),
                'b1map_number': scan_doc.get('B1map', None),
                'mt_number': scan_doc.get('52_MT', None),
                'rNOE_number': scan_doc.get('51_rnoe', None)
            }
        return subject_metadata
