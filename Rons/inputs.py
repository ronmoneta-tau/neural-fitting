import numpy as np
import xarray as xr
import pandas as pd
import pydicom
from pathlib import Path


class Inputs:
    def __init__(self, subject: Path):
        self.subject = subject
        self.subject_metadata = self.parse_scan_doc()
        self.t1_path = self.subject / f'{self.subject_metadata["t1map_number"]}'  # TODO: does it need to be t1w, or qt1?
        self.t2_path = self.subject / f'{self.subject_metadata["t2map_number"]}'  # TODO: does it need to be t2w, or qt2?
        self.b0_path = self.subject / f'{self.subject_metadata["b0map_number"]}'
        self.b1_path = self.subject / f'{self.subject_metadata["b1map_number"]}'  # TODO: folder for b1map exists, but sometimes empty - if empty assume perfect
        self.mt_path = self.subject / f'{self.subject_metadata["mt_number"]}'  # TODO: make sure its 31 (skip first one?)
        self.rnoe_path = self.subject / f'{self.subject_metadata["rNOE_number"]}'  # TODO: make sure its 31 (skip first one?)
        self.t1_wm_mask_path = self.subject / ''  # ignored for now
        self.t1_gm_mask_path = self.subject / ''  # ignored for now
        self.t1_map = self.load_auxiliary_data(self.t1_path / 'pdata/4/dicom', '3')  # TODO: Change to generic
        self.t2_map = self.load_auxiliary_data(self.t2_path / 'pdata/2/dicom', '3')  # TODO: Change to generic
        self.b0_map = self.load_auxiliary_data(self.b0_path / 'pdata/1/dicom', '02')  # TODO: Change to generic
        self.b1_map = self.load_auxiliary_data(self.b1_path / 'pdata/1/dicom', '2')  # TODO: Change to generic
        self.mt_map = self.load_mrf_data(self.mt_path)
        self.rnoe_map = self.load_mrf_data(self.rnoe_path)
        self.mt_params_path = self.extract_mrf_params(self.mt_path, 'MT')
        self.rnoe_params_path = self.extract_mrf_params(self.rnoe_path, 'rNOE')
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
        dicom_path = mrf_path / "pdata/1/dicom"
        images = [pydicom.dcmread(im).pixel_array for im in dicom_path.glob('MRIm*.dcm') if im.name != 'MRIm01.dcm']
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

    def extract_values_from_method_file(self, lines, start_index):
        """
        Extract values from a method file starting at a given index.
        """
        values = []
        for line in lines[start_index:]:
            if line.startswith("##$"):
                break
            values.extend(map(float, line.split()))
        return values

    def extract_mrf_params(self, scan_path: Path, name: str) -> Path:
        """
        Extract MRF acquisition protocol parameters from a method file and save them to a file.
        """
        method_file_path = scan_path / 'method'
        if not method_file_path.exists():
            raise FileNotFoundError(f"Method file not found at {method_file_path}")

        with open(method_file_path, 'r') as file:
            method_file = file.readlines()

        param_map = {
            "##$Fp_TRs": "TR_ms",
            "##$Fp_SatPows": "B1_uT",
            "##$Fp_SatOffset": "dwRF_Hz",
            "##$Fp_FlipAngle": "FA",
            "##$Fp_SatDur": "Tsat_ms"
        }

        data = {key: [] for key in param_map.values()}

        for i, line in enumerate(method_file):
            for key, param_name in param_map.items():
                if line.startswith(key):
                    data[param_name] = self.extract_values_from_method_file(method_file, i + 1)[1:]

        # Convert to DataFrame for easier handling
        df = pd.DataFrame(data)

        df['TR_ms'] = df['TR_ms'].round().astype(int)
        df['dwRF_Hz'] = df['dwRF_Hz'].round().astype(int)
        df['FA'] = df['FA'].round().astype(int)
        df['Tsat_ms'] = df['Tsat_ms'].round().astype(int)
        df['B1_uT'] = df['B1_uT'].round(2)

        # Save to a space-separated text file
        params_path = Path(f'{name}_mrf_params.txt')
        df.to_csv(params_path, sep=' ', index=False, header=True)

        return params_path