import re
import numpy as np
import xarray as xr
import pandas as pd
import pydicom
import os
import cv2
from pathlib import Path
from Rons.Parsers import parse_scan_doc
from get_paravision_t1_t2_map_ms import get_paravision_map_ms
from b0_mapping_functions import z_spec_rearranger, wassr_b0_mapping
from b1_mapping import calculate_b1_map


class MiceData:
    def __init__(self, scans_path: Path, working_path: Path):
        self.scans_metadata = parse_scan_doc(scans_path)
        self.t1_path = scans_path / f'{self.scans_metadata["t1map_number"]}'
        self.t2_path = scans_path / f'{self.scans_metadata["t2map_number"]}'
        self.wasser_path = scans_path / f'{self.scans_metadata["b0map_number"]}'
        self.b1_path = scans_path / (self.scans_metadata["b1map_number"] if self.scans_metadata["b1map_number"] else "None")
        self.mt_path = scans_path / f'{self.scans_metadata["mt_number"]}'
        self.rnoe_path = scans_path / f'{self.scans_metadata["rnoe_number"]}'
        self.amide_path = scans_path / f'{self.scans_metadata["amide_number"]}'
        self.t1_wm_mask_path = scans_path / ''  # ignored for now
        self.t1_gm_mask_path = scans_path / ''  # ignored for now
        self.brain_mask = self.load_mask(scans_path, working_path, 'brain_mask.npy')
        self.tumor_mask = self.load_mask(scans_path, working_path, 'tumor_mask.npy')
        self.contralateral_mask = self.load_mask(scans_path, working_path, 'contralateral_mask.npy')
        self.roi_mask = self.get_roi_mask()
        self.t1_qmap = self.get_qmaps(self.t1_path)
        self.t2_qmap = self.get_qmaps(self.t2_path)
        self.b0_map = self.get_b0_map()
        self.b1_map = self.get_b1_map()
        self.mt_map = self.load_mrf_data(self.mt_path)
        self.rnoe_map = self.load_mrf_data(self.rnoe_path)
        self.amide_map = self.load_mrf_data(self.amide_path)
        self.mt_params_path = self.extract_mrf_params(self.mt_path, 'MT')
        self.rnoe_params_path = self.extract_mrf_params(self.rnoe_path, 'rNOE')
        self.amide_params_path = self.extract_mrf_params(self.amide_path, 'Amide')
        self.dataset = xr.Dataset(
            {key: value for key, value in {
                'roi_mask_nans': self.roi_mask,
                'B1_fix_factor_map': self.b1_map if self.b1_map is not None else np.ones(self.roi_mask.shape),
                'B0_shift_ppm_map': self.b0_map,
                'T2ms': self.t2_qmap,
                'T1ms': self.t1_qmap,
                'MT_data': self.mt_map,
                'rNOE_data': self.rnoe_map,
                'AMIDE_data': self.amide_map,
                'white_mask': xr.DataArray(np.zeros(self.roi_mask.shape), dims=("height", "slice", "width")),
                'gray_mask': xr.DataArray(np.zeros(self.roi_mask.shape), dims=("height", "slice", "width"))
            }.items() if value is not None})

    def get_orderd_DICOM_files(self, path: Path) -> list:
        """
        Get ordered DICOM files from a given path
        :param path: Path to the DICOM files
        :return: List of ordered DICOM files
        """
        dicom_path = path / "pdata/1/dicom"

        # Sort files by numerical order based on their suffix
        dicom_files = sorted(dicom_path.glob('MRIm*.dcm'), key=lambda x: int(x.stem.replace('MRIm', '')))
        scans = [pydicom.dcmread(im).pixel_array.astype(float) for im in dicom_files]
        return scans

    def load_mrf_data(self, mrf_path: Path) -> xr.DataArray:
        """
        Load the MRF data (31 images) and inflates it to be of shape (31, height, 4, width)
        :param mrf_path: Path to the MRF maps
        :return: DataArray of inflated MRF images
        """
        images = self.get_orderd_DICOM_files(mrf_path)
        # Exclude the first image (Dummy) from the MRF data
        images.pop(0)
        mrf_data = np.stack(images, axis=0)
        mrf_data = np.expand_dims(mrf_data, axis=2)  # Add a new axis to the array
        return xr.DataArray(mrf_data, dims=("MRF_cycles", "height", "slice", "width"))

    def load_mask(self, scans_path: Path, working_path: Path, mask_name: str) -> np.array:
        """
        Load the desired mask
        :return: Brain/tumor/contralateral mask
        """
        mask_path = scans_path.parent.parent / "results" / working_path / mask_name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found at {mask_path}")

        if ".npy" in mask_name:
            mask = np.load(mask_path).astype(float)
        elif ".dcm" in mask_name:
            mask = pydicom.dcmread(mask_path).pixel_array
            np.save(scans_path.parent.parent / "results" / working_path / f'{mask_name.split(".")[0]}.npy', mask)
        else:
            raise ValueError(f"Unsupported mask file format: {mask_name}")

        mask = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST)

        return mask

    def get_roi_mask(self) -> xr.DataArray:
        """
        Convert the brain mask to appropriate behaviour (nans) and converts it to a DataArray.
        :return: DataArray of the ROI mask
        """
        roi_mask = np.ma.masked_equal(self.brain_mask, 0.).filled(np.nan)  # Changes mask from boolean to 1./nan
        roi_mask = np.expand_dims(roi_mask, axis=1)  # Add a new axis to the array
        return xr.DataArray(roi_mask, dims=("height", "slice", "width"))

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

    def get_qmaps(self, t_map_path: Path) -> xr.DataArray:
        """
        Generates quantitative map for T1/T2. Folder number 4 should be usd, but if doesn't exist, folder number 2 is used.
        :param t_map_path: Path to the T1/T2 map folder
        :return: Quantitative map
        """
        folder_4 = t_map_path / 'pdata' / '4'
        folder_2 = t_map_path / 'pdata' / '2'

        if os.path.isdir(folder_4):
            qmap = get_paravision_map_ms(folder_4 / 'dicom')
        elif os.path.isdir(folder_2):
            qmap = get_paravision_map_ms(folder_2 / 'dicom')

        qmap = np.expand_dims(qmap, axis=1)  # Add a new axis to the array
        return xr.DataArray(qmap, dims=("height", "slice", "width"))

    def get_b0_map(self) -> xr.DataArray:
        """
        Creates B0 map from WASSER data
        """
        images = self.get_orderd_DICOM_files(self.wasser_path)
        b0_data = np.stack(images, axis=0)  # (22, 64, 64) - all WASSER images, including M0

        m0_cest, z_spec_rearr = z_spec_rearranger(b0_data)
        normalized_b0_data = z_spec_rearr / m0_cest

        b0_map = wassr_b0_mapping(normalized_b0_data, self.brain_mask)
        # convert bo_map values from hz to ppm
        b0_map /= 298
        b0_map = np.expand_dims(b0_map, axis=1)  # Add a new axis to the array
        return xr.DataArray(b0_map, dims=("height", "slice", "width"))

    def get_b1_map(self) -> xr.DataArray:
        """
        Creates B1 map from Double Spin echo sequence
        """
        echos = self.get_orderd_DICOM_files(self.b1_path)

        if len(echos) < 2:
            print(ValueError("Not enough images found for B1 mapping. Assuming perfect B1 homogeneity."))
            return xr.DataArray(np.ones((64,1,64), dtype=np.float32), dims=("height", "slice", "width"))
        if len(echos) > 2:
            print(ValueError("Too many images found for B1 mapping. Assuming perfect B1 homogeneity."))
            return xr.DataArray(np.ones((64, 1, 64), dtype=np.float32), dims=("height", "slice", "width"))

        b1_map = calculate_b1_map(echos[0], echos[1], brain_mask=self.brain_mask, alpha1_deg=30, alpha2_deg=60)

        b1_map = np.expand_dims(b1_map, axis=1)  # Add a new axis to the array
        return xr.DataArray(b1_map, dims=("height", "slice", "width"))
