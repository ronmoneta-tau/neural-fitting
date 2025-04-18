# Converting paravision-generated T1 or T2 maps from arbitrary int15 to millisecond units
# Apr 9, 2023
# Or Perlman (orperlman@tauex.tau.ac.il)

import pydicom as dcm
import numpy as np
from pathlib import Path


def get_paravision_map_ms(path_to_paravision_dicom_folder: Path):
    """
    :param path_to_paravision_dicom_folder: points to folder "2" in the paravision-reconstrcuted T1/T2 map folder
    :return: paravision_map_ms - rescaled to ms units
    """
    data = (dcm.dcmread(path_to_paravision_dicom_folder / 'MRIm3.dcm'))
    paravision_map_int15 = data.pixel_array
    paravision_map_double = paravision_map_int15.astype(np.double)
    paravision_map_ms = paravision_map_double * data.RescaleSlope

    return paravision_map_ms  # map_ms