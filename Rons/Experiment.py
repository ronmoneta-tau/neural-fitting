import re
import pydicom
from pathlib import Path
from datetime import datetime
from typing import Union, Any
import numpy as np
import scipy.io as sio

from Rons.micedata import MiceData

IRREGULAR_PATHS = {'20241106_080713_OrPerlman_ped_tumor_immuno_1_baseline_1_1': {'cage': 'FALSE', 'marking': 'FALSE'},
                   '20241106_082248_OrPerlman_ped_tumor_immuno_right1_baseline_1_1': {'cage': 'C1', 'marking': '1R'},
                   '20241106_095306_OrPerlman_ped_tumor_immuno_left1_baseline_1_1': {'cage': 'C1', 'marking': '1L'},
                   '20241106_112334_OrPerlman_ped_tumor_immuno_Right2_baseline_1_1': {'cage': 'FALSE',
                                                                                      'marking': 'FALSE'},
                   '20241106_112437_OrPerlman_ped_tumor_immuno_Right2_baseline1_1_2': {'cage': 'C1', 'marking': '2R'},
                   '20241106_125013_OrPerlman_ped_tumor_immuno_Left2_baseline_1_1': {'cage': 'C1', 'marking': '2L'},
                   '20241106_142742_OrPerlman_ped_tumor_immuno_Stripes1_baselin_1_1': {'cage': 'C2', 'marking': '1R'},
                   '20241106_155504_OrPerlman_ped_tumor_immuno_Stripes2_baselin_1_1': {'cage': 'C2', 'marking': '1L'},
                   '20241106_171922_OrPerlman_ped_tumor_immuno_Stripes3_baselin_1_1': {'cage': 'C2', 'marking': 'None'},
                   }
FOLDER_PATTERN = r'(?P<date>\d{8}_\d{6})_OrPerlman_ped_tumor_immuno_(?:t1_)?(?P<cage>C\d)?_(?P<marking>.*?)_(?P<test_group>.*?)_(?P<scan_number>\d+)$'


class Experiment:
    def __init__(self, path: Path, output_dir: Path):
        self.scans_path = '/home/ron/pediatric-tumor-mice/' / path
        self.output_dir = output_dir
        self.date, self.cage, self.marking = self.extract_folder_info()
        self.name = f"{self.cage}_{self.marking}"
        self.working_path = Path(f"{self.name}_{self.date.date()}")
        self.data = MiceData(self.scans_path, self.working_path)
        print(f"Experiment {self.working_path} with path: {self.scans_path} created.\n")

    def extract_folder_info(self) -> tuple[datetime, Union[str, Any], Union[str, Any]]:
        """
        Extracts the date, cage and marking from the folder path
        """
        folder_name = self.scans_path.name
        pattern = re.compile(FOLDER_PATTERN)
        match = pattern.match(folder_name)
        if not match and folder_name not in IRREGULAR_PATHS:
            raise ValueError(
                f"Folder name '{folder_name}' does not match the expected pattern and is not previously known.")

        if folder_name in IRREGULAR_PATHS:
            scan_date = f"{folder_name.split('_')[0]}_{folder_name.split('_')[1]}"
            cage = IRREGULAR_PATHS.get(folder_name)['cage']
            marking = IRREGULAR_PATHS.get(folder_name)['marking']
        else:
            scan_date = match.group('date')
            cage = match.group('cage')
            marking = match.group('marking')

        scan_date = datetime.strptime(scan_date, '%Y%m%d_%H%M%S')
        if marking == 'NM':
            marking = 'None'

        return scan_date, cage, marking
