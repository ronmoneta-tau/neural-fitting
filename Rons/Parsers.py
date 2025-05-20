import argparse
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments. The following arguments are expected:
    --input_file: Path to the input text file. Each row should contain the desired data folder path.
    --generate_dict_MT (optional): Flag to indicate if MT dictionary should be generated.
    --generate_dict_rNOE (optional): Flag to indicate if rNOE dictionary should be generated.
    --MT_train (optional): Flag to indicate if MT training should be run.
    --rNOE_train (optional): Flag to indicate if rNOE training should be run.
    --MT_inference (optional): Flag to indicate if MT inference should be run.
    --rNOE_inference (optional): Flag to indicate if rNOE inference should be run.
    --output_dir (optional): Path to the output directory.
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Process mice data for MRF acquisition and analysis.")

    parser.add_argument(
        '--input_file',
        type=Path,
        required=True,
        help="Path to the input text file. Each row should contain: desired data folder path, T2 Highres dicom path."
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help="Flag to indicate if the training should be run."
    )
    parser.add_argument(
        '--inference',
        action='store_true',
        help="Flag to indicate if the inference should be run."
    )
    parser.add_argument(
        '--solutes',
        type=str,
        nargs='+',
        required=True,
        help="List of wanted solutes. Options: 'rNOE', 'Amide'."
    )
    parser.add_argument(
        '--uncertainty',
        action='store_true',
        help="Flag to indicate if the uncertainty pipeline should be run."
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default='results',
        help="Output directory to save results in. Default is 'results'."
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help="Flag to indicate if tumor and contralateral statistics should be run."
    )

    if not parser.parse_args().output_dir.exists():
        parser.parse_args().output_dir.mkdir(parents=True, exist_ok=True)

    return parser.parse_args()

def parse_input_file(input_path: Path) -> list[Path]:
    """
    Parse the input file into a dictionary of {Path:T2highres_number}.
    :param input_path: Path to the input file
    :return: Dictionary of experiment paths: T2 highres number
    """
    with open(input_path, 'r') as file:
        return [Path(path) for path in (line.strip().strip('"').strip("'") for line in file)]

def parse_scan_doc(scans_path: Path) -> {str: {str: str}}:
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
    scan_doc_path = scans_path / 'scan_doc.csv'
    with open(scan_doc_path, 'r') as file:
        scan_doc = {line.split(',')[2]: line.split(',')[0] for line in file}
        subject_metadata = {
            't1map_number': scan_doc.get('t1_map', None),
            't2map_number': scan_doc.get('t2_map', None),
            'b0map_number': scan_doc.get('wasser', None),
            'b1map_number': scan_doc.get('B1map', None),
            'mt_number': scan_doc.get('52_MT', None),
            'rnoe_number': scan_doc.get('51_rnoe', None),
            'amide_number': scan_doc.get('51_Amide', None)
        }
    return subject_metadata