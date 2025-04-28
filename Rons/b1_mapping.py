import numpy as np


def calculate_b1_map(echo1_data, echo2_data, brain_mask=None, alpha1_deg=30, alpha2_deg=60):
    """
    Calculate B1 map from two spin echo images with different flip angles.

    Parameters:
    -----------
    echo1_data : ndarray
        Signal intensity from first acquisition with flip angle alpha1
    echo2_data : ndarray
        Signal intensity from second acquisition with flip angle alpha2
    brain_mask : ndarray or None
        Binary mask indicating brain voxels (1 for brain, 0 for non-brain)
    alpha1_deg : float
        Nominal flip angle of first acquisition in degrees
    alpha2_deg : float
        Nominal flip angle of second acquisition in degrees

    Returns:
    --------
    b1_map : ndarray
        Map of B1 field relative to nominal value (unitless ratio)
    """
    # Convert flip angles to radians
    alpha1_rad = np.radians(alpha1_deg)
    alpha2_rad = np.radians(alpha2_deg)

    # Create a mask for valid data points (avoid division by zero)
    valid_mask = (echo1_data > 0) & (echo2_data > 0)

    # Apply brain mask if provided
    if brain_mask is not None:
        valid_mask = valid_mask & (brain_mask > 0)

    # Initialize B1 map with NaN values
    b1_map = np.full_like(echo1_data, np.nan, dtype=float)

    # Calculate signal ratio where valid
    signal_ratio = np.zeros_like(echo1_data, dtype=float)
    signal_ratio[valid_mask] = echo2_data[valid_mask] / (2 * echo1_data[valid_mask])

    # Calculate B1 deviation using the Double Angle method (https://qmrlab.org/mooc/b1-mapping)
    # If we have signals S1 = sin(α·B1) and S2 = sin(2α·B1), then:
    # B1 = arccos(S2/2S1) / α

    # clip signal_ratio to avoid invalid values
    signal_ratio = np.clip(signal_ratio, -1, 1)
    b1_map[valid_mask] = np.arccos(signal_ratio[valid_mask]) / alpha1_rad

    # Set non-physical values to NaN
    b1_map[(b1_map <= 0) | (b1_map > 2.0)] = np.nan

    # Explicitly set non-brain regions to NaN
    if brain_mask is not None:
        b1_map[brain_mask <= 0] = np.nan

    return b1_map
