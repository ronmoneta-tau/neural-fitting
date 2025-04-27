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
    signal_ratio[valid_mask] = echo2_data[valid_mask] / echo1_data[valid_mask]

    # Calculate B1 deviation using the arcsin method
    # If we have signals S1 = sin(α·B1) and S2 = sin(2α·B1), then:
    # B1 = arcsin(S2/S1 · sin(α)/sin(2α)) / α
    # b1_map[valid_mask] = np.arcsin(signal_ratio[valid_mask] / (2 * np.cos(alpha1_rad))) / alpha1_rad
    # if the angle are not with a factor of 2:
    # b1_map[valid_mask] = np.arcsin(signal_ratio[valid_mask] / (np.sin(alpha1_rad) / np.sin(alpha2_rad))) / alpha1_rad

    # For a double spin echo sequence:
    # The first echo signal is proportional to: sin(α₁·B1) * [sin²(α₂·B1/2)]
    # The second echo signal is proportional to: sin(α₁·B1) * [sin²(α₂·B1/2)]² * exp(-ΔTE/T2)
    # Where ΔTE is the additional time to the second echo
    # If TE2 = 2*TE1 (typical in DSE), the ratio removes T2 dependency and becomes: #TODO: can i assume that? didnt find in method
    # S₂/S₁ = sin²(α₂·B1/2)
    # Solving for B1:
    # B1 = (2/α₂) * arcsin(sqrt(S₂/S₁))

    # For your specific 30°-60° DSE:
    b1_map[valid_mask] = (2 / alpha2_rad) * np.arcsin(np.sqrt(signal_ratio[valid_mask]))

    # Set non-physical values to NaN
    b1_map[(b1_map <= 0) | (b1_map > 2.0)] = np.nan

    # Explicitly set non-brain regions to NaN
    if brain_mask is not None:
        b1_map[brain_mask <= 0] = np.nan

    return b1_map
