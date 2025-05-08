import numpy as np
from scipy import ndimage

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

    b1_map = bloch_siegert_median_filter(b1_map, brain_mask=brain_mask.astype(bool))

    # Set non-physical values to NaN
    b1_map[(b1_map <= 0) | (b1_map > 2.0)] = np.nan

    # Explicitly set non-brain regions to NaN
    if brain_mask is not None:
        b1_map[brain_mask <= 0] = np.nan

    return b1_map

def bloch_siegert_median_filter(b1_map: np.ndarray, brain_mask=None, window_size: int = 3, iterations: int = 10,
                                k_factor: float = 1.0) -> np.ndarray:
    """
    Implementation of the Bloch-Siegert median filter for B1 maps as described in
    "A Statistical Analysis of the Bloch-Siegert B1 Mapping Technique."
    (https://www.researchgate.net/publication/253647096_A_Statistical_Analysis_of_the_Bloch-Siegert_B1_Mapping_Technique)

    This filter is specifically adapted for B1 maps from a Double Spin Echo Double Angle method
    (https://qmrlab.org/mooc/b1-mapping), where the map contains brain area values and non-brain areas marked with NaNs.

    Parameters:
    -----------
    b1_map : ndarray
        The input B1 map with brain regions containing values and non-brain regions containing NaNs
    window_size : int, optional
        Size of the median filter window (default: 3)
    iterations : int, optional
        Number of iterations to apply the filter (default: 1)
    k_factor : float, optional
        Factor multiplying the noise standard deviation to determine threshold (default: 2.0)

    Returns:
    --------
    ndarray
        Smoothed B1 map with preserved brain/non-brain structure
    """

    # The Bloch-Siegert filter works better with squared B1 values.
    # This is because noise in B1² follows a more predictable distribution
    # Convert to B1² domain for filtering
    b1_squared = np.copy(b1_map) ** 2

    # Initialize a working copy that will only contain brain values (replacing NaNs with zeros)
    working_b1_squared = np.zeros_like(b1_squared)
    working_b1_squared[brain_mask] = b1_squared[brain_mask]

    for iteration in range(iterations):
        # Apply median filter to the entire image
        # (Scipy doesn't support masked median filtering directly)
        median_filtered = ndimage.median_filter(working_b1_squared, size=window_size)

        # Calculate residuals within brain mask only
        residuals = np.zeros_like(working_b1_squared)
        residuals[brain_mask] = working_b1_squared[brain_mask] - median_filtered[brain_mask]

        # Calculate noise standard deviation from residuals within brain mask
        noise_std = np.std(residuals[brain_mask])

        # Determine threshold based on noise level
        threshold = k_factor * noise_std

        # Identify voxels where residuals exceed threshold (outliers)
        outlier_mask = (np.abs(residuals) > threshold) & brain_mask

        # Update outliers with median filtered values
        working_b1_squared[outlier_mask] = median_filtered[outlier_mask]

    # Convert back from B1² to B1 domain
    filtered_result = np.sqrt(working_b1_squared)

    # Preserve original NaN structure for non-brain regions
    smoothed_b1 = np.copy(b1_map)
    smoothed_b1[brain_mask] = filtered_result[brain_mask]

    return smoothed_b1
