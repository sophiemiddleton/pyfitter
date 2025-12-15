# theo_components.py (Updated Theoretical Spectrum Definition)

import numpy as np
import math

# --- Physical Constants (Mu2e Standards 2025) ---
E_MAX = 104.97            # MeV
ALPHA = 1.0 / 137.035999  # Fine structure constant
M_E   = 0.510998          # Electron mass [MeV]

def LeadingLog(E):
    """
    Theoretical Leading Log conversion spectrum formula.
    Standard implementation for single-value evaluation.
    """
    prefactor = (1.0 / E_MAX) * (ALPHA / (2.0 * math.pi))
    log_term  = math.log(4.0 * E**2 / M_E**2) - 2.0
    energy_term = (E**2 + E_MAX**2) / (E_MAX * (E_MAX - E))
    
    val = prefactor * log_term * energy_term
    return max(0.0, val)

def binned_spectrum_CeLL(binwidth: float = 0.1):
    """
    Calculates the binned Conversion Electron spectrum data (values, edges).
    """
    # 1. Determine binning grid
    nbins = int(math.ceil(E_MAX / binwidth))
    upedge = binwidth * nbins
    
    edges = np.linspace(0.0, upedge, nbins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    
    # 2. Vectorized calculation (Replaces np.vectorize for speed)
    # Clip centers slightly away from E_MAX to avoid division by zero
    safe_centers = np.clip(centers, 1e-6, E_MAX - 1e-6)
    
    prefactor = (1.0 / E_MAX) * (ALPHA / (2.0 * np.pi))
    log_vals  = np.log(4.0 * safe_centers**2 / M_E**2) - 2.0
    en_vals   = (safe_centers**2 + E_MAX**2) / (E_MAX * (E_MAX - safe_centers))
    
    values = prefactor * log_vals * en_vals
    
    # 3. Apply physical boundaries
    values = np.where(centers < E_MAX, values, 0.0)
    values = np.maximum(0.0, values)
    
    # 4. Normalization and Endpoint handling
    # Handle the 'delta-like' behavior at the endpoint bin
    integral_excluding_last = np.sum(values[:-1] * binwidth)
    
    if integral_excluding_last < 1.0:
        values[-1] = (1.0 - integral_excluding_last) / binwidth
    else:
        # Fallback to standard unit normalization if integral already exceeds 1
        values /= np.sum(values * binwidth)
        
    return (values, edges)

