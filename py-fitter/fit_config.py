# fit_config.py: Centralized Configuration for Mu2e Momentum Fits

from typing import Dict, Tuple, Union

# --- 1. Global Fit Ranges ---
MOM_FIT_RANGE: Tuple[float, float] = (95.0, 110.0)
TIME_FIT_RANGE: Tuple[float, float] = (475.0, 1650.0)
from zfit import Space
OBS_MOM_STATIC = Space('x', limits=(95.0, 110.0)) # Use a safe, full range
# --- 2. Default Normalizations (Extended Likelihood Initial Values) ---
DEFAULT_NORMS: Dict[str, Union[int, float]] = {
    'CE': 600,             # Conversion Electron Signal
    'DIO': 55000,          # Decay in Orbit Background (Dominant)
    'Cosmic': 200,         # Cosmic Ray Background
    'RPC': 1               # Radiative Pion Capture Background
}

# --- 3. Configuration Paths (For Convolution Model 'theo_exp') ---
EFFICIENCY_PATH: str = '../common/efficiency.pkl'
DIO_THEORY_PATH: str = '/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v10_34_00/Offline/ConditionsService/data/czarnecki_szafron_Al_2016.tbl'
# Resolution/Loss component paths:
RES_PARAM_PATH: str = '../common/fitpars_flat_res_entrance_gcb.pkl'
LOSS_PARAM_PATH: str = '../common/fitpars_flat_loss_entrance_landau_unbinned.pkl'

# --- 4. Plotting Configuration ---
PLOT_N_BINS: int = 100

