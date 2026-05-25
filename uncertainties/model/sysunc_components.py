"""
Systematic uncertainty inventory and specifications.

Each systematic is characterized by:
  - 'type': 'shift' (fixed offset in MeV/c), 'frac' (fractional), or 'shape' (parameter variation)
  - 'process': physics process ('CE', 'DIO', 'RPC', 'Cosmics', 'all')
  - 'component': impact (['mom'], ['time'], or ['mom', 'time'])
  - 'value': [plus, minus] variation (± 1σ)
  - 'source': 'simulation' | 'data-driven' | 'theory'
  - 'status': 'implemented' | 'planned' | 'on-hold'
  - 'method': How to propagate (e.g., 'refit', 'reweight', 'constraint')
  - 'notes': Additional context

References:
  - SU2020: Sustainability update 2020
  - G4: Geant4 simulation studies
"""

sysunc_components = {
    ##### ===== GENERAL ===== #####
    
    'Abs_Mom_Scale': {
        'type': 'shift',
        'process': 'all',
        'component': ['mom'],
        'value': [0.1, 0.1],  # MeV
        'source': 'simulation',
        'status': 'implemented',
        'method': 'refit',
        'notes': 'Absolute momentum scale shift; impacts efficiency cuts (SU2020). Asymmetric variations allowed.'
    },
    
    'Mom_Resolution': {
        'type': 'shape',
        'process': 'all',
        'component': ['mom'],
        'value': [0.05, 0.05],  # sigma variation in MeV
        'source': 'simulation',
        'status': 'planned',
        'method': 'refit_convolution',
        'notes': 'Momentum resolution smearing; affects all processes. Requires simultaneous flat-e fit adjustment.'
    },
    
    ##### ===== SIGNAL (CE) ===== #####
    
    'CE_Tracking_Efficiency': {
        'type': 'frac',
        'process': 'CE',
        'component': ['mom', 'time'],
        'value': [0.01, 0.01],  # 1% placeholder (or absolute if nominal is 0)
        'abs_value': [1.0, 1.0],  # Absolute uncertainty fallback: ±1 event
        'source': 'simulation',
        'status': 'implemented',
        'method': 'constraint',
        'fit_param': 'N_CE',
        'notes': 'Signal tracking efficiency uncertainty; 1% of yield, or ±1 event if yield is 0.'
    },
    
    ##### ===== DIO BACKGROUND ===== #####
    
    'DIO_Theory': {
        'type': 'frac',
        'process': 'DIO',
        'component': ['mom', 'time'],
        'value': [0.025, 0.025],  # 2.5%
        'source': 'theory',
        'status': 'implemented',
        'method': 'constraint',
        'fit_param': 'N_DIO',
        'notes': 'DIO cross-section / form-factor uncertainty. Apply as Gaussian constraint on N_DIO.'
    },
    
    'DIO_PDF_Variant': {
        'type': 'shape',
        'process': 'DIO',
        'component': ['mom'],
        'value': [0.0, 0.0],  # qualitative (use template swap)
        'source': 'theory',
        'status': 'planned',
        'method': 'template_swap',
        'notes': 'Alternative DIO PDF shapes (e.g., Szafron vs other calculations); discrete variation.'
    },
    
    ##### ===== RPC BACKGROUND ===== #####
    
    'RPC_Rate': {
        'type': 'frac',
        'process': 'RPC',
        'component': ['mom', 'time'],
        'value': [0.093, 0.093],  # 9.3% from magnesium composition
        'source': 'simulation',
        'status': 'implemented',
        'method': 'constraint',
        'fit_param': 'N_RPC',
        'notes': 'RPC stopping power and composition uncertainty. Apply as constraint on N_RPC.'
    },
    
    'Pion_Rate': {
        'type': 'frac',
        'process': 'RPC',
        'component': ['mom', 'time'],
        'value': [0.27, 0.09],  # -27% to +9% from G4
        'source': 'simulation',
        'status': 'implemented',
        'method': 'constraint',
        'fit_param': 'N_RPC',
        'notes': 'Pion production in RPC affects RPC yield; asymmetric from Geant4 studies.'
    },
    
    'InternalConv_Rate': {
        'type': 'frac',
        'process': 'RPC',
        'component': ['mom', 'time'],
        'value': [0.0045, 0.0045],  # 0.45%
        'source': 'simulation',
        'status': 'implemented',
        'method': 'constraint',
        'fit_param': 'N_RPC',
        'notes': 'Internal conversion in RPC affects RPC yield; expected to improve with data-driven measurement.'
    },
    
    'OOT_RPC': {
        'type': 'frac',
        'process': 'RPC',
        'component': ['time'],
        'value': [0.00001, 0.00001],  # Placeholder
        'source': 'simulation',
        'status': 'on-hold',
        'method': 'constraint',
        'fit_param': 'N_RPC',
        'notes': 'Out-of-time RPC backgrounds affect RPC yield; requires time-window studies.'
    },
    
    ##### ===== COSMIC BACKGROUND ===== #####
    
    'CRV_Efficiency': {
        'type': 'frac',
        'process': 'Cosmics',
        'component': ['mom', 'time'],
        'value': [0.04, 0.04],  # 4%
        'source': 'simulation',
        'status': 'implemented',
        'method': 'constraint',
        'fit_param': 'N_Cosmic',
        'notes': 'Cosmic ray veto efficiency affects cosmic yield; from CRV detector studies (SU2020).'
    },
    
    'Cosmic_Generator': {
        'type': 'frac',
        'process': 'Cosmics',
        'component': ['mom', 'time'],
        'value': [0.20, 0.20],  # 20%
        'source': 'simulation',
        'status': 'on-hold',
        'method': 'constraint',
        'fit_param': 'N_Cosmic',
        'notes': 'Cosmic ray generator model differences affect cosmic yield; measured via cross-comparison.'
    },
    
    'CRV_Aging': {
        'type': 'frac',
        'process': 'Cosmics',
        'component': ['time'],
        'value': [0.10, 0.10],  # Placeholder
        'source': 'data-driven',
        'status': 'on-hold',
        'method': 'constraint',
        'fit_param': 'N_Cosmic',
        'notes': 'CRV detector aging effects affect cosmic yield; needs long-term data characterization.'
    },
    
    'c1_Cosmic': {
        'type': 'shape',
        'process': 'Cosmics',
        'component': ['mom'],
        'value': [0.022, 0.022],
        'param_value': 0.219,
        'source': 'simulation',
        'status': 'implemented',
        'method': 'constraint',
        'fit_param': 'c1_Cosmic',
        'notes': 'Gaussian prior on Chebyshev coefficient c1 for Cosmic poly2 spectrum shape.'
    },
    
    'c2_Cosmic': {
        'type': 'shape',
        'process': 'Cosmics',
        'component': ['mom'],
        'value': [0.022, 0.022],
        'param_value': -0.108803,
        'source': 'simulation',
        'status': 'implemented',
        'method': 'constraint',
        'fit_param': 'c2_Cosmic',
        'notes': 'Gaussian prior on Chebyshev coefficient c2 for Cosmic poly2 spectrum shape.'
    },

    'c1_RPC': {
        'type': 'shape',
        'process': 'RPC',
        'component': ['mom'],
        'value': [0.0062, 0.0062],
        'param_value': -0.54,
        'source': 'simulation',
        'status': 'implemented',
        'method': 'constraint',
        'fit_param': 'c1_RPC',
        'notes': 'Gaussian prior on Chebyshev coefficient c1 for RPC poly2 spectrum shape.'
    },
    
    'c2_RPC': {
        'type': 'shape',
        'process': 'RPC',
        'component': ['mom'],
        'value': [0.0682, 0.0682],
        'param_value': -0.1792,
        'source': 'simulation',
        'status': 'implemented',
        'method': 'constraint',
        'fit_param': 'c2_RPC',
        'notes': 'Gaussian prior on Chebyshev coefficient c2 for RPC poly2 spectrum shape.'
    },
    
    ##### ===== EXPERIMENTAL SYSTEMATICS ===== #####
    
    'Timing_Calibration': {
        'type': 'shift',
        'process': 'all',
        'component': ['time'],
        'value': [0.05, 0.05],  # ns
        'source': 'simulation',
        'status': 'planned',
        'method': 'refit',
        'fit_param': None,
        'notes': 'Time-of-flight calibration offset; affects all timing-based cuts.'
    },
    
    'Detector_Efficiency': {
        'type': 'frac',
        'process': 'all',
        'component': ['mom', 'time'],
        'value': [0.02, 0.02],  # 2%
        'source': 'simulation',
        'status': 'planned',
        'method': 'reweight',
        'fit_param': None,
        'notes': 'Overall detector tracking efficiency; scale all yields uniformly (affects all components).'
    },
    
    'Alignment': {
        'type': 'shape',
        'process': 'all',
        'component': ['mom'],
        'value': [0.05, 0.05],  # Effective scale variation
        'source': 'simulation',
        'status': 'on-hold',
        'method': 'refit',
        'notes': 'Tracker alignment; treated as secondary momentum scale effect.'
    },
}


# ============================================================================
# Utility functions
# ============================================================================

def validate_sysunc_spec(spec):
    """Validate a systematic uncertainty specification dictionary."""
    required_keys = {'type', 'process', 'component', 'value', 'source', 'status', 'method'}
    if not required_keys.issubset(spec.keys()):
        missing = required_keys - spec.keys()
        raise ValueError(f"Missing required keys: {missing}")
    
    if spec['type'] not in {'shift', 'frac', 'shape'}:
        raise ValueError(f"Invalid type: {spec['type']}")
    
    if spec['status'] not in {'implemented', 'planned', 'on-hold'}:
        raise ValueError(f"Invalid status: {spec['status']}")
    
    if not isinstance(spec['value'], (list, tuple)) or len(spec['value']) != 2:
        raise ValueError(f"value must be [plus, minus] pair, got {spec['value']}")
    
    return True


def get_implemented_systematics():
    """Return only systematics with status='implemented'."""
    return {k: v for k, v in sysunc_components.items() if v.get('status') == 'implemented'}


def get_systematics_by_process(process):
    """Filter systematics by physics process."""
    return {k: v for k, v in sysunc_components.items() if v.get('process') in {process, 'all'}}


def get_systematics_by_component(component):
    """Filter systematics affecting a given component ('mom' or 'time')."""
    return {k: v for k, v in sysunc_components.items() if component in v.get('component', [])}


def get_constraints_only():
    """Return systematics designed for constraint-based propagation."""
    return {k: v for k, v in sysunc_components.items() if v.get('method') == 'constraint' and v.get('status') == 'implemented'}
