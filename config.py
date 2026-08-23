# Global configuration settings

# ===================================================================
# GLOBAL VERBOSITY SETTING - used across all modules
# ===================================================================
# 0=errors only, 1=info, 2=debug, increase for more detail
GLOBAL_VERBOSITY = 1

# ===================================================================
# VERSION-SPECIFIC SELECTION CUTS
# ===================================================================
# Define cut configurations for different analysis versions
# Each version maps cut names to their threshold values
VERSION_CUTS = {
    '79_v02_1d': {
        'trkpid_threshold': 0.54,
        'tandip_lower': 0.57,
        'tandip_upper': 0.85,
        'trkqual_threshold': 0.22,
        't0err_threshold': 0.84,
        'nhits_threshold': 21,
        'crv_veto_dt_lower': 0.0,
        'crv_veto_dt_upper': 150.0,
        'mom_range_t0_lower': 540.0,
        'mom_range_t0_upper': 1650.0,
        'signal_region_p_lower': 103.34,
        'signal_region_p_upper': 104.74,
        'signal_region_t_lower': 640.0,
        'signal_region_t_upper': 1650.0,
        'upstream_veto_dt_lower': 40.0,
        'upstream_veto_dt_upper': 110.0,
        'multi_trk_veto_dt': 150.0,
        'd0_cut' : 1e10,
    },
    '79_v02_2d': {
        'trkpid_threshold': 0.54,
        'tandip_lower': 0.57,
        'tandip_upper': 0.85,
        'trkqual_threshold': 0.22,
        't0err_threshold': 0.84,
        'nhits_threshold': 21,
        'crv_veto_dt_lower': 0.0,
        'crv_veto_dt_upper': 150.0,
        'mom_range_t0_lower': 475.0,
        'mom_range_t0_upper': 1650.0,
        'signal_region_p_lower': 103.34,
        'signal_region_p_upper': 104.74,
        'signal_region_t_lower': 640.0,
        'signal_region_t_upper': 1650.0,
        'upstream_veto_dt_lower': 40.0,
        'upstream_veto_dt_upper': 110.0,
        'multi_trk_veto_dt': 150.0,
        'd0_cut' : 1e10,
    },
    '79_v03_2d': {
        'trkpid_threshold': 0.525,
        'tandip_lower': 0.57,
        'tandip_upper': 0.9,
        'trkqual_threshold': 0.20,
        't0err_threshold': 0.84,
        'nhits_threshold': 21,
        'crv_veto_dt_lower': 0.0,
        'crv_veto_dt_upper': 150.0,
        'mom_range_t0_lower': 475.0,
        'mom_range_t0_upper': 1650.0,
        'signal_region_p_lower': 103.34,
        'signal_region_p_upper': 104.74,
        'signal_region_t_lower': 640.0,
        'signal_region_t_upper': 1650.0,
        'upstream_veto_dt_lower': 40.0,
        'upstream_veto_dt_upper': 110.0,
        'multi_trk_veto_dt': 150.0,
        'd0_cut' : 90,
    },
    '79_v03_1d': {
        'trkpid_threshold': 0.525,
        'tandip_lower': 0.57,
        'tandip_upper': 0.9,
        'trkqual_threshold': 0.20,
        't0err_threshold': 0.84,
        'nhits_threshold': 21,
        'crv_veto_dt_lower': 0.0,
        'crv_veto_dt_upper': 150.0,
        'mom_range_t0_lower': 540.0,
        'mom_range_t0_upper': 1650.0,
        'signal_region_p_lower': 103.34,
        'signal_region_p_upper': 104.74,
        'signal_region_t_lower': 640.0,
        'signal_region_t_upper': 1650.0,
        'upstream_veto_dt_lower': 40.0,
        'upstream_veto_dt_upper': 110.0,
        'multi_trk_veto_dt': 150.0,
        'd0_cut' : 90,
    },
    '80_2d': {
        # Cut-set 80 (adjust these values based on actual requirements)
        'trkpid_threshold': 0.54,
        'tandip_lower': 0.575,
        'tandip_upper': 0.85,
        'trkqual_threshold': 0.155,
        't0err_threshold': 0.85,
        'nhits_threshold': 20,
        'crv_veto_dt_lower': 0.0,
        'crv_veto_dt_upper': 150.0,
        'mom_range_t0_lower': 475.0,
        'mom_range_t0_upper': 1650.0,
        'signal_region_p_lower': 103.34,
        'signal_region_p_upper': 104.74,
        'signal_region_t_lower': 640.0,
        'signal_region_t_upper': 1650.0,
        'upstream_veto_dt_lower': 40.0,
        'upstream_veto_dt_upper': 110.0,
        'multi_trk_veto_dt': 150.0,
        'd0_cut' : 1e10,
    }
    ,
    '80_1d': {
        # Cut-set 80 (adjust these values based on actual requirements)
        'trkpid_threshold': 0.54,
        'tandip_lower': 0.575,
        'tandip_upper': 0.85,
        'trkqual_threshold': 0.155,
        't0err_threshold': 0.85,
        'nhits_threshold': 20,
        'crv_veto_dt_lower': 0.0,
        'crv_veto_dt_upper': 150.0,
        'mom_range_t0_lower': 540.0,
        'mom_range_t0_upper': 1650.0,
        'signal_region_p_lower': 103.34,
        'signal_region_p_upper': 104.74,
        'signal_region_t_lower': 640.0,
        'signal_region_t_upper': 1650.0,
        'upstream_veto_dt_lower': 40.0,
        'upstream_veto_dt_upper': 110.0,
        'multi_trk_veto_dt': 150.0,
        'd0_cut' : 1e10,
    }
}

def get_version_cuts(version='79_v02'):
    """Retrieve cut configuration for a specified version.
    
    Args:
        version (str): Version identifier (e.g., '79_v02', '80')
    
    Returns:
        dict: Cut thresholds for the version, or default if version not found
    """
    return VERSION_CUTS.get(version, VERSION_CUTS.get('79_v02'))
