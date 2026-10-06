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
    '79_v04_2d': {
        'trkpid_threshold':  0.604992,
        'tandip_lower': 0.46,
        'tandip_upper': 0.987581,
        'trkqual_threshold': 0.228,
        't0err_threshold':  0.803592,
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
        'd0_cut_upper' : 76.5017,
        'd0_cut_lower' : -114.066,
    },
    '79_v04_1d': {
        'trkpid_threshold':  0.604992,
        'tandip_lower': 0.46,
        'tandip_upper': 0.987581,
        'trkqual_threshold': 0.228,
        't0err_threshold':  0.803592,
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
        'd0_cut_upper' : 76.5017,
        'd0_cut_lower' : -114.066,
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
    },
    '79_v06_1d': {
        # Cut-set 80 (adjust these values based on actual requirements)
        'trkpid_threshold': 0.609089868117903,
        'tandip_lower': 0.5259282860743348,
        'tandip_upper': 0.8538936366723017,
        'trkqual_threshold': 0.17556473613082713,
        't0err_threshold': 0.7826277436022875,
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
        'd0_cut_lower' : -80.21837,
        'd0_cut_upper' : 81.93914692839157,
    },

}

def get_version_cuts(version='79_v02'):
    """Retrieve cut configuration for a specified version.
    
    Args:
        version (str): Version identifier (e.g., '79_v02', '79_v06', '80')
    
    Returns:
        dict: Cut thresholds for the version, or default if version not found
    """
    # Try exact match first
    if version in VERSION_CUTS:
        return VERSION_CUTS[version]
    
    # Try with _1d suffix (for 1D fits)
    version_1d = f'{version}_1d'
    if version_1d in VERSION_CUTS:
        return VERSION_CUTS[version_1d]
    
    # Try with _2d suffix (for 2D fits)
    version_2d = f'{version}_2d'
    if version_2d in VERSION_CUTS:
        return VERSION_CUTS[version_2d]
    
    # Fallback to 79_v02
    return VERSION_CUTS.get('79_v02')
