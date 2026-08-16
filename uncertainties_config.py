"""
Standard physics systematic uncertainties for Mu2e analysis.

Defines log-normal (lnN) uncertainties typical for Mu2e CEL searches.
These are applied by background/signal when generating datacards.
"""

# Standard physics uncertainties for Mu2e
# Format: {
#   'syst_name': {
#     'type': 'lnN' (log-normal) or 'shape',
#     'description': Human-readable description,
#     'processes': {process_name: effect_factor, ...}
#       - effect_factor for lnN: 1.0 + fractional_uncertainty
#       - e.g., 1.10 = 10% uncertainty
#   }
# }

STANDARD_SYSTEMATICS = {
    'lumi': {
        'type': 'lnN',
        'description': 'Luminosity / beam intensity uncertainty',
        'processes': {
            'CE': 1.10,       # 10%
            'Cosmic': 1.10,
            'DIO': 1.10,
            'RPC_ext': 1.10,
            'RPC_int': 1.10,
            'RMC': 1.10,
        }
    },
    'ceHN': {
        'type': 'lnN',
        'description': 'Cosmic ray energy scale + acceptance uncertainty',
        'processes': {
            'Cosmic': 1.025,  # 2.5%
        }
    },
    'dioN': {
        'type': 'lnN',
        'description': 'DIO spectrum shape + rate + RPC scale uncertainty',
        'processes': {
            'DIO': 1.10,      # 10%
            'RPC': 1.10,      # 10%
        }
    },
    'rpcsN': {
        'type': 'lnN',
        'description': 'RPC detection efficiency and acceptance',
        'processes': {
            'RPC': 1.15,      # 15% (average of different detection modes)
        }
    },
}


def get_systematics_for_process(process_name: str) -> dict:
    """
    Get all systematics affecting a given process.
    
    Args:
        process_name: Name of process (e.g., 'Cosmic', 'DIO')
    
    Returns:
        dict: {syst_name: effect_factor, ...}
    
    Example:
        >>> get_systematics_for_process('Cosmic')
        {'lumi': 1.10, 'ceHN': 1.025}
    """
    result = {}
    for syst_name, syst_info in STANDARD_SYSTEMATICS.items():
        if process_name in syst_info['processes']:
            result[syst_name] = syst_info['processes'][process_name]
    return result


def get_all_systematics_names() -> list:
    """Get list of all systematic names."""
    return list(STANDARD_SYSTEMATICS.keys())


def get_systematics_description() -> str:
    """Get human-readable description of all systematics."""
    lines = ["Standard Mu2e Physics Systematics:\n"]
    for syst_name, syst_info in STANDARD_SYSTEMATICS.items():
        lines.append(f"\n{syst_name}:")
        lines.append(f"  Type: {syst_info['type']}")
        lines.append(f"  Description: {syst_info['description']}")
        lines.append(f"  Processes:")
        for proc, effect in syst_info['processes'].items():
            uncertainty_pct = abs(effect - 1.0) * 100
            lines.append(f"    {proc:15} {effect:.3f} ({uncertainty_pct:.1f}%)")
    return "\n".join(lines)


if __name__ == '__main__':
    # Print summary
    print(get_systematics_description())
    print("\n" + "="*70)
    print("\nExample: Get systematics for Cosmic")
    print(get_systematics_for_process('Cosmic'))
