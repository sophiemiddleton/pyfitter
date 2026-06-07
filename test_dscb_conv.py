#!/usr/bin/env python3
"""Test the dscb_conv model implementation."""

import numpy as np
import zfit
import sys

# Add path for imports
sys.path.insert(0, '/exp/mu2e/app/users/sophie/pyfitter')

from momentum_pdf_builder import MomPDFBuilder
from helper import make_HistogramPDF

def test_dscb_conv_basic():
    """Test basic dscb_conv model construction."""
    print("Testing dscb_conv model...")
    
    # Create test data (histogram)
    edges = np.linspace(95, 110, 50)
    prob = np.ones(len(edges) - 1)
    prob = prob / prob.sum()  # Normalize
    
    theory_pdf_tuple = (prob, edges)
    
    # Create observable
    obs = zfit.Space('mom', limits=(95, 110))
    
    # Create builder
    builder = MomPDFBuilder(
        comp_pars={'N_DIO': 1000},
        advanced_pars={
            'pdf_theo': 'dscb_conv',
            'treat_params_adv': 'float',
            'fitpars_in_formatted': {
                'theory_pdf': theory_pdf_tuple
            }
        }
    )
    
    try:
        # Build the PDF
        pdf, params = builder.build(obs=obs)
        print("✓ Successfully built dscb_conv PDF")
        print(f"  PDF type: {type(pdf)}")
        print(f"  PDF observable: {pdf.obs}")
        print(f"  Number of parameters: {len(params)}")
        
        # Try to sample from it
        sample = pdf.sample(n=100)
        print(f"✓ Successfully sampled {len(sample)} events from PDF")
        print(f"  Sample mean: {sample.value.numpy().mean():.2f}")
        print(f"  Sample std: {sample.value.numpy().std():.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_dscb_conv_basic()
    sys.exit(0 if success else 1)
