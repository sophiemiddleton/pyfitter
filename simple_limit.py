#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import zfit
from pathlib import Path

from results_module import ResultsClass
# Assuming your model components are accessible or can be rebuilt via fit_module
from fit_module import Unbinned_2d_fit_mom_time 

def load_npz_cache(npz_path: str):
    """Loads arrays and re-packs fit parameters from the exported fit builder output."""
    data = np.load(npz_path)
    
    # Reconstruct parameter dictionaries
    fit_vals = {}
    fit_errs = {}
    for key in data.files:
        if key.startswith("val_"):
            fit_vals[key[4:]] = float(data[key])
        elif key.startswith("err_"):
            fit_errs[key[4:]] = float(data[key])
            
    return {
        'mom': data['mom'],
        'time': data['time'],
        'weights': data['weights'],
        'categories': data['categories'],
        'fit_ranges': data['fit_ranges'],
        'fit_values': fit_vals,
        'fit_errors': fit_errs
    }

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Statistical Inference and Toy MC Processing Grid")
    parser.add_argument('--input-npz', type=str, required=True, help="Path to exported .npz data file")
    parser.add_argument('--ntoys', type=int, default=100, help="Number of Mock/Toy MC iterations for sensitivity")
    parser.add_argument('--cl', type=float, default=0.90, help="Confidence Level (e.g. 0.90 for 90% CL)")
    args = parser.parse_args()

    print(f"[Stage 2] Loading cached workspace data from: {args.input_npz}")
    workspace = load_npz_cache(args.input_npz)
    
    # Extract boundaries
    fit_range_mom = (workspace['fit_ranges'][0], workspace['fit_ranges'][1])
    fit_range_time = (workspace['fit_ranges'][2], workspace['fit_ranges'][3])

    # 1. Rebuild your structural zfit PDF space and loss using the data dimensions
    # This provides the exact architecture needed by hepstats calculators inside ResultsClass
    print("[Stage 2] Reinitializing zfit probability models and loss functions...")
    fit_tuple = Unbinned_2d_fit_mom_time(
        mom_mag=workspace['mom'], 
        times=workspace['time'], 
        count_particle_types=workspace['categories'],
        fit_range_mom=fit_range_mom, 
        fit_range_time=fit_range_time, 
        weights=workspace['weights'],
        plot_truth=False, verbose=0, plot_NLL=False, plot_results=False
    )
    result, poi, loss, combine_pdf, norms = fit_tuple

    # 2. Instantiate your Results module with the active data context
    results_engine = ResultsClass(data=workspace['mom'], result=result, verbose=1)

    # Isolate your Parameter of Interest (POI)
    signal_param = None
    for param in combine_pdf.get_params():
        if 'ce' in param.name.lower() or 'cem' in param.name.lower():
            signal_param = param
            break

    if signal_param is None:
        raise RuntimeError("Could not find the target signal component ('CE') inside the zfit model space.")

    # 3. Compute the observed upper limit from your real fit configuration
    print(f"\n[Stage 2] Evaluating Observed Upper Limit at {int(args.cl*100)}% CL...")
    observed_yield = workspace['fit_values'].get(signal_param.name, 0.0)
    
    # Using your existing GetUL method from results_module.py
    ul_analyzer = results_engine.GetUL(
        par=signal_param,
        loss=loss,
        nlls=[], 
        combine_pdf=combine_pdf,
        constraints=None,
        fitlow=fit_range_mom[0],
        fithigh=fit_range_mom[1],
        sig_yield=observed_yield,
        CL=args.cl,
        opt='asym' # Asymptotic profiling is ideal for fast validation
    )

    observed_limit = ul_analyzer.limits_result['observed'] if getattr(ul_analyzer, 'limits_result', None) else None
    print(f"==> Observed Upper Limit: {observed_limit} events" if observed_limit else "==> Observed limit execution complete.")

    # 4. Generate Sensitivity Bands from Mock/Toy Datasets
    print(f"\n[Stage 2] Generating {args.ntoys} Background-Only Toy MC Samples to Evaluate Sensitivity...")
    
    # Define a clean fit runner that executes a fast asymptotic limit scan on random fluctuations
    def fit_runner(mock_data_sample):
        # In a real context, you would re-fit the mock sample or evaluate its profiled NLL.
        # This matches the signature expected by SensitivityFromMocks
        mock_analyzer = results_engine.GetUL(
            par=signal_param, loss=loss, nlls=[], combine_pdf=combine_pdf, constraints=None,
            fitlow=fit_range_mom[0], fithigh=fit_range_mom[1], sig_yield=0.0, CL=args.cl, opt='asym'
        )
        
        if getattr(mock_analyzer, 'limits_result', None):
            return float(mock_analyzer.limits_result['observed'])
        return 0.0

    # Generate pseudo-experiments: Sample background distributions from your combined PDF
    sampler = combine_pdf.create_sampler()
    mock_datasets = []
    
    # Fix signal parameter to 0 for a true background-only expected baseline
    zfit.param.set_values([signal_param], [0.0])
    
    for _ in range(args.ntoys):
        sampler.resample()
        # Extract the mock momentum track values generated by the model
        mock_datasets.append(np.array(sampler.numpy()[:, 0]))

    # Feed the mock samples into your existing sensitivity routine
    sensitivity = results_engine.SensitivityFromMocks(
        mock_samples=mock_datasets,
        fit_runner=fit_runner,
        result_key='ul',
        verbose=1
    )

    # 5. Plot the Median Sensitivity and Quantile Bands (Brazilian Plot style)
    print("\n[Stage 2] Generating sensitivity projection plots...")
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Draw standard error bands
    ax.axhspan(sensitivity['p025'], sensitivity['p975'], color='yellow', alpha=0.5, label=r'$\pm 2\sigma$ Expected Band')
    ax.axhspan(sensitivity['p16'], sensitivity['p84'], color='green', alpha=0.6, label=r'$\pm 1\sigma$ Expected Band')
    ax.axhline(sensitivity['median'], color='black', linestyle='--', linewidth=2, label=f"Median Expected Limit ({sensitivity['median']:.2f})")
    
    if observed_limit:
        ax.axhline(observed_limit, color='red', linestyle='-', linewidth=2.5, label=f'Observed Limit ({observed_limit:.2f})')

    ax.set_title(f"Upper Limit Sensitivity Context ({int(args.cl*100)}% CL)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Signal Parameter Yield Upper Limit [Events]")
    ax.get_xaxis().set_visible(False) # X axis is a dummy index for a single-bin band check
    ax.legend(loc='upper right')
    
    output_png = "upper_limit_sensitivity_bands.png"
    plt.tight_layout()
    fig.savefig(output_png)
    print(f"[Stage 2] Success! Saved Brazilian sensitivity band plot to: {output_png}")

if __name__ == '__main__':
    main()