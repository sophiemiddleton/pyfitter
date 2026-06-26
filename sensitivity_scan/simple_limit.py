"""
simple sensitivity scan for background only assumption, requires:

- initial weighted fit build bkg input in form of snap shot
- to run : python simple_limit.py
"""
import numpy as np
import awkward as ak
import zfit
import tensorflow as tf
import matplotlib.pyplot as plt
from results_module import ResultsClass

# ==============================================================================
# 1. DEFINE CUSTOM DIO MOMENTUM CLASS
# ==============================================================================
from custom_models import poly58

# ==============================================================================
# 2. MODEL REBUILDER FROM NPZ CONFIG
# ==============================================================================
def rebuild_model_from_snapshot(config_path):
    config = np.load(config_path)
    fit_ranges = config['fit_ranges']
    fit_range_mom = (fit_ranges[0], fit_ranges[1])
    fit_range_time = (fit_ranges[2], fit_ranges[3])
    
    obs_mom = zfit.Space('mom', limits=fit_range_mom)
    obs_time = zfit.Space('time', limits=fit_range_time)
    obs_space = obs_mom * obs_time
    
    def get_val(keywords, default):
        for file_key in config.files:
            if file_key.startswith('val_'):
                if all(kw.lower() in file_key.lower() for kw in keywords):
                    return float(config[file_key])
        return default

    n_ce = get_val(['N_CE'], 0.3375)
    n_cosmic = get_val(['N_Cosmic'], 329.57)
    n_rpc = get_val(['N_RPC'], 9.41)
    n_dio = get_val(['N_DIO'], 1429.67)
    
    poi_parameter = zfit.Parameter('N_CE', n_ce, lower=0, upper=40.0)
    yield_cosmic = zfit.Parameter('N_Cosmic', n_cosmic, lower=0, upper=2000)
    yield_rpc = zfit.Parameter('N_RPC', n_rpc, lower=0, upper=200)
    yield_dio = zfit.Parameter('N_DIO', n_dio, lower=0, upper=5000)
    
    print(f"[Snapshot Map] Re-instantiated Model Yields:")
    print(f"  Signal (N_CE): {n_ce:.4f} | Cosmic: {n_cosmic:.2f} | RPC: {n_rpc:.2f} | DIO: {n_dio:.2f}")

    c1_cosmic = get_val(['c1_Cosmic'], 0.1409)
    c2_cosmic = get_val(['c2_Cosmic'], -0.1360)
    c1_rpc = get_val(['c1_RPC'], -0.1531)
    c2_rpc = get_val(['c2_RPC'], -0.0410)
    a5_dio = get_val(['a5_DIO'], 1.266e-17)
    a6_dio = get_val(['a6_DIO'], 3.774e-17)
    a7_dio = get_val(['a7_DIO'], -1.364e-19)
    a8_dio = get_val(['a8_DIO'], 9.981e-20)
    decay_rate_mu = get_val(['decay_rate_mu'], -0.001131)
    decay_rate_pi = get_val(['decay_rate_pi'], -0.0553)

    time_uniform = zfit.pdf.Uniform(obs=obs_time, low=fit_range_time[0], high=fit_range_time[1])
    time_muexp = zfit.pdf.Exponential(lambda_=decay_rate_mu, obs=obs_time)
    time_piexp = zfit.pdf.Exponential(lambda_=decay_rate_pi, obs=obs_time)

    cosmic_mom_pdf = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=[c1_cosmic, c2_cosmic])
    cosmic_base_2d = zfit.pdf.ProductPDF([cosmic_mom_pdf, time_uniform], obs=obs_space)
    
    rpc_mom_pdf = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=[c1_rpc, c2_rpc])
    rpc_base_2d = zfit.pdf.ProductPDF([rpc_mom_pdf, time_piexp], obs=obs_space)
    
    dio_mom_param_a5 = zfit.Parameter('a5_DIO_param', a5_dio, floating=False)
    dio_mom_param_a6 = zfit.Parameter('a6_DIO_param', a6_dio, floating=False)
    dio_mom_param_a7 = zfit.Parameter('a7_DIO_param', a7_dio, floating=False)
    dio_mom_param_a8 = zfit.Parameter('a8_DIO_param', a8_dio, floating=False)
    dio_mom_pdf = poly58(obs=obs_mom, a5=dio_mom_param_a5, a6=dio_mom_param_a6, a7=dio_mom_param_a7, a8=dio_mom_param_a8)
    dio_base_2d = zfit.pdf.ProductPDF([dio_mom_pdf, time_muexp], obs=obs_space)
    
    signal_mom_pdf = zfit.pdf.Gauss(mu=105.0, sigma=0.5, obs=obs_mom)
    signal_base_2d = zfit.pdf.ProductPDF([signal_mom_pdf, time_muexp], obs=obs_space)
    
    dio_extended = dio_base_2d.create_extended(yield_dio)
    cosmic_extended = cosmic_base_2d.create_extended(yield_cosmic)
    rpc_extended = rpc_base_2d.create_extended(yield_rpc)
    signal_extended = signal_base_2d.create_extended(poi_parameter)
    
    combined_pdf = zfit.pdf.SumPDF([dio_extended, cosmic_extended, rpc_extended, signal_extended])
    
    return combined_pdf, poi_parameter, obs_space, fit_range_mom, fit_range_time, [dio_mom_pdf, cosmic_mom_pdf, rpc_mom_pdf], [time_muexp, time_uniform, time_piexp]


# ==============================================================================
# 3. HELPER PLOTTING FUNCTION FOR TOYS
# ==============================================================================
def plot_toy_projections(toy_index, data_array, fit_range_mom, fit_range_time, mom_pdfs, time_pdfs):
    """Generates and saves 1D projections for debugging individual pseudo-experiments."""
    # Extract structural dimensions from the numpy structured sample array
    mom_data = data_array[:, 0]
    time_data = data_array[:, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Momentum Projection
    counts, bins, _ = ax1.hist(mom_data, bins=40, range=fit_range_mom, alpha=0.6, color='gray', label='Toy Data')
    x_plot = np.linspace(fit_range_mom[0], fit_range_mom[1], 200)
    
    # Scale shape curves roughly to the histogram contents for visual reference
    bin_width_mom = (fit_range_mom[1] - fit_range_mom[0]) / 40.0
    ax1.set_title(f"Toy {toy_index} - Momentum Space Projection")
    ax1.set_xlabel("Momentum [MeV/c]")
    ax1.set_ylabel(f"Events / {bin_width_mom:.2f} MeV/c")
    ax1.legend()
    
    # Panel 2: Time Projection
    counts_t, bins_t, _ = ax2.hist(time_data, bins=40, range=fit_range_time, alpha=0.6, color='gray', label='Toy Data')
    t_plot = np.linspace(fit_range_time[0], fit_range_time[1], 200)
    
    bin_width_time = (fit_range_time[1] - fit_range_time[0]) / 40.0
    ax2.set_title(f"Toy {toy_index} - Time Space Projection")
    ax2.set_xlabel("Time [ns]")
    ax2.set_ylabel(f"Events / {bin_width_time:.1f} ns")
    ax2.legend()
    
    plt.tight_layout()
    plot_name = f"toy_projection_{toy_index}.png"
    plt.savefig(plot_name, dpi=150)
    plt.close()
    print(f"   [Plotting] Saved component diagnostic: {plot_name}")


# ==============================================================================
# 4. EXECUTION ROUTINE WITH DISTRIBUTION PLOT
# ==============================================================================
def main():
    snapshot_file = "fit_snapshot.npz"
    rebuild_out = rebuild_model_from_snapshot(snapshot_file)
    combined_pdf, poi_parameter, obs_space, fit_range_mom, fit_range_time, mom_pdfs, time_pdfs = rebuild_out
    
    toy_minimizer = zfit.minimize.Minuit()
    import scipy.optimize as opt

    def frequentist_limit_runner(mock_data_sample):
        # 1. Convert raw numpy sample array into a live zfit Data object
        data_zfit = zfit.Data.from_numpy(array=mock_data_sample, obs=obs_space)
        
        # 2. Extract the background yield parameters dynamically from the imported model
        model_params = {p.name: p for p in combined_pdf.get_params()}
        fit_yield_cosmic = model_params.get('N_Cosmic')
        fit_yield_dio = model_params.get('N_DIO')
        fit_yield_rpc = model_params.get('N_RPC')
        
        if fit_yield_cosmic is None or fit_yield_dio is None:
            raise KeyError("Could not locate background yield parameters 'N_Cosmic' or 'N_DIO' within the combined PDF.")
        
        # 3. Read the nominal values directly from the imported parameters 
        nominal_cosmic = float(fit_yield_cosmic.value())
        nominal_dio = float(fit_yield_dio.value())
        nominal_rpc = float(fit_yield_rpc.value())
        
        cosmic_uncertainty = nominal_cosmic * 0.055   
        dio_uncertainty = nominal_dio * 0.026         
        rpc_uncertainty = nominal_rpc * 0.7        
        
        # 4. Construct the auxiliary Gaussian constraints dynamically
        constraints = [
            zfit.constraint.GaussianConstraint(params=[fit_yield_cosmic], observation=nominal_cosmic, uncertainty=cosmic_uncertainty),
            zfit.constraint.GaussianConstraint(params=[fit_yield_dio], observation=nominal_dio, uncertainty=dio_uncertainty),
            zfit.constraint.GaussianConstraint(params=[fit_yield_rpc], observation=nominal_rpc, uncertainty=rpc_uncertainty)
        ]
        
        # 5. Build the Extended Unbinned NLL
        loss_function = zfit.loss.ExtendedUnbinnedNLL(
            model=combined_pdf, 
            data=data_zfit, 
            constraints=constraints
        )
        
        # 4. Allow the POI to float into slightly negative space to prevent boundary traps
        poi_parameter.lower = -20.0
        poi_parameter.upper = 100.0
        poi_parameter.floating = True
        
        try:
            # Find the true global minimum for this toy dataset
            global_fit = toy_minimizer.minimize(loss_function)
            if not global_fit.valid:
                return np.nan
            
            nll_best = float(global_fit.fmin)
            poi_best_val = float(global_fit.params[poi_parameter]['value'])
        except Exception:
            return np.nan

        # 5. Profile Scanning over the full 2D shape space
        target_delta_nll = 1.3528  # 90% CL one-sided limit threshold
        poi_parameter.floating = False 
        
        def evaluate_shape_profile(test_yield):
            if test_yield < poi_best_val:
                return -target_delta_nll
                
            poi_parameter.set_value(test_yield)
            try:
                profile_fit = toy_minimizer.minimize(loss_function)
                return (float(profile_fit.fmin) - nll_best) - target_delta_nll
            except Exception:
                return 999.0  

        # 6. Search for the crossover point using a controlled bracket
        left_bracket = poi_best_val
        right_bracket = max(10.0, poi_best_val + 15.0)
        
        try:
            while evaluate_shape_profile(right_bracket) < 0:
                right_bracket += 10.0
                if right_bracket > 120.0:
                    return np.nan
            
            # Extract the exact upper limit from the full shape profile
            upper_limit_estimate = opt.brentq(evaluate_shape_profile, left_bracket, right_bracket, xtol=0.1)
            
            # Apply standard physical boundary protection at the very end
            if upper_limit_estimate < 0.0:
                upper_limit_estimate = 0.0
                
            return upper_limit_estimate
            
        except Exception:
            return np.nan

    print("\n[Sampling] Throwing background-only toys from the physics model...")
    num_external_toys = 2000  # Scaled to 1000 toys cleanly
    all_limits = []
    
    sampler = combined_pdf.create_sampler()
    
    print(f"\n[Execution] Launching Sensitivity checks over {num_external_toys} toys...")
    for i in range(num_external_toys):
        sampler.resample({poi_parameter: 0})
        raw_sample = sampler.numpy()
        
        # Plot and save diagnostic histograms for just the first 4 toys
        if i < 4:
            plot_toy_projections(i, raw_sample, fit_range_mom, fit_range_time, mom_pdfs, time_pdfs)
            
        # Execute the shape profile limit calculation exactly once per toy
        limit = frequentist_limit_runner(raw_sample)
        if not np.isnan(limit):
            all_limits.append(limit)
            
        if (i + 1) % 50 == 0:
            print(f"   --> Progress: Completed {i + 1} / {num_external_toys} toys...")

    # ==========================================================================
    # 5. COMPUTE STATISTICAL SENSITIVITY BANDS NATIVELY
    # ==========================================================================
    all_limits = np.array(all_limits)
    median = np.percentile(all_limits, 50.0)
    p16 = np.percentile(all_limits, 15.87)
    p84 = np.percentile(all_limits, 84.13)
    p025 = np.percentile(all_limits, 2.27)
    p975 = np.percentile(all_limits, 97.73)

    # ==========================================================================
    # 6. PLOT SENSITIVITY BAND DISTRIBUTION
    # ==========================================================================
    plt.figure(figsize=(10, 6.5))
    
    # Increase bins to 35 to resolve the detailed continuous shape of 1000 pseudo-experiments
    plt.hist(all_limits, bins=35, histtype='stepfilled', color='lightblue', 
             alpha=0.4, edgecolor='darkblue', linewidth=1.2, label='Toy Pseudo-experiments')
    
    # Draw Standard Error Bands as shaded regions
    plt.axvspan(p025, p975, color='yellow', alpha=0.15, label=r'$\pm$2$\sigma$ Expected Range (Yellow Band)')
    plt.axvspan(p16, p84, color='green', alpha=0.25, label=r'$\pm$1$\sigma$ Expected Range (Green Band)')
    plt.axvline(median, color='black', linestyle='--', linewidth=2, label=f'Median Expected Limit ({median:.2f})')
    
    # Annotate markers cleanly without overlapping titles
    ylim_max = plt.gca().get_ylim()[1]
    plt.text(median + 0.3, ylim_max * 0.85, f'Med: {median:.2f}', color='black', weight='bold', fontsize=11)
    plt.text(p16 - 2.5, ylim_max * 0.70, f'-1$\sigma$: {p16:.2f}', color='darkgreen', fontsize=10)
    plt.text(p84 + 0.3, ylim_max * 0.70, f'+1$\sigma$: {p84:.2f}', color='darkgreen', fontsize=10)
    
    plt.title("Expected 90% CL Upper Limit Sensitivity Distribution", fontsize=13, weight='bold')
    plt.xlabel("Upper Limit on Signal Yield ($N_{CE}$)", fontsize=11)
    plt.ylabel("Pseudo-experiments / Bin", fontsize=11)
    plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    plt.grid(axis='y', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("sensitivity_distribution_bands.png", dpi=150)
    print("\n[Plotting] Saved sensitivity summary chart: sensitivity_distribution_bands.png")
    
    print("\n" + "="*60)
    print("      SUCCESS: CALCULATIONS AND PLOTS COMPLETE (All TOYS)")
    print("="*60)
    print(f"Median Expected Upper Limit (90% CL): {median:.3f}")
    print(f"Green Band (-1σ / +1σ):       {p16:.3f} to {p84:.3f}")
    print(f"Yellow Band (-2σ / +2σ):      {p025:.3f} to {p975:.3f}")
    print("="*60)

if __name__ == '__main__':
    main()