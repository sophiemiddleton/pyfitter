import zfit
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def Unbinned_fit_efficiency(data_mc_np, fit_range, degree = 4):
    obs_mom = zfit.Space('mom', limits=fit_range)
    
    # build the zfit PDF for the efficiency
    N = zfit.Parameter('N', 10000, 0, 1e10)
    zpars_eff = {'N': N}

    # Chebychev polynomial coefficients
    coeffs = [zfit.Parameter(f'c_{i}', 0.1, -1, 1) for i in range(degree + 1)]
    for i, coeff in enumerate(coeffs):
        zpars_eff[f'c_{i}'] = coeff
    
    PDF = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=coeffs, extended=N)

    # convert numpy array to zfit Data
    data_zfit =  zfit.Data.from_numpy(obs=obs_mom, array=data_mc_np)

    # do the fit
    loss = zfit.loss.ExtendedUnbinnedNLL(model=PDF, data=data_zfit)
    minimizer = zfit.minimize.Minuit(
        gradient=True,    # use Minuit’s own gradient
        mode=2,           # full Hesse
        tol=1e-10,        # tighter EDM goal
        maxiter=2000      # more calls
    )
    result = minimizer.minimize(loss, params=zpars_eff.values())

    return result, PDF, N

def plot_fit_result(data_np, fit_range, PDF, N, log_scale=False):
    n_bins = 50
    scale = 1 / n_bins * (fit_range[1] - fit_range[0])
    data_hist, data_binedge = np.histogram(data_np, bins=n_bins, range=fit_range)
    data_bincenter = 0.5 * (data_binedge[1:] + data_binedge[:-1])

    fig, (ax1, ax2) = plt.subplots(2,1, height_ratios=[3,1])
    ax1.hist(data_np, color='black', bins=n_bins, range=fit_range, histtype='step')
    ax1.errorbar(data_bincenter, data_hist, yerr=np.sqrt(data_hist), color='None', ecolor='black', capsize=3)

    ax1.plot(data_bincenter, (PDF.pdf(data_bincenter, norm_range=fit_range) * N* scale).numpy(), '-r')

    for bincenter in data_bincenter:
        print(f"Bin center: {bincenter}, PDF value: {PDF.pdf(bincenter, norm_range=fit_range).numpy() * N * scale}, Data value: {data_hist[np.digitize(bincenter, data_binedge) - 1]}")

    # plot the residuals
    residuals = (data_hist - PDF.pdf(data_bincenter, norm_range=fit_range).numpy() * N * scale) / np.sqrt(data_hist)
    ax2.errorbar(data_bincenter, residuals, yerr=np.ones_like(residuals), fmt='o', color='black', markersize=3, capsize=3)
    ax2.set_xlim(ax1.get_xlim())

    if log_scale:
        ax1.set_yscale('log')
    
    return data_bincenter, residuals
