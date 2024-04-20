# Fit class
# Fit the data to a a product of PDFs defined in PDF_list

# Original Author: Leo Borrel
# Edits: Sophie Middleton
# Date: 2024-04-19

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

import tensorflow as tf
import zfit

from MomShapes_module import poly58

def Unbinned_fit_mom(data, fit_range_low, fit_range_hi):
    fit_range = (fit_range_low, fit_range_hi)
    obs_mom = zfit.Space('mom', limits=fit_range)

    # Parameters CE
    mu = zfit.Parameter('mu', 104, 103, 107)
    sigma = zfit.Parameter('sigma', 0.5, 0.1, 1)
    N_CE = zfit.Parameter('N_CE', 10, 0, 1e6)

    # Parameters DIO
    a5 = zfit.Parameter('a5', 8.6434e-17, 0, 1e-16)
    a6 = zfit.Parameter('a6', 1.16874e-17, 0, 1e-16)
    a7 = zfit.Parameter('a7', -1.87828e-19, -1e-18, 0)
    a8 = zfit.Parameter('a8', 9.16327e-20, 0, 1e-18)
    N_DIO = zfit.Parameter('N_DIO', 3000, 0, 1e6)

    # Parameters Cosmic
    N_Cosmic = zfit.Parameter('N_Cosmic', 10, 0, 1e6)

    # PDF components
    ce = zfit.pdf.Gauss(obs=obs_mom, mu=mu, sigma=sigma, extended=N_CE) #TODO - shouldnt be a Gaussian - try the crystall ball, or the CELL spectrum? Can we do a convolution?
    dio = poly58(obs=obs_mom, a5=a5, a6=a6, a7=a7, a8=a8, extended=N_DIO)
    cosmic = zfit.pdf.Uniform(low=fit_range[0], high=fit_range[1], obs=obs_mom, extended=N_Cosmic)

    combine_pdf = zfit.pdf.SumPDF([ce, dio, cosmic])

    # Convert data to zfit Data
    data_np = ak.to_numpy(data['deent','mom'])
    data_zfit = zfit.Data.from_numpy(array=data_np, obs=obs_mom)

    # Plot before fit with initial guess value
    plot_fit(data_np, fit_range, ce, N_CE, dio, N_DIO, cosmic, N_Cosmic)

    loss = zfit.loss.ExtendedUnbinnedNLL(model=combine_pdf, data=data_zfit)
    minimizer = zfit.minimize.Minuit()

    result = minimizer.minimize(loss, params=[mu, sigma, N_CE, N_DIO, N_Cosmic])
    param_errors, _ = result.errors()

    print('Full result info: ', result)

    # Plot after fit
    plot_fit(data_np, fit_range, ce, N_CE, dio, N_DIO, cosmic, N_Cosmic)


def plot_fit(data, fit_range, ce, N_CE, dio, N_DIO, cosmic, N_Cosmic):
    n_bins = 100 # TODO remove hardcoding
    mom_plot = np.linspace(fit_range[0], fit_range[1], n_bins)
    scale = 1 / n_bins * (fit_range[1] - fit_range[0])

    data_hist, data_binedge = np.histogram(data, bins=n_bins, range=fit_range)
    data_bincenter = 0.5 * (data_binedge[1:] + data_binedge[:-1])

    ce_plot = (ce.pdf(mom_plot) * N_CE * scale).numpy()
    dio_plot = (dio.pdf(mom_plot) * N_DIO * scale).numpy()
    cosmic_plot = (cosmic.pdf(mom_plot) * N_Cosmic * scale).numpy()
    combine_plot = ce_plot + dio_plot + cosmic_plot

    fig, (ax1, ax2) = plt.subplots(2,1, height_ratios=[3,1])
    ax1.hist(data, color='black', bins=n_bins, range=fit_range, histtype='stepfilled', alpha=0.0)
    ax1.hist(data, color='black', bins=n_bins, range=fit_range, histtype='step')
    ax1.errorbar(data_bincenter, data_hist, yerr=np.sqrt(data_hist), color='None', ecolor='black', capsize=3)

    ax1.plot(mom_plot, ce_plot, '--', color='blue', label='CE')
    ax1.plot(mom_plot, dio_plot, ':', color='green', label='DIO')
    ax1.plot(mom_plot, cosmic_plot, '-.', color='orange', label='Cosmic')
    ax1.plot(mom_plot, combine_plot, color='red', label='Total')

    ax1.grid(True)
    ax1.set_yscale('log')
    ax1.set_xlim(fit_range) #FIXME change range to variables
    ax1.set_ylim([1e-1, 1e3]) #TODO - shouldnt be hardcoded (what if we have a bigger data set?)
    ax1.set_xlabel('Momentum [MeV/c]')
    ax1.set_ylabel('# of events')
    ax1.legend()

    ax2.errorbar(mom_plot, np.abs(combine_plot - data_hist), yerr=np.sqrt(data_hist), color='None', marker='+', markerfacecolor='black', ecolor='black', capsize=3)

    ax2.grid(True)
    ax2.set_xlim(fit_range)
    ax2.set_xlabel('Momentum [MeV/c]')
    ax2.set_ylabel('residual')

    #plt.show()
