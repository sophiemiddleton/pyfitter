# Fit class
# Fit the data to a a product of PDFs defined in PDF_list

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

import tensorflow as tf
import zfit

from Mom_PDF import poly58
from RecoPlot_module import plotmom_fit

def Unbinned_fit_mom(data_np, fit_range_low, fit_range_hi):
    fit_range = (fit_range_low, fit_range_hi)
    obs_mom = zfit.Space('mom', limits=fit_range)

    # Parameters CE
    mu = zfit.Parameter('mu', 104, 103, 107)
    sigma = zfit.Parameter('sigma', 0.5, 0.1, 1.0)
    N_CE = zfit.Parameter('N_CE', 10, 0, 1e6)

    # Parameters DIO
    a5 = zfit.Parameter('a5', 8.6434e-17, 0, 1e-16)
    a6 = zfit.Parameter('a6', 1.16874e-17, 0, 1e-16)
    a7 = zfit.Parameter('a7', -1.87828e-19, -1e-18, 0)
    a8 = zfit.Parameter('a8', 9.16327e-20, 0, 1e-18)
    N_DIO = zfit.Parameter('N_DIO', 3000, 0, 1e6)

    # Parameters Cosmic
    N_cosmic = zfit.Parameter('N_cosmic', 10, 0, 1e6)

    # Parameters resolution
    mean_res = zfit.Parameter('mean_res', -0.5798, -10, 10)
    sigma_res = zfit.Parameter('sigma_res', 0.2671, 0, 10)
    alphal = zfit.Parameter('alphal', 0.422, 0, 10)
    nl = zfit.Parameter('nl', 25.1, 0, 100)
    alphar = zfit.Parameter('alphar', 2.227, 0, 100)
    nr = zfit.Parameter('nr', 5.954, 0, 100)

    # PDF components
    CE = zfit.pdf.DoubleCB(obs=obs_mom, mu=mu, sigma=sigma, alphal=alphal, nl=nl, alphar=alphar, nr=nr, extended=N_CE)
    DIO = poly58(obs=obs_mom, a5=a5, a6=a6, a7=a7, a8=a8, extended=N_DIO)
    cosmic = zfit.pdf.Uniform(low=fit_range[0], high=fit_range[1], obs=obs_mom, extended=N_cosmic)

    combine_pdf = zfit.pdf.SumPDF([CE, DIO, cosmic])

    list_pdfs = [('CE', CE, N_CE), ('DIO', DIO, N_DIO), ('cosmic', cosmic, N_cosmic)]

    # Convert data to zfit Data
    data_zfit = zfit.Data.from_numpy(array=data_np, obs=obs_mom)

    # Plot before fit with initial guess value
    #plotmom_fit(data_np, fit_range, list_pdfs) # FIXME - make this optional

    loss = zfit.loss.ExtendedUnbinnedNLL(model=combine_pdf, data=data_zfit)
    minimizer = zfit.minimize.Minuit()

    result = minimizer.minimize(loss, params=[mu, sigma, alphal, nl, alphar, nr, N_CE, N_DIO, N_cosmic])
    param_errors, _ = result.errors(method='minuit_minos')

    # Plot after fit
    plotmom_fit(data_np, fit_range, list_pdfs)

    return result


def Binned_fit(data): # Make using zfit
    print('binned_fit not implemented yet')
