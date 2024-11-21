# Fit class
# Fit the data to a a product of PDFs defined in PDF_list

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

import tensorflow as tf
import zfit

from momPDF_module import poly58
from recoplot_module import plotmom_fit

def Unbinned_fit_mom(array_trk, fit_range_low, fit_range_hi):
    fit_range = (fit_range_low, fit_range_hi)
    obs_mom = zfit.Space('mom', limits=fit_range)

    # Parameters CE
    mu = zfit.Parameter('mu', 104, 103, 107)
    sigma = zfit.Parameter('sigma', 0.5, 0.08, 2.0)
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
    data_np = ak.to_numpy(ak.flatten(array_trk['trksegs','mom.mag'], axis=None))
    data_zfit = zfit.Data.from_numpy(array=data_np, obs=obs_mom)

    loss = zfit.loss.ExtendedUnbinnedNLL(model=combine_pdf, data=data_zfit)
    minimizer = zfit.minimize.Minuit()

    result = minimizer.minimize(loss, params=[mu, sigma, alphal, nl, alphar, nr, N_CE, N_DIO, N_cosmic])
    param_errors, _ = result.errors(method='minuit_minos')

    # Plot after fit
    plotmom_fit(data_np, fit_range, list_pdfs)

    return result

def Unbinned_fit_time(data, fit_range_low, fit_range_hi, include_cosmic=True):
    '''
    Fit the time data to a exponential distribution

    Parameters:
        data (awkward array): Time data
        fit_range_low (float): Lower limit of the fit range
        fit_range_hi (float): Upper limit of the fit range
        include_cosmic (bool): Include cosmic distribution in the fit
    '''

    fit_range = (fit_range_low, fit_range_hi)
    obs_time = zfit.Space('time', limits=fit_range)

    #PDF components
    ## Exponential decay
    decay_rate = zfit.Parameter('decay_rate', -1/864, -1/10, -1/1000)
    N_exp = zfit.Parameter('N_exp', 6000, 0, 1e5)
    exp_t = zfit.pdf.Exponential(decay_rate, obs=obs_time, extended=N_exp)
    if (include_cosmic):
        ## Uniform distribution
        N_cosmic = zfit.Parameter('N_cosmic', 0, 0, 1e4)
        cosmic_t = zfit.pdf.Uniform(low=fit_range[0], high=fit_range[1], obs=obs_time, extended=N_cosmic)

    # Combine the time PDFs
    combine_pdf = exp_t
    list_pdfs = [('exp', exp_t, N_exp)]
    if (include_cosmic):
        combine_pdf = zfit.pdf.SumPDF([exp_t, cosmic_t])
        list_pdfs.append(('cosmic', cosmic_t, N_cosmic))

    # Convert data to zfit Data
    data_np = ak.to_numpy(ak.flatten(data['demfit']['time'], axis=None))
    data_zfit = zfit.Data.from_numpy(array=data_np, obs=obs_time)

    # Loss function and minimizer
    loss = zfit.loss.ExtendedUnbinnedNLL(model=combine_pdf, data=data_zfit)
    minimizer = zfit.minimize.Minuit()

    if (include_cosmic):
        result = minimizer.minimize(loss, params=[decay_rate, N_exp, N_cosmic])
    else:
        result = minimizer.minimize(loss, params=[decay_rate, N_exp])
    param_errors, _ = result.errors(method='minuit_minos')

    return result

def Unbinned_2d_fit_mom_time(data, fit_range_mom, fit_range_time, include_cosmic=True):
    '''
    Fit the 2D data to a product of 1D PDFs

    Parameters:
        data (awkward array): 2D data
        fit_range_mom (tuple): Fit range for momentum
        fit_range_time (tuple): Fit range for time
        include_cosmic (bool): Include cosmic distribution in the fit
    '''

    obs_mom = zfit.Space('mom', limits=fit_range_mom)
    obs_time = zfit.Space('time', limits=fit_range_time)
    obs_2D = obs_mom * obs_time

    # time PDF components
    ## Exponential decay
    decay_rate = zfit.Parameter('decay_rate', -1 / 864, -1 / 10, -1 / 1000)
    N_exp = zfit.Parameter('N_exp', 6000, 0, 1e5)
    exp_t = zfit.pdf.Exponential(decay_rate, obs=obs_time)
    if (include_cosmic):
        ## Uniform distribution
        N_cosmic = zfit.Parameter('N_cosmic', 0, 0, 1e4)
        cosmic_t = zfit.pdf.Uniform(low=fit_range_time[0], high=fit_range_time[1], obs=obs_time)

    # momentum PDF components
    ## CE
    mu = zfit.Parameter('mu', 104.329, 103, 107)
    sigma = zfit.Parameter('sigma', 0.434276, 0.08, 2.0)
    alphal = zfit.Parameter('alphal', 0.515, 0, 10)
    nl = zfit.Parameter('nl', 99.997, 0, 200)
    alphar = zfit.Parameter('alphar', 1.335, 0, 100)
    nr = zfit.Parameter('nr', 6.558, 0, 100)
    CE = zfit.pdf.DoubleCB(obs=obs_mom, mu=mu, sigma=sigma, alphal=alphal, nl=nl, alphar=alphar, nr=nr)

    ## DIO
    a5 = zfit.Parameter('a5', 8.6434e-17, 0, 1e-16)
    a6 = zfit.Parameter('a6', 1.16874e-17, 0, 1e-16)
    a7 = zfit.Parameter('a7', -1.87828e-19, -1e-18, 0)
    a8 = zfit.Parameter('a8', 9.16327e-20, 0, 1e-18)
    DIO = poly58(obs=obs_mom, a5=a5, a6=a6, a7=a7, a8=a8)

    if (include_cosmic):
        ## cosmic
        cosmic = zfit.pdf.Uniform(low=fit_range_mom[0], high=fit_range_mom[1], obs=obs_mom)

    # Combined PDFs
    N_CE = zfit.Parameter('N_CE', 34.069, 0, 1e6)
    combine_CE_pdf = zfit.pdf.ProductPDF([CE, exp_t], extended=N_CE)

    N_DIO = zfit.Parameter('N_DIO', 4398.87, 0, 1e6)
    combine_DIO_pdf = zfit.pdf.ProductPDF([DIO, exp_t], extended=N_DIO)

    combine_pdf = zfit.pdf.SumPDF([combine_CE_pdf, combine_DIO_pdf])
    list_pdfs = [('CE', CE, N_CE), ('DIO', DIO, N_DIO)]

    if (include_cosmic):
        N_cosmic = zfit.Parameter('N_cosmic', 0, 0, 1e6)
        combine_cosmic_pdf = zfit.pdf.ProductPDF([cosmic, exp_t], extended=N_cosmic)

        combine_pdf = zfit.pdf.SumPDF([combine_CE_pdf, combine_DIO_pdf, combine_cosmic_pdf])
        list_pdfs.append(('cosmic', cosmic, N_cosmic))

    # Convert data to zfit Data
    data_np_mom = ak.to_numpy(ak.flatten(data['demfit_mom'], axis=None))
    data_np_time = ak.to_numpy(ak.flatten(data['demfit']['time'], axis=None))
    data_zfit = zfit.Data.from_numpy(array=np.column_stack((data_np_mom, data_np_time)), obs=obs_2D)

    # Loss function and minimizer
    loss = zfit.loss.ExtendedUnbinnedNLL(model=combine_pdf, data=data_zfit)
    minimizer = zfit.minimize.Minuit()

    if (include_cosmic):
        result = minimizer.minimize(loss, params=[mu, sigma, alphal, nl, alphar, nr, N_CE, N_DIO, N_cosmic, decay_rate])
    else:
        result = minimizer.minimize(loss, params=[mu, sigma, alphal, nl, alphar, nr, N_CE, N_DIO, decay_rate])

    #param_errors, _ = result.errors(method='minuit_minos')
    result.hesse(method='minuit_hesse', name='Hesse')

    return result

def Binned_fit(data): # Make using zfit
    print('binned_fit not implemented yet')
