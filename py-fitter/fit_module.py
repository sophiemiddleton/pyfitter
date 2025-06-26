# Fit the data to a product of PDFs defined in PDF_list

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import tensorflow as tf
import zfit

from momPDF_module import MomModel, poly58
from timePDF_module import TimeModel
from recoplot_module import plotmom_fit, plot_time_fit
from mom_components import mom_components
from time_components import time_components

def Unbinned_fit_mom(mom_mag, track_cat, fit_range_low, fit_range_hi, plot_cat=False, verbose=0):
    """
    Configures and calls the unbinned maximum likelihood fit for momentum using zfit

    Parameters
    ----------
    mom_mag : awkward array of floats
        magnitude of momenta at chosen SID
    track_cat : awkward array of floats
        gives track catagory (corresponds to index in component list)
    fit_range_low, fit_range_hi : float, float
        min and max of fit range (args in the main function)
    plot_cat: bool
        show the MC truth processes on the histogram
    verbose : 1
        print progress statements and debug printouts

    """
    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ initializing fit")
    fit_range = (fit_range_low, fit_range_hi)
    obs_mom = zfit.Space('x', limits=fit_range)

    # PDF components
    pars = []
    pdfs = {}
    norms = {}
    constraints = []
    nlls = []
    if verbose > 0:
          print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ components", mom_components)
    for proc in mom_components:
        pdf = mom_components[proc]['pdf']
        pardict = mom_components[proc]['pars']
        treat_params = mom_components[proc]['treat_params']
        pdfs[proc], norms[proc] = MomModel(obs_mom, pars, proc, pdf, pardict, treat_params, fit_range, constraints)
        if 'nll' in mom_components[proc].keys():
            nlls.extend(mom_components[proc]['nll'].get_nll(pars))

    # build combined PDF
    combine_pdf = zfit.pdf.SumPDF(list(pdfs.values()))

    # Convert data to zfit Data
    mom_np = ak.to_numpy(ak.flatten(mom_mag, axis=None))
    mom_zfit = zfit.Data.from_numpy(array=mom_np, obs=obs_mom)

    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ running minimizer")

    loss = zfit.loss.ExtendedUnbinnedNLL(model=combine_pdf, data=mom_zfit, constraints=constraints)
    for nll in nlls:
        loss = loss+nll

    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss, params=pars)
    
    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ finished minimizing")
    try:
        param_errors, _ = result.errors(method='minuit_minos')
    except:
        print('[py-fitter/fit_module/Unbinned_fit_mom] ❌ ERROR! Invalid fit, postfit parameters may not be optimal')

    if result.valid == True:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ fit is valid")
    else:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ⚠️ WARNING! fit is not valid")
    # Plot after fit
 
    #cat = ak.to_numpy(ak.flatten(track_cat, axis=None)) if plot_cat else None
    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ plotting")
    plotmom_fit(mom_np, fit_range, [(proc,pdfs[proc],norms[proc]) for proc in mom_components.keys()], track_cat) 
    plt.show()


    return result, pars[1], loss, nlls, combine_pdf, constraints

def Unbinned_fit_time(data, fit_range_low, fit_range_hi, plot_cat=False, verbose=0):
    """
    Configures and calls the unbinned maximum likelihood fit for time using zfit

    Parameters
    ----------
    data : awkward array (with cuts applied)
        your data array post-processing
    fit_range_low, fit_range_hi : float, float
        min and max of fit range (args in the main function)
    plot_cat: bool
        show the MC truth processes on the histogram
    verbose : 1
        print progress statements and debug printouts

    """
    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_fit_time] ✅ initializing fit")
    fit_range = (fit_range_low, fit_range_hi)
    obs_time = zfit.Space('time', limits=fit_range)
    
    #build PDF components
    pars = []
    pdfs = {}
    norms = {}
    for proc in time_components:
        pdf = time_components[proc]['pdf']
        pardict = time_components[proc]['pars']
        pdfs[proc], norms[proc] = TimeModel(obs_time, pars, proc, pdf, pardict, fit_range)

    # build combined PDF
    combine_pdf = zfit.pdf.SumPDF(list(pdfs.values()))

    # Convert data to zfit Data
    data_np = ak.to_numpy(ak.flatten(data['trkfit']['trksegs']['time'], axis=None))
    data_zfit = zfit.Data.from_numpy(array=data_np, obs=obs_time)

    # Loss function and minimizer
    loss = zfit.loss.ExtendedUnbinnedNLL(model=combine_pdf, data=data_zfit)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss, params=pars)
    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_fit_time] ✅ finished minimizing")
    try:
        param_errors, _ = result.errors(method='minuit_minos')
    except:
        print('[py-fitter/fit_module] ❌ WARNING! Invalid fit, postfit parameters may not be optimal')
    if result.valid == True:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ fit is valid")
    else:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ⚠️ WARNING! fit is not valid")
    # Plot after fit
    cat = ak.to_numpy(ak.flatten(data['trksegs','cat'], axis=None)) if plot_cat else None
    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_fit_time] ✅ plotting")
    plot_time_fit(data_np, fit_range, [(proc,pdfs[proc],norms[proc]) for proc in time_components.keys()], cat)

    return result

def Unbinned_2d_fit_mom_time(data, fit_range_mom, fit_range_time, plot_cat=False, verbose=0): #FIXME - the following code is not optimal, needs upgrading to new interface
    """
    Configures and calls the unbinned maximum likelihood fit for momentum and time using zfit

    Parameters
    ----------
    data : awkward array (with cuts applied)
        your data array post-processing
    fit_range_mom, fit_range_time : [float, float] [float, float]
        min and max of fit ranges for each dimension (args in the main function)
    plot_cat: bool
        show the MC truth processes on the histogram
    verbose : 1
        print progress statements and debug printouts

    """

    obs_mom = zfit.Space('mom', limits=fit_range_mom)
    obs_time = zfit.Space('time', limits=fit_range_time)
    obs_2D = obs_mom * obs_time

    # time PDF components
    ## Exponential decay for muons
    decay_rate = zfit.Parameter('decay_rate', -1 / 864, -1 / 10, -1 / 1000)
    exp_t = zfit.pdf.Exponential(decay_rate, obs=obs_time)
    ## Exponential decay for pions
    decay_rate_RPC = zfit.Parameter('decay_rate_rpc', -1 / 864, -1 / 10, -1 / 1000)
    exp_t_RPC = zfit.pdf.Exponential(decay_rate_RPC, obs=obs_time)
    ## Uniform distribution
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

    ## RPC
    mu_RPC = zfit.Parameter('mu_rpc', 100, 95, 115)
    sigma_RPC = zfit.Parameter('sigma_rpc', 0.5, 1e-3, 1e3)
    RPC = zfit.pdf.Gauss(obs=obs_mom, mu=mu_RPC, sigma=sigma_RPC)

    ## cosmic
    cosmic = zfit.pdf.Uniform(low=fit_range_mom[0], high=fit_range_mom[1], obs=obs_mom)

    # Combined PDFs
    N_CE = zfit.Parameter('N_CE', 34.069, 0, 1e6)
    combine_CE_pdf = zfit.pdf.ProductPDF([CE, exp_t], extended=N_CE)

    N_DIO = zfit.Parameter('N_DIO', 4398.87, 0, 1e6)
    combine_DIO_pdf = zfit.pdf.ProductPDF([DIO, exp_t], extended=N_DIO)

    N_RPC = zfit.Parameter('N_rpc', 0, 0, 1e6)
    combine_RPC_pdf = zfit.pdf.ProductPDF([RPC, exp_t_RPC], extended=N_RPC)

    N_cosmic = zfit.Parameter('N_cosmic', 0, 0, 1e6)
    combine_cosmic_pdf = zfit.pdf.ProductPDF([cosmic, cosmic_t], extended=N_cosmic)

    combine_pdf = zfit.pdf.SumPDF([combine_CE_pdf, combine_DIO_pdf, combine_RPC_pdf, combine_cosmic_pdf])
    list_pdfs = [('CE', CE, N_CE), ('DIO', DIO, N_DIO), ('RPC', RPC, N_RPC),('cosmic', cosmic, N_cosmic)]

    # Convert data to zfit Data
    data_np_mom = ak.to_numpy(ak.flatten(data['trkfit']['trksegs']['mom.mag'], axis=None))
    data_np_time = ak.to_numpy(ak.flatten(data['trkfit']['trksegs']['time'], axis=None))
    data_zfit = zfit.Data.from_numpy(array=np.column_stack((data_np_mom, data_np_time)), obs=obs_2D)

    # Loss function and minimizer
    loss = zfit.loss.ExtendedUnbinnedNLL(model=combine_pdf, data=data_zfit)
    minimizer = zfit.minimize.Minuit()

    result = minimizer.minimize(loss, params=[mu, sigma, alphal, nl, alphar, nr, N_CE, N_DIO, decay_rate, N_cosmic, mu_RPC, sigma_RPC, N_RPC])
    # the null hypothesis

    #FIXME - can we add some plotting functionality here?

    param_errors, _ = result.errors(method='minuit_minos')
    #result.hesse(method='minuit_hesse', name='Hesse')

    return result
