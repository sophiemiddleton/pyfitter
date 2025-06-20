import hepunits as u
import matplotlib.pyplot as plt
import mplhep
import numpy as np
import zfit
import zfit.z.numpy as znp
import zfit_physics as zphys
from utils import *

plt.rcParams['figure.figsize'] = (8, 6)

mu_true = 4180 * u.MeV
sigma_true = 50 * u.MeV

# number of signal and background
n_sig_rare = 120
n_bkg_rare = 700

# create some data
signal_np = np.random.normal(loc=mu_true, scale=sigma_true, size=n_sig_rare)
bkg_np_raw = np.random.exponential(size=20000, scale=700)
bkg_np = bkg_np_raw[bkg_np_raw < 1000][:n_bkg_rare] + 5000  # just cutting right, but zfit could also cut

# Firstly, the observable and its range is defined
obs = zfit.Space('Bmass', 5000, 6000, label="$B_{mass} [MeV/c^2]$")  # for whole range

# load data into zfit and let zfit concatenate the data
signal_data = zfit.Data(signal_np, obs=obs)
bkg_data = zfit.Data(bkg_np, obs=obs)
data = zfit.data.concat([signal_data, bkg_data])
# (we could also do it manually)
# data = zfit.Data(array=np.concatenate([signal_np, bkg_np], axis=0), obs=obs)

# Parameters are specified:  (name , initial, lower, upper) whereas lower, upper are optional
mu = zfit.Parameter('mu', 5279, 5100, 5400, label=r"$\mu$ [MeV/c^2]$")
sigma = zfit.Parameter('sigma', 20, 1, 200, label=r"$\sigma$ [MeV/c^2]$")
sig_yield = zfit.Parameter('sig_yield', n_sig_rare + 30,
                           step_size=3)  # step size: default is small, use appropriate
signal = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs, extended=sig_yield)

lam = zfit.Parameter('lambda', -0.002, -0.1, -0.00001, step_size=0.001)  # floating, also without limits
bkg_yield = zfit.Parameter('bkg_yield', n_bkg_rare - 40, step_size=1)
comb_bkg = zfit.pdf.Exponential(lam, obs=obs, extended=bkg_yield)

# The final model is the combination of the signal and backgrond PDF
model = zfit.pdf.SumPDF([comb_bkg, signal])

model.plot.plotpdf()

observ = 5270
constraint = zfit.constraint.GaussianConstraint(mu, observation=observ * u.MeV, sigma=15 * u.MeV)
constr_space = zfit.Space('mu', 4500, 6000)
gaussconstr = zfit.pdf.Gauss(mu=mu, sigma=15, obs=constr_space, extended=1)
constr_val = zfit.Data.from_numpy(array=np.array(observ), obs=constr_space)

nll = zfit.loss.ExtendedUnbinnedNLL(model, data, constraints=constraint)
nllalt = zfit.loss.ExtendedUnbinnedNLL(model, data)
nllconstraint = zfit.loss.ExtendedUnbinnedNLL(gaussconstr, constr_val)
nll2 = nllalt + nllconstraint
init_vals = {mu: mu.value(), sigma: sigma.value(), sig_yield: sig_yield.value(), lam: lam.value(),
             bkg_yield: bkg_yield.value()}
zfit.param.set_values(init_vals)
minimizer = zfit.minimize.Minuit(gradient="zfit")
result = minimizer.minimize(nll2)
result.hesse();


n_sig_reso = 40000
n_bkg_reso = 3000

# create some data
signal_np_reso = np.random.normal(loc=mu_true, scale=sigma_true * 0.7, size=n_sig_reso)
bkg_np_raw_reso = np.random.exponential(size=20000, scale=900)
bkg_np_reso = bkg_np_raw_reso[bkg_np_raw_reso < 1000][:n_bkg_reso] + 5000

# load data into zfit
obs_reso = zfit.Space('Bmass_reso', 5000, 6000)
signal_data_reso = zfit.Data(signal_np_reso, obs=obs_reso)
bkg_data_reso = zfit.Data(bkg_np_reso, obs=obs_reso)
data_reso = zfit.data.concat([signal_data_reso, bkg_data_reso])



print(result)

print(result)

print(result)

# Firstly, we create a free scaling parameter
sigma_scaling = zfit.Parameter('sigma_scaling', 0.9, 0.1, 10, step_size=0.1)


def sigma_scaled_fn(sigma, sigma_scaling):
    return sigma * sigma_scaling  # this can be an arbitrary function


sigma_scaled = zfit.ComposedParameter('sigma scaled',  # name
                                      sigma_scaled_fn,  # function
                                      params=[sigma, sigma_scaling],  # the objects used inside the function
                                      unpack_params=True  # we could also just use a `params` argument, a dict
                                      )

reso_sig_yield = zfit.Parameter('reso_sig_yield', n_sig_reso - 100, 0, n_sig_reso * 3,
                                step_size=1)
signal_reso = zfit.pdf.Gauss(mu=mu,  # the same as for the rare mode
                             sigma=sigma_scaled,
                             obs=obs_reso,
                             extended=reso_sig_yield)

lambda_reso = zfit.Parameter('lambda_reso', -0.002, -0.01, 0.0001)
reso_bkg_yield = zfit.Parameter('reso_bkg_yield', n_bkg_reso + 70, 0, 2e5, step_size=1)
comb_bkg_reso = zfit.pdf.Exponential(lambda_reso, obs=obs_reso, extended=reso_bkg_yield)


model_reso = zfit.pdf.SumPDF([comb_bkg_reso, signal_reso])

nll_rare = zfit.loss.ExtendedUnbinnedNLL(model, data)
nll_reso = zfit.loss.ExtendedUnbinnedNLL(model_reso, data_reso)
nll_simultaneous = nll_rare + nll_reso

signal_reso.get_yield()

result_simultaneous = minimizer.minimize(nll_simultaneous)

result_simultaneous.hesse()

print(result_simultaneous.params)


print(result_simultaneous.params[sig_yield])

from hepstats.hypotests.parameters import POI

# the null hypothesis
print("sig_yield",sig_yield)
sig_yield_poi = POI(sig_yield, 0)
from hepstats.hypotests.calculators import (AsymptoticCalculator,
                                            FrequentistCalculator)

# construction of the calculator instance
"""
calculator = FrequentistCalculator(input=nll_simultaneous, minimizer=minimizer)
calculator.bestfit = result_simultaneous

# equivalent to above
calculator = FrequentistCalculator(input=result_simultaneous, minimizer=minimizer)
"""

# construction of the calculator instance
calculator = AsymptoticCalculator(input=nll_simultaneous, minimizer=minimizer)
calculator.bestfit = result_simultaneous

# equivalent to above
calculator = AsymptoticCalculator(input=result_simultaneous, minimizer=minimizer)

from hepstats.hypotests import Discovery
print("discovery")
discovery = Discovery(calculator=calculator, poinull=sig_yield_poi)
discovery.result()

print(discovery.result())


# Sets the values of the parameters to the result of the simultaneous fit
zfit.param.set_values(nll_simultaneous.get_params(), result_simultaneous)
sigma_scaling.floating = False

# Creates a sampler that will draw events from the model
sampler = model.create_sampler()

# Creates new simultaneous loss
nll_simultaneous_low_sig = zfit.loss.ExtendedUnbinnedNLL(model, sampler) + nll_reso

# Samples with sig_yield = 10. Since the model is extended the number of
# signal generated is drawn from a poisson distribution with lambda = 10.
sampler.resample({sig_yield: 10})

calculator_low_sig = AsymptoticCalculator(input=nll_simultaneous_low_sig, minimizer=minimizer)

discovery_low_sig = Discovery(calculator=calculator_low_sig, poinull=sig_yield_poi)
discovery_low_sig.result()
print(f"\n {calculator_low_sig.bestfit.params} \n")
from hepstats.hypotests import UpperLimit
from hepstats.hypotests.parameters import POIarray

#Background only hypothesis.
bkg_only = POI(sig_yield, 0)
# Range of Nsig values to scan.
sig_yield_scan = POIarray(sig_yield, np.linspace(0, 70, 10))

ul = UpperLimit(calculator=calculator_low_sig, poinull=sig_yield_scan, poialt=bkg_only)
ul.upperlimit(alpha=0.05);

plotlimit(ul, CLs=False)





