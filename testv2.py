import numpy as np
import zfit
from zfit.loss import ExtendedUnbinnedNLL
from zfit.minimize import Minuit
from hepstats.hypotests.calculators import AsymptoticCalculator
from hepstats.hypotests import UpperLimit
from hepstats.hypotests.parameters import POI, POIarray

bounds = (0.1, 3.0)
zfit.Space('x', limits=bounds)

bkg = np.random.exponential(0.5, 300)
peak = np.random.normal(1.2, 0.1, 10)
data = np.concatenate((bkg, peak))
data = data[(data > bounds[0]) & (data < bounds[1])]
N = data.size
data = zfit.data.Data.from_numpy(obs=obs, array=data)

lambda_ = zfit.Parameter("lambda", -2.0, -4.0, -1.0)
Nsig = zfit.Parameter("Ns", 20., -20., N)
Nbkg = zfit.Parameter("Nbkg", N, 0., N*1.1)
signal = Nsig * zfit.pdf.Gauss(obs=obs, mu=1.2, sigma=0.1)
background = Nbkg * zfit.pdf.Exponential(obs=obs, lambda_=lambda_)
loss = ExtendedUnbinnedNLL(model=signal + background, data=data)



calculator = AsymptoticCalculator(loss, Minuit())
poinull = POIarray(Nsig, np.linspace(0.0, 25, 20))
poialt = POI(Nsig, 0)
ul = UpperLimit(calculator, poinull, poialt)
ul.upperlimit(alpha=0.05, CLs=True)

