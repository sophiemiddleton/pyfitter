# Custom_PDF
# Define custom PDF
# TODO : separate PDFs by process e.g. DIOs all together (mom, time)

import numpy as np
import tensorflow as tf
import zfit

class poly58(zfit.pdf.ZPDF):
    """ for DIO parameterization """
    _N_OBS = 1
    _PARAMS = ['a5', 'a6', 'a7', 'a8']

    def _unnormalized_pdf(self, x):
        x = zfit.z.unstack_x(x) # convert the zfit Data object into a tf tensor
        a5 = self.params['a5']
        a6 = self.params['a6']
        a7 = self.params['a7']
        a8 = self.params['a8']
        E_mu = 105.194 # mass of the muon [MeV]
        m_Al = 25133 # mass of the Aluminum atom [MeV]
        delta = tf.nn.relu(E_mu - x - x**2 / (2 * m_Al))    # Use relu function to make pdf = 0 for x > E_mu

        return a5 * delta**5 + a6 * delta**6 + a7 * delta**7 + a8 * delta**8


def CeModel(obs_mom, model=None):
  """ model will be version controlled """
  #N_CE = 0
  #CE =
  # Parameters CE
  #if model = 'dscb':
  mu = zfit.Parameter('mu', 104, 103, 107)
  sigma = zfit.Parameter('sigma', 0.5, 0.08, 2.0)
  # Parameters resolution
  mean_res = zfit.Parameter('mean_res', -0.5798, -10, 10)
  sigma_res = zfit.Parameter('sigma_res', 0.2671, 0, 10)
  alphal = zfit.Parameter('alphal', 0.422, 0, 10)
  nl = zfit.Parameter('nl', 25.1, 0, 100)
  alphar = zfit.Parameter('alphar', 2.227, 0, 100)
  nr = zfit.Parameter('nr', 5.954, 0, 100)


  N_CE= zfit.Parameter('N_CE', 10, 0, 1e6)
  CE = zfit.pdf.DoubleCB(obs=obs_mom, mu=mu, sigma=sigma, alphal=alphal, nl=nl, alphar=alphar, nr=nr, extended=N_CE)
  return CE, N_CE
    
def DIOModel(obs_mom, model=None):

  # Parameters DIO
  a5 = zfit.Parameter('a5', 8.6434e-17, 0, 1e-16)
  a6 = zfit.Parameter('a6', 1.16874e-17, 0, 1e-16)
  a7 = zfit.Parameter('a7', -1.87828e-19, -1e-18, 0)
  a8 = zfit.Parameter('a8', 9.16327e-20, 0, 1e-18)
  N_DIO = zfit.Parameter('N_DIO', 3000, 0, 1e6)

  DIO = poly58(obs=obs_mom, a5=a5, a6=a6, a7=a7, a8=a8, extended=N_DIO)
  return DIO, N_DIO
