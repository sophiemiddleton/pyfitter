# Custom_PDF
# Define custom PDF (Partial Distribution Functions that are used in the PDF_definition script

# Author: Leo Borrel
# Date: 2024-04-02

import numpy as np
import tensorflow as tf
import zfit

# FIXME convert for zfit
class PDF:
    """ 1-dimensional PDF implementation """
    def __init__(self, suffix, start_value):
        #self.x = x
        #self.params = params
        self.suffix = suffix
        self.start_value = start_value

# FIXME convert for zfit
class gaussian(PDF):
    def __init__(self, suffix, start_value):
        super().__init__(suffix, start_value)
        self.start_mean = start_value['mean']
        self.start_sigma = start_value['sigma']

    def pdf(self, x, mean, sigma):
        return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mean) ** 2 / 2. / sigma ** 2)

    def set_start_value(self):
        print("test")
        for param in self.start_value:
            print("start_value")
            print(param)
        return self.start_value


class poly58(zfit.pdf.ZPDF):
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
