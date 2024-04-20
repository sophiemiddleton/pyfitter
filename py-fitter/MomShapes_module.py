# mom_shapes
# Define custom PDF Functions that are used in the PDF_definition script

# Original Author: Leo Borrel
# Edits: Sophie Middleton
# Date: 2024-04-19

import numpy as np
import tensorflow as tf
import zfit

#class CE(zfit.pdf.ZPDF): ### TODO

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
