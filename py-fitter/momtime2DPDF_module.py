# to run 2D fitting
# place holder example ...

import tensorflow as tf

import zfit
from zfit import z

class PDF2D(zfit.pdf.BasePDF):
    """2 dimensional pdf where the axes are: Mom and Time."""

    def __init__(
        self,
        param1,
        param2,
        param3,
        obs,
        name="PDF2D",
    ):
        params = {
            "super_param": param1,
            "param2": param2,
            "param3": param3,
        }
        super().__init__(obs, params, name=name)

    def _unnormalized_pdf(self, x, params):
        time = x[0]
        momentum = x[1]
        param1 = params["super_param"]
        param2 = params["param2"]
        param3 = params["param3"]

        # just a fantasy function
        return #timeComp + momComp
