import numpy as np
import math

# from https://github.com/Mu2e/Offline/blob/main/Mu2eUtilities/src/ConversionSpectrum.cc
eMax = 104.97
alpha = 1./137.035999139
me = 0.511
def LeadingLog(E):
    val = (1./eMax)*(alpha/(2*math.pi))*(math.log(4*E*E/me/me)-2.)*((E*E+eMax*eMax)/eMax/(eMax-E))
    if val < 0.0: val = 0.0
    return val
def binned_spectrum_CeLL(binwidth=0.1):
    nbins = math.floor(eMax/binwidth)
    if binwidth*nbins < eMax: nbins += 1
    upedge = binwidth*nbins
    edges = np.linspace(0., upedge, nbins+1)
    centers = (edges[:-1]+edges[1:])/2.
    vectorize_LL = np.vectorize(LeadingLog)
    values = vectorize_LL(centers)
    values[-1] = (1-np.sum(values[:-1])*binwidth)/binwidth
    return (values,edges)

