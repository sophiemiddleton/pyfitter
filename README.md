# Introduction

This repo contains the initial python based fitting code.

The structure is such that:

1) main.py - calls the main driver script which talks to the other classes
2) cuts.py - applies chosen set of cuts
3) fitter.py - calls chosen python based minimizer
4) shapes.py - sets up functional form for pdfs
5) import-reco.py - imports standard NTuples and utilizes awkward array
