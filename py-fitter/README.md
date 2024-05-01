# Mu2e Analysis


Python based analysis tool for analysis of reconstructed Mu2e data or MC.

# Building the python environment:

Eventually we aim to have a setup script inside this Repo, meaning that all this would be obsolete. For now if you do want to do this you can simple pull the "requirements.txt":

```
$ virtualenv <env_name>
$ source <env_name>/bin/activate
(<env_name>)$ pip install -r path/to/requirements.txt
```

This can be a bit dangerous as you may have missed dependencies, but it will get you some way towards replicating the developer environments.

# Running the analysis code

The code is currently object orientated with a set of distinct classes:

* Main.py - the driver function. The user can define several input parameters:

```
# example use: python main.py --filelist "trkana7.root" --treename "TrkAnaNeg" --branchname "trkana" --fitrange_mom_low 95 fitrange_mom_hi 115
```

* The Main function imports the given root NTuple via the ImportClass defined in Import_module.py. The code currently assumes the Mu2e/TrkAna will be in input NTuple but the user parameters allow some flexibility.

* The Fit_module.py calls zfit and the various parameterizations of the signal (CE) and backgrounds (currently DIO, Cosmics only). The PDFs are written only for the momentum parameter currently.
We plan to expand to a 2D momentum and time fit eventually.

# The Momentum PDF Parameterizations

* conversion e- signal (CE) is currently parameterized as a Gaussian, assuming their has been multiple scattering, energy losses and detector distortions
* decay in orbit (DIO) is currently parameterized using the work of Czernecki et al and the polynomial functional form derived in [REF]
* cosmic induced background is currently parameterized as a uniform distribution

These distributions are all defined inside of the "mom_shapes" script. Each function is represented by a class.

# The Time PDF Parameterizations

* TODO

# Results

The Results module store the final fit results in terms of expected yield from each of the sources of events. This should be adapted to interface with our Bayesian tools eventually.
