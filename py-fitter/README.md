# Mu2e Analysis

Python based analysis tool for analysis of reconstructed Mu2e data or MC.

# Developers

The current code base has been primarily developed by Leo Borrel and Sophie Middleton. Moving forward it will be part of the joint Mu2e Caltech/LBNL/Berkeley.... Working Group.

# Current Code

The current code base (May 2024)  was developed using MDC2018 TrkAna NTuples. Please run it with these (e.g. /pnfs/mu2e/scratch/users/sophie/MDC2018/trkana7.root). See the Mu2e wiki page for more information on MDC2018.

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

* conversion e- signal (CE) is currently parameterized as a Crystal Ball, assuming there has been multiple scattering, energy losses and detector distortions.
* decay in orbit (DIO) is currently parameterized using the work of Czernecki et al and the polynomial functional form derived in [Phys. Rev. D 94, 051301]
* cosmic induced background is currently parameterized as a uniform distribution

These distributions are all defined inside of the "Custom_PDF" script. This infrastructure needs to evolve as we develop more complexity.

# The Time PDF Parameterizations

* TODO

# 2D fits

* TODO

# Resoluton and Efficency parameterizations

* TODO

# Systematics and Nusiance Parameters

* TODO

# Results

The Results module store the final fit results in terms of expected yield from each of the sources of events. This should be adapted to interface with our Bayesian tools eventually.
