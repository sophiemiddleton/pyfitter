# ✨ The `zfit` Fitting Package

`zfit` is a scalable, Pythonic model fitting library built on top of **TensorFlow**. It is specifically designed to meet the rigorous demands of **High Energy Physics (HEP)** likelihood fitting.

## 💡 Why `zfit`? Key Advantages

The primary choice of `zfit` as the fitting engine for `py-fitter` is based on its ability to handle complex models and large datasets with high performance, while remaining flexible and Pythonic.

| Advantage | Description | Relevance to Mu2e Analysis |
| :--- | :--- | :--- |
| **Scalability & Performance** | Built on TensorFlow, it inherently supports **GPU and multi-CPU parallelization**. It is optimized for computationally intensive tasks like large-scale fits and toy Monte Carlo studies. | Essential for processing the large volumes of Mu2e data and performing robust statistical tests (e.g., confidence interval calculation). |
| **Automatic Differentiation** | TensorFlow provides **automatic gradients**. This removes the need for manual gradient calculations, simplifying the implementation of complex, custom probability density functions (PDFs) and speeding up minimization. | Allows rapid iteration and implementation of new, custom analysis models without worrying about complex derivatives. |
| **Pythonic Design** | Offers a clean, high-level, object-oriented **API** (Application Programming Interface). It integrates seamlessly with the existing SciPy and NumPy ecosystem. | Easy to integrate with `pyutils`, plotting libraries (like `matplotlib`), and standard Python data analysis tools. |
| **Extensibility** | It provides base classes that allows users to easily implement **custom losses**, **custom minimizers**, and **arbitrary PDF shapes** with full control. | Necessary for advanced techniques like adding complex constraints (e.g., Gaussian, log-normal) to the likelihood function. |



## 🛠️ `zfit` Core Workflow

The `zfit` workflow follows a few conceptual steps, which mirror the statistical analysis chain:

1.  **Define Parameter:** Create parameters with initial values and limits using `zfit.Parameter()`.
2.  **Define Model:** Create a PDF (e.g., Gaussian, Exponential) using the defined parameters.
3.  **Load Data:** Use the `zfit.Data()` object to wrap NumPy arrays or other data sources.
4.  **Create Loss:** Define the cost function, typically the negative log-likelihood (`zfit.loss.UnbinnedNLL`).
5.  **Minimize:** Instantiate a minimizer (e.g., `zfit.minimize.Minuit`) and run the fit.
6.  **Analyze Result:** Extract the best-fit parameters and calculate uncertainties (e.g., using `FitResult.hesse()` or `FitResult.errors()`).

## 👨‍💻 Basic Usage Examples

The following snippets illustrate some simple examples of how to get started with `zfit`. It is recommended to look over these examples, and the more detailed examples in the tutorial listed below, before diving in with the more complex `py-fitter` customized code.

### Example 1: Defining a Simple Parameter

This is the fundamental building block for any fit parameter, including signal yield or background rate.

```
import zfit

# Define the number of signal events, floating between 0 and 100
N_signal = zfit.Parameter("N_signal", 50.0, lower=0.0, upper=100.0)

# Define a fixed background rate parameter
tau_bkg = zfit.Parameter("Tau_Bkg", 1.5, floating=False)

```

### Example 2: Creating a Gaussian PDF

A probability density function (PDF) requires parameters and a defined observation space.

```python
# 1. Define the observable range (e.g., reconstructed energy)
obs_space = zfit.Space("energy", limits=(100.0, 106.0))

# 2. Define the PDF's intrinsic parameters
mu = zfit.Parameter("mu", 104.9, lower=104.8, upper=105.0)
sigma = zfit.Parameter("sigma", 0.05, lower=0.01, upper=0.1)

# 3. Instantiate the Gaussian model
gaussian_pdf = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs_space)
```

### Example 3: Running a Simple Fit

The typical process of fitting a model to a dataset (`data` is a pre-loaded `zfit.Data` object):

```python
# 1. Define the Loss function (Negative Log-Likelihood)
loss = zfit.loss.UnbinnedNLL(model=gaussian_pdf, data=data)

# 2. Instantiate and run the minimizer (Minuit is often the preferred choice in HEP)
minimizer = zfit.minimize.Minuit()
result = minimizer.minimize(loss)

# 3. Print the fit result summary
print(result.params)
```

#$ Tutorial
To learn more please look over the tutorial material: https://zfit-tutorials.readthedocs.io/en/latest/tutorials/guides/README.html.
