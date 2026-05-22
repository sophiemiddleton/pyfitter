# 📊 Frequentist vs. Bayesian Interpretations

The Mu2e analysis strategy is twofold, utilizing both Frequentist (via **py-fitter/zfit**) and Bayesian (via **BAT.jl**) statistical methods to interpret the results of the same extended likelihood function. Although the eventual procedures for calculating uncertainties and final limits are very different, they share a common infrastructure for the likelihood model.

---

## Frequentist Approach

The Frequentist approach is **descriptive**, focusing on characterizing the measurement itself.

| Aspect | Description |
| :--- | :--- |
| **Probability** | Defined in the *frequency sense*: the outcome that would be achieved as a fraction of hypothetical repetitions of the measurement process. |
| **Parameters** | Treated as fixed, true values. |
| **Result (Point Estimate)** | The **Maximum Likelihood Estimate ($\hat{\theta}_{\text{MLE}}$)**, obtained by maximizing the full likelihood function $\mathcal{L}(\theta, \vec{\nu})$ simultaneously with respect to the Parameter of Interest ($\theta$) and the nuisance parameters ($\vec{\nu}$). |
| **Handling Nuisance Parameters ($\vec{\nu}$)** | Nuisance parameters, which account for systematic uncertainties, are constrained directly in the likelihood function (e.g., using Gaussian or Log-Normal terms derived from external measurements). |
| **Uncertainty Interval** | A **Confidence Interval (CI)**. This interval is constructed such that, if the experiment were repeated many times, the interval would contain the true, fixed parameter value in a specified fraction of those hypothetical experiments. |
| **CI Construction Method** | Commonly uses the **profile likelihood ratio test statistic, $\lambda(\theta)$**. The boundaries of the CI are determined by comparing $\lambda(\theta)$ to the quantiles of the asymptotic $\chi^2$ distribution. |

## In our code

A Frequentist interpretation of the results can be produced using the `results_module.py` (see [documentation](code-specifics/results_module_py.md)).

---

## Bayesian Approach

The Bayesian approach is **interpretive**, focusing on updating the degree of belief in the true value of a parameter.

| Aspect | Description |
| :--- | :--- |
| **Probability** | Assigned to degrees of belief in different possibilities for the true parameter value. |
| **Parameters** | Treated as random variables. |
| **Derivation** | Derived using **Bayes' Theorem** by combining: |
| | 1. **Prior Distribution** ($p(\theta, \vec{\nu})$): Encapsulates initial beliefs about the parameters (e.g., uniform priors are currently assumed for signal yields). |
| | 2. **Likelihood Function** ($\mathcal{L}(\theta, \vec{\nu}) = p(\text{data}|\theta, \vec{\nu})$): Represents the probability of the observed data given the parameter values. |
| **Handling Nuisance Parameters ($\vec{\nu}$)** | The effect of nuisance parameters is removed by **marginalization**—integrating the joint posterior over the nuisance parameters ($\vec{\nu}$) to obtain the marginalized posterior for the POI ($\theta$). |
| **Uncertainty Interval** | A **Credible Interval (CI)**. This interval represents a direct probabilistic statement: a 95% CI means there is a 95% probability that the true parameter value lies within the interval. |
| **Implementation** | The **BAT.jl** framework is used to reconstruct the likelihood and calculate the posterior, often involving the generation of Toy Monte Carlo (Toy-MC) experiments (on the order of 100-1000). |

## In our code

A Bayesian interpretation of the results can be produced by reimplementing into **BAT.jl**. Our analysis group also maintains **BayesAna**: https://github.com/HighEeM0/BayesAna/tree/main. The `results_module.py`  can be used to produce selected data and inputs to the custom Bayesian analysis framework.
