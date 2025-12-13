## ⚛️ $\mu^{-} \rightarrow e^{-}$ Analysis Framework

**Python-based analysis tool for fitting and interpreting reconstructed Mu2e data and Monte Carlo (MC) simulations.**

This repository contains the primary analysis framework used by our Mu2e Analysis Group to search for the $\mu^{-} \rightarrow e^{-}$ conversion process.

---

### 📖 Documentation

The following resources provide detailed technical information on the framework and the analysis procedure:

| Resource | Description |
| :--- | :--- |
| [**Getting Started**](documentation/GettingStarted.md) | **Step-by-step guide** for setup, installation, and running your first analysis job. |
| [**`zfit`**](documentation/IntroducingZFit.md) | The tool that underpins our custom fitting package. |
| [**The `pyfitter` framework**](documentation/ThePyfitterFramework.md) | Detailed explanation of the main class structure and workflow. |
| [**Our Likelihood Definition**](documentation/LikelihoodDefinition.md) | Mathematical definition of the likelihood function and its parameters. |
| [**Results and Interpretation**](documentation/ResultsInterpretation.md) | Guide on plotting results, statistical methods, and interpreting confidence limits. |
| [**Uncertainties**](documentation/Uncertainties.md) | Comprehensive list and treatment of systematic and statistical uncertainties. |

---
---

### 📦 Code Structure

The repository is organized under the `LikelihoodAnalysis` package to separate core analysis components from auxiliary tasks.

| Directory | Purpose | Key Contents |
| :--- | :--- | :--- |
| `py-fitter` | **Core Analysis Framework** | Main classes for data loading, model definition, and likelihood fitting. |
| `common` | **Subsidiary Inputs** | Files necessary for incorporating systematic measurements, corrections, and background normalizations into the main fit. |
| `extra` | **Subsidiary Analysis** | Scripts and notebooks for analyzing control regions, determining resolution, and performing closure tests. |

---

### 🤝 Code Review and Contribution Policy

All changes, no matter how minor, require a Pull Request (PR). Please see the dedicated **[CONTRIBUTING.md]** file for full details.

#### **PR Submission Checklist (Author Responsibility)**
Before submitting, the author **must** verify the following:
* [ ] The code **converges** successfully on all available nominal samples.
* [ ] The changes **do not introduce regression** and produce meaningful, consistent results compared to the baseline.
* [ ] All relevant **docstrings** (within the code) have been updated or added.

#### **PR Review Turnaround Times**

To ensure a smooth and efficient development cycle, we expect the following review turnaround times:

| Change Scope | Description | Expected Turnaround |
| :--- | :--- | :--- |
| **Minor** | A few lines of change, no expected breaks (e.g., typos, comment fixes). | **1 Business Day** |
| **Moderate** | Changes to multiple files, potential for side-effects, but focused. | **3 Business Days** |
| **Major** | Significant restructuring, new features, or changes requiring substantial validation. | **~1 Week, but depends on on-going discussions** |

> **Escalation Policy:** If a designated reviewer is unresponsive or inactive for the specified period, the Group Leader is authorized to merge the PR and assume responsibility for the review. **If the reviewer is on vacation, they must communicate their return date, and the turnaround time will resume upon their return.**

---

### 👥 Development Team

This code is maintained by the Mu2e Analysis Group.

* **Current Team Members:** R. Bonventre, L. Borrel, D. Brown, E. Callaghan, S. Dittmer, B. Echenard, A. Edmonds, S. Garg, D. Hitlin, H. Jafree, C. Kampa, Y. Kolomensky, S. Middleton, F. Porter, M. Schmitt, A. Trumic, S. Zhou, J. Wang
* **Contact for Participation:** To be added as a new participant, please contact sophie@fnal.gov.

*Individual contributions can be tracked using the repository's Git History.*
