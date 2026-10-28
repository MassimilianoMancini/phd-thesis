# PhD Thesis — Fairness Measurement in Classification

**Thesis:** *Fairness Measurement in Classification: From Binary to Multi-Valued Outcomes and Groups*  
**Author:** Massimiliano Mancini  
**Institution:** University of Florence (Università degli Studi di Firenze), PhD programme in Statistics, Computer Science and Mathematics

## Overview

This repository collects Python implementations, experimental scripts, and supporting data associated with the doctoral research on algorithmic fairness in classification.

The thesis investigates fairness measurement beyond conventional binary settings, with particular attention to:

- **Independence**, operationalised through Statistical Parity;
- **Separation**, operationalised through Total Accuracy;
- **Sufficiency**, operationalised through Calibration;
- fairness evaluation for **multi-valued scores** and **multiple demographic groups**;
- pairwise and vector-based approaches to group comparison;
- experimental evaluation of classification models and exploratory fairness-aware learning methods.

The repository contains research code developed over different stages of the project. Some directories preserve earlier implementations or experimental variants; they are not necessarily interchangeable or packaged as a single software application.

## Repository structure

The main directories currently include:

| Directory | Contents |
| --- | --- |
| `fairness_measures-main/fairness_measures-main/` | Fairness-measures API, example usage, and Portuguese student-performance datasets. |
| `calibration-main/calibration-main/src/` | Calibration-related implementations and examples, including student-performance data. |
| `multigroup-main/multigroup-main/src/` | Multi-group fairness experiments and Law School-related classification scripts. |
| `multigroup-v2/` | Further multi-group experiments, LSAT score discretisation, classification evaluations, and plots. |
| `fairness-experiments-main/fairness-experiments-main/` | Exploratory fairness-aware decision-tree implementations and experiment scripts. |
| `wails2026-main/wails2026-main/` | Data preparation, prediction, and fairness-measure scripts for the school-related experiments. |

Some folders contain an additional nested directory with the same name, reflecting the original project layout.

## Fairness measures

### Statistical Parity — Independence

Statistical Parity compares the rates of positive predictions across demographic groups. The corresponding rates can also be computed on observed outcomes rather than predictions; the thesis refers to this baseline assessment as **True Statistical Parity**.

### Total Accuracy — Separation

Total Accuracy compares the proportion of correct predictions across groups.

### Calibration — Sufficiency

Calibration evaluates the relationship between observed outcomes and predicted scores across groups. The research explores its operationalisation for multi-valued scores.

### Multi-group assessment

The research considers both pairwise comparisons between groups and vector-based comparisons against reference distributions, including equality- and equity-oriented assessments.

## Data and experiments

The repository contains code related to several experimental settings, including:

- Portuguese secondary-school student-performance data;
- Law School / LSAT-related classification experiments;
- Italian school-related data preparation and fairness analysis;
- exploratory fairness-aware decision trees.

**Data availability varies by experiment.** Some datasets are included in the repository, while others may need to be obtained from their original sources or prepared before running a script. Please inspect each experiment directory and respect the applicable data-source terms and privacy requirements.

## Getting started

### Requirements

- Python (version requirements may vary across the individual projects);
- the Python packages imported by the scripts being run.

Some subprojects contain a `pyproject.toml` and `uv.lock`; others consist of standalone Python scripts. There is currently no single verified environment specification covering the entire repository.

Clone the repository:

```bash
git clone https://github.com/MassimilianoMancini/phd-thesis.git
cd phd-thesis
```

Create and activate a virtual environment if needed:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

For a subproject with a `pyproject.toml`, inspect its declared dependencies and use the package manager specified by that project. For standalone scripts, inspect imports and data-file paths before installing dependencies or running the code.

**Important:** The repository contains multiple research prototypes and scripts. There is no universal command that has been verified to reproduce every result in the thesis. Some scripts may require local path adjustments, dataset preparation, or additional configuration.

## Example: fairness-measures API

The directory `fairness_measures-main/fairness_measures-main/` includes:

- `fairness_measures_api.py` — API implementation;
- `fair_api_example.py` — usage example;
- `student-mat.csv` and `student-por.csv` — student-performance datasets.

Its local README documents methods including `true_statistical_parity`, `statistical_parity`, `total_accuracy`, `calibration`, and `ofi`. Refer to that directory for the API's expected arguments and example usage.

## Testing, validation, and reproducibility

The research workflow includes data preprocessing, implementation of fairness measures, classifier training and evaluation, and checks of the resulting calculations. Individual directories contain scripts with names indicating testing or evaluation, but the repository should **not** be assumed to provide a unified automated test suite.

For reproducible use:

1. Identify the thesis experiment and the corresponding directory.
2. Check the input datasets and preprocessing steps.
3. Record the Python version and installed dependency versions.
4. Inspect model settings, random seeds, and validation procedures in the relevant scripts.
5. Run the experiment and compare its outputs with the results reported in the thesis.

The code and available data are provided to support transparency and further investigation; exact reproduction of every published result has not been independently verified through this README.

## Citation

If you use this repository, please cite the doctoral thesis:

> Mancini, Massimiliano. *Fairness Measurement in Classification: From Binary to Multi-Valued Outcomes and Groups*. PhD thesis, University of Florence, 2026.

Please also cite the relevant research publications and original dataset sources where appropriate.

## Repository

https://github.com/MassimilianoMancini/phd-thesis

## Licence and reuse

No repository-wide licence is specified in this README. Public availability on GitHub does not by itself grant permission to redistribute or reuse all code and data. Consult the repository's licence information, if added, and the terms associated with each dataset before reuse.

## Contact

For questions about the research or experimental materials, please use the repository's GitHub issue tracker.
