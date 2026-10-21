# Bayesian Dynamic Borrowing with Deep Neural Networks for Multiple Time-to-Event Endpoints

R code for the simulation studies and the case study in the paper. Each folder is
self-contained: run its scripts from inside that folder.

| Folder | Content |
|---|---|
| `Simulation_DNN/` | Simulation study, DNN trained on MCMC posterior labels (piecewise-exponential hazards, 15 scenarios) |
| `Simulation_NPE/` | Simulation study, DNN trained on neural posterior estimation (NPE) labels, with NPE-vs-MCMC check |
| `Simulation_MAP/` | Simulation study comparator: MAP prior, one script per scenario (01–15) |
| `CaseStudy_DNN/` | Case study, MCMC-labelled DNN (constant hazards, two endpoints, HR = 1, 1.25, 1.5, 1.75, 2) |
| `CaseStudy_NPE/` | Case study, NPE-labelled DNN |
| `CaseStudy_MAP/` | Case study comparator: meta-analytic-predictive (MAP) prior via RBesT |

## Structure of the DNN folders

```
<folder>/
  BDB_DNN_functions.r          # data simulation, Stan fitting, NPE and DNN helper functions
  *.stan                       # hierarchical borrowing model (and history-only model for NPE)
  multiple_TTE_endpoints/
    BDB_DNN_train.r            # generate training data, label it (MCMC or NPE), train DNNs
    BDB_DNN_test.r             # evaluate type I error / power on the test scenarios
    *.keras, DNN_scale_parameters_TTE   # trained networks and input scaling
```

Training proceeds in two stages: stage 1 networks (`DNN_first_prob`, `DNN_first_mean`) approximate the
posterior probability and posterior mean of the treatment effect given the current and historical data;
stage 2 networks (`DNN_second_H1/H2/H12`) learn the decision thresholds that control the
(family-wise) type I error. The trained models are included, so `BDB_DNN_test.r` can be run without retraining.

## Running

```sh
cd Simulation_NPE          # or Simulation_DNN, CaseStudy_DNN, CaseStudy_NPE
Rscript multiple_TTE_endpoints/BDB_DNN_train.r   # optional, long-running
Rscript multiple_TTE_endpoints/BDB_DNN_test.r
```

```sh
cd CaseStudy_MAP
Rscript run_all_MAP.R          # runs all five HR cases, then selects the common cutoff and summarises
Rscript analyze_MAP.R          # re-summarise only (optionally --cutoff=0.975)
```

```sh
cd Simulation_MAP
Rscript MAP_01_global_null_compatible.R    # one scenario; MAP_02 ... MAP_15 likewise (or submit the .sbatch files on SLURM)
Rscript analysis_MAP.R                     # summarise the 15 result files
```

Each `MAP_XX_*.R` writes `results_XX_*_0.975.csv` to the working directory, which `analysis_MAP.R` then reads.
Fill in the account/partition placeholders in the `.sbatch` templates for your cluster.

`CaseStudy_MAP/setup_packages.R` installs the R packages it needs. Results are written to
`CaseStudy_MAP/results/`.

Simulation and case-study results are not included in this repository; they are reproduced by the scripts above.

`Simulation_*/multiple_TTE_endpoints/Treatment_effect_PWE.R` computes the true average treatment effects
in `PWE_true_treatment_effect_scenarios_01_15.csv`.

## Requirements

- R (>= 4.1) with `rstan`, `RBesT`, `keras3`, `tensorflow`, `reticulate`, `foreach`, `doParallel`,
  `bindata`, `tibble`, `withr`, and a C++ toolchain for Stan.
- A Python environment with TensorFlow/Keras for `keras3`. Point reticulate to it with the
  `RETICULATE_PYTHON` environment variable if it is not found automatically.

## Case study data

The case study uses only published summary statistics of two historical control groups
(KIDCARE, n = 49; Wang et al., n = 40), entered directly in the scripts. No individual patient data are used.
