## source functions file
script_dir <- if (!is.null(sys.frames()[[1]]$ofile)) {
  dirname(normalizePath(sys.frames()[[1]]$ofile))
} else if (file.exists(file.path(getwd(), "DNN_first_prob.keras"))) {
  getwd()
} else {
  file.path(getwd(), "multiple_TTE_endpoints")
}
root_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(root_dir, "BDB_DNN_functions.r"))

## 1. Load trained DNNs ----

## Python with TensorFlow/Keras for keras3: set RETICULATE_PYTHON beforehand if needed,
## e.g. Sys.setenv(RETICULATE_PYTHON = "/path/to/python")

library(keras3)
library(reticulate)

model_dir <- script_dir
first_prob_model_path <- file.path(model_dir, "DNN_first_prob.keras")
first_mean_model_path <- file.path(model_dir, "DNN_first_mean.keras")

DNN.first.prob <- keras3::load_model(
  first_prob_model_path,
  compile = FALSE
)
DNN.first.mean <- keras3::load_model(
  first_mean_model_path,
  compile = FALSE
)
DNN.second.H1 <- keras3::load_model(file.path(model_dir, "DNN_second_H1_TTE.keras"), compile = FALSE)
DNN.second.H2 <- keras3::load_model(file.path(model_dir, "DNN_second_H2_TTE.keras"), compile = FALSE)
DNN.second.H12 <- keras3::load_model(file.path(model_dir, "DNN_second_H12_TTE.keras"), compile = FALSE)

## load scaling parameters
load(file = file.path(model_dir, "DNN_scale_parameters_TTE"))

if (is.null(DNN_scale_parameters$col_means_second_H1) ||
    is.null(DNN_scale_parameters$col_means_second_H2) ||
    is.null(DNN_scale_parameters$col_means_second_H12)) {
  stop("This test script requires models retrained with the Stage-2 cutoff DNNs. ",
       "Please run BDB_DNN_train.r first.")
}

stage1_label_method <- DNN_scale_parameters$Stage1_label_method
if (is.null(stage1_label_method)) {
  stage1_label_method <- "MCMC"
}
message("Loaded first-stage DNNs trained with ", stage1_label_method, " labels.")

true_effect_path <- file.path(root_dir, "PWE_true_treatment_effect_scenarios_01_15.csv")
true_effect_results <- read.csv(true_effect_path, stringsAsFactors = FALSE)


#### 2. Simulation Setup ====

scenarios_to_run <- c(
  "01_global_null_compatible",
  "02_global_null_mild_conflict",
  "03_global_null_moderate_conflict",
  "04_single_null_s1_delayed_compatible",
  "05_single_null_s1_early_compatible",
  "06_single_null_s2_delayed_compatible",
  "07_single_null_s2_early_compatible",
  "08_single_null_s1_delayed_conflict",
  "09_single_null_s1_early_conflict",
  "10_single_null_s2_delayed_conflict",
  "11_single_null_s2_early_conflict",
  "12_global_alt_delayed_compatible",
  "13_global_alt_early_compatible",
  "14_global_alt_delayed_conflict",
  "15_global_alt_early_conflict"
)

# trial design
nsim <- 1000

E <- 2  # ! if these two need to be changed, scenarios below also need to be changed, including censor design
K <- 6
cuts <- seq(0, 6, by = 1)
n_ctrl <- 150
n_trt  <- 150

# baseline hazards
base_lam_hist_mat <- rbind(
  c(0.12, 0.10, 0.09, 0.08, 0.07, 0.06),
  c(0.08, 0.07, 0.06, 0.05, 0.045, 0.04)
)

# censoring design
analysis_time <- 6
entry_dist <- function() 0  # all participants enter at time 0

# define treatment effect scenarios
make_global_null_beta <- function(E, K) {
  matrix(0, nrow = E, ncol = K)
}

make_delayed_beta <- function() {
  HR_mat <- rbind(
    c(1, 1, 0.75, 0.6, 0.5, 0.5),
    c(1, 1, 0.8, 0.65, 0.55, 0.55)
  )
  log(HR_mat)
}

make_early_beta <- function() {
  HR_mat <- rbind(
    c(0.5, 0.55, 0.65, 0.8, 1, 1),
    c(0.55, 0.6, 0.7, 0.85, 1, 1)
  )
  log(HR_mat)
}

make_s1_delayed_beta <- function() {
  HR_mat <- rbind(
    c(1, 1, 0.75, 0.6, 0.5, 0.5),
    c(1, 1, 1, 1, 1, 1)
  )
  log(HR_mat)
}

make_s1_early_beta <- function() {
  HR_mat <- rbind(
    c(0.5, 0.55, 0.65, 0.8, 1, 1),
    c(1, 1, 1, 1, 1, 1)
  )
  log(HR_mat)
}

make_s2_delayed_beta <- function() {
  HR_mat <- rbind(
    c(1, 1, 1, 1, 1, 1),
    c(1, 1, 0.8, 0.65, 0.55, 0.55)
  )
  log(HR_mat)
}

make_s2_early_beta <- function() {
  HR_mat <- rbind(
    c(1, 1, 1, 1, 1, 1),
    c(0.55, 0.6, 0.7, 0.85, 1, 1)
  )
  log(HR_mat)
}

# map scenario -> parameters
get_scenario_params <- function(scenario_name) {
  if (scenario_name == "01_global_null_compatible") {
    base_lam_current_ctrl_mat <- base_lam_hist_mat
    beta_mat <- make_global_null_beta(E, K)
  } else if (scenario_name == "02_global_null_mild_conflict") {
    base_lam_current_ctrl_mat <- 1.25 * base_lam_hist_mat
    beta_mat <- make_global_null_beta(E, K)
  } else if (scenario_name == "03_global_null_moderate_conflict") {
    base_lam_current_ctrl_mat <- 1.50 * base_lam_hist_mat
    beta_mat <- make_global_null_beta(E, K)
  } else if (scenario_name == "04_single_null_s1_delayed_compatible") {
    base_lam_current_ctrl_mat <- base_lam_hist_mat
    beta_mat <- make_s1_delayed_beta()
  } else if (scenario_name == "05_single_null_s1_early_compatible") {
    base_lam_current_ctrl_mat <- base_lam_hist_mat
    beta_mat <- make_s1_early_beta()
  } else if (scenario_name == "06_single_null_s2_delayed_compatible") {
    base_lam_current_ctrl_mat <- base_lam_hist_mat
    beta_mat <- make_s2_delayed_beta()
  } else if (scenario_name == "07_single_null_s2_early_compatible") {
    base_lam_current_ctrl_mat <- base_lam_hist_mat
    beta_mat <- make_s2_early_beta()
  } else if (scenario_name == "08_single_null_s1_delayed_conflict") {
    base_lam_current_ctrl_mat <- 1.5 * base_lam_hist_mat
    beta_mat <- make_s1_delayed_beta()
  } else if (scenario_name == "09_single_null_s1_early_conflict") {
    base_lam_current_ctrl_mat <- 1.5 * base_lam_hist_mat
    beta_mat <- make_s1_early_beta()
  } else if (scenario_name == "10_single_null_s2_delayed_conflict") {
    base_lam_current_ctrl_mat <- 1.5 * base_lam_hist_mat
    beta_mat <- make_s2_delayed_beta()
  } else if (scenario_name == "11_single_null_s2_early_conflict") {
    base_lam_current_ctrl_mat <- 1.5 * base_lam_hist_mat
    beta_mat <- make_s2_early_beta()
  } else if (scenario_name == "12_global_alt_delayed_compatible") {
    base_lam_current_ctrl_mat <- base_lam_hist_mat
    beta_mat <- make_delayed_beta()
  } else if (scenario_name == "13_global_alt_early_compatible") {
    base_lam_current_ctrl_mat <- base_lam_hist_mat
    beta_mat <- make_early_beta()
  } else if (scenario_name == "14_global_alt_delayed_conflict") {
    base_lam_current_ctrl_mat <- 1.5 * base_lam_hist_mat
    beta_mat <- make_delayed_beta()
  } else if (scenario_name == "15_global_alt_early_conflict") {
    base_lam_current_ctrl_mat <- 1.5 * base_lam_hist_mat
    beta_mat <- make_early_beta()
  } else {
    stop("Unknown scenario")
  }
  
  list(
    base_lam_current_ctrl_mat = base_lam_current_ctrl_mat,
    beta_mat = beta_mat
  )
}

run_scenario <- function(scenario_name, scenario_seed = 123) {
scenario_params <- get_scenario_params(scenario_name)
base_lam_current_ctrl_mat <- scenario_params$base_lam_current_ctrl_mat
beta_mat <- scenario_params$beta_mat

true_effect_scenario <- true_effect_results[
  true_effect_results$scenario_name == scenario_name,
]
true_effect_scenario <- true_effect_scenario[
  match(paste0("H", seq_len(E)), true_effect_scenario$endpoint),
]

if (any(is.na(true_effect_scenario$true_beta_bar))) {
  stop("Missing true beta/HR values for scenario: ", scenario_name)
}

true_beta_ref <- true_effect_scenario$true_beta_bar
true_HR_ref <- true_effect_scenario$true_HR_bar

cat("Running scenario:", scenario_name, "\n")
cat("HR matrix:\n")
print(exp(beta_mat))
cat("Reference true beta/HR from PWE_true_treatment_effect_scenarios_01_15.csv:\n")
print(true_effect_scenario[, c("endpoint", "true_beta_bar", "true_HR_bar")])

## 3. Run simulations----

flatten_features <- function(d_c, Y_c, d_t, Y_t) {
  c(
    as.numeric(d_c),
    as.numeric(Y_c),
    as.numeric(d_t),
    as.numeric(Y_t)
  )
}

calc_person_time_weighted_beta <- function(beta_mat, Y_ctrl, Y_trt) {
  total_Y <- Y_ctrl + Y_trt
  beta_bar <- numeric(nrow(beta_mat))
  
  for (e in seq_len(nrow(beta_mat))) {
    if (sum(total_Y[e, ]) == 0) {
      beta_bar[e] <- NA_real_
    } else {
      weights <- total_Y[e, ] / sum(total_Y[e, ])
      beta_bar[e] <- sum(weights * beta_mat[e, ])
    }
  }
  
  beta_bar
}

# Stage-2 inputs are true simulation parameters.  This is appropriate for the
# operating-characteristic simulation, but not a real-data analysis.
predict_stage2_cutoff <- function(model, x, center, scale_value) {
  x <- matrix(as.numeric(x), nrow = 1)
  as.numeric(pmax(pmin(predict(model, scale(x, center = center, scale = scale_value)), 1), 0))
}
loglam_mean_vec <- rowMeans(log(base_lam_current_ctrl_mat))
c1 <- predict_stage2_cutoff(DNN.second.H1,
                            c(loglam_mean_vec, beta_mat[2, ]),
                            DNN_scale_parameters$col_means_second_H1,
                            DNN_scale_parameters$col_stddevs_second_H1)
c2 <- predict_stage2_cutoff(DNN.second.H2,
                            c(loglam_mean_vec, beta_mat[1, ]),
                            DNN_scale_parameters$col_means_second_H2,
                            DNN_scale_parameters$col_stddevs_second_H2)
c12 <- predict_stage2_cutoff(DNN.second.H12, loglam_mean_vec,
                             DNN_scale_parameters$col_means_second_H12,
                             DNN_scale_parameters$col_stddevs_second_H12)
c1.final <- max(c1, c12)
c2.final <- max(c2, c12)
cat("Stage-2 DNN cutoffs: c1 =", c1, ", c2 =", c2, ", c12 =", c12, "\n")
cat("Closure-adjusted testing cutoffs: c1.final =", c1.final,
    ", c2.final =", c2.final, ", c12 =", c12, "\n")

## store iteration-level results

results.mat <- matrix(
  NA,
  nrow = nsim,
  ncol = 22
)

colnames(results.mat) <- c(
  "post_prob_H1",
  "post_prob_H2",
  "est_beta_H1",
  "est_beta_H2",
  "est_HR_H1",
  "est_HR_H2",
  "reject_H1",
  "reject_H2",
  "reject_H12",
  "sim_true_beta_H1",
  "sim_true_beta_H2",
  "ref_true_beta_H1",
  "ref_true_beta_H2",
  "ref_true_HR_H1",
  "ref_true_HR_H2",
  "bias_beta_H1",
  "bias_beta_H2",
  "bias_HR_H1",
  "bias_HR_H2",
  "crit_H1",
  "crit_H2",
  "crit_H12"
)

## simulation loop

set.seed(scenario_seed)

for (sim in 1:nsim) {
  
  test_dat <- sim_current_twoarm_counts(
    n_ctrl = n_ctrl,
    n_trt  = n_trt,
    E = E,
    K = K,
    cuts = cuts,
    base_lam_ctrl_mat = base_lam_current_ctrl_mat,
    beta_mat = beta_mat,
    analysis_time = analysis_time,
    entry_dist = entry_dist
  )
  
  d_c <- test_dat$d_ctrl
  Y_c <- test_dat$Y_ctrl
  
  d_t <- test_dat$d_trt
  Y_t <- test_dat$Y_trt
  
  x_prob <- flatten_features(
    d_c = d_c,
    Y_c = Y_c,
    d_t = d_t,
    Y_t = Y_t
  )
  
  x_prob <- matrix(x_prob, nrow = 1)
  
  x_prob_scale <- scale(
    x_prob,
    center = DNN_scale_parameters$col_means_first_prob,
    scale  = DNN_scale_parameters$col_stddevs_first_prob
  )
  x_mean_scale <- scale(
    x_prob,
    center = DNN_scale_parameters$col_means_first_mean,
    scale  = DNN_scale_parameters$col_stddevs_first_mean
  )
  
  post_prob <- predict(
    DNN.first.prob,
    x_prob_scale
  )
  
  post_prob <- as.numeric(post_prob)
  
  post_prob <- pmax(
    pmin(post_prob, 1),
    0
  )
  
  est_beta <- predict(
    DNN.first.mean,
    x_mean_scale
  )
  est_beta <- as.numeric(est_beta)
  est_HR <- exp(est_beta)
  
  true_beta_bar <- calc_person_time_weighted_beta(
    beta_mat = beta_mat,
    Y_ctrl = Y_c,
    Y_trt = Y_t
  )
  
  reject_H1 <- as.integer(post_prob[1] > c1.final)
  reject_H2 <- as.integer(post_prob[2] > c2.final)
  reject_H12 <- as.integer(any(post_prob > c12))
  
  results.mat[sim, ] <- c(
    post_prob[1],
    post_prob[2],
    est_beta[1],
    est_beta[2],
    est_HR[1],
    est_HR[2],
    reject_H1,
    reject_H2,
    reject_H12,
    true_beta_bar[1],
    true_beta_bar[2],
    true_beta_ref[1],
    true_beta_ref[2],
    true_HR_ref[1],
    true_HR_ref[2],
    est_beta[1] - true_beta_ref[1],
    est_beta[2] - true_beta_ref[2],
    est_HR[1] - true_HR_ref[1],
    est_HR[2] - true_HR_ref[2],
    c1.final,
    c2.final,
    c12
  )
  
  cat("Scenario:", scenario_name, "Iteration:", sim, "\n")
}

results.df <- as.data.frame(results.mat)

power.summary <- data.frame(
  
  Scenario = rep(scenario_name, 3),
  
  Hypothesis = c("H1", "H2", "H12"),
  
  Power = c(
    mean(results.df$reject_H1),
    mean(results.df$reject_H2),
    mean(results.df$reject_H12)
  ),
  
  Critical_value = c(
    c1.final,
    c2.final,
    c12
  ),
  
  True_beta = c(
    true_beta_ref,
    NA_real_
  ),
  
  Mean_est_beta = c(
    mean(results.df$est_beta_H1),
    mean(results.df$est_beta_H2),
    NA_real_
  ),
  
  Bias_beta = c(
    mean(results.df$bias_beta_H1),
    mean(results.df$bias_beta_H2),
    NA_real_
  ),
  
  MSE_beta = c(
    mean(results.df$bias_beta_H1^2),
    mean(results.df$bias_beta_H2^2),
    NA_real_
  ),
  
  True_HR = c(
    true_HR_ref,
    NA_real_
  ),
  
  Mean_est_HR = c(
    mean(results.df$est_HR_H1),
    mean(results.df$est_HR_H2),
    NA_real_
  ),
  
  Bias_HR = c(
    mean(results.df$bias_HR_H1),
    mean(results.df$bias_HR_H2),
    NA_real_
  ),
  
  MSE_HR = c(
    mean(results.df$bias_HR_H1^2),
    mean(results.df$bias_HR_H2^2),
    NA_real_
  )
)

print(power.summary)

crit_tag <- paste(round(c(c1.final, c2.final, c12), 4), collapse = "_")
write.csv(results.df, paste0("DNN_results_", scenario_name, "_", crit_tag, ".csv"), row.names = FALSE)
write.csv(power.summary, paste0("DNN_power_", scenario_name, "_", crit_tag, ".csv"), row.names = FALSE)

power.summary
}

summary.results <- do.call(
  rbind,
  lapply(seq_along(scenarios_to_run), function(i) {
    run_scenario(
      scenario_name = scenarios_to_run[i],
      scenario_seed = 123 + i - 1
    )
  })
)

write.csv(
  summary.results,
  "DNN_summary_results_all_scenarios.csv",
  row.names = FALSE
)

print(summary.results)
