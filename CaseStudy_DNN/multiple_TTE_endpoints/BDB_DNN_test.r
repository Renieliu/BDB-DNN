# Locate this script (works with Rscript and RStudio source()) to resolve relative paths.
script_file <- local({
  source_files <- vapply(sys.frames(), function(x) {
    if (is.null(x$ofile)) "" else as.character(x$ofile)
  }, character(1))
  cli_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  candidates <- c(rev(source_files[nzchar(source_files)]),
                  gsub("~+~", " ", cli_file, fixed = TRUE))
  directory <- normalizePath(getwd(), mustWork = TRUE)
  repeat {
    candidates <- c(candidates,
      file.path(directory, "CaseStudy_DNN", "multiple_TTE_endpoints", "BDB_DNN_test.r"),
      file.path(directory, "multiple_TTE_endpoints", "BDB_DNN_test.r"))
    parent <- dirname(directory)
    if (identical(parent, directory)) break
    directory <- parent
  }
  candidates <- unique(normalizePath(candidates[file.exists(candidates)], mustWork = TRUE))
  candidates <- candidates[grepl(
    "/CaseStudy_DNN/multiple_TTE_endpoints/BDB_DNN_test\\.r$", candidates)]
  if (!length(candidates)) {
    stop("Cannot locate CaseStudy_DNN. Source its BDB_DNN_test.r using the full path.")
  }
  candidates[1]
})
script_dir <- dirname(normalizePath(script_file, mustWork = TRUE))
root_dir <- dirname(script_dir)
## Python with TensorFlow/Keras for keras3: set RETICULATE_PYTHON beforehand if needed,
## e.g. Sys.setenv(RETICULATE_PYTHON = "/path/to/python")
source(file.path(root_dir, "BDB_DNN_functions.r"))
library(keras3)
library(reticulate)

model_dir <- script_dir
scale_file <- file.path(model_dir, "DNN_scale_parameters_TTE")
if (!file.exists(scale_file)) stop("Run the constant-hazard BDB_DNN_train.r first: missing ", scale_file)
scale_env <- new.env()
load(scale_file, envir = scale_env)
DNN_scale_parameters <- scale_env$DNN_scale_parameters
if (!identical(DNN_scale_parameters$model_type, "constant_hazard") ||
    !identical(DNN_scale_parameters$benefit_direction, "HR > 1") ||
    DNN_scale_parameters$K != 1L || DNN_scale_parameters$E != 2L) {
  stop("These are not the constant-hazard recovery models. Retrain BDB_DNN_train.r.")
}
for (part in c("first_prob", "first_mean", "second_H1", "second_H2", "second_H12")) {
  center <- DNN_scale_parameters[[paste0("col_means_", part)]]
  scl <- DNN_scale_parameters[[paste0("col_stddevs_", part)]]
  expected <- if (grepl("first", part)) 8L else if (part == "second_H12") 2L else 3L
  if (length(center) != expected || length(scl) != expected ||
      any(!is.finite(center)) || any(!is.finite(scl)) || any(scl <= 0))
    stop("Incompatible scaling parameters for ", part, "; retrain first.")
}
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

#### 2. Constant-hazard scenarios ----
nsim <- 1000L
E <- 2L
K <- 1L
# Read the design used for training, rather than silently testing another n.
n_ctrl <- n_trt <- DNN_scale_parameters$n_new
analysis_time <- DNN_scale_parameters$analysis_time
stopifnot(length(n_ctrl) == 1L, n_ctrl > 0, length(analysis_time) == 1L, analysis_time > 0)
cuts <- c(0, analysis_time)
entry_dist <- function() 0
HR.grid <- c(1, 1.25, 1.5, 1.75, 2)
historical_mean_days <- rbind(KIDCARE = c(fever = 2.5, discharge = 4.5),
                              Wang2020 = c(fever = 0.750, discharge = 6.7))
stopifnot(isTRUE(all.equal(DNN_scale_parameters$historical_mean_days, historical_mean_days)),
          identical(as.integer(DNN_scale_parameters$n.vec.hist), c(49L, 40L)))
current_control_means <- rbind(
  A_KIDCARE = historical_mean_days[1, ],
  B_middle = sqrt(historical_mean_days[1, ] * historical_mean_days[2, ]),
  C_Wang = historical_mean_days[2, ]
)
# Five testing scenarios only: fixed KIDCARE current control, equal endpoint HRs.
scenario_table <- data.frame(
  scenario_name = paste("A_KIDCARE_both", sprintf("HR%.2f", HR.grid), sep = "_"),
  control = "A_KIDCARE", effect = "both", HR_grid = HR.grid,
  HR_fever = HR.grid, HR_discharge = HR.grid
)
scenarios_to_run <- scenario_table$scenario_name
output_dir <- file.path(script_dir, "constant_test_results_5HR_plugin")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(scenario_table, file.path(output_dir, "scenario_settings.csv"), row.names = FALSE)

get_scenario_params <- function(scenario_name) {
  row <- scenario_table[match(scenario_name, scenario_table$scenario_name), ]
  if (is.na(row$scenario_name)) stop("Unknown scenario: ", scenario_name)
  list(base_lam_current_ctrl_mat = matrix(1 / current_control_means[row$control, ], E, K),
       beta_mat = matrix(log(c(row$HR_fever, row$HR_discharge)), E, K), row = row)
}

run_scenario <- function(scenario_name, scenario_seed = 123) {
scenario_params <- get_scenario_params(scenario_name)
base_lam_current_ctrl_mat <- scenario_params$base_lam_current_ctrl_mat
beta_mat <- scenario_params$beta_mat

# Constant effects are known exactly; no PWE reference CSV or weighting.
true_beta_ref <- as.numeric(beta_mat[, 1])
true_HR_ref <- exp(true_beta_ref)
cat("Running scenario:", scenario_name, "; recovery HRs:", true_HR_ref, "\n")

## 3. Run simulations----

flatten_features <- function(d_c, Y_c, d_t, Y_t) {
  c(
    as.numeric(d_c),
    as.numeric(Y_c),
    as.numeric(d_t),
    as.numeric(Y_t)
  )
}

# Stage-2 models approximate parameter-to-cutoff maps. At test time, plug in
# estimates from each observed current dataset, using the saved training scales.
# Estimates are not clipped to training support; extrapolation and plug-in
# uncertainty require a new error-rate evaluation with this testing procedure.
predict_stage2_cutoff <- function(model, x, center, scale_value) {
  x <- matrix(as.numeric(x), nrow = 1)
  value <- as.numeric(predict(model, scale(x, center = center, scale = scale_value), verbose = 0))
  if (length(value) != 1L || !is.finite(value)) stop("Invalid Stage-2 cutoff prediction.")
  pmax(pmin(value, 1), 0)
}

## store iteration-level results

results.mat <- matrix(
  NA,
  nrow = nsim,
  ncol = 36
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
  "crit_H12",
  "crit_raw_H1", "crit_raw_H2",
  "stage2_loglam_ctrl_H1", "stage2_loglam_ctrl_H2",
  "stage2_beta_H1", "stage2_beta_H2",
  "d_ctrl_H1", "d_ctrl_H2", "Y_ctrl_H1", "Y_ctrl_H2",
  "d_trt_H1", "d_trt_H2", "Y_trt_H1", "Y_trt_H2"
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

  stage2_input <- estimate_stage2_inputs(d_c, Y_c, d_t, Y_t)
  c1 <- predict_stage2_cutoff(DNN.second.H1, stage2_input$H1,
    DNN_scale_parameters$col_means_second_H1,
    DNN_scale_parameters$col_stddevs_second_H1)
  c2 <- predict_stage2_cutoff(DNN.second.H2, stage2_input$H2,
    DNN_scale_parameters$col_means_second_H2,
    DNN_scale_parameters$col_stddevs_second_H2)
  c12 <- predict_stage2_cutoff(DNN.second.H12, stage2_input$H12,
    DNN_scale_parameters$col_means_second_H12,
    DNN_scale_parameters$col_stddevs_second_H12)
  c1.final <- max(c1, c12)
  c2.final <- max(c2, c12)
  
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
    x_prob_scale, verbose = 0
  )
  
  post_prob <- as.numeric(post_prob)
  
  post_prob <- pmax(
    pmin(post_prob, 1),
    0
  )
  
  est_beta <- predict(
    DNN.first.mean,
    x_mean_scale, verbose = 0
  )
  est_beta <- as.numeric(est_beta)
  if (length(post_prob) != E || length(est_beta) != E ||
      any(!is.finite(c(post_prob, est_beta)))) stop("Invalid DNN predictions.")
  est_HR <- exp(est_beta)
  
  true_beta_bar <- true_beta_ref

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
    c12,
    c1, c2,
    stage2_input$loglam_ctrl, stage2_input$beta,
    as.numeric(d_c), as.numeric(Y_c), as.numeric(d_t), as.numeric(Y_t)
  )
  
  cat("Scenario:", scenario_name, "Iteration:", sim, "\n")
}

results.df <- as.data.frame(results.mat)

power.summary <- data.frame(
  
  Scenario = rep(scenario_name, 3),
  
  Hypothesis = c("H1", "H2", "H12"),
  
  Rejection_rate = c(
    mean(results.df$reject_H1),
    mean(results.df$reject_H2),
    mean(results.df$reject_H12)
  ),
  
  # Critical_value is the mean of the per-replicate thresholds, not one cutoff.
  Critical_value = colMeans(results.df[, c("crit_H1", "crit_H2", "crit_H12")]),
  Critical_value_SD = vapply(results.df[, c("crit_H1", "crit_H2", "crit_H12")], sd, numeric(1)),
  Critical_value_min = vapply(results.df[, c("crit_H1", "crit_H2", "crit_H12")], min, numeric(1)),
  Critical_value_max = vapply(results.df[, c("crit_H1", "crit_H2", "crit_H12")], max, numeric(1)),
  Stage2_input_method = "current_data_plugin_MLE",
  
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

# Under null hypotheses these rates estimate error, not power.
power.summary$Null_true <- c(true_beta_ref <= 0, all(true_beta_ref <= 0))
power.summary$MCSE <- sqrt(power.summary$Rejection_rate * (1 - power.summary$Rejection_rate) / nsim)
power.summary$Control <- scenario_params$row$control
power.summary$Effect <- scenario_params$row$effect
power.summary$HR_grid <- scenario_params$row$HR_grid
power.summary$n_ctrl <- n_ctrl
power.summary$n_trt <- n_trt
power.summary$Reject_any_endpoint <- mean(results.df$reject_H1 | results.df$reject_H2)
power.summary$Reject_both_endpoints <- mean(results.df$reject_H1 & results.df$reject_H2)
null_endpoints <- which(true_beta_ref <= 0)
power.summary$FWER <- if (length(null_endpoints)) {
  mean(rowSums(results.df[, c("reject_H1", "reject_H2")[null_endpoints], drop = FALSE]) > 0)
} else NA_real_

print(power.summary)

write.csv(results.df, file.path(output_dir, paste0("DNN_results_", scenario_name, "_plugin.csv")), row.names = FALSE)
write.csv(power.summary, file.path(output_dir, paste0("DNN_power_", scenario_name, "_plugin.csv")), row.names = FALSE)

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
  file.path(output_dir, "DNN_summary_results_all_scenarios.csv"),
  row.names = FALSE
)

print(summary.results)
