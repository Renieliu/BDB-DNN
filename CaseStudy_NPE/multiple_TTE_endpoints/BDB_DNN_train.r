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
      file.path(directory, "CaseStudy_NPE", "multiple_TTE_endpoints", "BDB_DNN_train.r"),
      file.path(directory, "multiple_TTE_endpoints", "BDB_DNN_train.r"))
    parent <- dirname(directory)
    if (identical(parent, directory)) break
    directory <- parent
  }
  candidates <- unique(normalizePath(candidates[file.exists(candidates)], mustWork = TRUE))
  candidates <- candidates[grepl(
    "/CaseStudy_NPE/multiple_TTE_endpoints/BDB_DNN_train\\.r$", candidates)]
  if (!length(candidates)) {
    stop("Cannot locate CaseStudy_NPE. Source its BDB_DNN_train.r using the full path.")
  }
  candidates[1]
})
script_dir <- dirname(normalizePath(script_file, mustWork = TRUE))
root_dir <- dirname(script_dir)
## Python with TensorFlow/Keras for keras3: set RETICULATE_PYTHON beforehand if needed,
## e.g. Sys.setenv(RETICULATE_PYTHON = "/path/to/python")
setwd(root_dir)
sub_folder_name <- "multiple_TTE_endpoints"

## source functions file
functions_file <- file.path(root_dir, "BDB_DNN_functions.r")
source(functions_file)
if (!exists("make_npe_training_data", mode = "function", inherits = FALSE)) {
  stop("Missing make_npe_training_data after sourcing ", functions_file,
       ". Use the updated CaseStudy_NPE/BDB_DNN_functions.r file.")
}

library(keras3)
library(reticulate)
library(tensorflow)
library(tibble)
library(RBesT)
library(doParallel)
library(bindata)

## --- packages
library(foreach)
library(doParallel)
library(withr)

py_config()

# --------- user settings ----------
stage1.label.method <- "NPE" # "NPE" or "MCMC" (Stan posterior labels)
E <- 2 # number of outcomes
K <- 1L # one interval: exponential event times and constant log-HR
analysis_time <- 28 # days of administrative follow-up per current participant
cuts <- c(0, analysis_time)
entry_dist <- function() 0  # all participants enter at time 0

n.cluster <- 4 # number of parallel workers
type.1.error.working <- 0.05 ## working type I error rate

n_new <- 150 # current control/treatment arm sample size
n.train <- 1000 # number of training samples for the first-stage DNN
n.validation <- 240 # architecture-selection simulations, balanced across truth patterns
n.evaluation <- 240 # final independent diagnostics; never used for fitting or selection
n.second.train <- 600 # total Stage-2 configurations: 200 each for H1, H2, H12
n.second.train.itt <- 1000 # null replicates used to label each Stage-2 configuration
# Recovery/discharge events: HR > 1 is beneficial.
# Continuous training support; the five evaluation HRs are included in this range.
beta.lower <- log(1.0)
beta.upper <- log(2.0)
HR.grid <- c(1, 1.25, 1.5, 1.75, 2)

# Rows = historical studies; columns = fever and hospital discharge (days).
# Literature-calibrated complete-event summaries: D=n, Y=n*mean.
# These are fixed historical controls, not current treatment means, and are
# not regenerated with random study effects. SD is not an exponential input.
historical_mean_days <- rbind(
  KIDCARE = c(fever = 2.5, discharge = 4.5),
  Wang2020 = c(fever = 0.750, discharge = 6.7)
)
n.vec.hist <- c(KIDCARE = 49L, Wang2020 = 40L)
Jhist <- length(n.vec.hist)
historical_hazards <- 1 / historical_mean_days
d_hist <- array(rep(as.integer(n.vec.hist), E), dim = c(Jhist, E, K))
Y_hist <- array(historical_mean_days * n.vec.hist, dim = c(Jhist, E, K))

# Current controls may match either history, lie between them, or vary over
# a broad range covering all three. Rates have units day^-1.
current_control_rates <- rbind(
  KIDCARE = historical_hazards[1, ],
  middle = sqrt(historical_hazards[1, ] * historical_hazards[2, ]),
  Wang2020 = historical_hazards[2, ]
)
lograte.lower <- log(c(0.25, 0.10))
lograte.upper <- log(c(1.8, 0.35))
stopifnot(K == 1L, Jhist == 2L,
          all(c(n.train, n.validation, n.evaluation) %% 4 == 0))

# Independent conditional-prior simulations for fitting the NPE density.
n.npe.train <- 10000L
n.npe.validation <- 2000L
n.npe.evaluation <- 2000L

if (stage1.label.method == "MCMC") {
  library(rstan)
  rstan_options(auto_write = TRUE)
  options(mc.cores = 1)
  stan_file <- "multi_tte_constant.stan"
  sm <- rstan::stan_model(file = stan_file)
  saveRDS(sm, file.path(getwd(), "multi_tte_constant_sm.rds"))
}

# training feature matrix: flatten d/Y for ctrl + trt
p_feat <- 2 * 2 * E * K
# Keep the validation simulations separate from the very beginning.  This is
# deliberately not a Keras validation_split: first-stage labels are created
# after the train/validation/evaluation split.
n.stage1.total <- n.train + n.validation + n.evaluation
data.stage1.all <- matrix(NA_real_, nrow = n.stage1.total, ncol = p_feat)
beta_bar.stage1.true <- matrix(NA_real_, nrow = n.stage1.total, ncol = E)

# helper to flatten
flatten_features <- function(d_c, Y_c, d_t, Y_t) {
  c(as.numeric(d_c), as.numeric(Y_c), as.numeric(d_t), as.numeric(Y_t))
}

# --------- simulate training inputs ----------
set.seed(1)
for (i in 1:n.stage1.total) {
  
  baseline.pattern <- sample(c(rownames(current_control_rates), "random"), 1)
  if (baseline.pattern == "random") {
    current_rates <- exp(runif(E, lograte.lower, lograte.upper))
  } else {
    current_rates <- current_control_rates[baseline.pattern, ]
  }
  lam_c1 <- current_rates[1]
  lam_c2 <- current_rates[2]

  # draw constant treatment effects (4 truth patterns: global null, two single nulls, global alternative)
  beta <- matrix(0, nrow = E, ncol = K)
  if (i <= n.stage1.total/4) {
    beta[,] <- 0
  } else if (i <= 2*n.stage1.total/4) {
    beta[2, ] <- runif(1, beta.lower, beta.upper)
  } else if (i <= 3*n.stage1.total/4) {
    beta[1, ] <- runif(1, beta.lower, beta.upper)
  } else {
    beta[1, ] <- runif(1, beta.lower, beta.upper)
    beta[2, ] <- runif(1, beta.lower, beta.upper)
  }
  
  lam_t1 <- lam_c1 * exp(beta[1, ])
  lam_t2 <- lam_c2 * exp(beta[2, ])
  
  # Simulate current trial with administrative censoring at day 28.
  s_c1 <- sim_pwe_arm_counts(n_new, lam_c1, cuts, analysis_time, entry_dist)
  s_c2 <- sim_pwe_arm_counts(n_new, lam_c2, cuts, analysis_time, entry_dist)
  s_t1 <- sim_pwe_arm_counts(n_new, lam_t1, cuts, analysis_time, entry_dist)
  s_t2 <- sim_pwe_arm_counts(n_new, lam_t2, cuts, analysis_time, entry_dist)
  
  d_c <- rbind(s_c1$d, s_c2$d)
  Y_c <- rbind(s_c1$Y, s_c2$Y)
  d_t <- rbind(s_t1$d, s_t2$d)
  Y_t <- rbind(s_t1$Y, s_t2$Y)
  
  data.stage1.all[i, ] <- flatten_features(d_c, Y_c, d_t, Y_t)
  beta_bar.stage1.true[i, ] <- as.numeric(beta[, 1])
}

# These objects are intentionally kept disjoint.  In particular, neither the
# feature scaling nor architecture selection sees the independent validation
# simulations during fitting.
# Stratified three-way split: architecture validation and final evaluation both
# contain global-null, each single-null, and global-alternative trials.  A
# contiguous split would put only one truth pattern in each set because the
# simulations are generated in four blocks.
truth_pattern <- rep(seq_len(4), each = n.stage1.total / 4)
set.seed(20260622)
stratified_indices <- split(seq_len(n.stage1.total), truth_pattern)
validation_idx <- unlist(lapply(stratified_indices, sample, size = n.validation / 4))
evaluation_idx <- unlist(Map(function(idx, val_idx) {
  sample(setdiff(idx, val_idx), size = n.evaluation / 4)
}, stratified_indices, split(validation_idx, truth_pattern[validation_idx])))
train_idx <- setdiff(seq_len(n.stage1.total), c(validation_idx, evaluation_idx))

# --------- Stage-1 labels ----------
if (stage1.label.method == "NPE") {
  # Fit history once, then draw current parameters conditional on that history.
  # The historical D/Y therefore determine the NPE training distribution.
  npe_data <- make_npe_training_data(
    d_hist, Y_hist, n_new, analysis_time,
    n.npe.train, n.npe.validation, n.npe.evaluation,
    stan_file = file.path(root_dir, "npe_history_constant.stan"),
    output_dir = file.path(root_dir, sub_folder_name), seed = 20260927L)
  data.NPE.train <- as_tibble(npe_data$train$x)
  data.NPE.validation <- as_tibble(npe_data$validation$x)
  col_means_npe <- colMeans(as.matrix(data.NPE.train))
  col_stddevs_npe <- apply(as.matrix(data.NPE.train), 2, sd)
  col_stddevs_npe[!is.finite(col_stddevs_npe) | col_stddevs_npe == 0] <- 1
  data.NPE.train.scale <- scale(data.NPE.train, center = col_means_npe, scale = col_stddevs_npe)
  data.NPE.validation.scale <- scale(data.NPE.validation, center = col_means_npe, scale = col_stddevs_npe)
  data.NPE.all.scale <- scale(as_tibble(data.stage1.all), center = col_means_npe, scale = col_stddevs_npe)

  NPE.beta_bar <- NPE.gaussian.fit(
    data.train.scale.in = data.NPE.train.scale,
    beta_bar.train.in = npe_data$train$beta,
    data.validation.scale.in = data.NPE.validation.scale,
    beta_bar.validation.in = npe_data$validation$beta,
    drop.rate.in = 0.1,
    active.name.in = "relu",
    n.node.in = 120,
    n.layer.in = 3,
    max.epoch.in = 1000,
    batch_size_in = 100,
    n.endpoint.in = E
  )
  
  npe.stage1.posterior <- predict_npe_gaussian(
    NPE.model.in = NPE.beta_bar$model,
    data.scale.in = data.NPE.all.scale,
    n.endpoint.in = E
  )
  
  label.stage1.all <- cbind(npe.stage1.posterior$prob_gt0, npe.stage1.posterior$mean)
  # Independent parameter-recovery/coverage diagnostics, not distillation error.
  check <- predict_npe_gaussian(NPE.beta_bar$model,
    scale(npe_data$evaluation$x, center = col_means_npe, scale = col_stddevs_npe), E)
  truth <- npe_data$evaluation$beta
  write.csv(data.frame(endpoint = c("fever", "discharge"),
    mean_RMSE = sqrt(colMeans((check$mean - truth)^2)),
    coverage_95 = colMeans(abs(check$mean - truth) <= qnorm(.975) * check$sd),
    sign_Brier = colMeans((check$prob_gt0 - (truth > 0))^2)),
    file.path(sub_folder_name, "NPE_independent_parameter_diagnostics.csv"), row.names = FALSE)
} else if (stage1.label.method == "MCMC") {
  cl <- parallel::makeCluster(n.cluster)
  registerDoParallel(cl)
  
  label.stage1.all <- foreach(i = 1:n.stage1.total, .combine = "rbind") %dopar% {
    
    library(rstan)
    sm_worker <- readRDS("multi_tte_constant_sm.rds")
    
    # unflatten
    idx <- 1
    d_c <- matrix(as.integer(data.stage1.all[i, idx:(idx+E*K-1)]), E, K); idx <- idx + E*K
    Y_c <- matrix(data.stage1.all[i, idx:(idx+E*K-1)], E, K); idx <- idx + E*K
    d_t <- matrix(as.integer(data.stage1.all[i, idx:(idx+E*K-1)]), E, K); idx <- idx + E*K
    Y_t <- matrix(data.stage1.all[i, idx:(idx+E*K-1)], E, K)
    
    stan_data <- list(
      Jhist = Jhist, E = E, K = K,
      d_hist = d_hist,
      Y_hist = Y_hist,
      d_c = d_c, Y_c = Y_c,
      d_t = d_t, Y_t = Y_t
    )
    
    fit <- rstan::sampling(
      sm_worker, data = stan_data,
      chains = 2, iter = 2000, warmup = 1000,
      refresh = 0, seed = 1000 + i, cores = 1,
      control = list(adapt_delta = 0.99, max_treedepth = 12)
    )
    
    post <- rstan::extract(fit)
    pr_beta_gt0 <- colMeans(post$beta_bar > 0)
    beta_mean <- colMeans(post$beta_bar)
    
    c(pr_beta_gt0, beta_mean)
  }
  
  stopCluster(cl)
} else {
  stop("Unknown stage1.label.method. Use 'NPE' or 'MCMC'.")
}

colnames(label.stage1.all) <- c(
  paste0("Pr_beta_bar_gt0_end", 1:E),
  paste0("E_beta_bar_end", 1:E)
)

data.train <- data.stage1.all[train_idx, , drop = FALSE]
label.train <- label.stage1.all[train_idx, , drop = FALSE]
data.validation <- data.stage1.all[validation_idx, , drop = FALSE]
label.validation <- label.stage1.all[validation_idx, , drop = FALSE]
data.evaluation <- data.stage1.all[evaluation_idx, , drop = FALSE]
label.evaluation <- label.stage1.all[evaluation_idx, , drop = FALSE]
label.evaluation.prob <- label.evaluation[, 1:E, drop = FALSE]
label.evaluation.mean <- label.evaluation[, (E + 1):(2 * E), drop = FALSE]

###############################################################################################
## fit DNN on posterior means (E_beta)
## input: current control summaries only (d_c + Y_c)

p_block <- E * K

data.train.mean <- data.train  # d_c + Y_c + d_t + Y_t
data.DNN.train.mean <- as_tibble(data.train.mean)

## standardize training data
col_means_train_mean <- colMeans(as.matrix(data.DNN.train.mean))
col_stddevs_train_mean <- apply(as.matrix(data.DNN.train.mean), 2, sd)
col_stddevs_train_mean[!is.finite(col_stddevs_train_mean) | col_stddevs_train_mean == 0] <- 1
data.DNN.train.mean.scale <- scale(data.DNN.train.mean,
                                   center = col_means_train_mean,
                                   scale = col_stddevs_train_mean)

## output label: posterior mean of constant beta (E endpoints)
label.train.mean <- label.train[, (E + 1):(2 * E)]  # E_beta_bar_end1..E

## Choose each first-stage architecture against an independent simulation set.
## Do not reuse the beta-regression architecture for probability calibration.
DNN.arch.grid <- data.frame(
  layer = c(2, 2, 3, 3, 4, 4),
  node  = c(40, 80, 60, 100, 80, 120)
)

tune_dnn_architecture <- function(x_train, y_train, x_validation, y_validation,
                                  output_activation) {
  tuning <- DNN.arch.grid
  tuning$train_MSE <- tuning$validation_MSE <- NA_real_

  for (cross.ind in seq_len(nrow(tuning))) {
    fit <- DNN.fit(
    data.train.scale.in  = x_train,
    data.train.label.in  = y_train,
    drop.rate.in         = 0.1,
    active.name.in       = "relu",
    n.node.in            = tuning$node[cross.ind],
    n.layer.in           = tuning$layer[cross.ind],
    max.epoch.in         = 1000,
    batch_size_in        = 100,
    validation.prop.in   = 0,
    n.endpoint.in        = ncol(y_train),
    output_activation    = output_activation,
    loss_name            = "mse"
    )
    fitted_validation <- predict(fit$model, x_validation)
    tuning$train_MSE[cross.ind] <- tail(fit$history$metrics$mse, 1)
    tuning$validation_MSE[cross.ind] <- mean((fitted_validation - y_validation)^2)
  }
  tuning
}

data.DNN.validation.mean.scale <- scale(
  as_tibble(data.validation),
  center = col_means_train_mean, scale = col_stddevs_train_mean
)
label.validation.mean <- label.validation[, (E + 1):(2 * E), drop = FALSE]

DNN.mean.tuning <- tune_dnn_architecture(
  data.DNN.train.mean.scale, label.train.mean,
  data.DNN.validation.mean.scale, label.validation.mean,
  output_activation = "linear"
)
DNN.mean.best.layer <- DNN.mean.tuning$layer[which.min(DNN.mean.tuning$validation_MSE)]
DNN.mean.best.node  <- DNN.mean.tuning$node[which.min(DNN.mean.tuning$validation_MSE)]

## fit final mean DNN on full data (no validation split)
DNN.first.mean <- DNN.fit(
  data.train.scale.in  = data.DNN.train.mean.scale,
  data.train.label.in  = label.train.mean,
  drop.rate.in         = 0.1,
  active.name.in       = "relu",
  n.node.in            = DNN.mean.best.node,
  n.layer.in           = DNN.mean.best.layer,
  max.epoch.in         = 1000,
  batch_size_in        = 100,
  validation.prop.in   = 0,
  n.endpoint.in        = ncol(label.train.mean),
  output_activation    = "linear",
  loss_name            = "mse"
)

label.mean.pred <- DNN.first.mean$model %>% predict(data.DNN.train.mean.scale)
DNN.first.mean.weight <- get_weights(DNN.first.mean$model)

print(DNN.first.mean)

###############################################################################################
## fit DNN on posterior probabilities Pr(beta>0)
## input: full current-trial summaries (d_c, Y_c, d_t, Y_t)

data.train.prob <- data.train
data.DNN.train.prob <- as_tibble(data.train.prob)

## standardize
col_means_train_prob <- colMeans(as.matrix(data.DNN.train.prob))
col_stddevs_train_prob <- apply(as.matrix(data.DNN.train.prob), 2, sd)
col_stddevs_train_prob[!is.finite(col_stddevs_train_prob) | col_stddevs_train_prob == 0] <- 1
data.DNN.train.prob.scale <- scale(data.DNN.train.prob,
                                   center = col_means_train_prob,
                                   scale = col_stddevs_train_prob)

## output label: posterior prob constant beta>0 (E endpoints)
label.train.prob <- label.train[, 1:E]

data.DNN.validation.prob.scale <- scale(
  as_tibble(data.validation),
  center = col_means_train_prob, scale = col_stddevs_train_prob
)
label.validation.prob <- label.validation[, 1:E, drop = FALSE]

DNN.prob.tuning <- tune_dnn_architecture(
  data.DNN.train.prob.scale, label.train.prob,
  data.DNN.validation.prob.scale, label.validation.prob,
  output_activation = "sigmoid"
)
DNN.prob.best.layer <- DNN.prob.tuning$layer[which.min(DNN.prob.tuning$validation_MSE)]
DNN.prob.best.node  <- DNN.prob.tuning$node[which.min(DNN.prob.tuning$validation_MSE)]

DNN.first.prob <- DNN.fit(
  data.train.scale.in  = data.DNN.train.prob.scale,
  data.train.label.in  = label.train.prob,
  drop.rate.in         = 0.1,
  active.name.in       = "relu",
  n.node.in            = DNN.prob.best.node,
  n.layer.in           = DNN.prob.best.layer,
  max.epoch.in         = 1000,
  batch_size_in        = 100,
  validation.prop.in   = 0,
  n.endpoint.in        = ncol(label.train.prob),
  output_activation    = "sigmoid",
  loss_name            = "mse"
)

label.prob.pred <- DNN.first.prob$model %>% predict(data.DNN.train.prob.scale)
DNN.first.prob.weight <- get_weights(DNN.first.prob$model)

print(DNN.first.prob)

#########################################################################################
## Stage 2: train three DNNs mapping true simulation configurations to cutoff
## values. Training uses data-generating parameters to learn the cutoff surface;
## testing plugs in current-data estimates and must validate the resulting error rates.
## For c12, the label solves Pr{(S1 > c12) OR (S2 > c12) | H12} <= alpha.

stopifnot(n.second.train %% 3 == 0)
n.second.per.hypothesis <- n.second.train / 3
beta_cols <- as.vector(outer(paste0("beta_e", 1:E), paste0("_k", 1:K), paste0))
make_stage2_candidates <- function(null_hypothesis, n) {
  out <- matrix(0, nrow = n, ncol = E + E * K)
  colnames(out) <- c(paste0("loglam_mean_e", 1:E), beta_cols)
  for (s in seq_len(n)) {
    out[s, 1:E] <- runif(E, lograte.lower, lograte.upper)
    beta <- matrix(0, nrow = E, ncol = K)
    if (null_hypothesis == "H1") beta[2, ] <- runif(1, beta.lower, beta.upper)
    if (null_hypothesis == "H2") beta[1, ] <- runif(1, beta.lower, beta.upper)
    out[s, (E + 1):ncol(out)] <- as.numeric(beta)
  }
  as.data.frame(out)
}

set.seed(20260624)
cand.H12 <- make_stage2_candidates("H12", n.second.per.hypothesis)
cand.H1 <- make_stage2_candidates("H1", n.second.per.hypothesis)
cand.H2 <- make_stage2_candidates("H2", n.second.per.hypothesis)

cl <- parallel::makeCluster(n.cluster)
doParallel::registerDoParallel(cl)
parallel::clusterExport(cl, c(
  "critical.value.func.tte", "sim_pwe_arm_counts", "make_lam_profile",
  "pred.DNN.normal", "pred.DNN.sigmoid", "sigmoid", "relu", "cuts", "analysis_time", "entry_dist",
  "n.second.train.itt", "n_new", "E", "K", "col_means_train_prob",
  "col_stddevs_train_prob", "DNN.first.prob.weight", "DNN.prob.best.layer",
  "type.1.error.working", "beta_cols"
), envir = environment())
stage2_labels <- function(candidates) {
  foreach(s = seq_len(nrow(candidates)), .combine = c) %dopar% {
    loglam_mean_vec <- as.numeric(candidates[s, paste0("loglam_mean_e", 1:E)])
    beta_mat <- matrix(as.numeric(candidates[s, beta_cols]), nrow = E, ncol = K)
    critical.value.func.tte(
      n.itt.in = n.second.train.itt, n.new.in = n_new, E = E, K = K, cuts = cuts,
      loglam_mean_vec = loglam_mean_vec, beta_mat = beta_mat, shape_sd = 0,
      col_means_train.in = col_means_train_prob,
      col_stddevs_train.in = col_stddevs_train_prob,
      DNN.first.weight.in = DNN.first.prob.weight,
      DNN.first.layer.in = DNN.prob.best.layer,
      type.1.error.in = type.1.error.working,
      analysis_time.in = analysis_time, entry_dist.in = entry_dist
    )
  }
}
label.H12 <- stage2_labels(cand.H12)
label.H1 <- stage2_labels(cand.H1)
label.H2 <- stage2_labels(cand.H2)
parallel::stopCluster(cl)

DNN.stage2.arch.grid <- data.frame(layer = c(1, 2, 2, 3), node = c(20, 40, 80, 60))
fit_stage2_dnn <- function(x, y, seed) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  y <- as.numeric(y)
  if (any(!is.finite(x))) stop("Stage-2 inputs contain NA, NaN, or Inf.")
  if (any(!is.finite(y))) stop("Stage-2 cutoff labels contain NA, NaN, or Inf.")
  if (any(y < 0 | y > 1)) stop("Stage-2 cutoff labels must lie in [0, 1].")
  set.seed(seed)
  validation_idx <- sample(seq_len(nrow(x)), size = ceiling(0.2 * nrow(x)))
  train_idx <- setdiff(seq_len(nrow(x)), validation_idx)
  x_train <- as.matrix(x[train_idx, , drop = FALSE])
  x_validation <- as.matrix(x[validation_idx, , drop = FALSE])
  center <- colMeans(x_train)
  scale_value <- apply(x_train, 2, sd)
  scale_value[scale_value == 0] <- 1
  x_train <- scale(x_train, center = center, scale = scale_value)
  x_validation <- scale(x_validation, center = center, scale = scale_value)
  tuning <- DNN.stage2.arch.grid
  tuning$validation_MSE <- NA_real_
  for (j in seq_len(nrow(tuning))) {
    fit <- DNN.fit(x_train, matrix(y[train_idx], ncol = 1), 0.1, "relu",
                   tuning$node[j], tuning$layer[j], 1000, 100, 0, 1,
                   output_activation = "sigmoid", loss_name = "mse")
    pred <- as.numeric(predict(fit$model, x_validation))
    tuning$validation_MSE[j] <- mean((pred - y[validation_idx])^2)
  }
  valid_models <- which(is.finite(tuning$validation_MSE))
  if (!length(valid_models)) {
    stop("All Stage-2 architecture validation losses are non-finite. ",
         "Check the Stage-2 cutoff labels before fitting.")
  }
  best <- valid_models[which.min(tuning$validation_MSE[valid_models])]
  x_all <- as.matrix(x)
  center <- colMeans(x_all)
  scale_value <- apply(x_all, 2, sd)
  scale_value[scale_value == 0] <- 1
  model <- DNN.fit(scale(x_all, center = center, scale = scale_value), matrix(y, ncol = 1),
                   0.1, "relu", tuning$node[best], tuning$layer[best], 1000, 100, 0, 1,
                   output_activation = "sigmoid", loss_name = "mse")
  list(DNN.model = model, mean_center = center, sd_center = scale_value,
       tuning = tuning, best_layer = tuning$layer[best], best_node = tuning$node[best])
}

DNN.second.H12.fit <- fit_stage2_dnn(cand.H12[, paste0("loglam_mean_e", 1:E)], label.H12, 101)
DNN.second.H1.fit <- fit_stage2_dnn(cand.H1[, c(paste0("loglam_mean_e", 1:E), paste0("beta_e2_k", 1:K))], label.H1, 102)
DNN.second.H2.fit <- fit_stage2_dnn(cand.H2[, c(paste0("loglam_mean_e", 1:E), paste0("beta_e1_k", 1:K))], label.H2, 103)
write.csv(rbind(cbind(task = "H1", DNN.second.H1.fit$tuning),
                cbind(task = "H2", DNN.second.H2.fit$tuning),
                cbind(task = "H12", DNN.second.H12.fit$tuning)),
          file.path(sub_folder_name, "DNN_stage2_architecture_tuning_TTE.csv"), row.names = FALSE)

#########################################################################################
## write fitted label to a file

data.train.pred.com <- cbind(data.train, label.train, label.prob.pred, label.mean.pred)

colnames(data.train.pred.com) <- c(
  paste0("x", 1:ncol(data.train)),
  paste0("label_prob_beta_bar_end", 1:E),
  paste0("label_mean_beta_bar_end", 1:E),
  paste0("fit_prob_beta_bar_end", 1:E),
  paste0("fit_mean_beta_bar_end", 1:E)
)

write.csv(data.train.pred.com, file = file.path(sub_folder_name, "DNN_train_label_TTE_stage1.csv"),
          row.names = FALSE)

######################################################################################
## Save trained DNNs to files

if (stage1.label.method == "NPE") {
  keras3::save_model(
    NPE.beta_bar$model,
    filepath = file.path(sub_folder_name, "NPE_beta_bar.keras"),
    overwrite = TRUE
  )
}

keras3::save_model(
  DNN.first.mean$model,
  filepath = file.path(sub_folder_name, "DNN_first_mean.keras"),
  overwrite = TRUE
)

keras3::save_model(
  DNN.first.prob$model,
  filepath = file.path(sub_folder_name, "DNN_first_prob.keras"),
  overwrite = TRUE
)
keras3::save_model(DNN.second.H1.fit$DNN.model$model,
                   filepath = file.path(sub_folder_name, "DNN_second_H1_TTE.keras"), overwrite = TRUE)
keras3::save_model(DNN.second.H2.fit$DNN.model$model,
                   filepath = file.path(sub_folder_name, "DNN_second_H2_TTE.keras"), overwrite = TRUE)
keras3::save_model(DNN.second.H12.fit$DNN.model$model,
                   filepath = file.path(sub_folder_name, "DNN_second_H12_TTE.keras"), overwrite = TRUE)

# Load later
#DNN.first.mean$model <- keras3::load_model(
#  file.path(sub_folder_name, "DNN_first_mean.keras")
#)

## save scale parameters for DNNs
DNN_scale_parameters = list("model_type" = "constant_hazard",
                       "K" = K, "E" = E, "n_new" = n_new,
                       "analysis_time" = analysis_time,
                       "historical_mean_days" = historical_mean_days,
                       "n.vec.hist" = n.vec.hist,
                       "benefit_direction" = "HR > 1", "HR.grid" = HR.grid,
                       "NPE_history_conditioned" = stage1.label.method == "NPE",
                       "Stage1_label_method" = stage1.label.method,
                       "NPE_col_means" = if (exists("col_means_npe")) col_means_npe else NULL,
                       "NPE_col_stddevs" = if (exists("col_stddevs_npe")) col_stddevs_npe else NULL,
                       "NPE_layer" = if (stage1.label.method == "NPE") 3 else NULL,
                       "NPE_node" = if (stage1.label.method == "NPE") 120 else NULL,
                       "NPE_posterior_family" = if (stage1.label.method == "NPE") "diagonal_gaussian_beta_bar" else NULL,
                       "col_means_first_mean" = col_means_train_mean, 
                       "col_stddevs_first_mean" = col_stddevs_train_mean,
                       "col_means_first_prob" = col_means_train_prob, 
                       "col_stddevs_first_prob" = col_stddevs_train_prob,
                       "DNN_mean_best_layer" = DNN.mean.best.layer,
                       "DNN_mean_best_node" = DNN.mean.best.node,
                       "DNN_prob_best_layer" = DNN.prob.best.layer,
                       "DNN_prob_best_node" = DNN.prob.best.node,
                       "col_means_second_H1" = DNN.second.H1.fit$mean_center,
                       "col_stddevs_second_H1" = DNN.second.H1.fit$sd_center,
                       "col_means_second_H2" = DNN.second.H2.fit$mean_center,
                       "col_stddevs_second_H2" = DNN.second.H2.fit$sd_center,
                       "col_means_second_H12" = DNN.second.H12.fit$mean_center,
                       "col_stddevs_second_H12" = DNN.second.H12.fit$sd_center,
                       "Stage2_H1_best_layer" = DNN.second.H1.fit$best_layer,
                       "Stage2_H1_best_node" = DNN.second.H1.fit$best_node,
                       "Stage2_H2_best_layer" = DNN.second.H2.fit$best_layer,
                       "Stage2_H2_best_node" = DNN.second.H2.fit$best_node,
                       "Stage2_H12_best_layer" = DNN.second.H12.fit$best_layer,
                       "Stage2_H12_best_node" = DNN.second.H12.fit$best_node,
                       "Stage2_cutoff_rule" = ">"
)

save(DNN_scale_parameters, file = file.path(sub_folder_name, "DNN_scale_parameters_TTE"))

## ---- Stage 1 performance on TRAIN (in-sample) ----
# mean DNN: beta mean regression
mse_beta_train <- colMeans((label.mean.pred - label.train.mean)^2)
rmse_beta_train <- sqrt(mse_beta_train)

cat("Stage1 mean-DNN TRAIN RMSE (beta):\n")
print(rmse_beta_train)

# prob DNN: Pr(beta>0)
mse_prob_train <- colMeans((label.prob.pred - label.train.prob)^2)
rmse_prob_train <- sqrt(mse_prob_train)

cat("Stage1 prob-DNN TRAIN RMSE (Pr(beta>0)):\n")
print(rmse_prob_train)

cat("Stage-1 architecture choices (independent validation MSE):\n")
print(list(mean = c(layer = DNN.mean.best.layer, node = DNN.mean.best.node),
           probability = c(layer = DNN.prob.best.layer, node = DNN.prob.best.node)))

## ---- Final independent diagnostics (not used to fit or select any model) ----
data.DNN.evaluation.prob.scale <- scale(
  as_tibble(data.evaluation),
  center = col_means_train_prob, scale = col_stddevs_train_prob
)
data.DNN.evaluation.mean.scale <- scale(
  as_tibble(data.evaluation),
  center = col_means_train_mean, scale = col_stddevs_train_mean
)
evaluation.prob.pred <- as.matrix(predict(DNN.first.prob$model,
                                          data.DNN.evaluation.prob.scale))
evaluation.mean.pred <- as.matrix(predict(DNN.first.mean$model,
                                          data.DNN.evaluation.mean.scale))
evaluation.pattern <- truth_pattern[evaluation_idx]

validation.diagnostics <- data.frame(
  beta_MSE_H1 = mean((evaluation.mean.pred[, 1] - label.evaluation.mean[, 1])^2),
  beta_MSE_H2 = mean((evaluation.mean.pred[, 2] - label.evaluation.mean[, 2])^2),
  probability_MSE_H1 = mean((evaluation.prob.pred[, 1] - label.evaluation.prob[, 1])^2),
  probability_MSE_H2 = mean((evaluation.prob.pred[, 2] - label.evaluation.prob[, 2])^2)
)
cat("Independent distillation diagnostics (NPE labels are not exact posterior truth):\n")
print(validation.diagnostics)
write.csv(validation.diagnostics,
          file = file.path(sub_folder_name, "DNN_independent_validation_diagnostics_TTE.csv"),
          row.names = FALSE)
