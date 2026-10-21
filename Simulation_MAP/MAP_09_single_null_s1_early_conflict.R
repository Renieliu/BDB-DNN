library(rstan)
library(posterior)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

source("BDB_DNN_functions.r")

#### 1. Simulation Setup ====

# choose scenario
scenario_name <- "09_single_null_s1_early_conflict"

# "01_global_null_compatible"
# "02_global_null_mild_conflict"
# "03_global_null_moderate_conflict"

# "04_single_null_s1_delayed_compatible"
# "05_single_null_s1_early_compatible"
# "06_single_null_s2_delayed_compatible"
# "07_single_null_s2_early_compatible"
# "08_single_null_s1_delayed_conflict"
# "09_single_null_s1_early_conflict"
# "10_single_null_s2_delayed_conflict"
# "11_single_null_s2_early_conflict"

# "12_global_alt_delayed_compatible"
# "13_global_alt_early_compatible"
# "14_global_alt_delayed_conflict"
# "15_global_alt_early_conflict"

# trial design
nsim <- 1000
decision_threshold <- 0.975 # critical value c

E <- 2  # ! if these two need to be changed, scenarios below also need to be changed, including censor design
K <- 6
cuts <- seq(0, 6, by = 1)
n_ctrl <- 150
n_trt  <- 150


# historical data design
n.vec.hist <- c(100, 100, 200, 200, 300, 300)
tau_hist <- 0.5
r_endpt <- 0.2

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
    c(1, 1, 0.9, 0.8, 0.7, 0.7),
    c(1, 1, 0.95, 0.85, 0.75, 0.75)
  )
  log(HR_mat)
}

make_early_beta <- function() {
  HR_mat <- rbind(
    c(0.7, 0.75, 0.85, 0.95, 1, 1),
    c(0.75, 0.8, 0.9, 0.95, 1, 1)
  )
  log(HR_mat)
}

make_s1_delayed_beta <- function() {
  HR_mat <- rbind(
    c(1, 1, 0.9, 0.8, 0.7, 0.7),
    c(1, 1, 1, 1, 1, 1)
  )
  log(HR_mat)
}

make_s1_early_beta <- function() {
  HR_mat <- rbind(
    c(0.7, 0.75, 0.85, 0.95, 1, 1),
    c(1, 1, 1, 1, 1, 1)
  )
  log(HR_mat)
}

make_s2_delayed_beta <- function() {
  HR_mat <- rbind(
    c(1, 1, 1, 1, 1, 1),
    c(1, 1, 0.95, 0.85, 0.75, 0.75)
  )
  log(HR_mat)
}

make_s2_early_beta <- function() {
  HR_mat <- rbind(
    c(1, 1, 1, 1, 1, 1),
    c(0.75, 0.8, 0.9, 0.95, 1, 1)
  )
  log(HR_mat)
}

# map scenario -> parameters

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

cat("Running scenario:", scenario_name, "\n")
cat("HR matrix:\n")
print(exp(beta_mat))

#### 2. Compile Stan models ====

mod_hist <- rstan::stan_model("map_hist_model.stan")
mod_current <- rstan::stan_model("map_current_model.stan")

#### 3. Extract MAP prior mean/sd from historical fit ====

extract_map_prior <- function(fit_hist, E, K) {
  theta_new_arr <- rstan::extract(fit_hist, pars = "theta_new")$theta_new
  # expected dims: iterations x E x K
  
  theta_prior_mean <- apply(theta_new_arr, c(2, 3), mean)
  theta_prior_sd   <- apply(theta_new_arr, c(2, 3), sd)
  
  list(
    theta_prior_mean = theta_prior_mean,
    theta_prior_sd   = theta_prior_sd
  )
}

# 4. One simulation replicate ====

one_rep_map <- function(
    seed,
    n.vec.hist,
    n_ctrl,
    n_trt,
    E, K, cuts,
    base_lam_hist_mat,
    base_lam_current_ctrl_mat,
    tau_hist,
    r_endpt,
    beta_mat,
    mod_hist,
    mod_current,
    analysis_time = cuts[length(cuts)],
    entry_dist = function() 0,
    hist_iter_warmup = 500,
    hist_iter_sampling = 500,
    cur_iter_warmup = 500,
    cur_iter_sampling = 500,
    hist_chains = 4,
    cur_chains = 4,
    hist_cores = 4,
    cur_cores = 4,
    prob_threshold = decision_threshold
) {
  set.seed(seed)
  
  stopifnot(all(dim(base_lam_hist_mat) == c(E, K)))
  stopifnot(all(dim(base_lam_current_ctrl_mat) == c(E, K)))
  stopifnot(all(dim(beta_mat) == c(E, K)))
  
  # Historical data
  hist_dat <- sim_hist_pwe_counts(
    n.vec.hist = n.vec.hist,
    E = E,
    K = K,
    cuts = cuts,
    base_lam_mat = base_lam_hist_mat,
    tau = tau_hist,
    r_endpt = r_endpt,
    analysis_time = analysis_time,
    entry_dist = entry_dist
  )
  
  stan_data_hist <- list(
    J = length(n.vec.hist),
    E = E,
    K = K,
    d = hist_dat$d_hist,
    Y = hist_dat$Y_hist
  )
  
  fit_hist <- rstan::sampling(
    object = mod_hist,
    data = stan_data_hist,
    seed = seed + 1L,
    chains = hist_chains,
    cores = hist_cores,
    iter = hist_iter_warmup + hist_iter_sampling,
    warmup = hist_iter_warmup,
    refresh = 0
  )
  
  prior_obj <- extract_map_prior(fit_hist, E, K)
  theta_prior_mean <- prior_obj$theta_prior_mean
  theta_prior_sd   <- pmax(prior_obj$theta_prior_sd, 1e-3)
  
  # Current trial data
  cur_dat <- sim_current_twoarm_counts(
    n_ctrl = n_ctrl,
    n_trt = n_trt,
    E = E,
    K = K,
    cuts = cuts,
    base_lam_ctrl_mat = base_lam_current_ctrl_mat,
    beta_mat = beta_mat,
    analysis_time = analysis_time,
    entry_dist = entry_dist
  )
  
  stan_data_current <- list(
    E = E,
    K = K,
    d_ctrl = cur_dat$d_ctrl,
    d_trt  = cur_dat$d_trt,
    Y_ctrl = unname(cur_dat$Y_ctrl),
    Y_trt  = unname(cur_dat$Y_trt),
    theta_prior_mean = theta_prior_mean,
    theta_prior_sd   = theta_prior_sd
  )
  
  fit_cur <- rstan::sampling(
    object = mod_current,
    data = stan_data_current,
    seed = seed + 2L,
    chains = cur_chains,
    cores = cur_cores,
    iter = cur_iter_warmup + cur_iter_sampling,
    warmup = cur_iter_warmup,
    refresh = 0
  )
  
  beta_draws <- rstan::extract(fit_cur, pars = "beta")$beta
  hr_draws   <- rstan::extract(fit_cur, pars = "hr")$hr
  # expected dims: iterations x E x K
  
  post_prob <- numeric(E)
  beta_mean <- numeric(E)
  hr_mean   <- numeric(E)
  
  for (e in seq_len(E)) {
    weights <- cur_dat$Y_ctrl[e, ] + cur_dat$Y_trt[e, ]
    
    beta_mat_e <- beta_draws[, e, ]  # iterations x K
    
    beta_avg_draw <- apply(
      beta_mat_e,
      1, # by row
      function(x) weighted.mean(x, weights)
    )
    
    hr_avg_draw <- exp(beta_avg_draw)
    
    post_prob[e] <- mean(beta_avg_draw < 0)
    beta_mean[e] <- mean(beta_avg_draw)
    hr_mean[e]   <- mean(hr_avg_draw)
  }
  
  success_e1  <- as.integer(post_prob[1] > prob_threshold)
  success_e2  <- as.integer(post_prob[2] > prob_threshold)
  success_any <- as.integer((success_e1 == 1) | (success_e2 == 1))
  
  out <- c(
    success_e1   = success_e1,
    success_e2   = success_e2,
    success_any  = success_any,
    postprob_e1  = post_prob[1],
    postprob_e2  = post_prob[2],
    beta_mean_e1 = beta_mean[1],
    beta_mean_e2 = beta_mean[2],
    hr_mean_e1   = hr_mean[1],
    hr_mean_e2   = hr_mean[2]
  )
  out
}

#### 5. Run many replicates ====

run_simulation <- function(
    nsim,
    seeds,
    n.vec.hist,
    n_ctrl,
    n_trt,
    E, K, cuts,
    base_lam_hist_mat,
    base_lam_current_ctrl_mat,
    tau_hist,
    r_endpt,
    beta_mat,
    mod_hist,
    mod_current,
    analysis_time = cuts[length(cuts)],
    entry_dist = function() 0,
    hist_iter_warmup = 500,
    hist_iter_sampling = 500,
    cur_iter_warmup = 500,
    cur_iter_sampling = 500,
    hist_chains = 4,
    cur_chains = 4,
    hist_cores = 4,
    cur_cores = 4,
    prob_threshold = decision_threshold
) {
  stopifnot(length(seeds) >= nsim)
  
  res_list <- vector("list", nsim)
  
  for (i in seq_len(nsim)) {
    cat("Running replicate", i, "of", nsim, "...\n")
    
    res_list[[i]] <- one_rep_map(
      seed = seeds[i],
      n.vec.hist = n.vec.hist,
      n_ctrl = n_ctrl,
      n_trt = n_trt,
      E = E,
      K = K,
      cuts = cuts,
      base_lam_hist_mat = base_lam_hist_mat,
      base_lam_current_ctrl_mat = base_lam_current_ctrl_mat,
      tau_hist = tau_hist,
      r_endpt = r_endpt,
      beta_mat = beta_mat,
      mod_hist = mod_hist,
      mod_current = mod_current,
      analysis_time = analysis_time,
      entry_dist = entry_dist,
      hist_iter_warmup = hist_iter_warmup,
      hist_iter_sampling = hist_iter_sampling,
      cur_iter_warmup = cur_iter_warmup,
      cur_iter_sampling = cur_iter_sampling,
      hist_chains = hist_chains,
      cur_chains = cur_chains,
      hist_cores = hist_cores,
      cur_cores = cur_cores,
      prob_threshold = prob_threshold
    )
  }
  
  res <- do.call(rbind, res_list)
  res <- as.data.frame(res)
  rownames(res) <- NULL
  res
}

#### 6. Summarize operating characteristics ====

summarize_operating_characteristics <- function(res) {
  out <- list()
  
  out$empirical_power_e1  <- mean(res$success_e1)
  out$empirical_power_e2  <- mean(res$success_e2)
  out$empirical_power_any <- mean(res$success_any)
  
  endpoint_summary <- data.frame(
    endpoint = c(1, 2),
    empirical_power = c(mean(res$success_e1), mean(res$success_e2)),
    mean_postprob   = c(mean(res$postprob_e1), mean(res$postprob_e2)),
    mean_beta_avg   = c(mean(res$beta_mean_e1), mean(res$beta_mean_e2)),
    mean_hr_avg     = c(mean(res$hr_mean_e1), mean(res$hr_mean_e2))
  )
  
  out$endpoint_summary <- endpoint_summary
  out
}

#### 7. Empirical censoring calculator ====

estimate_censoring_rate <- function(
    n = 10000,
    lam,
    cuts,
    analysis_time = cuts[length(cuts)],
    entry_dist = function() 0
) {
  n_censored <- 0L
  
  for (i in seq_len(n)) {
    entry_time <- entry_dist()
    followup_limit <- max(0, analysis_time - entry_time)
    
    t_event <- 0
    event_occured <- FALSE
    
    for (k in seq_along(lam)) {
      width_k <- cuts[k + 1] - cuts[k]
      wait_k <- rexp(1, rate = lam[k])
      
      if (wait_k < width_k) {
        t_event <- cuts[k] + wait_k
        event_occured <- TRUE
        break
      }
    }
    
    if (!event_occured) {
      t_event <- cuts[length(lam) + 1]
    }
    
    delta <- as.integer(event_occured && (t_event <= followup_limit))
    if (delta == 0) n_censored <- n_censored + 1L
  }
  
  n_censored / n
}


#### 8. Optional: estimate censoring rates ====

#cat("\nApproximate censoring rates under current control hazards:\n")
#for (e in seq_len(E)) {
#  c_rate <- estimate_censoring_rate(
#    n = 5000,
#    lam = base_lam_current_ctrl_mat[e, ],
#    cuts = cuts,
#    analysis_time = analysis_time,
#    entry_dist = entry_dist
#  )
#  cat("Endpoint", e, "control censoring rate:", round(c_rate, 3), "\n")
#}

#cat("\nApproximate censoring rates under treatment hazards:\n")
#for (e in seq_len(E)) {
#  c_rate <- estimate_censoring_rate(
#    n = 5000,
#    lam = base_lam_current_ctrl_mat[e, ] * exp(beta_mat[e, ]),
#    cuts = cuts,
#    analysis_time = analysis_time,
#    entry_dist = entry_dist
#  )
#  cat("Endpoint", e, "treatment censoring rate:", round(c_rate, 3), "\n")
#}

#### 9. Simulation run ====
seeds <- sample.int(1e7, nsim)

res <- run_simulation(
  nsim = nsim,
  seeds = seeds,
  n.vec.hist = n.vec.hist,
  n_ctrl = n_ctrl,
  n_trt = n_trt,
  E = E,
  K = K,
  cuts = cuts,
  base_lam_hist_mat = base_lam_hist_mat,
  base_lam_current_ctrl_mat = base_lam_current_ctrl_mat,
  tau_hist = tau_hist,
  r_endpt = r_endpt,
  beta_mat = beta_mat,
  mod_hist = mod_hist,
  mod_current = mod_current,
  analysis_time = analysis_time,
  entry_dist = entry_dist,
  hist_iter_warmup = 500,
  hist_iter_sampling = 500,
  cur_iter_warmup = 500,
  cur_iter_sampling = 500,
  hist_chains = 4,
  cur_chains = 4,
  hist_cores = 4,
  cur_cores = 4,
  prob_threshold = decision_threshold
)

#sum_res <- summarize_operating_characteristics(res)

#cat("\nScenario summary:\n")
#print(sum_res$endpoint_summary)

#at("\nH1: P(mean_k beta[1,k] < 0 | data) > 0.975 =", sum_res$empirical_power_e1, "\n")
#cat("H2: P(mean_k beta[2,k] < 0 | data) > 0.975 =", sum_res$empirical_power_e2, "\n")
#cat("H12: reject endpoint 1 or 2 =", sum_res$empirical_power_any, "\n")

#### 10. Save results ====

write.csv(res, paste0("results_", scenario_name, "_", decision_threshold,".csv"), row.names = FALSE)
