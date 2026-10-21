## source functions file
if (!file.exists("BDB_DNN_functions.r") && file.exists("../BDB_DNN_functions.r")) setwd("..")
source("BDB_DNN_functions.r")

#### 0. Simulation helpers ====

make_entry_dist <- function(accrual_end) {
  force(accrual_end)
  function() stats::runif(1, min = 0, max = accrual_end)
}

sim_pwe_arm_counts <- function(
    n,
    lam,
    cuts,
    analysis_time = cuts[length(cuts)],
    entry_dist = function() 0
) {
  K <- length(lam)
  d <- numeric(K)
  Y <- numeric(K)
  
  stopifnot(length(cuts) == K + 1)
  stopifnot(analysis_time >= cuts[length(cuts)])
  
  for (i in seq_len(n)) {
    entry_time <- entry_dist()
    followup_limit <- max(0, analysis_time - entry_time)
    
    t_event <- 0
    event_occured <- FALSE
    
    for (k in seq_len(K)) {
      width_k <- cuts[k + 1] - cuts[k]
      wait_k <- rexp(1, rate = lam[k])
      
      if (wait_k < width_k) {
        t_event <- cuts[k] + wait_k
        event_occured <- TRUE
        break
      }
    }
    
    if (!event_occured) {
      t_event <- cuts[K + 1]
    }
    
    t_obs <- min(t_event, followup_limit)
    delta <- as.integer(event_occured && (t_event <= followup_limit))
    
    for (k in seq_len(K)) {
      left <- cuts[k]
      right <- cuts[k + 1]
      
      time_in_k <- max(0, min(t_obs, right) - left)
      Y[k] <- Y[k] + time_in_k
      
      if (delta == 1 && t_event >= left && t_event < right) {
        d[k] <- d[k] + 1
      }
    }
  }
  
  list(d = d, Y = Y)
}

sim_current_twoarm_counts <- function(
    n_ctrl, n_trt,
    E, K, cuts,
    base_lam_ctrl_mat,
    beta_mat,
    analysis_time = cuts[length(cuts)],
    entry_dist = function() 0
) {
  d_ctrl <- matrix(0L, nrow = E, ncol = K)
  Y_ctrl <- matrix(0,  nrow = E, ncol = K)
  d_trt  <- matrix(0L, nrow = E, ncol = K)
  Y_trt  <- matrix(0,  nrow = E, ncol = K)
  
  stopifnot(all(dim(base_lam_ctrl_mat) == c(E, K)))
  stopifnot(all(dim(beta_mat) == c(E, K)))
  
  for (e in seq_len(E)) {
    sim_c <- sim_pwe_arm_counts(
      n = n_ctrl,
      lam = base_lam_ctrl_mat[e, ],
      cuts = cuts,
      analysis_time = analysis_time,
      entry_dist = entry_dist
    )
    
    d_ctrl[e, ] <- as.integer(sim_c$d)
    Y_ctrl[e, ] <- sim_c$Y
    
    lam_trt <- base_lam_ctrl_mat[e, ] * exp(beta_mat[e, ])
    
    sim_t <- sim_pwe_arm_counts(
      n = n_trt,
      lam = lam_trt,
      cuts = cuts,
      analysis_time = analysis_time,
      entry_dist = entry_dist
    )
    
    d_trt[e, ] <- as.integer(sim_t$d)
    Y_trt[e, ] <- sim_t$Y
  }
  
  list(
    d_ctrl = d_ctrl,
    Y_ctrl = Y_ctrl,
    d_trt  = d_trt,
    Y_trt  = Y_trt
  )
}

#### 1. Simulation Setup ====

# choose scenario
scenario_name <- "05_single_null_s1_early_compatible"

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
accrual_end <- 2
entry_dist <- make_entry_dist(accrual_end)

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

scenario_params <- get_scenario_params(scenario_name)
base_lam_current_ctrl_mat <- scenario_params$base_lam_current_ctrl_mat
beta_mat <- scenario_params$beta_mat

cat("Running scenario:", scenario_name, "\n")
cat("HR matrix:\n")
print(exp(beta_mat))

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

estimate_true_treatment_effect <- function(
    scenario_name,
    n_true_sim = 10000,
    seed = 20260609
) {
  scenario_params <- get_scenario_params(scenario_name)
  beta_mat <- scenario_params$beta_mat
  base_lam_current_ctrl_mat <- scenario_params$base_lam_current_ctrl_mat
  
  beta_bar_mat <- matrix(NA_real_, nrow = n_true_sim, ncol = E)
  colnames(beta_bar_mat) <- paste0("true_beta_bar_end", seq_len(E))
  
  set.seed(seed)
  
  for (sim in seq_len(n_true_sim)) {
    sim_dat <- sim_current_twoarm_counts(
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
    
    beta_bar_mat[sim, ] <- calc_person_time_weighted_beta(
      beta_mat = beta_mat,
      Y_ctrl = sim_dat$Y_ctrl,
      Y_trt = sim_dat$Y_trt
    )
    
    if (sim %% 1000 == 0) {
      cat("Scenario:", scenario_name, "simulation:", sim, "\n")
    }
  }
  
  beta_mean <- colMeans(beta_bar_mat, na.rm = TRUE)
  beta_sd <- apply(beta_bar_mat, 2, stats::sd, na.rm = TRUE)
  
  data.frame(
    scenario_name = scenario_name,
    n_true_sim = n_true_sim,
    endpoint = paste0("H", seq_len(E)),
    true_beta_bar = as.numeric(beta_mean),
    true_HR_bar = exp(as.numeric(beta_mean)),
    mc_sd = as.numeric(beta_sd),
    mc_se = as.numeric(beta_sd / sqrt(n_true_sim)),
    unweighted_beta = rowMeans(beta_mat),
    unweighted_HR = exp(rowMeans(beta_mat)),
    row.names = NULL
  )
}

scenarios_for_true_effect <- c(
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

n_true_sim <- 10000

null_scenario_results <- do.call(
  rbind,
  lapply(c(
    "01_global_null_compatible",
    "02_global_null_mild_conflict",
    "03_global_null_moderate_conflict"
  ), function(scenario_name) {
    data.frame(
      scenario_name = scenario_name,
      n_true_sim = 0,
      endpoint = paste0("H", seq_len(E)),
      true_beta_bar = rep(0, E),
      true_HR_bar = rep(1, E),
      mc_sd = rep(0, E),
      mc_se = rep(0, E),
      unweighted_beta = rep(0, E),
      unweighted_HR = rep(1, E),
      row.names = NULL
    )
  })
)

simulated_scenario_results <- do.call(
  rbind,
  lapply(seq_along(scenarios_for_true_effect), function(i) {
    estimate_true_treatment_effect(
      scenario_name = scenarios_for_true_effect[i],
      n_true_sim = n_true_sim,
      seed = 20260609 + i
    )
  })
)

true_effect_results <- rbind(
  null_scenario_results,
  simulated_scenario_results
)

print(true_effect_results)

write.csv(
  true_effect_results,
  file = "PWE_true_treatment_effect_scenarios_01_15.csv",
  row.names = FALSE
)
