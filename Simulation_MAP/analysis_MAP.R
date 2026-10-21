alpha <- 0.05

## Run from Simulation_MAP/ after the 15 MAP_XX_*.R scripts have written their results_*.csv files
base_dir <- "."

result_files <- sprintf(
  "results_%02d_%s",
  1:15,
  c(
    "global_null_compatible_0.975.csv",
    "global_null_mild_conflict_0.975.csv",
    "global_null_moderate_conflict_0.975.csv",
    "single_null_s1_delayed_compatible_0.975.csv",
    "single_null_s1_early_compatible_0.975.csv",
    "single_null_s2_delayed_compatible_0.975.csv",
    "single_null_s2_early_compatible_0.975.csv",
    "single_null_s1_delayed_conflict_0.975.csv",
    "single_null_s1_early_conflict_0.975.csv",
    "single_null_s2_delayed_conflict_0.975.csv",
    "single_null_s2_early_conflict_0.975.csv",
    "global_alt_delayed_compatible_0.975.csv",
    "global_alt_early_compatible_0.975.csv",
    "global_alt_delayed_conflict_0.975.csv",
    "global_alt_early_conflict_0.975.csv"
  )
)

result_paths <- file.path(base_dir, result_files)

scenario_names <- sub("^results_", "", result_files)
scenario_names <- sub("_0\\.975\\.csv$", "", scenario_names)

find_common_crit_grid <- function(p1, p2, alpha = 0.05, grid = seq(0, 1, by = 0.0001)) {
  type1_H1 <- sapply(grid, function(cutoff) {
    mean(p1 >= cutoff, na.rm = TRUE)
  })

  type1_H2 <- sapply(grid, function(cutoff) {
    mean(p2 >= cutoff, na.rm = TRUE)
  })

  type1_H12 <- sapply(grid, function(cutoff) {
    mean(pmax(p1, p2) >= cutoff, na.rm = TRUE)
  })

  max_type1 <- pmax(type1_H1, type1_H2, type1_H12)
  valid <- which(max_type1 < alpha)

  if (length(valid) == 0) {
    return(data.frame(
      common_threshold = NA,
      type1_H1 = NA,
      type1_H2 = NA,
      type1_H12 = NA,
      max_type1 = NA
    ))
  }

  best_id <- valid[1]

  data.frame(
    common_threshold = grid[best_id],
    type1_H1 = type1_H1[best_id],
    type1_H2 = type1_H2[best_id],
    type1_H12 = type1_H12[best_id],
    max_type1 = max_type1[best_id]
  )
}

null_result <- read.csv(file.path(base_dir, "results_01_global_null_compatible_0.975.csv"))

critical.common <- find_common_crit_grid(
  p1 = null_result$postprob_e1,
  p2 = null_result$postprob_e2,
  alpha = alpha
)

critical_value <- critical.common$common_threshold
#critical_value <- 0.95

true_effects <- read.csv(file.path(base_dir, "PWE_true_treatment_effect_scenarios_01_15.csv"))

get_true_value <- function(scenario_name, endpoint, column) {
  value <- true_effects[
    true_effects$scenario_name == scenario_name &
      true_effects$endpoint == endpoint,
    column
  ]

  if (length(value) != 1) {
    stop("Cannot find one true value for ", scenario_name, " / ", endpoint)
  }

  value
}

summaries <- lapply(seq_along(result_files), function(i) {
  result <- read.csv(result_paths[i])
  scenario_name <- scenario_names[i]

  true_beta_e1 <- get_true_value(scenario_name, "H1", "true_beta_bar")
  true_beta_e2 <- get_true_value(scenario_name, "H2", "true_beta_bar")

  reject_e1 <- result$postprob_e1 >= critical_value
  reject_e2 <- result$postprob_e2 >= critical_value
  reject_any <- reject_e1 | reject_e2

  data.frame(
    scenario = scenario_name,
    critical_value = critical_value,
    true_beta_e1 = true_beta_e1,
    true_beta_e2 = true_beta_e2,
    metric_e1 = ifelse(true_beta_e1 == 0, "type I error", "power"),
    metric_e2 = ifelse(true_beta_e2 == 0, "type I error", "power"),
    metric_any = ifelse(true_beta_e1 == 0 & true_beta_e2 == 0, "type I error", "power"),
    rejection_rate_e1 = mean(reject_e1, na.rm = TRUE),
    rejection_rate_e2 = mean(reject_e2, na.rm = TRUE),
    rejection_rate_any = mean(reject_any, na.rm = TRUE),
    beta_mean_e1 = mean(result$beta_mean_e1, na.rm = TRUE),
    beta_mean_e2 = mean(result$beta_mean_e2, na.rm = TRUE),
    bias_e1 = mean(result$beta_mean_e1 - true_beta_e1, na.rm = TRUE),
    bias_e2 = mean(result$beta_mean_e2 - true_beta_e2, na.rm = TRUE),
    mse_e1 = mean((result$beta_mean_e1 - true_beta_e1)^2, na.rm = TRUE),
    mse_e2 = mean((result$beta_mean_e2 - true_beta_e2)^2, na.rm = TRUE)
  )
})

summary_df <- do.call(rbind, summaries)

write.csv(summary_df, file.path(base_dir, "analysis_summary_MAP_II.csv"), row.names = FALSE)

print(critical.common)
print(summary_df)



