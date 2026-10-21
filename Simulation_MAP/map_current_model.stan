data {
  int<lower=1> E;
  int<lower=1> K;

  array[E, K] int<lower=0> d_ctrl;
  array[E, K] int<lower=0> d_trt;

  matrix[E, K] Y_ctrl;
  matrix[E, K] Y_trt;

  matrix[E, K] theta_prior_mean;
  matrix[E, K] theta_prior_sd;
}

parameters {
  matrix[E, K] theta_ctrl;
  matrix[E, K] beta;

  vector<lower=0>[E] sigma_beta;
}

model {
  // MAP prior for current control hazards
  for (e in 1:E) {
    for (k in 1:K) {
      theta_ctrl[e, k] ~ normal(theta_prior_mean[e, k],
                                theta_prior_sd[e, k]);
    }
  }

  // Prior for smoothness of interval-specific treatment effects
  sigma_beta ~ normal(0, 0.5);

  // Flexible treatment effect model:
  // no fixed early/delayed pattern assumed,
  // only mild temporal smoothing across adjacent intervals
  for (e in 1:E) {
    beta[e, 1] ~ normal(0, 0.5);
    for (k in 2:K) {
      beta[e, k] ~ normal(beta[e, k - 1], sigma_beta[e]);
    }
  }

  // Likelihood
  for (e in 1:E) {
    for (k in 1:K) {
      d_ctrl[e, k] ~ poisson_log(
        log(Y_ctrl[e, k] + 1e-12) + theta_ctrl[e, k]
      );

      d_trt[e, k] ~ poisson_log(
        log(Y_trt[e, k] + 1e-12) + theta_ctrl[e, k] + beta[e, k]
      );
    }
  }
}

generated quantities {
  matrix[E, K] hr;
  vector[E] beta_avg;
  vector[E] hr_avg;

  for (e in 1:E) {
    beta_avg[e] = 0;
    hr_avg[e] = 0;
    for (k in 1:K) {
      hr[e, k] = exp(beta[e, k]);
      beta_avg[e] += beta[e, k];
      hr_avg[e] += hr[e, k];
    }
    beta_avg[e] /= K;
    hr_avg[e] /= K;
  }
}
