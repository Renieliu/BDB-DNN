// Hierarchical PWE borrowing model (Gaussian random-walk priors on mu
// and beta), written with non-centered random walks.  The joint distribution
// is unchanged; the reparameterization removes the funnel between the walks
// and their scales (sigma_time, sigma_beta) that caused divergent transitions
// in the centered historical fit.
data {
  int<lower=1> Jhist;              // # historical studies
  int<lower=1> E;                  // # endpoints
  int<lower=1> K;                  // # time intervals

  array[Jhist, E, K] int<lower=0> d_hist;
  array[Jhist, E, K] real<lower=0> Y_hist;

  array[E, K] int<lower=0> d_c;    // current control counts
  array[E, K] real<lower=0> Y_c;

  array[E, K] int<lower=0> d_t;    // current treatment counts
  array[E, K] real<lower=0> Y_t;
}

parameters {
  // correlation among endpoints for study effects + time innovations
  vector<lower=0>[E] sigma_endpt;
  cholesky_factor_corr[E] L_R;

  // random walk smoothness + between-study heterogeneity per interval
  real<lower=0> sigma_time;
  vector<lower=0>[K] tau;

  // standardized innovations of the random walk on mu
  matrix[E, K] mu_raw;

  // non-centered study effects (historical + current control) per interval
  array[K] matrix[E, Jhist + 1] z_study;

  // standardized innovations of the random walk on beta
  matrix[E, K] beta_raw;

  // endpoint-specific innovation scales and cross-endpoint correlation
  // for the treatment-effect random walk
  vector<lower=0>[E] sigma_beta;
  cholesky_factor_corr[E] L_R_beta;
}

transformed parameters {
  matrix[E,E] L_Sigma = diag_pre_multiply(sigma_endpt, L_R);
  matrix[E,E] L_Sigma_beta = diag_pre_multiply(sigma_beta, L_R_beta);

  // across-study mean log-hazards per endpoint & interval
  matrix[E, K] mu;
  // interval-specific treatment effects with temporal smoothing
  matrix[E, K] beta;

  // theta for studies: historical 1..Jhist, current control is Jhist+1
  array[K] matrix[E, Jhist + 1] theta_ctrl;

  // mu[,1] ~ normal(log(0.1), 2);
  // mu[,k] ~ multi_normal_cholesky(mu[,k-1], sigma_time * L_Sigma)
  mu[,1] = log(0.1) + 2 * mu_raw[,1];
  for (k in 2:K) {
    mu[,k] = mu[,k-1] + sigma_time * (L_Sigma * mu_raw[,k]);
  }

  // beta[,1] ~ multi_normal_cholesky(0, L_Sigma_beta);
  // beta[,k] ~ multi_normal_cholesky(beta[,k-1], L_Sigma_beta)
  beta[,1] = L_Sigma_beta * beta_raw[,1];
  for (k in 2:K) {
    beta[,k] = beta[,k-1] + L_Sigma_beta * beta_raw[,k];
  }

  for (k in 1:K) {
    // theta_ctrl[,j,k] = mu[,k] + tau[k] * L_Sigma * z
    theta_ctrl[k] = rep_matrix(mu[,k], Jhist + 1) + (tau[k] * (L_Sigma * z_study[k]));
  }
}

model {
  // priors
  sigma_endpt ~ normal(0, 1);
  L_R ~ lkj_corr_cholesky(2);

  sigma_time ~ normal(0, 0.3);
  tau ~ normal(0, 1); // half-normal via <lower=0>

  to_vector(mu_raw) ~ std_normal();

  // non-centered study effects
  for (k in 1:K) to_vector(z_study[k]) ~ std_normal();

  // multivariate first-order Gaussian random walk for time-varying
  // treatment effects; L_Sigma_beta is the innovation Cholesky factor
  sigma_beta ~ normal(0, 0.5);
  L_R_beta ~ lkj_corr_cholesky(2);
  to_vector(beta_raw) ~ std_normal();

  // likelihood: historical controls
  for (j in 1:Jhist) {
    for (e in 1:E) {
      for (k in 1:K) {
        d_hist[j,e,k] ~ poisson(Y_hist[j,e,k] * exp(theta_ctrl[k][e,j]));
      }
    }
  }

  // likelihood: current control (index Jhist+1)
  for (e in 1:E) {
    for (k in 1:K) {
      d_c[e,k] ~ poisson(Y_c[e,k] * exp(theta_ctrl[k][e, Jhist+1]));
    }
  }

  // likelihood: current treatment, theta_t = theta_c + interval-specific beta
  for (e in 1:E) {
    for (k in 1:K) {
      d_t[e,k] ~ poisson(Y_t[e,k] * exp(theta_ctrl[k][e, Jhist+1] + beta[e,k]));
    }
  }
}

generated quantities {
  vector[E] beta_bar;
  vector[E] Pr_beta_bar_lt0;

  for (e in 1:E) {
    real total_Y = 0;
    beta_bar[e] = 0;

    for (k in 1:K) {
      total_Y += Y_c[e,k] + Y_t[e,k];
    }

    for (k in 1:K) {
      beta_bar[e] += ((Y_c[e,k] + Y_t[e,k]) / total_Y) * beta[e,k];
    }

    Pr_beta_bar_lt0[e] = beta_bar[e] < 0 ? 1 : 0;
  }
}

