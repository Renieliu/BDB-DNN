// Historical-only part of multi_tte.stan, used to build the NPE training
// distribution.  Priors and the hierarchical structure are identical to
// multi_tte.stan; the current-control column and the treatment parameters are
// dropped because they do not enter the historical likelihood.  Their
// distribution given D_H is therefore (predictive) prior, and is simulated in R.
data {
  int<lower=1> Jhist;              // # historical studies
  int<lower=1> E;                  // # endpoints
  int<lower=1> K;                  // # time intervals

  array[Jhist, E, K] int<lower=0> d_hist;
  array[Jhist, E, K] real<lower=0> Y_hist;
}

parameters {
  vector<lower=0>[E] sigma_endpt;
  cholesky_factor_corr[E] L_R;

  real<lower=0> sigma_time;
  vector<lower=0>[K] tau;

  // non-centered random walk for mu: same distribution as the centered
  // random walk in multi_tte.stan, but avoids the funnel between mu and
  // sigma_time that produced divergent transitions
  matrix[E, K] mu_raw;

  // non-centered study effects, historical studies only
  array[K] matrix[E, Jhist] z_study;
}

transformed parameters {
  matrix[E,E] L_Sigma = diag_pre_multiply(sigma_endpt, L_R);
  matrix[E, K] mu;
  array[K] matrix[E, Jhist] theta_hist;

  mu[,1] = log(0.1) + 2 * mu_raw[,1];
  for (k in 2:K) {
    mu[,k] = mu[,k-1] + sigma_time * (L_Sigma * mu_raw[,k]);
  }

  for (k in 1:K) {
    theta_hist[k] = rep_matrix(mu[,k], Jhist) + (tau[k] * (L_Sigma * z_study[k]));
  }
}

model {
  // priors (same as multi_tte.stan)
  sigma_endpt ~ normal(0, 1);
  L_R ~ lkj_corr_cholesky(2);

  sigma_time ~ normal(0, 0.3);
  tau ~ normal(0, 1);

  // implies mu[,1] ~ normal(log(0.1), 2) and
  // mu[,k] ~ multi_normal_cholesky(mu[,k-1], sigma_time * L_Sigma), as in multi_tte.stan
  to_vector(mu_raw) ~ std_normal();

  for (k in 1:K) to_vector(z_study[k]) ~ std_normal();

  // likelihood: historical controls
  for (j in 1:Jhist) {
    for (e in 1:E) {
      for (k in 1:K) {
        d_hist[j,e,k] ~ poisson(Y_hist[j,e,k] * exp(theta_hist[k][e,j]));
      }
    }
  }
}

