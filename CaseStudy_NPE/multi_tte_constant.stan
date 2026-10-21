// Constant-hazard counterpart of multi_tte.stan. Keep the K=1 data interface
// and beta_bar output for the existing two-stage DNN training pipeline.
data {
  int<lower=1> Jhist;
  int<lower=1> E;
  int<lower=1,upper=1> K;
  array[Jhist,E,K] int<lower=0> d_hist;
  array[Jhist,E,K] real<lower=0> Y_hist;
  array[E,K] int<lower=0> d_c;
  array[E,K] real<lower=0> Y_c;
  array[E,K] int<lower=0> d_t;
  array[E,K] real<lower=0> Y_t;
}
parameters {
  vector[E] mu;
  vector<lower=0>[E] tau;
  cholesky_factor_corr[E] L_R;
  matrix[E,Jhist+1] z_study;
  vector[E] beta;
}
transformed parameters {
  matrix[E,Jhist+1] theta_ctrl = rep_matrix(mu,Jhist+1)
    + diag_pre_multiply(tau,L_R)*z_study;
}
model {
  mu ~ normal(0,1.5);
  tau ~ normal(0,0.5);
  L_R ~ lkj_corr_cholesky(2);
  to_vector(z_study) ~ std_normal();
  beta ~ normal(0,1);
  for(e in 1:E) {
    for(j in 1:Jhist)
      target += d_hist[j,e,1]*theta_ctrl[e,j] - Y_hist[j,e,1]*exp(theta_ctrl[e,j]);
    target += d_c[e,1]*theta_ctrl[e,Jhist+1] - Y_c[e,1]*exp(theta_ctrl[e,Jhist+1]);
    target += d_t[e,1]*(theta_ctrl[e,Jhist+1]+beta[e])
      - Y_t[e,1]*exp(theta_ctrl[e,Jhist+1]+beta[e]);
  }
}
generated quantities {
  vector[E] beta_bar = beta;
}
