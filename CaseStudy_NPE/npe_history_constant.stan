// Posterior over shared hyperparameters conditional on the two fixed histories.
data {
  int<lower=1> Jhist;
  int<lower=1> E;
  array[Jhist,E] int<lower=0> d;
  matrix<lower=0>[Jhist,E] Y;
}
parameters {
  vector[E] mu;
  vector<lower=0>[E] tau;
  cholesky_factor_corr[E] L_R;
  matrix[E,Jhist] z;
}
transformed parameters {
  matrix[E,Jhist] theta = rep_matrix(mu,Jhist) + diag_pre_multiply(tau,L_R)*z;
}
model {
  mu ~ normal(0,1.5);
  tau ~ normal(0,0.5);
  L_R ~ lkj_corr_cholesky(2);
  to_vector(z) ~ std_normal();
  for(j in 1:Jhist) for(e in 1:E)
    target += d[j,e]*theta[e,j] - Y[j,e]*exp(theta[e,j]);
}
