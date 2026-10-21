data {
  int<lower=1> J;
  int<lower=1> E;
  int<lower=1> K;

  array[J,E,K] int d;
  array[J,E,K] real Y;
}

parameters {
  matrix[E,K] mu;
  vector<lower=0>[E] tau;
  cholesky_factor_corr[E] L_Omega;

  matrix[J,E] z_b;
}

transformed parameters {
  matrix[E,E] L_Sigma;
  matrix[J,E] b;

  L_Sigma = diag_pre_multiply(tau, L_Omega);
  b = z_b * L_Sigma';
}

model {

  to_vector(mu) ~ normal(-2.5, 1.5);

  tau ~ normal(0,0.5);

  L_Omega ~ lkj_corr_cholesky(2);

  to_vector(z_b) ~ normal(0,1);

  for(j in 1:J)
    for(e in 1:E)
      for(k in 1:K)
        d[j,e,k] ~ poisson_log(
          log(Y[j,e,k] + 1e-12)
          + mu[e,k]
          + b[j,e]
        );
}

generated quantities {

  corr_matrix[E] Omega;
  cov_matrix[E] Sigma_b;

  vector[E] b_new;

  matrix[E,K] theta_new;
  matrix[E,K] lambda_new;

  Omega = multiply_lower_tri_self_transpose(L_Omega);

  Sigma_b = quad_form_diag(Omega, tau);

  b_new = multi_normal_rng(rep_vector(0,E), Sigma_b);

  for(e in 1:E)
    for(k in 1:K){

      theta_new[e,k] = mu[e,k] + b_new[e];

      lambda_new[e,k] = exp(theta_new[e,k]);

    }

}
