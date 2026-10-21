data {
  int<lower=1> M;
  array[2] int<lower=1,upper=M> nmix;
  matrix<lower=0>[2,M] w;
  matrix<lower=0>[2,M] a;
  matrix<lower=0>[2,M] b;
  array[2] int<lower=0> dc;
  array[2] int<lower=0> dt;
  vector<lower=0>[2] yc;
  vector<lower=0>[2] yt;
  real<lower=0> beta_sd;
}
parameters {
  vector<lower=0>[2] lambda;
  vector[2] beta;
}
model {
  beta ~ normal(0,beta_sd);
  for(e in 1:2) {
    vector[nmix[e]] lp;
    for(m in 1:nmix[e]) lp[m] = log(w[e,m]) + gamma_lpdf(lambda[e]|a[e,m],b[e,m]);
    target += log_sum_exp(lp);
    target += dc[e]*log(lambda[e]) - yc[e]*lambda[e];
    target += dt[e]*(log(lambda[e])+beta[e]) - yt[e]*lambda[e]*exp(beta[e]);
  }
}
