# Shared simulation and analysis implementation for five independent HR scripts.
args <- if(interactive()) character() else commandArgs(trailingOnly=TRUE)
if(isTRUE(getOption('kd.map.check',FALSE))) args <- c(args,'--check')
if(isTRUE(getOption('kd.map.smoke',FALSE))) args <- c(args,'--smoke')
if(any(!grepl('^--(smoke|check)$|^--(output|cores|cutoff)=.+$',args))) stop('Use --smoke, --check, --cores=2, --cutoff=0.975, or --output=/path')
z <- args[startsWith(args,'--output=')]
if(length(z)>1) stop('Only one --output argument is allowed')
if(!length(z) && !is.null(getOption('kd.map.output'))) {
  output_option <- getOption('kd.map.output')
  stopifnot(is.character(output_option),length(output_option)==1L,
            !is.na(output_option),nzchar(output_option))
  z <- paste0('--output=',path.expand(output_option))
}
local_smoke <- '--smoke' %in% args || identical(Sys.getenv('KD_SMOKE'),'1')
# Days throughout. Fixed literature-calibrated summaries, NOT harmonized IPD.
kcfg <- list(
  seed = 20260927L, n_ctrl = 150L, n_trt = 150L, followup = 28,
  hr = c(1, 1.25, 1.5, 1.75, 2), alpha = 0.05,
  hist_n = c(49L, 40L),
  hist_mean = rbind(KIDCARE = c(fever = 2.5, discharge = 4.5),
                    Wang2020 = c(fever = 0.750, discharge = 6.7)),
  tau_sd = 0.5, mu_mean = 0, mu_sd = 1.5, beta_sd = 1,
  # Endpoints are simulated independently within patients.
  # This is separate from the DNN model's correlated study random effects.
  patient_rho = 0,
  nsim = 1000L,
  chains = 4L, iter = 2000L, warmup = 1000L, cores = 1L,
  # Use a separate run_name for each design and rebuild/recalibrate.
  run_name = 'constant_n150_common_cutoff'
)
kcfg$controls <- rbind(A_KIDCARE=kcfg$hist_mean[1,])
# A very small integration run; never use its results for inference.
if (local_smoke) {
  kcfg$nsim <- 4L
  kcfg$chains <- 2L
  kcfg$iter <- 400L; kcfg$warmup <- 200L; kcfg$run_name <- 'smoke_common_cutoff'
}

# Default to sequential chains to avoid the observed RStudio PSOCK socket stall.
core_arg <- args[startsWith(args,'--cores=')]
if(length(core_arg)>1) stop('Only one --cores argument is allowed')
cpus <- if(length(core_arg)) suppressWarnings(as.integer(sub('--cores=','',core_arg,fixed=TRUE))) else getOption('kd.map.cores',1L)
if(length(cpus)!=1 || !is.finite(cpus) || cpus<1 || cpus!=as.integer(cpus)) stop('cores must be a positive integer')
kcfg$cores <- min(kcfg$chains,as.integer(cpus))

run_root <- if(length(z)) substring(z,10L) else file.path(script_dir,'results',kcfg$run_name)
stan_path <- file.path(script_dir,'map_current_constant.stan')
progress_every <- getOption('kd.map.progress.every',1L)
checkpoint_every <- getOption('kd.map.checkpoint.every',10L)
stopifnot(length(progress_every)==1L, is.finite(progress_every), progress_every>=1,
          length(checkpoint_every)==1L, is.finite(checkpoint_every), checkpoint_every>=1)
format_duration <- function(seconds) {
  if(!is.finite(seconds)) return('estimating')
  seconds <- round(max(0,seconds))
  sprintf('%02d:%02d:%02d',seconds%/%3600,(seconds%%3600)%/%60,seconds%%60)
}
progress_log <- function(...) {
  line <- paste0('[',format(Sys.time(),'%Y-%m-%d %H:%M:%S'),'] ',paste0(...,collapse=''))
  message(line)
  flush.console()
  dir.create(run_root,recursive=TRUE,showWarnings=FALSE)
  cat(line,'\n',file=file.path(kout,'progress.log'),append=TRUE)
  writeLines(line,file.path(kout,'status.txt'))
  invisible(line)
}
need <- function(pkgs) for(p in pkgs) {
  tryCatch(loadNamespace(p),error=function(e) stop(
    'Cannot load R package ',p,': ',conditionMessage(e),
    '\nR library paths: ',paste(.libPaths(),collapse='; '),
    '\nIn local R/RStudio, source setup_packages.R once, then restart R and retry.',
    call.=FALSE))
}
identity_config <- function() kcfg[setdiff(names(kcfg),'cores')]
save_artifact <- function(x,name) {
  tmp <- tempfile('checkpoint_',tmpdir=kout)
  saveRDS(list(config=identity_config(),value=x),tmp)
  if(!file.rename(tmp,file.path(kout,name))) stop('Could not save checkpoint: ',name)
}
load_artifact <- function(name, directory=kout) {
  p <- file.path(directory,name)
  if(!file.exists(p)) stop('Missing prerequisite: ',p)
  x <- readRDS(p)
  if(!identical(x$config,identity_config())) stop('Configuration changed; use a new run_name and rebuild: ',p)
  x$value
}
# Save posterior summaries and beta draws for each simulated trial. Inputs and seeds are
# checked so a partial run cannot silently resume a different simulation.
fit_many <- function(xs,sm,mixes,seed_base,checkpoint,label) {
  n <- length(xs)
  results <- vector('list',n)
  completed <- 0L
  if(file.exists(file.path(kout,checkpoint))) {
    cached <- load_artifact(checkpoint)
    stopifnot(identical(cached$inputs,xs),identical(cached$seed_base,seed_base),
              length(cached$results)==n)
    results <- cached$results
    done <- which(!vapply(results,is.null,logical(1)))
    completed <- length(done)
    stopifnot(identical(done,seq_len(completed)))
  }
  initial <- completed
  saved <- completed
  started <- Sys.time()
  persist <- function() save_artifact(list(inputs=xs,seed_base=seed_base,results=results),checkpoint)
  on.exit({ if(completed>saved) persist() },add=TRUE)
  progress_log(label,': ',completed,'/',n,' completed; ',n-completed,' remaining.')
  if(completed<n) for(i in seq.int(completed+1L,n)) {
    results[[i]] <- fit_stan(sm,map_data(mixes,xs[[i]]),seed_base+i)
    completed <- i
    if(i%%checkpoint_every==0L || i==n) {
      persist()
      saved <- completed
    }
    if(i==initial+1L || i%%progress_every==0L || i==n) {
      elapsed <- as.numeric(difftime(Sys.time(),started,units='secs'))
      eta <- elapsed/(i-initial)*(n-i)
      progress_log(label,': ',i,'/',n,' (',sprintf('%.1f',100*i/n),'%)',
                   ' | elapsed ',format_duration(elapsed),' | stage ETA ',format_duration(eta))
    }
  }
  results
}
history_data <- function() list(J=2L,
  dh=matrix(rep(kcfg$hist_n,2),2,2), yh=kcfg$hist_mean*kcfg$hist_n)
# Gaussian copula parameter is latent-normal correlation, not Pearson time correlation.
sim_arm <- function(n, rate) {
  stopifnot(length(rate)==2, all(rate>0), abs(kcfg$patient_rho)<1)
  z <- matrix(rnorm(n*2),n,2) %*% chol(matrix(c(1,kcfg$patient_rho,kcfg$patient_rho,1),2))
  u <- pmin(pmax(pnorm(z),1e-15),1-1e-15)
  tt <- sweep(-log(u),2,rate,'/')
  list(d=as.integer(colSums(tt<=kcfg$followup)),y=colSums(pmin(tt,kcfg$followup)))
}
sim_trial <- function(rate, beta) {
  cc <- sim_arm(kcfg$n_ctrl,rate); tt <- sim_arm(kcfg$n_trt,rate*exp(beta))
  list(dc=cc$d,yc=cc$y,dt=tt$d,yt=tt$y)
}
features <- function(x) c(x$dc,x$yc,x$dt,x$yt)
unfeatures <- function(x) list(dc=as.integer(x[1:2]),yc=x[3:4],dt=as.integer(x[5:6]),yt=x[7:8])
scenarios <- function() {
  # Five testing scenarios; both endpoints share a constant recovery HR.
  data.frame(control='A_KIDCARE', path='both', HR=kcfg$hr,
             HR1=kcfg$hr, HR2=kcfg$hr)
}
local_model_cache <- new.env(parent=emptyenv())
stan_model <- function() {
  need('rstan')
  # Concurrent HR jobs must not write a shared compiled cache beside the source.
  rstan::rstan_options(auto_write=FALSE)
  if(!exists('local_stan_model',envir=local_model_cache,inherits=FALSE)) {
    progress_log('Compiling current-trial Stan model; this step has no percentage progress.')
    local_model_cache$local_stan_model <- rstan::stan_model(model_code=paste(readLines(stan_path),collapse='\n'),auto_write=FALSE)
    progress_log('Stan compilation complete.')
  }
  local_model_cache$local_stan_model
}
fit_stan <- function(sm,data,seed) {
  fit <- rstan::sampling(sm,data=data,seed=seed,chains=kcfg$chains,
    iter=kcfg$iter,warmup=kcfg$warmup,cores=kcfg$cores,refresh=0,
    control=list(adapt_delta=0.99,max_treedepth=12))
  ss <- rstan::summary(fit)$summary
  rh <- ss[,'Rhat']; ne <- ss[,'n_eff']
  sp <- rstan::get_sampler_params(fit,inc_warmup=FALSE)
  dg <- c(max_rhat=max(rh,na.rm=TRUE),min_neff=min(ne,na.rm=TRUE),
    divergences=sum(vapply(sp,function(x) sum(x[,'divergent__']),numeric(1))),
    treedepth_hits=sum(vapply(sp,function(x) sum(x[,'treedepth__']>=12),numeric(1))))
  ok <- is.finite(dg[1]) && dg[1]<=1.05 && dg[2]>=100 && dg[3]==0 && dg[4]==0
  if(!ok && !local_smoke) stop('MCMC diagnostic failure: ',paste(names(dg),dg,collapse=', '))
  b <- rstan::extract(fit,pars='beta')$beta
  list(prob=colMeans(b>0),mean=colMeans(b),diagnostics=dg,beta_draws=b)
}
map_data <- function(mixes,x) {
  nm <- vapply(mixes,ncol,integer(1)); M <- max(nm)
  w <- matrix(0,2,M); a <- b <- matrix(1,2,M)
  for(e in 1:2) {
    w[e,seq_len(nm[e])] <- mixes[[e]][1,]
    a[e,seq_len(nm[e])] <- mixes[[e]][2,]
    b[e,seq_len(nm[e])] <- mixes[[e]][3,]
  }
  c(list(M=M,nmix=nm,w=w,a=a,b=b,beta_sd=kcfg$beta_sd),x)
}
build_prior <- function() {
set.seed(kcfg$seed)
hd <- history_data(); mixes <- list(); diag_rows <- list()
for(e in 1:2) {
  progress_log('Historical prior ',e,'/2: ',c('fever','discharge')[e],'; fitting MCMC and mixture.')
  dat <- data.frame(study=factor(c('KIDCARE','Wang2020')),D=hd$dh[,e],Y=hd$yh[,e])
  fit <- RBesT::gMAP(D ~ 1 | study,family=poisson,data=dat,offset=log(dat$Y),
    tau.dist='HalfNormal',tau.prior=kcfg$tau_sd,beta.prior=matrix(c(kcfg$mu_mean,kcfg$mu_sd),nrow=1),
    chains=kcfg$chains,iter=kcfg$iter,warmup=kcfg$warmup,thin=1,cores=kcfg$cores)
  ss <- rstan::summary(fit$fit)$summary
  sp <- rstan::get_sampler_params(fit$fit,inc_warmup=FALSE)
  dg <- data.frame(max_rhat=max(ss[,'Rhat'],na.rm=TRUE),
    min_neff=min(ss[,'n_eff'],na.rm=TRUE),
    divergences=sum(vapply(sp,function(x) sum(x[,'divergent__']),numeric(1))))
  write.csv(dg,file.path(kout,paste0('gMAP_diagnostics_',e,'.csv')),row.names=FALSE)
  if(!local_smoke && (dg$max_rhat>1.05 || dg$min_neff<100 || dg$divergences>0))
    stop('gMAP convergence failed; increase iterations/review diagnostics for endpoint ',e)
  save_artifact(fit,paste0('gMAP_endpoint_',e,'.rds'))
  # gMAP's response-scale prediction is the hazard (unit exposure), not D or Y*hazard.
  mixes[[e]] <- RBesT::automixfit(fit)
  stopifnot(inherits(mixes[[e]],'gammaMix'),all(is.finite(mixes[[e]])),all(mixes[[e]]>0))
  qq <- c(.025,.1,.5,.9,.975)
  write.csv(data.frame(probability=qq,
    MCMC=as.numeric(quantile(as.matrix(fit)[,'theta_resp_pred'],qq)),
    mixture=RBesT::qmix(mixes[[e]],qq)),
    file.path(kout,paste0('MAP_mixture_quantiles_',e,'.csv')),row.names=FALSE)
  pdf(file.path(kout,paste0('MAP_prior_endpoint_',e,'.pdf')))
  plot(mixes[[e]]); dev.off()
  capture.output(print(fit),file=file.path(kout,paste0('gMAP_summary_',e,'.txt')))
  # ESS must use the exponential interpretation, not the default Poisson scale.
  tmp <- mixes[[e]]; RBesT::likelihood(tmp) <- 'exp'
  diag_rows[[e]] <- data.frame(endpoint=c('fever','discharge')[e],
    prior_ESS=RBesT::ess(tmp),components=ncol(tmp))
  progress_log('Historical prior ',e,'/2 complete.')
}
save_artifact(mixes,'MAP_mixtures.rds')
write.csv(do.call(rbind,diag_rows),file.path(kout,'MAP_prior_ESS.csv'),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(kout,'MAP_sessionInfo.txt'))

}


# Grid search for the smallest common cutoff with error rate strictly below alpha; reject if p >= cutoff.
find_common_crit_grid <- function(p1,p2,alpha=0.05,grid=seq(0,1,by=0.0001)) {
  stopifnot(length(p1)>0L,length(p1)==length(p2),
            all(is.finite(p1)),all(is.finite(p2)),
            all(p1>=0 & p1<=1),all(p2>=0 & p2<=1))
  e1 <- vapply(grid,function(c) mean(p1>=c),numeric(1))
  e2 <- vapply(grid,function(c) mean(p2>=c),numeric(1))
  e12 <- vapply(grid,function(c) mean(pmax(p1,p2)>=c),numeric(1))
  valid <- which(pmax(e1,e2,e12)<alpha)
  i <- if(length(valid)) valid[1] else NA_integer_
  data.frame(common_threshold=grid[i],type1_H1=e1[i],type1_H2=e2[i],
             type1_H12=e12[i],max_type1=pmax(e1,e2,e12)[i])
}

run_case <- function(scenario) {
  stopifnot(length(scenario)==1L,scenario %in% 1:5)
  if(any(startsWith(args,'--cutoff='))) stop('--cutoff is only for analyze_MAP.R')
  need(c('rstan','RBesT'))
  if('--check' %in% args) {
    stopifnot(nrow(scenarios())==5,rstan::stanc(file=stan_path)$status)
    message('PASS: dependencies, five scenarios, Stan syntax.')
    return(invisible(NULL))
  }
  row <- scenarios()[scenario,]
  kout <<- file.path(run_root,sprintf('HR_%.2f',row$HR))
  dir.create(kout,recursive=TRUE,showWarnings=FALSE)
  lock <- file.path(kout,'.run_lock')
  if(!dir.create(lock,showWarnings=FALSE)) stop('Case running or stale lock: ',lock)
  on.exit(unlink(lock,recursive=TRUE),add=TRUE)
  fn <- paste0('MAP_scenario_',scenario,'.rds')
  if(file.exists(file.path(kout,fn))) {
    result <- load_artifact(fn)
    progress_log('Reusing completed HR ',row$HR)
  } else {
    # Same historical data and seed in each case; no shared writable prior cache.
    if(!file.exists(file.path(kout,'MAP_mixtures.rds'))) build_prior()
    mixes <- load_artifact('MAP_mixtures.rds')
    data_seed <- 123L+scenario-1L
    seed_base <- kcfg$seed+200000L+scenario*10000L
    set.seed(data_seed)
    xs <- lapply(seq_len(kcfg$nsim),function(i)
      sim_trial(1/kcfg$controls[row$control,],log(c(row$HR1,row$HR2))))
    zz <- fit_many(xs,stan_model(),mixes,seed_base,paste0('partial_',fn),
                   paste0('HR=',sprintf('%.2f',row$HR)))
    pp <- do.call(rbind,lapply(zz,`[[`,'prob'))
    bm <- do.call(rbind,lapply(zz,`[[`,'mean'))
    dg <- do.call(rbind,lapply(zz,`[[`,'diagnostics'))
    trials <- data.frame(simulation=seq_len(kcfg$nsim),scenario=scenario,HR=row$HR,
      data_seed=data_seed,mcmc_seed=seed_base+seq_len(kcfg$nsim),
      postprob_e1=pp[,1],postprob_e2=pp[,2],
      beta_mean_e1=bm[,1],beta_mean_e2=bm[,2],dg)
    result <- list(scenario=row,trials=trials,inputs=xs,
                   beta_draws=lapply(zz,`[[`,'beta_draws'))
    save_artifact(result,fn)
  }
  write.csv(result$trials,file.path(kout,'posterior_per_simulation.csv'),row.names=FALSE)
  if(scenario==1L) {
    cut <- find_common_crit_grid(result$trials$postprob_e1,result$trials$postprob_e2,kcfg$alpha)
    write.csv(cut,file.path(kout,'common_cutoff.csv'),row.names=FALSE)
    if(is.na(cut$common_threshold)) message('No grid cutoff achieves error < 0.05; posterior results saved.')
  }
  writeLines(capture.output(sessionInfo()),file.path(kout,'sessionInfo.txt'))
  progress_log('HR ',row$HR,' complete. Run analyze_MAP.R after all five cases finish.')
  invisible(result)
}

analyze_results <- function() {
  grid <- scenarios()
  results <- lapply(seq_len(5),function(g) {
    x <- load_artifact(paste0('MAP_scenario_',g,'.rds'),
                      file.path(run_root,sprintf('HR_%.2f',grid$HR[g])))
    stopifnot(identical(x$scenario,grid[g,]),nrow(x$trials)==kcfg$nsim)
    x$trials
  })
  chosen <- find_common_crit_grid(results[[1]]$postprob_e1,results[[1]]$postprob_e2,kcfg$alpha)
  ca <- args[startsWith(args,'--cutoff=')]
  if(!length(ca) && !is.null(getOption('kd.map.cutoff')))
    ca <- paste0('--cutoff=',getOption('kd.map.cutoff'))
  if(length(ca)>1L) stop('Only one --cutoff is allowed')
  cv <- if(length(ca)) suppressWarnings(as.numeric(sub('--cutoff=','',ca,fixed=TRUE))) else chosen$common_threshold
  if(length(cv)!=1L || !is.finite(cv) || cv<0 || cv>1)
    stop('No valid cutoff. Posterior results are preserved; supply --cutoff in [0,1] or increase simulations.')
  # Distinct analysis directories preserve previous cutoff analyses and raw results.
  out <- file.path(run_root,paste0('analysis_cutoff_',format(cv,digits=15,trim=TRUE)))
  dir.create(out,recursive=TRUE,showWarnings=FALSE)
  write.csv(chosen,file.path(out,'grid_search_cutoff.csv'),row.names=FALSE)
  summaries <- lapply(seq_len(5),function(g) {
    x <- results[[g]]
    r <- cbind(x$postprob_e1>=cv,x$postprob_e2>=cv)
    truth <- log(c(grid$HR1[g],grid$HR2[g]))
    null <- truth<=0
    fw <- if(any(null)) mean(rowSums(r[,null,drop=FALSE])>0) else NA_real_
    x$success_e1 <- r[,1]; x$success_e2 <- r[,2]
    x$success_any <- rowSums(r)>0; x$success_both <- rowSums(r)==2
    x$critical_value <- cv
    write.csv(x,file.path(out,sprintf('HR_%.2f_decisions.csv',grid$HR[g])),row.names=FALSE)
    data.frame(grid[g,],method='MAP',n_ctrl=kcfg$n_ctrl,n_trt=kcfg$n_trt,nsim=nrow(x),
      critical_value=cv,cutoff_source=if(length(ca)) 'manual' else 'case_1_grid_search',
      reject_fever=mean(r[,1]),reject_discharge=mean(r[,2]),
      reject_any=mean(x$success_any),reject_both=mean(x$success_both),FWER=fw,
      FWER_MCSE=sqrt(fw*(1-fw)/nrow(x)),
      bias_beta_fever=mean(x$beta_mean_e1)-truth[1],
      bias_beta_discharge=mean(x$beta_mean_e2)-truth[2],
      mse_beta_fever=mean((x$beta_mean_e1-truth[1])^2),
      mse_beta_discharge=mean((x$beta_mean_e2-truth[2])^2))
  })
  summary <- do.call(rbind,summaries)
  write.csv(summary,file.path(out,'MAP_summary_all_5HR.csv'),row.names=FALSE)
  message('Common cutoff = ',cv,'; results: ',out)
  invisible(summary)
}
