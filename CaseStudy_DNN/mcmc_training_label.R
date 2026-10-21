# Retry the SAME data with more conservative sampling; never accept failed labels.
fit_training_label <- function(sm, stan_data, simulation, checkpoint_dir,
    model_code, schedule,
    sample_fun = rstan::sampling, summary_fun = rstan::summary,
    sampler_fun = rstan::get_sampler_params, extract_fun = rstan::extract) {
  signature <- list(data=stan_data,model=model_code,version=1L)
  path <- file.path(checkpoint_dir,sprintf('label_%05d.rds',simulation))
  if(file.exists(path)) {
    old <- readRDS(path)
    if(identical(old$signature,signature) && isTRUE(old$ok)) return(old)
  }
  history <- list(); result <- NULL
  for(a in seq_len(nrow(schedule))) {
    setting <- schedule[a,]
    warnings_seen <- character()
    result <- tryCatch(withCallingHandlers({
      fit <- sample_fun(sm,data=stan_data,chains=setting$chains,
        iter=setting$iter,warmup=setting$warmup,cores=1,refresh=0,
        seed=1000L+simulation+(a-1L)*100000L,
        control=list(adapt_delta=setting$adapt_delta,max_treedepth=setting$max_treedepth))
      ss <- summary_fun(fit,pars='beta_bar')$summary
      sp <- sampler_fun(fit,inc_warmup=FALSE)
      rh <- ss[,'Rhat']; ess <- ss[,'n_eff']
      div <- sum(vapply(sp,function(x) sum(x[,'divergent__']),numeric(1)))
      tree <- sum(vapply(sp,function(x) sum(x[,'treedepth__']>=setting$max_treedepth),numeric(1)))
      ok <- all(is.finite(c(rh,ess))) && all(rh<=1.05) && all(ess>=100) && div==0 && tree==0
      labels <- NULL
      if(ok) {
        beta <- extract_fun(fit,pars='beta_bar')$beta_bar
        labels <- c(colMeans(beta>0),colMeans(beta))
        ok <- length(labels)==2*stan_data$E && all(is.finite(labels))
      }
      list(ok=ok,label=labels,diagnostics=list(max_Rhat=max(rh),min_ESS=min(ess),
        divergences=div,treedepth_hits=tree),error=NULL)
    },warning=function(w) {
      warnings_seen <<- c(warnings_seen,conditionMessage(w)); invokeRestart('muffleWarning')
    }),error=function(e) list(ok=FALSE,label=NULL,diagnostics=NULL,error=conditionMessage(e)))
    history[[a]] <- list(setting=setting,diagnostics=result$diagnostics,
                        warnings=unique(warnings_seen),error=result$error)
    if(isTRUE(result$ok)) break
  }
  out <- list(signature=signature,ok=isTRUE(result$ok),label=result$label,
              attempts=history,simulation=simulation)
  temp <- tempfile('label_',tmpdir=checkpoint_dir)
  saveRDS(out,temp)
  if(!file.rename(temp,path)) stop('Could not save MCMC checkpoint: ',path)
  out
}
