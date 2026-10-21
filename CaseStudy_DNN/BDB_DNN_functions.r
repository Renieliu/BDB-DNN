# Shared piecewise-exponential simulation helpers for DNN training/testing and MAP.
make_entry_dist <- function(accrual_end) {
  force(accrual_end)
  function() stats::runif(1, min = 0, max = accrual_end)
}

sim_pwe_arm_counts <- function(
    n, lam, cuts,
    analysis_time = cuts[length(cuts)],
    entry_dist = function() 0
) {
  K <- length(lam)
  d <- numeric(K)
  Y <- numeric(K)

  stopifnot(length(cuts) == K + 1)
  stopifnot(analysis_time >= cuts[length(cuts)])

  for (i in seq_len(n)) {
    entry_time <- entry_dist()
    followup_limit <- max(0, analysis_time - entry_time)
    t_event <- 0
    event_occurred <- FALSE

    for (k in seq_len(K)) {
      wait_k <- rexp(1, rate = lam[k])
      if (wait_k < cuts[k + 1] - cuts[k]) {
        t_event <- cuts[k] + wait_k
        event_occurred <- TRUE
        break
      }
    }
    if (!event_occurred) t_event <- cuts[K + 1]

    t_obs <- min(t_event, followup_limit)
    delta <- as.integer(event_occurred && t_event <= followup_limit)

    for (k in seq_len(K)) {
      left <- cuts[k]
      right <- cuts[k + 1]
      Y[k] <- Y[k] + max(0, min(t_obs, right) - left)
      if (delta == 1 && t_event >= left && t_event < right) d[k] <- d[k] + 1
    }
  }

  list(d = d, Y = Y)
}

sim_hist_pwe_counts <- function(
    n.vec.hist, E, K, cuts, base_lam_mat,
    tau = 0.4, r_endpt = 0.0,
    analysis_time = cuts[length(cuts)], entry_dist = function() 0
) {
  Jhist <- length(n.vec.hist)
  R <- diag(E)
  if (E == 2) R[1, 2] <- R[2, 1] <- r_endpt else stop("This template currently assumes E = 2.")
  L <- chol(R)
  d_hist <- array(0L, dim = c(Jhist, E, K))
  Y_hist <- array(0, dim = c(Jhist, E, K))

  for (j in seq_len(Jhist)) {
    b <- as.vector(t(L) %*% rnorm(E)) * tau
    for (e in seq_len(E)) {
      sim <- sim_pwe_arm_counts(
        n = n.vec.hist[j], lam = base_lam_mat[e, ] * exp(b[e]), cuts = cuts,
        analysis_time = analysis_time, entry_dist = entry_dist
      )
      d_hist[j, e, ] <- as.integer(sim$d)
      Y_hist[j, e, ] <- sim$Y
    }
  }
  list(d_hist = d_hist, Y_hist = Y_hist)
}

sim_current_twoarm_counts <- function(
    n_ctrl, n_trt, E, K, cuts, base_lam_ctrl_mat, beta_mat,
    analysis_time = cuts[length(cuts)], entry_dist = function() 0
) {
  stopifnot(all(dim(base_lam_ctrl_mat) == c(E, K)), all(dim(beta_mat) == c(E, K)))
  d_ctrl <- matrix(0L, E, K); Y_ctrl <- matrix(0, E, K)
  d_trt <- matrix(0L, E, K); Y_trt <- matrix(0, E, K)

  for (e in seq_len(E)) {
    sim_c <- sim_pwe_arm_counts(n_ctrl, base_lam_ctrl_mat[e, ], cuts, analysis_time, entry_dist)
    sim_t <- sim_pwe_arm_counts(n_trt, base_lam_ctrl_mat[e, ] * exp(beta_mat[e, ]), cuts, analysis_time, entry_dist)
    d_ctrl[e, ] <- as.integer(sim_c$d); Y_ctrl[e, ] <- sim_c$Y
    d_trt[e, ] <- as.integer(sim_t$d); Y_trt[e, ] <- sim_t$Y
  }
  list(d_ctrl = d_ctrl, Y_ctrl = Y_ctrl, d_trt = d_trt, Y_trt = Y_trt)
}



# Plug-in estimates for the two-endpoint, constant-hazard Stage-2 models.
# These use only observed current-trial counts and exposure, not simulation truth.
estimate_stage2_inputs <- function(d_c, Y_c, d_t, Y_t) {
  summaries <- list(d_c, Y_c, d_t, Y_t)
  if (!all(vapply(summaries, function(x) {
    is.matrix(x) && identical(dim(x), c(2L, 1L)) && all(is.finite(x))
  }, logical(1)))) stop("Stage-2 estimation requires finite 2-by-1 current-data summaries.")
  if (any(d_c < 0) || any(d_t < 0) || any(Y_c <= 0) || any(Y_t <= 0))
    stop("Stage-2 estimation requires nonnegative events and positive exposure.")
  if (any(d_c == 0) || any(d_t == 0))
    stop("Zero events give non-finite log-hazard estimates; specify a zero-event estimator before testing.")
  loglam_ctrl <- as.numeric(log(d_c) - log(Y_c))
  loglam_trt <- as.numeric(log(d_t) - log(Y_t))
  beta <- loglam_trt - loglam_ctrl
  list(loglam_ctrl = loglam_ctrl, beta = beta,
       H1 = c(loglam_ctrl, beta[2]), H2 = c(loglam_ctrl, beta[1]),
       H12 = loglam_ctrl)
}

## functions for deep historical borrowing for multiple endpoints
##################################################################################################
## function to get responders based on historical response rate
sim.hist.data.func = function(sim.hist.rate.vec.in){
  ## data vector with first 6 for the first endpoint, then 6 for the second endpoint
  sim.hist.data.vec = sapply(1:(n.endpoint*n.hist), 
                             function(x){rbinom(1, rep(n.vec.hist, n.endpoint)[x],
                                                rep(sim.hist.rate.vec.in, each = n.hist)[x])})
  
  ## turn to a matrix, row as endpoints, column as studies
  sim.hist.data.mat = matrix(sim.hist.data.vec, nrow = n.endpoint, 
                             ncol = n.hist, byrow= TRUE)
  
  ## return the vector and matrix
  new.list = list("vec" = sim.hist.data.vec,
                  "mat" = sim.hist.data.mat)
}

## function for DNN fitting. The last layer activation function. Can also try sigmoid activation 
## function because the posterior probs are between 0 and 1. However, sigmoid function is easily
## to get saturated (small gradient) at boundary. 

DNN.fit <- function(data.train.scale.in, data.train.label.in, 
                    drop.rate.in, active.name.in, n.node.in, 
                    n.layer.in, max.epoch.in, batch_size_in, validation.prop.in,
                    n.endpoint.in,
                    output_activation = "linear",
                    loss_name = "mse") {
  
  library(keras3)
  
  x_train <- as.matrix(data.train.scale.in)
  y_train <- as.matrix(data.train.label.in)
  n_features <- ncol(x_train)
  
  inputs <- layer_input(shape = n_features)
  x <- inputs
  for (i in seq_len(n.layer.in)) {
    x <- x |>
      layer_dense(units = n.node.in, activation = active.name.in) |>
      layer_dropout(rate = drop.rate.in)
  }
  outputs <- x |> layer_dense(units = n.endpoint.in, activation = output_activation)
  
  model <- keras_model(inputs = inputs, outputs = outputs) |>
    compile(
      optimizer = optimizer_rmsprop(),
      loss = loss_name,
      metrics = "mse"
    )
  
  history <- model |>
    fit(
      x = x_train,
      y = y_train,
      epochs = max.epoch.in,
      batch_size = batch_size_in,
      validation_split = validation.prop.in,
      verbose = 0
    )
  
  list(model = model, history = history)
}

## function to obtain the critical value
critical.value.func.tte <- function(
    n.itt.in,                # number of Monte Carlo iterations for cutoff
    n.new.in,                # sample size per arm
    E, K, cuts,              # TTE structure
    loglam_mean_vec,         # length E: control mean log-hazard
    beta_mat,                # E x K interval-specific beta matrix
    shape_sd = 0.15,         # SD of log-hazard deviations across intervals
    # first-stage prob DNN objects (weights + scaling)
    col_means_train.in,
    col_stddevs_train.in,
    DNN.first.weight.in,
    DNN.first.layer.in,
    type.1.error.in,
    analysis_time.in = cuts[length(cuts)],
    entry_dist.in = function() 0
) {
  
  p_hat_mat <- matrix(NA_real_, nrow = n.itt.in, ncol = E)
  
  for (itt in 1:n.itt.in) {
    # progress print
    if (itt %% 100 == 0) cat("itt=", itt, "\n")
    
    # generate hazards per endpoint
    lam_c_list <- lapply(1:E, function(e) make_lam_profile(loglam_mean_vec[e], K, shape_sd))
    lam_t_list <- lapply(1:E, function(e) lam_c_list[[e]] * exp(beta_mat[e, ]))
    
    # simulate aggregated counts/person-time
    d_c <- matrix(0L, E, K); Y_c <- matrix(0, E, K)
    d_t <- matrix(0L, E, K); Y_t <- matrix(0, E, K)
    
    for (e in 1:E) {
      sc <- sim_pwe_arm_counts(
        n.new.in, lam_c_list[[e]], cuts,
        analysis_time = analysis_time.in,
        entry_dist = entry_dist.in
      )
      st <- sim_pwe_arm_counts(
        n.new.in, lam_t_list[[e]], cuts,
        analysis_time = analysis_time.in,
        entry_dist = entry_dist.in
      )
      
      d_c[e, ] <- as.integer(sc$d); Y_c[e, ] <- sc$Y
      d_t[e, ] <- as.integer(st$d); Y_t[e, ] <- st$Y
    }
    
    # build first-stage feature vector (same order as the training data)
    x <- t(as.matrix(c(as.numeric(d_c), as.numeric(Y_c), as.numeric(d_t), as.numeric(Y_t))))
    # fast DNN forward (sigmoid output)
    p_hat <- pred.DNN.sigmoid(
      DNN.final.weights.in = DNN.first.weight.in,
      n.layer.in = DNN.first.layer.in,
      data.train.in = x,
      col_means_train.in = col_means_train.in,
      col_stddevs_train.in = col_stddevs_train.in
    )
    
    p_hat_mat[itt, ] <- as.numeric(p_hat)[1:E]
  }
  
  # determine cutoff depending on null pattern
  beta_is_null <- rowSums(abs(beta_mat)) == 0
  if (all(beta_is_null)) {
    # Global null: Pr{(S1 > c12) OR (S2 > c12) | H12} <= alpha.
    max_prob <- apply(p_hat_mat, 1, max)
    cutoff <- sort(max_prob)[ceiling((1 - type.1.error.in) * (n.itt.in + 1))]
  } else {
    # single-null cases: use the null endpoint's (1-alpha) quantile
    null_idx <- which(beta_is_null)
    # with more than one null endpoint, use the first one
    cutoff <- sort(p_hat_mat[, null_idx[1]])[ceiling((1 - type.1.error.in) * (n.itt.in + 1))]
  }
  
  cutoff
}

## DNN fitting function for critical values to accommodate different input data dimension
DNN.combine.fit <- function(DNN.second.mat.in, DNN.second.label.in,
                            DNN.second.layer.in, DNN.second.node.in) {
  
  # Scale returns a matrix; keep as.matrix for keras
  data.DNN.second.train.scale <- scale(as.matrix(DNN.second.mat.in))
  
  col_means_train_second <- attr(data.DNN.second.train.scale, "scaled:center")
  col_stddevs_train_second <- attr(data.DNN.second.train.scale, "scaled:scale")
  
  DNN.second.temp <- DNN.fit(
    data.train.scale.in  = data.DNN.second.train.scale,
    data.train.label.in  = as.matrix(DNN.second.label.in),
    drop.rate.in         = 0.1,
    active.name.in       = "relu",
    n.node.in            = DNN.second.node.in,
    n.layer.in           = DNN.second.layer.in,
    max.epoch.in         = 1000,
    batch_size_in        = 100,
    validation.prop.in   = 0,
    n.endpoint.in        = 1,
    output_activation    = "sigmoid",
    loss_name            = "mse"
  )
  
  list(
    DNN.model   = DNN.second.temp,
    mean_center = col_means_train_second,
    sd_center   = col_stddevs_train_second
  )
}

## get fitted values from DNN, which can be used in parallel computing
pred.DNN.normal = function(DNN.final.weights.in, n.layer.in, data.train.in,
                           col_means_train.in, col_stddevs_train.in){
  
  w1.scale = DNN.final.weights.in[[1]]
  b1.scale = as.matrix(DNN.final.weights.in[[2]])
  w1 = t(w1.scale/matrix(rep(col_stddevs_train.in, dim(w1.scale)[2]),
                         nrow = dim(w1.scale)[1], ncol = dim(w1.scale)[2]))
  b1 = b1.scale - t(w1.scale)%*%as.matrix(col_means_train.in/col_stddevs_train.in)
  
  for (wb.itt in 2:(n.layer.in+1)){
    w.text = paste0("w", wb.itt, "=t(DNN.final.weights.in[[", wb.itt*2-1, "]])")
    b.text = paste0("b", wb.itt, "= as.matrix(DNN.final.weights.in[[", wb.itt*2, "]])")
    
    eval(parse(text=w.text))
    eval(parse(text=b.text))
  }
  
  eval_f_whole_text1 = paste0(
    "eval_f <- function( x ) {x.mat = as.matrix(as.numeric(x), nrow = length(x), ncol = 1);
    w1x = (w1)%*%x.mat + b1;sw1x = as.matrix(c(relu(w1x)))")
  
  eval_f_whole_text2 = NULL
  for (wb.itt in 2:(n.layer.in)){
    wx.text = paste0("w", wb.itt, "x = (w", wb.itt, ")%*%sw", wb.itt-1,
                     "x + b", wb.itt)
    swx.text = paste0("sw", wb.itt, "x = as.matrix(c(relu(w", wb.itt, "x)))")
    eval_f_whole_text2 = paste(eval_f_whole_text2, wx.text, swx.text, sep = ";")
  }
  
  wb.itt.final = n.layer.in + 1
  wx.text = paste0("w", wb.itt.final, "x = (w", wb.itt.final, ")%*%sw", wb.itt.final-1,
                   "x + b", wb.itt.final)
  swx.text = paste0("sw",n.layer.in+1,"x =(w", wb.itt.final, "x)")
  eval_f_whole_text2 = paste(eval_f_whole_text2, wx.text, swx.text, sep = ";")
  
  eval_f_whole_text3 = paste0(";return(sw", n.layer.in+1, "x)}")
  
  eval_f_whole_text = paste(eval_f_whole_text1, eval_f_whole_text2,
                            eval_f_whole_text3)
  
  eval(parse(text=eval_f_whole_text))
  
  final.pred = sapply(1:(dim(data.train.in)[1]),
                      function(y){eval_f(x = as.vector(data.train.in[y,]))})
  return(final.pred)
}  

## ReLU activation function
relu = function(x){
  return(pmax(0, x))
}

sigmoid <- function(x) 1/(1 + exp(-x))

pred.DNN.sigmoid <- function(DNN.final.weights.in, n.layer.in, data.train.in,
                             col_means_train.in, col_stddevs_train.in) {
  lin <- pred.DNN.normal(
    DNN.final.weights.in = DNN.final.weights.in,
    n.layer.in = n.layer.in,
    data.train.in = data.train.in,
    col_means_train.in = col_means_train.in,
    col_stddevs_train.in = col_stddevs_train.in
  )
  sigmoid(lin)
}

make_lam_profile <- function(loglam_mean, K, shape_sd = 0.15) {
  # log hazards: random perturbations around the mean
  eps <- rnorm(K, mean = 0, sd = shape_sd)
  lam <- exp(loglam_mean + eps)
  lam
}

make_beta_profile <- function(beta_mean, K, shape_sd = 0.15) {
  if (beta_mean == 0) {
    return(rep(0, K))
  }

  eps <- rnorm(K, mean = 0, sd = shape_sd)
  beta_mean + eps - mean(eps)
}

draw_beta_profile <- function(K, beta.lower, beta.upper, jitter_sd = 0.05) {
  pattern <- sample(c("constant", "delayed", "early", "heterogeneous"), size = 1)
  
  beta <- rep(0, K)
  b <- runif(1, beta.lower, beta.upper)
  
  if (pattern == "constant") {
    beta[] <- b
  }
  
  if (pattern == "delayed") {
    delay <- sample(1:(K - 1), size = 1)
    active_len <- K - delay
    active <- seq(log(0.9), b, length.out = active_len)
    beta[(delay + 1):K] <- active
  }
  
  if (pattern == "early") {
    stop_at <- sample(1:(K - 1), size = 1)
    active <- seq(b, log(0.9), length.out = stop_at)
    beta[1:stop_at] <- active
  }
  
  if (pattern == "heterogeneous") {
    beta[] <- runif(K, beta.lower, beta.upper)
  }
  
  active_idx <- beta != 0
  beta[active_idx] <- beta[active_idx] + rnorm(sum(active_idx), 0, jitter_sd)
  
  beta
}
