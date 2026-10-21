#!/usr/bin/env Rscript
# Supports both Rscript and RStudio source().
ff <- vapply(sys.frames(),function(x) if(is.null(x$ofile)) "" else as.character(x$ofile),character(1))
ff <- ff[nzchar(ff)]
cli <- grep('^--file=',commandArgs(FALSE),value=TRUE)
if(length(ff)) {
  script_file <- tail(ff,1)
} else if(length(cli)) {
  script_file <- gsub('~+~',' ',sub('^--file=','',cli[1]),fixed=TRUE)
} else stop('Use source() or Rscript to run this file.')
script_dir <- dirname(normalizePath(script_file,mustWork=TRUE))
source(file.path(script_dir,'MAP_common.R'))
if('--check' %in% args) {
  run_case(1L)
} else {
  # Run cases sequentially to limit local memory/CPU use.
  for(g in seq_len(5)) run_case(g)
  analyze_results()
}
