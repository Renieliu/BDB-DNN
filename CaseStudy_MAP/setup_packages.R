# Run once in local R/RStudio with Source. Existing loadable packages are kept.
options(repos=c(CRAN='https://cloud.r-project.org'),timeout=600)
cat(R.version.string,'\nR libraries:\n'); print(.libPaths())
for(pkg in c('codetools','rstan','RBesT')) {
  ok <- tryCatch({loadNamespace(pkg); TRUE},error=function(e) {
    message(pkg,': ',conditionMessage(e)); FALSE
  })
  if(!ok) install.packages(pkg,Ncpus=1L)
  tryCatch(loadNamespace(pkg),error=function(e)
    stop('Cannot load ',pkg,': ',conditionMessage(e),
         '\nReview installation output; restart R after installing/updating dependencies.'))
  cat(pkg,as.character(packageVersion(pkg)),'at',find.package(pkg),'\n')
}
cat('Packages ready. Source MAP_HR_1.00.R with options(kd.map.check=TRUE) to check Stan syntax.\n')
