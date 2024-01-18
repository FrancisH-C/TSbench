# http://www.unstarched.net/r-examples/rugarch/a-short-introduction-to-the-rugarch-package/

suppressMessages(suppressWarnings(library(rugarch)))

train_ruGARCH <- function(series, alpha_lag, beta_lag) {
  #' This function returns trained GARCH model's parameters
  #'
  #' @param series The array containing the data
  #' @param alpha_lag The lag for returns
  #' @param beta_lag The lag for vol

  series <- as.data.frame(series)

  # if you want to use cluters
  # cluster = makePSOCKcluster(2)

  ## train
  spec <- ugarchspec(variance.model =
                           list(model = "sGARCH",
                                garchOrder = c(alpha_lag, beta_lag)))
  fit <- ugarchfit(spec, series, solver = "hybrid")
  return(coef(fit))
}
