#' Gaussian Process Regression MCMC Sampling
#'
#' @param y A numeric vector of outcomes
#' @param X A numeric matrix of predictors
#' @param nburn An integer vector of length one; how many burn-in
#'   (unrecorded) iterations should there be? Default is 100.
#' @param nsample An integer vector of length one; how many posterior
#'   samples should be produced? Default is 1000.
#' @param b A numeric vector of length ncol(X) + 1 giving the prior mean for
#'   beta (the +1 is for the intercept); default is a zero vector
#' @param B A numeric matrix with ncol(X) + 1 rows and columns giving the prior
#'   covariance matrix for beta; default is a diagonal matrix whose diagonal
#'   entries are 10
#' @param e A numeric vector of length ncol(X) + 2 giving the prior mean for
#'   eta (the +2 is for sy2 and sf2); default is a zero vector
#' @param E A numeric matrix with ncol(X) + 2 rows and columns giving the prior
#'   covariance matrix for eta; default is a diagonal matrix whose diagonal
#'   entries are 1
#' @param shape A numeric vector of length one giving the shape for the inverse
#'   gamma prior on sy2
#' @param scale A numeric vector of length one giving the scale for the inverse
#'   gamma prior on sy2
#' @param minloops An integer vector of length one giving the minimum number of
#'   tuning loops to complete; the default is 5
#' @param maxloops An integer vector of length one giving the maximum number of
#'   tuning loops to complete; the default is 20
#' @param ntune An integer vector of length one giving the number of iterations
#'   to complete in each tuning loop; the default is 100
#' @param target_rate A numeric vector of length one giving the desired
#'   proposal acceptance rate; the default is 0.234
#' @param rate_tol A numeric vector of length one giving the tolerance on the
#'   proposal acceptance rate; the default is 0.03
#' @param prop_scale A numeric vector of length one giving the initial scaling
#'   constant to use for the proposal covariance; the default is 2.38
#' @param weight A numeric vector of length one between 0 and 1 giving the
#'   weight to place on the current observed covariance between draws relative
#'   to the proposal covariance when updating the proposal covariance; the
#'   default is 0.5
#' @param verbose A logical vector of length one; should detailed proposal
#'   tuning information be printed to the console? The default is TRUE.
#' @param mean_zero A logical vector of length one; should a mean zero function
#'   be used instead of a linear mean? Default is FALSE
#'
#' @return
#'
#' @export
gprMCMC <- function(
    y,
    X,
    burn_iters = 1000,
    sample_iters = 10000,
    b = rep(0, ncol(as.matrix(X)) + 1),
    B = diag(x = 10, nrow = ncol(as.matrix(X)) + 1),
    e = rep(0, ncol(as.matrix(X)) + 1),
    E = diag(x = 1, nrow = ncol(as.matrix(X)) + 1),
    shape = 1,
    scale = 1,
    minloops = 5,
    maxloops = 20,
    ntune = 500,
    target_rate = 0.234,
    rate_tol = 0.03,
    prop_scale = 2.38 / sqrt(ncol(X) + 2),
    weight = 0.75,
    verbose = TRUE,
    mean_zero = FALSE
) {
    if ( mean_zero ) {    res <- .gprMCMCZ(
        y,
        X,
        burn_iters,
        sample_iters,
        e,
        E,
        shape,
        scale,
        minloops,
        maxloops,
        ntune,
        target_rate,
        rate_tol,
        prop_scale,
        weight,
        verbose
    )
    colnames(res) <- c(
        paste("f", 1:length(y), sep = "_"),
        "sf2", paste("ell", 1:ncol(as.matrix(X)), sep = "_"), "sy2"
    )
    } else {
        res <- .gprMCMC(
            y,
            X,
            burn_iters,
            sample_iters,
            b,
            B,
            e,
            E,
            shape,
            scale,
            minloops,
            maxloops,
            ntune,
            target_rate,
            rate_tol,
            prop_scale,
            weight,
            verbose
        )
        colnames(res) <- c(
            paste("f", 1:length(y), sep = "_"),
            paste("beta", 0:ncol(as.matrix(X)), sep = "_"),
            "sf2", paste("ell", 1:ncol(as.matrix(X)), sep = "_"), "sy2"
        )
    }
    return(res)
}
