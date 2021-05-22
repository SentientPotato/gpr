## Load packages
library(gpr)

## Define R-side function for ARD kernel
covSEard <- function(X, Z = X, sf2 = 1, ell2 = rep(1, ncol(X))) {
    if ( ncol(X) != ncol(Z) ) {
        stop("X and Z should have the same number of columns")
    }
    n <- nrow(X)
    m <- nrow(Z)
    d <- ncol(X)
    ell2 <- 1 / ell2
    res <- matrix(NA_real_, nrow = n, ncol = m)
    for ( j in 1:m ) {
        for ( i in 1:n ) {
            res[i, j] <- sf2 * exp(-0.5 * sum(ell2 * (X[i, ] - Z[j, ])^2))
        }
    }
    return(res)
}

## Simulate data
# set.seed(138)
# beta <- rnorm(3, sd = 2) ## intercept, running variable, treatment effect
# n <- 250
# x <- rnorm(n)
# d <- as.numeric(x > 0)
# X <- cbind(x, d)
# H <- cbind(1, X)
# f <- H %*% beta
# e <- rnorm(n)
# y <- f + e
# plot(x, y, pch = 19, col = "#33333333")
# o <- order(x)
# lines(x[o], f[o])
set.seed(138)
beta <- rnorm(3, sd = 2) ## intercept, running variable, treatment effect
ell2 <- abs(rnorm(2, sd = 2))
sf2 <- abs(rnorm(1, sd = 2))
sy2 <- abs(rnorm(1))
n <- 500
x <- rnorm(n)
d <- as.numeric(x > 0)
X <- cbind(x, d)
H <- cbind(1, X)
m <- H %*% beta
f <- c(mvtnorm::rmvnorm(1, sigma = covSEard(X, sf2 = sf2, ell2 = ell2)))
e <- rnorm(n, sd = sqrt(sy2))
y <- f + m + e
plot(x, y, pch = 19, col = "#33333333")
o <- order(x)
lines(x[o], f[o] + m[o])

## Sample posterior
set.seed(42)
samples <- gprMCMC(
    y, X, ntune = 500#, mean_zero = TRUE
)
saveRDS(samples, "../samples500.rds")
samples <- readRDS("../samples200.rds")

## Look at mean function coefficient estimates
ests <- colMeans(samples)
low  <- apply(samples, 2, quantile, probs = 0.025)
high <- apply(samples, 2, quantile, probs = 0.975)
idx  <- (n+1):length(ests)
comparison <- data.frame(
    truth = c(beta, c(sf2, ell2, sy2)),
    estimate = ests[idx],
    ci = sprintf("[% 6.2f,% 6.2f]", low[idx], high[idx])
)
rownames(comparison) <- c("intercept", "beta", "tau",
                          "sf2", "ell_x", "ell_tau", "sy2")
print(comparison, digits = 2)

## Look at traceplots of the parameters
vars <- expression(intercept, beta[x], beta[d], sigma[f], l[x], l[d], sigma[y])
vals <- c(beta, c(sf2, ell2, sy2))
for ( j in (n+1):(n+length(vals)) ) {
    lim <- range(c(samples[ , j], vals[j - n]))
    lim[1] <- max(lim[1], -10 * abs(vals[j - n]))
    lim[2] <- max(lim[1],  10 * abs(vals[j - n]))
    plot(
        samples[ , j], type = "l", ylim = lim,
        main = vars[j - n], xlab = "Iteration", ylab = "Draw"
    )
    abline(h = vals[j - n], lty = 2)
}

draws <- matrix(NA_real_, nrow = nrow(samples), ncol = n)
for ( s in 1:nrow(samples) ) {
    draws[s, ] <- samples[s, 1:n] + H %*% samples[s, (n+1):(n+3)]
}
means <- apply(draws, 2, mean)
highs <- apply(draws, 2, quantile, probs = 0.975)
lows  <- apply(draws, 2, quantile, probs = 0.025)

plot(x[o], y[o], pch = 19, col = "#80808080")
lines(x[o], f[o] + m[o])
lines(x[o], means[o], lty = 2)
polygon(x = c(x[o], rev(x[o])), y = c(highs[o], rev(lows[o])),
        border = NA, col = "#aaaaaa33")

## Estimate treatment effect
Z <- matrix(c(0, 0, 1, 0), ncol = 2)
set.seed(314)
preds <- gpr:::.gpr_predict(samples, y, X, Z)
tau_draws <- apply(preds, 1, function(x) x[1] - x[2])
tau_est   <- mean(tau_draws)
tau_low   <- quantile(tau_draws, probs = 0.025)
tau_high  <- quantile(tau_draws, probs = 0.975)

## Compare to truth
Kss <- covSEard(Z, sf2 = sf2, ell2 = ell2)
Ks  <- covSEard(Z, X, sf2 = sf2, ell2 = ell2)
K   <- covSEard(X, sf2 = sf2, ell2 = ell2)
Ki  <- solve(K + sy2 * diag(n))
C   <- Kss - Ks %*% Ki %*% t(Ks)
mZ  <- c(cbind(1, Z) %*% beta + Ks %*% Ki %*% f)
tau_mean <- mZ[1] - mZ[2]
tau_sd   <- sqrt(sum(C))
comparison <- data.frame(
    Source  = c("Truth", "GPR"),
    Effect  = c(tau_mean, tau_est),
    CI_Low  = c(qnorm(0.025, mean = tau_mean, sd = tau_sd), tau_low),
    CI_High = c(qnorm(0.975, mean = tau_mean, sd = tau_sd), tau_high),
    row.names = NULL
)
comparison

library(rdrobust)
summary(rdrobust(y, x, c = 0))

plot(abs(samples[ , n+1]), type = "l")
abline(h = sf2, lty = 2, col = "red")
plot(abs(samples[ , n+2]), type = "l")
abline(h = ell2[1], lty = 2, col = "red", lwd = 2)
plot(abs(samples[ , n+3]), type = "l")
abline(h = ell2[2], lty = 2, col = "red", lwd = 2)
