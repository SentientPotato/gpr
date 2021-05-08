## Load packages
library(gpr)

## Simulate data
set.seed(138)
beta <- rnorm(3, sd = 2) ## intercept, running variable, treatment effect
n <- 100
x <- rnorm(n)
d <- as.numeric(x > 0)
X <- cbind(x, d)
H <- cbind(1, X)
f <- H %*% beta
e <- rnorm(n)
y <- f + e
plot(x, y, pch = 19, col = "#33333333")
o <- order(x)
lines(x[o], f[o])

## Sample posterior
set.seed(42)
samples <- gprMCMC(100, y, X, b = rep(0, 3), B = diag(10, nrow = 3))

## Look at mean function coefficient estimates
ests <- rowMeans(samples)
low  <- apply(samples[(n+1):nrow(samples), ], 1, quantile, probs = 0.025)
high <- apply(samples[(n+1):nrow(samples), ], 1, quantile, probs = 0.975)
comparison <- data.frame(
    truth = beta,
    estimate = tail(ests, 3),
    ci = sprintf("[% 6.2f,% 6.2f]", low, high)
)
rownames(comparison) <- c("intercept", "beta", "tau")
print(comparison, digits = 2)

## Look at traceplots of the parameters
vars <- expression(sigma[y], sigma[f], l[x], l[d], intercept, beta[x], beta[d])
vals <- c(sigma_y, sigma_f, ell, beta)
for ( j in 101:107 ) {
    lim <- range(c(samples[ , j], vals[j - 100]))
    plot(
        samples[ , j], type = "l", ylim = lim,
        main = vars[j - 100], xlab = "Iteration", ylab = "Draw"
    )
    abline(h = vals[j - 100], lty = 2)
}
