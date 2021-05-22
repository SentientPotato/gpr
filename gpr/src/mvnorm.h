#ifndef GPR_MVNORM_H
#define GPR_MVNORM_H

#include <RcppArmadillo.h> // OK as I know they use include guards
#include "matops.h" // OK because I know I use include guards

// normal density function; log only, one observation only
inline double dmvnorm(const arma::vec& x,   // observations
                      const arma::vec& mu,  // mean
                      const arma::mat& S) { // variance
    arma::vec    z = x - mu;
    arma::mat    U = arma::chol(regularize(S));
    double log_det = arma::sum(arma::log(U.diag())); // really log(det(S))/2
    double     tmp = -1.0 * (S.n_cols/2.0) * M_LN_2PI - log_det;
                 U = arma::inv(arma::trimatu(U));
    return tmp - 0.5 * arma::as_scalar(z.t() * U * U.t() * z);
}

// random generator, one observation only
inline arma::vec rmvnorm(const arma::vec& mu, const arma::mat& S) {
    arma::uword m = S.n_cols, i;
    arma::vec res(m);
    for ( i = 0; i < m; ++i ) {
        res[i] = R::rnorm(0.0, 1.0);
    }
    return arma::chol(regularize(S), "lower") * res + mu;
}

// Where [x; y] ~ N([mx; my], [Vx, Vyx; Vxy, Vy]), draw from x|y distribution
inline arma::vec condrmvnorm(const arma::vec& mx,  // mean of x
                             const arma::vec& my,  // mean of y
                             const arma::mat& Vx,  // x variance
                             const arma::mat& Vy,  // y variance
                             const arma::mat& Vxy, // covariance btw x & y
                             const arma::mat& Vyx, // covariance btw y & x
                             const arma::vec& y) { // y observations 
    using arma::chol;
    using arma::solve;
    using arma::trimatl;
    using arma::trimatu;
    arma::mat L  = chol(regularize(Vy), "lower");
    arma::vec mu = mx + Vxy * solve(trimatu(L.t()), solve(trimatl(L), y - my));
    arma::mat M  = arma::solve(arma::trimatl(L), Vyx);
    arma::mat S  = Vx - M.t() * M;
    return rmvnorm(mu, S);
}

#endif

