#define ARMA_DONT_PRINT_ERRORS
#include <RcppArmadillo.h>
#include "covSEard.h"
#include "matops.h"
#include "mvnorm.h"


// [[Rcpp::export(.gpr_predict)]]
arma::mat gpr_predict(
        const arma::mat& P, // posterior samples
        const arma::vec& y, // training outcomes
        const arma::mat& X, // training predictors
        const arma::mat& Z  // testing predictors
        ) {
    /* Set up bookkeeping variables */
    arma::uword n = y.n_elem;
    arma::uword m = Z.n_rows;
    arma::uword d = X.n_cols;
    arma::uword S = P.n_rows;
    /* Generate some matrices that do not change */
    arma::mat   I = arma::eye(n, n);
    arma::mat   H = arma::join_horiz(arma::ones(n), X);
    arma::mat   G = arma::join_horiz(arma::ones(m), Z);
    arma::cube dX = squared_distance(X, X);
    arma::cube dZ = squared_distance(Z, Z);
    arma::cube dd = squared_distance(Z, X);
    arma::vec  zX = arma::zeros(n);
    arma::vec  zZ = arma::zeros(m);
    /* Set up objects to store data that does change */
    arma::mat K(n, n);
    arma::mat Ks(m, n);
    arma::mat Kss(m, m);
    arma::vec f(n);
    arma::vec beta(d+1);
    arma::vec eta(d+1);
    arma::vec muX(n);
    arma::vec muZ(m);
    arma::vec Gbeta(m);
    arma::mat KsKyi(m, n);
    arma::mat res(S, m);
    /* For every posterior sample... */
    for ( arma::uword s = 0; s < S; ++s ) {
        /* Update progress */
        Rcpp::Rcout << "\rDrawing predictions: " << s+1 << " / " << S;
        /* Store this sample's parameters */
        for ( arma::uword i = 0; i < n; ++i ) {
            f[i] = P(s, i);
        }
        beta[0] = P(s, n);
        eta[0]  = P(s, n+d+1);
        for ( arma::uword k = 0; k < d; ++k ) {
            beta[k+1] = P(s, n+k+1);
            eta[k+1]  = P(s, n+d+k+2);
        }
        /* Evaluate covariance function */
        K   = covSEard(dX, eta);
        Ks  = covSEard(dd, eta);
        Kss = covSEard(dZ, eta);
        /* Calculate Ks * Ky.i() */
        // KsKyi = Ks * arma::inv(K + (std::abs(eta[0]) * I));
        /* Evaluate mean function */
        // muX = H * beta;
        // muZ = G * beta;
        Gbeta = G * beta;
        /* Draw and store prediction */
        // arma::vec pred = rmvnorm(muZ + KsKyi * (y-f-muX), Kss - KsKyi * Ks.t());
        arma::vec pred = Gbeta + condrmvnorm(zZ, zX, Kss, K, Ks, Ks.t(), f);
        res.row(s) = pred.t();
    }
    /* Close out progress display and return results */
    Rcpp::Rcout << "\n";
    return res;
}

// [[Rcpp::export(.gpr_predictZ)]]
arma::mat gpr_predictZ(
        const arma::mat& P, // posterior samples
        const arma::vec& y, // training outcomes
        const arma::mat& X, // training predictors
        const arma::mat& Z  // testing predictors
        ) {
    /* Set up bookkeeping variables */
    arma::uword n = y.n_elem;
    arma::uword m = Z.n_rows;
    arma::uword d = X.n_cols;
    arma::uword S = P.n_rows;
    /* Generate some matrices that do not change */
    arma::mat   I = arma::eye(n, n);
    arma::cube dX = squared_distance(X, X);
    arma::cube dZ = squared_distance(Z, Z);
    arma::cube dd = squared_distance(Z, X);
    arma::vec  zX = arma::zeros(n);
    arma::vec  zZ = arma::zeros(m);
    /* Set up objects to store data that does change */
    arma::mat K(n, n);
    arma::mat Ks(m, n);
    arma::mat Kss(m, m);
    arma::vec f(n);
    arma::vec eta(d+1);
    arma::mat res(S, m);
    /* For every posterior sample... */
    for ( arma::uword s = 0; s < S; ++s ) {
        /* Update progress */
        Rcpp::Rcout << "\rDrawing predictions: " << s+1 << " / " << S;
        /* Store this sample's parameters */
        for ( arma::uword i = 0; i < n; ++i ) {
            f[i] = P(s, i);
        }
        eta[0] = P(s, n);
        for ( arma::uword k = 0; k < d; ++k ) {
            eta[k+1] = P(s, n+k+1);
        }
        /* Evaluate covariance function */
        K   = covSEard(dX, eta);
        Ks  = covSEard(dd, eta);
        Kss = covSEard(dZ, eta);
        /* Draw and store prediction */
        arma::vec pred = condrmvnorm(zZ, zX, Kss, K, Ks, Ks.t(), f);
        res.row(s) = pred.t();
    }
    /* Close out progress display and return results */
    Rcpp::Rcout << "\n";
    return res;
}

