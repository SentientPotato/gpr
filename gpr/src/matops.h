/* This file contains various matrix operation helper functions */
#ifndef GPR_MATOPS_H
#define GPR_MATOPS_H

/* I can include RcppArmadillo here since they use include guards */
#include <RcppArmadillo.h>

inline arma::mat regularize(const arma::mat& X, const double r = 0.01) {
    arma::uword d = X.n_cols;
    arma::mat   R = arma::eye(d, d);
    R.diag() += r;
    return X + R;
}

inline arma::cube squared_distance(const arma::mat& X, const arma::mat& Y) {
    arma::uword n = X.n_rows;
    arma::uword m = Y.n_rows;
    arma::uword d = X.n_cols;
    arma::cube res(n, m, d);
    for ( arma::uword k = 0; k < d; ++k ) {
        for ( arma::uword j = 0; j < m; ++j ) {
            for ( arma::uword i = 0; i < n; ++i ) {
                double diff  = X(i, k) - Y(j, k);
                res(i, j, k) = diff * diff;
            }
        }
    }
    return res;
}

#endif
