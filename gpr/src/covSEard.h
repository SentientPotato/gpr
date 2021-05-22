/* This file contains functions for the squared exponential kernel */
#ifndef GPR_COVSEISO_H
#define GPR_COVSEISO_H

/* I can include RcppArmadillo here since they use include guards */
#include <RcppArmadillo.h>

inline arma::mat covSEard(
        const arma::cube& sqd, // squared dist btwn the 2 matrices on ea dim
        const arma::vec& eta   // vector of (log) hypers log[(sy2, sf2, ell2)]
        ) {
    arma::uword n = sqd.n_rows;
    arma::uword m = sqd.n_cols;
    arma::uword d = eta.n_elem - 1;
    arma::mat res(n, m);
    double sf2 = std::abs(eta[0]);
    arma::vec ell2 = 1.0 / arma::abs(eta.tail(d));
    for ( arma::uword j = 0; j < m; ++j ) {
        for ( arma::uword i = 0; i < n; ++i ) {
            double elem = 0.0;
            for ( arma::uword k = 0; k < d; ++k ) {
                elem += std::exp(-0.5 * sqd(i, j, k) * ell2[k]);
            }
            res(i, j) = sf2 * elem;
        }
    }
    return res;
}

#endif

