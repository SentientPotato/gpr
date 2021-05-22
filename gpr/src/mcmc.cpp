/* We don't want warnings about computational singularity being printed to
 * the console, particularly during the early stages of proposal tuning
 * where it can happen a lot, so we tell Armadillo not to print them
 */
#define ARMA_DONT_PRINT_ERRORS

/* Now we include the header needed for the model */
#include "GPRModel.h"

// [[Rcpp::export(.gprMCMC)]]
arma::mat gprMCMC(
        const arma::vec& y,         // outcomes
        const arma::mat& X,         // predictors
        const arma::uword nburn,    // number of burn-in iterations
        const arma::uword nsample,  // number of posterior samples to draw
        const arma::vec& b,         // prior mean for beta
        const arma::mat& B,         // prior cov for beta
        const arma::vec& e,         // prior mean for eta
        const arma::mat& E,         // prior cov for eta
        const double shape,
        const double scale,
        const arma::uword minloops, // min tuning loops
        const arma::uword maxloops, // max tuning loops
        const arma::uword ntune,    // iters per tuning loop
        const double target_rate,   // target accrate
        const double rate_tol,      // accrate tolerance
        double prop_scale,          // scaling constant for proposal cov
        const double weight,        // weight on obs cov when tuning proposals
        const bool verbose          // print tuning info?
        ) {

    /* Setup model and results storage */
    GPRModel mod(y, X, b, B, e, E, shape, scale);
    arma::mat res(nsample, mod.p);

    /* Tune proposals for h */
    mod.tune_proposals(minloops, maxloops, ntune, target_rate, rate_tol,
            prop_scale, weight, verbose);

    /* Run burn-in */
    for ( arma::uword iter = 0; iter < nburn; ++iter ) {
        Rcpp::Rcout << "\rBurn-in:  " << iter + 1 << " / " << nburn;
        mod.update_beta();
        mod.update_f();
        mod.update_sy2();
        mod.update_eta();
    }
    Rcpp::Rcout << "\n";

    /* Draw posterior samples */
    for ( arma::uword iter = 0; iter < nsample; ++iter ) {
        Rcpp::Rcout << "\rSampling: " << iter + 1 << " / " << nsample;
        mod.update_beta();
        mod.update_f();
        mod.update_sy2();
        mod.update_eta();
        arma::vec params(mod.p);
        params.head(mod.p - 1) = arma::join_vert(mod.f, mod.beta, mod.eta);
        params[mod.p - 1] = mod.sy2;
        res.row(iter) = params.t();
    }
    Rcpp::Rcout << "\n";

    /* return results */
    return res;
}

// [[Rcpp::export(.gprMCMCZ)]]
arma::mat gprMCMCZ(
        const arma::vec& y,         // outcomes
        const arma::mat& X,         // predictors
        const arma::uword nburn,    // number of burn-in iterations
        const arma::uword nsample,  // number of posterior samples to draw
        const arma::vec& e,         // prior mean for eta
        const arma::mat& E,         // prior cov for eta
        const double shape,
        const double scale,
        const arma::uword minloops, // min tuning loops
        const arma::uword maxloops, // max tuning loops
        const arma::uword ntune,    // iters per tuning loop
        const double target_rate,   // target accrate
        const double rate_tol,      // accrate tolerance
        double prop_scale,          // scaling constant for proposal cov
        const double weight,        // weight on obs cov when tuning proposals
        const bool verbose          // print tuning info?
        ) {

    /* Setup model and results storage */
    GPRZModel mod(y, X, e, E, shape, scale);
    arma::mat res(nsample, mod.p);

    /* Tune proposals for h */
    mod.tune_proposals(minloops, maxloops, ntune, target_rate, rate_tol,
            prop_scale, weight, verbose);

    /* Run burn-in */
    for ( arma::uword iter = 0; iter < nburn; ++iter ) {
        Rcpp::Rcout << "\rBurn-in:  " << iter + 1 << " / " << nburn;
        mod.update_f();
        mod.update_sy2();
        mod.update_eta();
    }
    Rcpp::Rcout << "\n";

    /* Draw posterior samples */
    for ( arma::uword iter = 0; iter < nsample; ++iter ) {
        Rcpp::Rcout << "\rSampling: " << iter + 1 << " / " << nsample;
        mod.update_f();
        mod.update_sy2();
        mod.update_eta();
        arma::vec params(mod.p);
        params.head(mod.p - 1) = arma::join_vert(mod.f, mod.eta);
        params[mod.p - 1] = mod.sy2;
        res.row(iter) = params.t();
    }
    Rcpp::Rcout << "\n";

    /* return results */
    return res;
}

