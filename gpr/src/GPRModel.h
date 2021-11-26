#ifndef GPR_MODEL_H
#define GPR_MODEL_H

/* Model.h requires these headers; safe to include here as they use guards */
#include <RcppArmadillo.h>
#include "covSEard.h"
#include "matops.h"
#include "mvnorm.h"
#include "igamma.h"
#include "tuning.h"

class GPRModel {
    public:
        /* :----- Attributes  -----: */
            /* Data */
            arma::vec y; // outcomes
            arma::mat X; // predictors
            arma::mat H; // design matrix
            /* Parameters */
            arma::vec f;    // de-meaned function outputs
            arma::vec beta; // mean function coefficients
            arma::vec eta;  // covariance function hypers, (sf2, ell)
            double sy2;     // likelihood function hyper
            /* Hyper-parameters for priors */
            arma::vec z;  // vector of zeros of length n (f prior mean)
            arma::vec b;  // beta prior mean
            arma::mat B;  // beta prior covariance
            arma::vec e;  // eta prior mean
            arma::mat E;  // eta prior covariance
            double shape; // shape parameter for sy2 prior
            double scale; // scale parameter for sy2 prior
            /* Derived, bookkeeping, and convenience quantities */
            arma::uword n;    // number of observations
            arma::uword d;    // number of predictors
            arma::uword p;    // number of parameters
            arma::mat H_beta; // mean function evaluation
            arma::mat K;      // covariance function evaluation
            arma::mat I;      // identity matrix of size n x n
            arma::mat sy2I;   // sy2 * I (needed for likelihood evaluation)
            arma::cube sqd;   // squared distance btwn obs for each predictor
            arma::mat KKyi;   // = K * arma::inv(K + sy2I)
            arma::vec Hb;     // = H * b
            arma::mat HBHT;   // = H * B * H.t()
            arma::mat HB;     // = H * B
            arma::mat BHT;    // = HB.t()
            arma::mat C;      // proposal covariance for h

        /* :----- Constructor  -----: */
            GPRModel(
                    const arma::vec& y_, // outcomes
                    const arma::mat& X_, // predictors
                    const arma::vec& b_, // beta prior mean
                    const arma::mat& B_, // beta prior covariance
                    const arma::vec& e_, // eta prior mean
                    const arma::mat& E_, // eta prior covariance
                    const double shape_,
                    const double scale_
                    ) {
                /* Store data and prior hypers */
                y = y_;
                X = X_;
                b = b_;
                B = B_;
                e = e_;
                E = E_;
                shape = shape_;
                scale = scale_;
                /* Set bookkeeping variables */
                n = X.n_rows;
                d = X.n_cols;
                p = n + 2 * d + 3;
                /* Add intercept for design matrix */
                H = arma::join_horiz(arma::ones(n), X);
                /* Calculate quantities derived from H, b, and B */
                Hb = H * b;
                HBHT = H * B * H.t();
                HB = H * B;
                BHT = HB.t();
                /* Draw initial covariance function parameters */
                eta = rmvnorm(e, E);
                /* Set likelihood covariance */
                sy2 = rinvgamma(shape, scale);
                I = arma::eye(n, n);
                sy2I = sy2 * I;
                /* Calculate initial kernel and quantities derived from it */
                sqd = squared_distance(X, X);
                K = covSEard(sqd, eta);
                KKyi = K * arma::inv(K + sy2I);
                /* Draw initial f values */
                z = arma::zeros(n);
                f = rmvnorm(z, K);
                /* Draw initial beta values and calculate H * beta */
                beta = rmvnorm(b, B);
                H_beta = H * beta;
                /* Store identity matrix for initial C value */
                C = arma::eye(d+1, d+1);
            }

        /* :----- Methods  -----: */
            /* Parameter updating */
            void update_beta() {
                // carry out a calculation common to posterior mean & cov
                arma::mat tmp = BHT * arma::inv(sy2I + HBHT);
                // draw from the full conditional
                beta = rmvnorm(b + tmp * (y - f - Hb), B - tmp * HB);
                // update derived quantity H_beta
                H_beta = H * beta;
            }
            void update_f() {
                // draw from full conditional
                f = rmvnorm(KKyi * (y - H_beta), K - KKyi * K);
            }
            void update_sy2() {
                arma::vec tmp = y - f - H_beta;
                double post_shape = shape + 0.5 * (double)n;
                double post_scale = scale + 0.5 * arma::dot(tmp, tmp);
                sy2  = rinvgamma(post_shape, post_scale);
                sy2I = sy2 * I;
            }
            /* notice update_eta() returns 1 if the proposal is accepted
             * and 0 if it is not; this is useful for proposal tuning
             */
            int update_eta() {
                // draw a proposed eta from normal centered on eta w cov C
                arma::vec eta_star = rmvnorm(eta, C);
                // determine acceptance probability r
                double r  = dmvnorm(f, z, covSEard(sqd, eta_star));
                       r -= dmvnorm(f, z, K);
                       r += dmvnorm(eta_star, e, E);
                       r -= dmvnorm(eta, e, E);
                // draw a uniform random variate & if it's < r, accept proposal
                if ( std::log(R::runif(0.0, 1.0)) < r ) {
                    eta = eta_star;
                    K = covSEard(sqd, eta);
                    KKyi = K * arma::inv(K + sy2I);
                    return 1;
                }
                return 0;
            }
            /* Proposal tuning */
            void tune_proposals(
                    const arma::uword minloops,
                    const arma::uword maxloops,
                    const arma::uword ntune,
                    const double target_rate,
                    const double rate_tol,
                    double prop_scale,
                    const double weight,
                    const bool verbose
                    ) {
                // Create object to store parameter draws
                // (for updating proposal covariance)
                arma::mat param_draws(ntune, d+1);
                // Create object for *unscaled* proposal covariance
                arma::mat S = C;
                C *= prop_scale;
                // Keep track of progress
                double prog = 0.0;
                double incr = 0.0;
                if ( verbose ) {
                    incr = 1000.0 / (double)ntune;
                } else {
                    incr = 1000.0 / (maxloops * (double)ntune);
                }
                double accrate = 0.0;
                int acceptances = 0;
                bool stop_tuning = false;
                int loops_w_hi_rates = 0; // for detecting problems
                /* Warm up beta */
                f.fill(0);
                sy2I = I;
                for ( arma::uword s = 0; s < 1000; ++s ) {
                    Rprintf("\rWarming up beta: %i / 1000", s+1);
                    update_beta();
                }
                for ( arma::uword i = 0; i < maxloops; ++i ) {
                    if ( stop_tuning ) {
                        break;
                    }
                    acceptances = 0;
                    for ( arma::uword s = 0; s < ntune; ++s ) {
                        // Periodically update progress and check for user interrupt
                        if ( ((s+1) % 10) == 0 ) {
                            prog += incr;
                            if ( verbose ) {
                                Rprintf("\rTuning loop %2i: ", i+1);
                                Rprintf("%6.2f%% complete", prog);
                            } else {
                                Rprintf("\rTuning:   %6.2f%% complete", prog);
                            }
                            Rcpp::checkUserInterrupt();
                        }
                        // Update parameters
                        // (incrementing acceptances as necessary)
                        update_beta();
                        update_f();
                        update_sy2();
                        acceptances += update_eta();
                        // Store h draw
                        param_draws.row(s) = eta.t();
                    }
                    // Check for high acceptance rate
                    accrate = acceptances / (double)ntune;
                    if ( accrate > 0.75 ) {
                        loops_w_hi_rates += 1;
                    } else {
                        loops_w_hi_rates  = 0;
                    }
                    // Check for stopping condition
                    if ( accrate < (target_rate + rate_tol)
                         && accrate > (target_rate - rate_tol)
                         && (i + 1) >= minloops
                       ) {
                        stop_tuning = true;
                        continue;
                    }
                    // If condition not met, update proposal dist parameters
                    if ( (loops_w_hi_rates > 1) & ((maxloops - i) > 1) ) {
                        /* While we are still tuning the proposal density,
                         * it's possible to end up in a bad equilibrium
                         * where we're accepting too many upward proposals.
                         * We have to snap out of that feedback loop if it occurs.
                         */
                        loops_w_hi_rates = 0;
                        eta = arma::zeros(d+1);
                        prop_scale = -2.0 * phi_inv(target_rate * 0.5);
                        C = prop_scale * arma::eye(d+1, d+1);
                        if ( verbose ) {
                            Rprintf("\n    Divergence detected; resetting\n\n");
                        }
                    } else {
                        // Otherwise, we update as normal
                        // (note we protect against dividing by 0 or -Inf here)
                        double minrate = 1.0 / (double)ntune;
                        double maxrate = (ntune - 1) / (double)ntune;
                        accrate = std::min(maxrate, std::max(minrate, accrate));
                        prop_scale = update_step_size(prop_scale, accrate, target_rate);
                        S = weight * arma::cov(param_draws) + (1 - weight) * S;
                        C = prop_scale * S;
                        if ( verbose ) {
                            prog = 0.0;
                            Rprintf("\rTuning loop %2i: ", i+1);
                            Rprintf("%0.2f accept rate\n", accrate);
                            Rprintf("%0.2f scale, ", prop_scale);
                            Rcpp::Rcout << "C =\n" << C
                                        << "eta = " << arma::abs(eta.t())
                                        << "\n"
                                        << "beta = " << beta.t()
                                        << "\n";
                        }
                    }
                }
                if ( verbose ) {
                    Rcpp::Rcout << "\nUsing proposal covariance\n" << C << "\n";
                } else {
                    Rprintf("\rTuning:   100.00%% complete\n");
                }
            }
};

class GPRZModel {
    public:
        /* :----- Attributes  -----: */
            /* Data */
            arma::vec y; // outcomes
            arma::mat X; // predictors
            /* Parameters */
            arma::vec f;    // de-meaned function outputs
            arma::vec eta;  // covariance function hypers, (sf2, ell)
            double sy2;     // likelihood function hyper
            /* Hyper-parameters for priors */
            arma::vec z;  // vector of zeros of length n (f prior mean)
            arma::vec e;  // eta prior mean
            arma::mat E;  // eta prior covariance
            double shape; // shape parameter for sy2 prior
            double scale; // scale parameter for sy2 prior
            /* Derived, bookkeeping, and convenience quantities */
            arma::uword n;    // number of observations
            arma::uword d;    // number of predictors
            arma::uword p;    // number of parameters
            arma::mat K;      // covariance function evaluation
            arma::mat I;      // identity matrix of size n x n
            arma::mat sy2I;   // sy2 * I (needed for likelihood evaluation)
            arma::cube sqd;   // squared distance btwn obs for each predictor
            arma::mat KKyi;   // = K * arma::inv(K + sy2I)
            arma::mat C;      // proposal covariance for eta

        /* :----- Constructor  -----: */
            GPRZModel(
                    const arma::vec& y_, // outcomes
                    const arma::mat& X_, // predictors
                    const arma::vec& e_, // eta prior mean
                    const arma::mat& E_, // eta prior covariance
                    const double shape_,
                    const double scale_
                    ) {
                /* Store data and prior hypers */
                y = y_;
                X = X_;
                e = e_;
                E = E_;
                shape = shape_;
                scale = scale_;
                /* Set bookkeeping variables */
                n = X.n_rows;
                d = X.n_cols;
                p = n + d + 2;
                /* Draw initial covariance function parameters */
                eta = rmvnorm(e, E);
                /* Set likelihood covariance */
                sy2 = rinvgamma(shape, scale);
                I = arma::eye(n, n);
                sy2I = sy2 * I;
                /* Calculate initial kernel and quantities derived from it */
                sqd = squared_distance(X, X);
                K = covSEard(sqd, eta);
                KKyi = K * arma::inv(K + sy2I);
                /* Draw initial f values */
                z = arma::zeros(n);
                f = rmvnorm(z, K);
                /* Store identity matrix for initial C value */
                C = arma::eye(d+1, d+1);
            }

        /* :----- Methods  -----: */
            /* Parameter updating */
            void update_f() {
                // draw from full conditional
                f = rmvnorm(KKyi * y, K - KKyi * K);
            }
            void update_sy2() {
                arma::vec tmp = y - f;
                double post_shape = shape + 0.5 * (double)n;
                double post_scale = scale + 0.5 * arma::dot(tmp, tmp);
                sy2  = rinvgamma(post_shape, post_scale);
                sy2I = sy2 * I;
            }
            /* notice update_eta() returns 1 if the proposal is accepted
             * and 0 if it is not; this is useful for proposal tuning
             */
            int update_eta() {
                // draw a proposed eta from normal centered on eta w cov C
                arma::vec eta_star = rmvnorm(eta, C);
                // determine acceptance probability r
                double r  = dmvnorm(f, z, covSEard(sqd, eta_star));
                       r -= dmvnorm(f, z, K);
                       r += dmvnorm(eta_star, e, E);
                       r -= dmvnorm(eta, e, E);
                // draw a uniform random variate & if it's < r, accept proposal
                if ( std::log(R::runif(0.0, 1.0)) < r ) {
                    eta = eta_star;
                    K = covSEard(sqd, eta);
                    KKyi = K * arma::inv(K + sy2I);
                    return 1;
                }
                return 0;
            }
            /* Proposal tuning */
            void tune_proposals(
                    const arma::uword minloops,
                    const arma::uword maxloops,
                    const arma::uword ntune,
                    const double target_rate,
                    const double rate_tol,
                    double prop_scale,
                    const double weight,
                    const bool verbose
                    ) {
                // Create object to store parameter draws
                // (for updating proposal covariance)
                arma::mat param_draws(ntune, d+1);
                // Create object for *unscaled* proposal covariance
                arma::mat S = C;
                C *= prop_scale;
                // Keep track of progress
                double prog = 0.0;
                double incr = 0.0;
                if ( verbose ) {
                    incr = 1000.0 / (double)ntune;
                } else {
                    incr = 1000.0 / (maxloops * (double)ntune);
                }
                double accrate = 0.0;
                int acceptances = 0;
                bool stop_tuning = false;
                int loops_w_hi_rates = 0; // for detecting problems
                for ( arma::uword i = 0; i < maxloops; ++i ) {
                    if ( stop_tuning ) {
                        break;
                    }
                    acceptances = 0;
                    for ( arma::uword s = 0; s < ntune; ++s ) {
                        // Periodically update progress and check for user interrupt
                        if ( ((s+1) % 10) == 0 ) {
                            prog += incr;
                            if ( verbose ) {
                                Rprintf("\rTuning loop %2i: ", i+1);
                                Rprintf("%6.2f%% complete", prog);
                            } else {
                                Rprintf("\rTuning:   %6.2f%% complete", prog);
                            }
                            Rcpp::checkUserInterrupt();
                        }
                        // Update parameters
                        // (incrementing acceptances as necessary)
                        update_f();
                        update_sy2();
                        acceptances += update_eta();
                        // Store h draw
                        param_draws.row(s) = eta.t();
                    }
                    // Check for high acceptance rate
                    accrate = acceptances / (double)ntune;
                    if ( accrate > 0.75 ) {
                        loops_w_hi_rates += 1;
                    } else {
                        loops_w_hi_rates  = 0;
                    }
                    // Check for stopping condition
                    if ( accrate < (target_rate + rate_tol)
                         && accrate > (target_rate - rate_tol)
                         && (i + 1) >= minloops
                       ) {
                        stop_tuning = true;
                        continue;
                    }
                    // If condition not met, update proposal dist parameters
                    if ( (loops_w_hi_rates > 1) & ((maxloops - i) > 1) ) {
                        /* While we are still tuning the proposal density,
                         * it's possible to end up in a bad equilibrium
                         * where we're accepting too many upward proposals.
                         * We have to snap out of that feedback loop if it occurs.
                         */
                        loops_w_hi_rates = 0;
                        eta = arma::zeros(d+1);
                        prop_scale = -2.0 * phi_inv(target_rate * 0.5);
                        C = prop_scale * arma::eye(d+1, d+1);
                        if ( verbose ) {
                            Rprintf("\n    Divergence detected; resetting\n\n");
                        }
                    } else {
                        // Otherwise, we update as normal
                        // (note we protect against dividing by 0 or -Inf here)
                        double minrate = 1.0 / (double)ntune;
                        double maxrate = (ntune - 1) / (double)ntune;
                        accrate = std::min(maxrate, std::max(minrate, accrate));
                        prop_scale = update_step_size(prop_scale, accrate, target_rate);
                        S = weight * arma::cov(param_draws) + (1 - weight) * S;
                        C = prop_scale * S;
                        if ( verbose ) {
                            prog = 0.0;
                            Rprintf("\rTuning loop %2i: ", i+1);
                            Rprintf("%0.2f accept rate\n", accrate);
                            Rprintf("%0.2f scale, ", prop_scale);
                            Rcpp::Rcout << "C =\n" << C
                                        << "eta = " << arma::abs(eta.t())
                                        << "\n";
                        }
                    }
                }
                if ( verbose ) {
                    Rcpp::Rcout << "\nUsing proposal covariance\n" << C << "\n";
                } else {
                    Rprintf("\rTuning:   100.00%% complete\n");
                }
            }
};

#endif
