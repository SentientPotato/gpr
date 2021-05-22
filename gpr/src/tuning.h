#ifndef GPR_TUNING_H
#define GPR_TUNING_H

#include <Rcpp.h> // OK since I know they use include guards

inline double phi_inv(double p) {
    return R::qnorm(p, 0.0, 1.0, 1, 0);
}

inline double update_step_size(double delta, double current, double target) {
    return ((delta * phi_inv(target * 0.5)) / phi_inv(current * 0.5));
}

#endif
