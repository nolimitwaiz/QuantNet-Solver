#pragma once

#include <Eigen/Dense>
#include <functional>
#include <cmath>
#include <limits>

namespace quantnet::solver {

// Line search result
struct LineSearchResult {
    double alpha = 1.0;    // Step size
    double merit = 0.0;    // Merit function value at x + alpha * d
    int evaluations = 0;   // Number of function evaluations
    bool success = true;   // Whether line search found valid step
};

// Merit function: phi(x) = 0.5 * ||F(x)||^2
template<typename Func>
double merit_function(Func&& F, const Eigen::VectorXd& x) {
    Eigen::VectorXd r = F(x);
    return 0.5 * r.squaredNorm();
}

// Armijo backtracking line search
// Find alpha such that phi(x + alpha*d) <= phi(x) + c * alpha * grad_phi' * d
// where grad_phi = J' * F(x)
//
// Parameters:
//   F: residual function R^n -> R^m
//   x: current point
//   d: search direction (Newton step)
//   J: Jacobian at x
//   c: Armijo parameter (typically 1e-4)
//   rho: backtracking factor (typically 0.5)
//   max_iters: maximum backtracking iterations
template<typename Func>
LineSearchResult armijo_backtrack(
    Func&& F,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& d,
    const Eigen::MatrixXd& J,
    double c = 1e-4,
    double rho = 0.5,
    int max_iters = 20
) {
    LineSearchResult result;

    // Compute merit at current point
    Eigen::VectorXd r0 = F(x);
    const double phi0 = 0.5 * r0.squaredNorm();
    result.evaluations = 1;

    // Compute directional derivative: grad_phi' * d = (J' * r)' * d = r' * J * d
    const double dphi0 = r0.dot(J * d);

    // If dphi0 >= 0, d is not a descent direction
    if (dphi0 >= 0) {
        // Try using gradient descent direction instead
        result.alpha = 0.0;
        result.merit = phi0;
        result.success = false;
        return result;
    }

    double alpha = 1.0;

    for (int i = 0; i < max_iters; ++i) {
        Eigen::VectorXd x_new = x + alpha * d;
        Eigen::VectorXd r_new = F(x_new);
        const double phi_new = 0.5 * r_new.squaredNorm();
        result.evaluations++;

        // Armijo condition: phi(x + alpha*d) <= phi(x) + c * alpha * dphi0
        if (phi_new <= phi0 + c * alpha * dphi0) {
            result.alpha = alpha;
            result.merit = phi_new;
            result.success = true;
            return result;
        }

        // Backtrack
        alpha *= rho;
    }

    // Failed to find acceptable step
    result.alpha = alpha;
    result.merit = merit_function(std::forward<Func>(F), x + alpha * d);
    result.evaluations++;
    result.success = false;
    return result;
}

// Simple backtracking: just reduce step until merit decreases
template<typename Func>
LineSearchResult simple_backtrack(
    Func&& F,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& d,
    double rho = 0.5,
    int max_iters = 20
) {
    LineSearchResult result;

    const double phi0 = merit_function(F, x);
    result.evaluations = 1;

    double alpha = 1.0;

    for (int i = 0; i < max_iters; ++i) {
        Eigen::VectorXd x_new = x + alpha * d;
        const double phi_new = merit_function(F, x_new);
        result.evaluations++;

        if (phi_new < phi0) {
            result.alpha = alpha;
            result.merit = phi_new;
            result.success = true;
            return result;
        }

        alpha *= rho;
    }

    result.alpha = alpha;
    result.merit = merit_function(std::forward<Func>(F), x + alpha * d);
    result.evaluations++;
    result.success = false;
    return result;
}

} // namespace quantnet::solver
