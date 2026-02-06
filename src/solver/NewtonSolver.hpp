#pragma once

#include <Eigen/Dense>
#include <functional>
#include <optional>
#include <stdexcept>
#include "Diagnostics.hpp"
#include "FiniteDiff.hpp"
#include "LineSearch.hpp"

namespace quantnet::solver {

// Configuration for Newton solver
struct NewtonConfig {
    double tol = 1e-10;           // Convergence tolerance on residual norm
    int max_iters = 100;          // Maximum iterations
    double fd_step = 1e-7;        // Finite difference step size
    bool central_diff = true;     // Use central (vs forward) differences
    double lambda_init = 1e-6;    // Initial Levenberg regularization
    double lambda_max = 1e6;      // Maximum regularization
    double lambda_factor = 10.0;  // Factor to increase/decrease lambda
    double armijo_c = 1e-4;       // Armijo condition parameter
    double armijo_rho = 0.5;      // Backtracking factor
    bool use_line_search = true;  // Whether to use line search
    bool verbose = false;         // Print iteration info
};

// Result of Newton solve
struct NewtonResult {
    Eigen::VectorXd x;            // Solution
    SolverTrace trace;            // Full iteration trace
    bool converged = false;
    int iterations = 0;
    double final_residual = 0.0;
};

// Newton solver for F(x) = 0
// Uses damped Newton with Levenberg regularization and Armijo line search
class NewtonSolver {
public:
    explicit NewtonSolver(NewtonConfig config = {});

    // Set callback to receive iteration updates (for telemetry)
    void set_callback(IterationCallback callback);

    // Solve F(x) = 0 starting from x0
    // F: R^n -> R^n (same dimension for square system)
    template<typename Func>
    NewtonResult solve(Func&& F, Eigen::VectorXd x0);

    // Get current configuration
    const NewtonConfig& config() const { return config_; }

    // Modify configuration
    NewtonConfig& config() { return config_; }

private:
    NewtonConfig config_;
    std::optional<IterationCallback> callback_;
};

// Implementation

inline NewtonSolver::NewtonSolver(NewtonConfig config)
    : config_(std::move(config)) {}

inline void NewtonSolver::set_callback(IterationCallback callback) {
    callback_ = std::move(callback);
}

template<typename Func>
NewtonResult NewtonSolver::solve(Func&& F, Eigen::VectorXd x0) {
    NewtonResult result;
    result.x = std::move(x0);

    const int n = static_cast<int>(result.x.size());
    double lambda = config_.lambda_init;

    // Initial residual
    Eigen::VectorXd r = F(result.x);
    if (r.size() != n) {
        throw std::runtime_error(
            "Newton solver requires square system: F: R^n -> R^n. "
            "Got input dim " + std::to_string(n) +
            ", output dim " + std::to_string(r.size()));
    }

    double residual_norm = r.norm();

    for (int iter = 0; iter < config_.max_iters; ++iter) {
        IterationStats stats;
        stats.iteration = iter;
        stats.residual_norm = residual_norm;
        stats.lambda = lambda;

        // Check convergence
        if (residual_norm < config_.tol) {
            stats.converged = true;
            stats.status = "Converged";
            result.trace.add_iteration(stats);
            if (callback_) (*callback_)(stats, result.x);

            result.converged = true;
            result.iterations = iter;
            result.final_residual = residual_norm;
            result.trace.success = true;
            result.trace.termination_reason = "Converged: residual below tolerance";
            return result;
        }

        // Compute Jacobian via finite differences
        Eigen::MatrixXd J = compute_jacobian(
            std::forward<Func>(F), result.x, config_.fd_step, config_.central_diff);

        // Estimate condition number (using SVD, expensive but informative)
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(J);
        const double cond = svd.singularValues()(0) /
            svd.singularValues()(svd.singularValues().size() - 1);
        stats.jacobian_cond = cond;

        // Solve with Levenberg regularization: (J'J + lambda*I) * d = -J' * r
        // This is equivalent to solving the damped normal equations
        Eigen::MatrixXd JtJ = J.transpose() * J;
        Eigen::VectorXd Jtr = J.transpose() * r;

        Eigen::VectorXd d;
        bool solve_success = false;

        // Try solving with increasing regularization if needed
        for (int reg_try = 0; reg_try < 10 && !solve_success; ++reg_try) {
            Eigen::MatrixXd A = JtJ + lambda * Eigen::MatrixXd::Identity(n, n);

            // Use robust solver
            Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
            if (lu.isInvertible()) {
                d = lu.solve(-Jtr);
                solve_success = true;
            } else {
                lambda *= config_.lambda_factor;
            }
        }

        if (!solve_success) {
            stats.status = "Failed: Jacobian singular";
            result.trace.add_iteration(stats);
            if (callback_) (*callback_)(stats, result.x);

            result.converged = false;
            result.iterations = iter;
            result.final_residual = residual_norm;
            result.trace.success = false;
            result.trace.termination_reason = "Failed: Jacobian singular";
            return result;
        }

        stats.step_norm = d.norm();

        // Line search
        double alpha = 1.0;
        Eigen::VectorXd x_new;
        Eigen::VectorXd r_new;
        double new_residual_norm;

        if (config_.use_line_search) {
            auto ls_result = armijo_backtrack(
                std::forward<Func>(F), result.x, d, J,
                config_.armijo_c, config_.armijo_rho);

            alpha = ls_result.alpha;
            x_new = result.x + alpha * d;
            r_new = F(x_new);
            new_residual_norm = r_new.norm();

            // Adjust regularization based on progress
            if (new_residual_norm < residual_norm) {
                lambda = std::max(config_.lambda_init, lambda / config_.lambda_factor);
            } else {
                lambda = std::min(config_.lambda_max, lambda * config_.lambda_factor);
            }
        } else {
            x_new = result.x + d;
            r_new = F(x_new);
            new_residual_norm = r_new.norm();
        }

        stats.alpha = alpha;
        stats.status = "Iteration complete";

        // Update
        result.x = x_new;
        r = r_new;
        residual_norm = new_residual_norm;

        result.trace.add_iteration(stats);
        if (callback_) (*callback_)(stats, result.x);

        if (config_.verbose) {
            std::printf("Iter %3d: ||r|| = %.6e, ||d|| = %.6e, alpha = %.4f, lambda = %.2e\n",
                iter, residual_norm, stats.step_norm, alpha, lambda);
        }
    }

    // Max iterations reached
    result.converged = false;
    result.iterations = config_.max_iters;
    result.final_residual = residual_norm;
    result.trace.success = false;
    result.trace.termination_reason = "Max iterations reached";
    return result;
}

} // namespace quantnet::solver
