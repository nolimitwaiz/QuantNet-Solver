#pragma once

#include <Eigen/Dense>
#include <functional>
#include <cmath>

namespace quantnet::solver {

// Compute Jacobian matrix using finite differences
// F: R^n -> R^m
// J_ij = dF_i/dx_j
template<typename Func>
Eigen::MatrixXd compute_jacobian(
    Func&& F,
    const Eigen::VectorXd& x,
    double h = 1e-7,
    bool central = true
) {
    const int n = static_cast<int>(x.size());
    const Eigen::VectorXd f0 = F(x);
    const int m = static_cast<int>(f0.size());

    Eigen::MatrixXd J(m, n);

    if (central) {
        // Central difference: (F(x+h) - F(x-h)) / (2h)
        // More accurate: O(h^2) error vs O(h) for forward diff
        for (int j = 0; j < n; ++j) {
            Eigen::VectorXd x_plus = x;
            Eigen::VectorXd x_minus = x;
            x_plus(j) += h;
            x_minus(j) -= h;

            Eigen::VectorXd f_plus = F(x_plus);
            Eigen::VectorXd f_minus = F(x_minus);

            J.col(j) = (f_plus - f_minus) / (2.0 * h);
        }
    } else {
        // Forward difference: (F(x+h) - F(x)) / h
        for (int j = 0; j < n; ++j) {
            Eigen::VectorXd x_plus = x;
            x_plus(j) += h;

            Eigen::VectorXd f_plus = F(x_plus);
            J.col(j) = (f_plus - f0) / h;
        }
    }

    return J;
}

// Adaptive step size for finite differences based on x magnitude
inline double adaptive_fd_step(double x_j, double base_h = 1e-7) {
    const double abs_x = std::abs(x_j);
    if (abs_x > 1.0) {
        return base_h * abs_x;
    }
    return base_h;
}

// Compute Jacobian with adaptive step sizes per variable
template<typename Func>
Eigen::MatrixXd compute_jacobian_adaptive(
    Func&& F,
    const Eigen::VectorXd& x,
    double base_h = 1e-7
) {
    const int n = static_cast<int>(x.size());
    const Eigen::VectorXd f0 = F(x);
    const int m = static_cast<int>(f0.size());

    Eigen::MatrixXd J(m, n);

    for (int j = 0; j < n; ++j) {
        const double h = adaptive_fd_step(x(j), base_h);

        Eigen::VectorXd x_plus = x;
        Eigen::VectorXd x_minus = x;
        x_plus(j) += h;
        x_minus(j) -= h;

        Eigen::VectorXd f_plus = F(x_plus);
        Eigen::VectorXd f_minus = F(x_minus);

        J.col(j) = (f_plus - f_minus) / (2.0 * h);
    }

    return J;
}

} // namespace quantnet::solver
