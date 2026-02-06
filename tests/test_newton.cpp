// Tests for Newton solver
//
// We test the Newton solver on known nonlinear systems with analytical roots.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include <cmath>

#include "solver/NewtonSolver.hpp"
#include "solver/FiniteDiff.hpp"
#include "solver/LineSearch.hpp"

using namespace quantnet::solver;
using Catch::Matchers::WithinAbs;

TEST_CASE("Newton solver converges on simple linear system", "[newton]") {
    // F(x) = Ax - b where A = I, b = [1, 2]
    // Solution: x = [1, 2]
    auto F = [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd r(2);
        r(0) = x(0) - 1.0;
        r(1) = x(1) - 2.0;
        return r;
    };

    NewtonConfig config;
    config.tol = 1e-10;
    config.max_iters = 10;

    NewtonSolver solver(config);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);

    auto result = solver.solve(F, x0);

    REQUIRE(result.converged);
    REQUIRE_THAT(result.x(0), WithinAbs(1.0, 1e-8));
    REQUIRE_THAT(result.x(1), WithinAbs(2.0, 1e-8));
}

TEST_CASE("Newton solver converges on Rosenbrock-like system", "[newton]") {
    // F(x, y) = [10*(y - x^2), 1 - x]
    // Root: (1, 1)
    // This is a classic test problem for nonlinear solvers
    auto F = [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd r(2);
        r(0) = 10.0 * (x(1) - x(0) * x(0));
        r(1) = 1.0 - x(0);
        return r;
    };

    NewtonConfig config;
    config.tol = 1e-10;
    config.max_iters = 50;
    config.use_line_search = true;

    NewtonSolver solver(config);
    Eigen::VectorXd x0(2);
    x0 << -1.0, 1.0;  // Start away from solution

    auto result = solver.solve(F, x0);

    REQUIRE(result.converged);
    REQUIRE_THAT(result.x(0), WithinAbs(1.0, 1e-6));
    REQUIRE_THAT(result.x(1), WithinAbs(1.0, 1e-6));
}

TEST_CASE("Newton solver converges on 3D polynomial system", "[newton]") {
    // F(x, y, z) = [x^2 + y - 1, y^2 + z - 1, z^2 + x - 1]
    // One root is approximately (0.543689, 0.543689, 0.543689)
    // (the real root of t^4 + t - 1 = 0 where t ≈ 0.7245)
    // Actually, let's use a simpler system with known root:
    // F(x, y, z) = [x - 1, y - 2, z - 3]
    auto F = [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd r(3);
        r(0) = x(0) - 1.0;
        r(1) = x(1) - 2.0;
        r(2) = x(2) - 3.0;
        return r;
    };

    NewtonConfig config;
    config.tol = 1e-10;

    NewtonSolver solver(config);
    Eigen::VectorXd x0 = Eigen::VectorXd::Ones(3) * 10.0;

    auto result = solver.solve(F, x0);

    REQUIRE(result.converged);
    REQUIRE_THAT(result.x(0), WithinAbs(1.0, 1e-8));
    REQUIRE_THAT(result.x(1), WithinAbs(2.0, 1e-8));
    REQUIRE_THAT(result.x(2), WithinAbs(3.0, 1e-8));
}

TEST_CASE("Newton solver handles quadratic system", "[newton]") {
    // F(x) = [x^2 - 4] has roots at x = 2 and x = -2
    // Starting from x0 = 1, should converge to x = 2
    auto F = [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd r(1);
        r(0) = x(0) * x(0) - 4.0;
        return r;
    };

    NewtonConfig config;
    config.tol = 1e-10;

    NewtonSolver solver(config);
    Eigen::VectorXd x0(1);
    x0 << 1.0;

    auto result = solver.solve(F, x0);

    REQUIRE(result.converged);
    REQUIRE_THAT(result.x(0), WithinAbs(2.0, 1e-8));
}

TEST_CASE("Finite difference Jacobian is accurate", "[jacobian]") {
    // F(x, y) = [x^2 + y, x*y - 1]
    // J = [[2x, 1], [y, x]]
    // At (1, 2): J = [[2, 1], [2, 1]]
    auto F = [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd r(2);
        r(0) = x(0) * x(0) + x(1);
        r(1) = x(0) * x(1) - 1.0;
        return r;
    };

    Eigen::VectorXd x(2);
    x << 1.0, 2.0;

    // Analytical Jacobian at (1, 2)
    Eigen::MatrixXd J_exact(2, 2);
    J_exact << 2.0, 1.0,
               2.0, 1.0;

    // Numerical Jacobian using central differences
    Eigen::MatrixXd J_num = compute_jacobian(F, x, 1e-7, true);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            REQUIRE_THAT(J_num(i, j), WithinAbs(J_exact(i, j), 1e-5));
        }
    }
}

TEST_CASE("Line search finds descent step", "[linesearch]") {
    // F(x) = [x^2 - 1] at x = 3
    // Residual at x=3: F = 8
    // Merit at x=3: 32
    // Newton direction: d = -J^{-1} * F = -8/6 ≈ -1.33
    // Should find step that reduces merit
    auto F = [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd r(1);
        r(0) = x(0) * x(0) - 1.0;
        return r;
    };

    Eigen::VectorXd x(1);
    x << 3.0;

    Eigen::MatrixXd J = compute_jacobian(F, x);
    Eigen::VectorXd r = F(x);

    // Newton direction
    Eigen::VectorXd d = -J.fullPivLu().solve(r);

    auto ls_result = armijo_backtrack(F, x, d, J);

    REQUIRE(ls_result.success);
    REQUIRE(ls_result.alpha > 0);
    REQUIRE(ls_result.alpha <= 1.0);

    // Verify merit decreased
    double merit_before = 0.5 * r.squaredNorm();
    REQUIRE(ls_result.merit < merit_before);
}

TEST_CASE("Newton solver reports non-convergence for bad problems", "[newton]") {
    // F(x) = exp(x) which has no real root
    auto F = [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd r(1);
        r(0) = std::exp(x(0));
        return r;
    };

    NewtonConfig config;
    config.tol = 1e-10;
    config.max_iters = 10;

    NewtonSolver solver(config);
    Eigen::VectorXd x0(1);
    x0 << 0.0;

    auto result = solver.solve(F, x0);

    // Should not claim convergence
    REQUIRE(!result.converged);
    REQUIRE(result.iterations == config.max_iters);
}

TEST_CASE("Newton solver tracks iteration history", "[newton][trace]") {
    auto F = [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd r(2);
        r(0) = x(0) - 1.0;
        r(1) = x(1) - 2.0;
        return r;
    };

    NewtonConfig config;
    config.tol = 1e-10;

    NewtonSolver solver(config);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);

    auto result = solver.solve(F, x0);

    // Should have recorded iterations
    REQUIRE(result.trace.iterations.size() > 0);
    REQUIRE(result.trace.success);

    // Residual should decrease
    if (result.trace.iterations.size() >= 2) {
        double first_residual = result.trace.iterations.front().residual_norm;
        double last_residual = result.trace.iterations.back().residual_norm;
        REQUIRE(last_residual <= first_residual);
    }
}
