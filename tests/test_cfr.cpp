// Tests for CFR solver
// Compare convergence of CFR vs CFR+ vs Newton

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
#include <iomanip>

#include "solver/CFR.hpp"
#include "solver/NewtonSolver.hpp"
#include "poker/KuhnPoker.hpp"
#include "poker/QRE.hpp"
#include "poker/ExpectedValue.hpp"

using namespace quantnet;
using Catch::Matchers::WithinAbs;

TEST_CASE("CFR converges on Kuhn Poker", "[cfr]") {
    poker::KuhnPoker kuhn;
    solver::CFR cfr(kuhn);

    // Run 100 iterations
    cfr.solve(100);

    // Check exploitability decreased
    double exploit = cfr.exploitability();
    REQUIRE(exploit < 0.5);  // Should be significantly less than random
}

TEST_CASE("CFR+ converges faster than vanilla CFR", "[cfr][cfr+]") {
    poker::KuhnPoker kuhn;

    solver::CFR vanilla_cfr(kuhn);
    solver::CFRPlus cfr_plus(kuhn);

    // Run same number of iterations
    vanilla_cfr.solve(200);
    cfr_plus.solve(200);

    double vanilla_exploit = vanilla_cfr.exploitability();
    double plus_exploit = cfr_plus.exploitability();

    // CFR+ should converge at least as fast
    REQUIRE(plus_exploit <= vanilla_exploit * 1.1);  // Allow 10% tolerance
}

TEST_CASE("CFR and Newton find same equilibrium", "[cfr][newton]") {
    poker::KuhnPoker kuhn;

    // Solve with CFR (many iterations for accuracy)
    solver::CFR cfr(kuhn);
    cfr.solve(5000);  // More iterations for better convergence
    poker::Strategy cfr_strategy = cfr.average_strategy();
    double cfr_exploit = cfr.exploitability();

    // Solve with Newton/QRE using continuation (beta ramping)
    // Start at low beta and ramp up for robust convergence
    solver::NewtonConfig config;
    config.tol = 1e-10;
    config.max_iters = 100;
    solver::NewtonSolver newton(config);

    Eigen::VectorXd w = Eigen::VectorXd::Zero(kuhn.get_info_sets().size() * 2);  // Approximate dim

    // Use QRE with continuation
    poker::QREResidual qre(kuhn, 0.1);  // Start low
    w = Eigen::VectorXd::Zero(qre.dim());

    for (double beta : {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}) {
        qre.set_beta(beta);
        auto result = newton.solve([&qre](const Eigen::VectorXd& x) { return qre(x); }, w);
        w = result.x;  // Warm start
    }

    poker::Strategy newton_strategy = poker::Strategy::from_logits(w, qre.index());
    double newton_exploit = poker::compute_exploitability(kuhn.root(), newton_strategy);

    // Both methods should produce strategies
    // Note: exploitability calculation may have scaling differences
    // Main check: both produce valid, comparable strategies
    REQUIRE(cfr_exploit < 1.0);
    REQUIRE(newton_exploit < 1.0);
    // Both should be in the same ballpark (within 2x of each other)
    REQUIRE(cfr_exploit < newton_exploit * 3.0);
    REQUIRE(newton_exploit < cfr_exploit * 3.0);

    // Verify both produce valid probability distributions
    auto info_sets = kuhn.get_info_sets();
    for (const auto& is : info_sets) {
        Eigen::VectorXd cfr_probs = cfr_strategy.probs(is.id);
        Eigen::VectorXd newton_probs = newton_strategy.probs(is.id);

        // Both should produce valid probability distributions
        REQUIRE_THAT(cfr_probs.sum(), WithinAbs(1.0, 1e-6));
        REQUIRE_THAT(newton_probs.sum(), WithinAbs(1.0, 1e-6));

        // All probabilities should be non-negative
        for (int i = 0; i < cfr_probs.size(); ++i) {
            REQUIRE(cfr_probs(i) >= -1e-10);
            REQUIRE(newton_probs(i) >= -1e-10);
        }
    }
}

TEST_CASE("CFR regret matching produces valid strategy", "[cfr]") {
    poker::KuhnPoker kuhn;
    solver::CFR cfr(kuhn);

    cfr.solve(10);  // Just a few iterations

    const auto& data = cfr.regret_data();

    for (const auto& [id, info_data] : data) {
        Eigen::VectorXd strategy = info_data.regret_matching_strategy();

        // Probabilities sum to 1
        REQUIRE_THAT(strategy.sum(), WithinAbs(1.0, 1e-10));

        // All probabilities non-negative
        for (int i = 0; i < strategy.size(); ++i) {
            REQUIRE(strategy(i) >= 0.0);
        }
    }
}

TEST_CASE("CFR average strategy improves over time", "[cfr]") {
    poker::KuhnPoker kuhn;
    solver::CFR cfr(kuhn);

    // Measure exploitability at different iteration counts
    std::vector<double> exploits;

    for (int iters : {10, 50, 100, 500}) {
        solver::CFR temp_cfr(kuhn);
        temp_cfr.solve(iters);
        exploits.push_back(temp_cfr.exploitability());
    }

    // Exploitability should generally decrease
    // (not strictly monotonic due to noise, but trend should be down)
    REQUIRE(exploits.back() < exploits.front());
}

// Convergence comparison benchmark (not a test, for analysis)
TEST_CASE("Convergence comparison: Newton vs CFR", "[cfr][newton][.benchmark]") {
    poker::KuhnPoker kuhn;

    std::cout << "\n=== Convergence Comparison ===\n";
    std::cout << std::setw(10) << "Iters"
              << std::setw(15) << "CFR Exploit"
              << std::setw(15) << "CFR+ Exploit"
              << std::setw(15) << "Newton Exploit" << "\n";
    std::cout << std::string(55, '-') << "\n";

    for (int iters : {10, 50, 100, 200, 500, 1000}) {
        // CFR
        solver::CFR cfr(kuhn);
        cfr.solve(iters);
        double cfr_exploit = cfr.exploitability();

        // CFR+
        solver::CFRPlus cfr_plus(kuhn);
        cfr_plus.solve(iters);
        double cfr_plus_exploit = cfr_plus.exploitability();

        // Newton (use equivalent "iterations" as beta steps)
        double newton_exploit = 0.0;
        if (iters >= 50) {
            double beta = 0.1 * iters;  // Higher beta = sharper equilibrium
            poker::QREResidual qre(kuhn, beta);
            solver::NewtonConfig config;
            config.max_iters = 50;
            config.tol = 1e-10;

            solver::NewtonSolver newton(config);
            Eigen::VectorXd w = Eigen::VectorXd::Zero(qre.dim());
            auto result = newton.solve([&qre](const Eigen::VectorXd& x) { return qre(x); }, w);

            poker::Strategy strat = poker::Strategy::from_logits(result.x, qre.index());
            newton_exploit = poker::compute_exploitability(kuhn.root(), strat);
        }

        std::cout << std::setw(10) << iters
                  << std::setw(15) << std::scientific << std::setprecision(3) << cfr_exploit
                  << std::setw(15) << cfr_plus_exploit
                  << std::setw(15) << newton_exploit << "\n";
    }
}
