#pragma once

#include <vector>
#include <string>
#include <optional>
#include <functional>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

namespace quantnet::solver {

// Statistics for a single Newton iteration
struct IterationStats {
    int iteration = 0;
    double residual_norm = 0.0;
    double step_norm = 0.0;
    double alpha = 1.0;           // Line search step size
    double lambda = 0.0;          // Levenberg regularization parameter
    double jacobian_cond = 0.0;   // Condition number of Jacobian
    bool converged = false;
    std::string status;           // Description of iteration outcome

    nlohmann::json to_json() const {
        return {
            {"iteration", iteration},
            {"residual_norm", residual_norm},
            {"step_norm", step_norm},
            {"alpha", alpha},
            {"lambda", lambda},
            {"jacobian_cond", jacobian_cond},
            {"converged", converged},
            {"status", status}
        };
    }
};

// Full trace of solver execution
struct SolverTrace {
    std::vector<IterationStats> iterations;
    bool success = false;
    int total_iterations = 0;
    double final_residual = 0.0;
    std::string termination_reason;

    void add_iteration(const IterationStats& stats) {
        iterations.push_back(stats);
        total_iterations = static_cast<int>(iterations.size());
        final_residual = stats.residual_norm;
    }

    nlohmann::json to_json() const {
        nlohmann::json j;
        j["success"] = success;
        j["total_iterations"] = total_iterations;
        j["final_residual"] = final_residual;
        j["termination_reason"] = termination_reason;
        j["iterations"] = nlohmann::json::array();
        for (const auto& it : iterations) {
            j["iterations"].push_back(it.to_json());
        }
        return j;
    }
};

// Callback type for iteration updates
// Receives iteration stats and the current solution vector
using IterationCallback = std::function<void(const IterationStats&, const Eigen::VectorXd&)>;

} // namespace quantnet::solver
