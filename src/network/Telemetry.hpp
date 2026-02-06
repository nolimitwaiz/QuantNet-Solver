#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include "../solver/Diagnostics.hpp"
#include "../poker/Strategy.hpp"

namespace quantnet::network {

// Snapshot of solver state for telemetry
struct TelemetrySnapshot {
    int iteration = 0;
    double residual_norm = 0.0;
    double step_norm = 0.0;
    double alpha = 1.0;
    double lambda = 0.0;
    double beta = 1.0;
    nlohmann::json strategy;
    nlohmann::json action_evs;
    std::optional<double> exploitability;
    std::optional<double> expected_value;
    std::string game_name;

    nlohmann::json to_json() const {
        nlohmann::json j;
        j["type"] = "iteration";
        j["iteration"] = iteration;
        j["residual_norm"] = residual_norm;
        j["step_norm"] = step_norm;
        j["alpha"] = alpha;
        j["lambda"] = lambda;
        j["beta"] = beta;
        j["strategy"] = strategy;
        j["game"] = game_name;
        if (!action_evs.is_null()) {
            j["action_evs"] = action_evs;
        }
        if (exploitability.has_value()) {
            j["exploitability"] = exploitability.value();
        }
        if (expected_value.has_value()) {
            j["expected_value"] = expected_value.value();
        }
        return j;
    }

    static TelemetrySnapshot from_solver_stats(
        const quantnet::solver::IterationStats& stats,
        double beta,
        const quantnet::poker::Strategy& sigma,
        const std::string& game_name,
        std::optional<double> exploit = std::nullopt,
        std::optional<double> ev = std::nullopt,
        nlohmann::json evs = nullptr
    ) {
        TelemetrySnapshot snap;
        snap.iteration = stats.iteration;
        snap.residual_norm = stats.residual_norm;
        snap.step_norm = stats.step_norm;
        snap.alpha = stats.alpha;
        snap.lambda = stats.lambda;
        snap.beta = beta;
        snap.strategy = sigma.to_json();
        snap.game_name = game_name;
        snap.exploitability = exploit;
        snap.expected_value = ev;
        snap.action_evs = evs;
        return snap;
    }
};

} // namespace quantnet::network
