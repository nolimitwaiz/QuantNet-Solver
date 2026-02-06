#pragma once

#include <Eigen/Dense>
#include <map>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include "../poker/GameTree.hpp"
#include "../poker/GameTypes.hpp"
#include "../poker/Strategy.hpp"

namespace quantnet::solver {

// Regret and strategy accumulator for one information set
struct InfoSetData {
    Eigen::VectorXd cumulative_regret;    // Sum of regrets over iterations
    Eigen::VectorXd cumulative_strategy;  // Sum of reach-weighted strategies
    int num_actions = 0;

    InfoSetData() = default;

    explicit InfoSetData(int n_actions)
        : cumulative_regret(Eigen::VectorXd::Zero(n_actions))
        , cumulative_strategy(Eigen::VectorXd::Zero(n_actions))
        , num_actions(n_actions) {}

    // Regret matching: convert regrets to strategy
    // σ(a) = max(R(a), 0) / Σ max(R(b), 0)
    Eigen::VectorXd regret_matching_strategy() const {
        Eigen::VectorXd positive_regret = cumulative_regret.cwiseMax(0.0);
        double sum = positive_regret.sum();

        if (sum > 0) {
            return positive_regret / sum;
        } else {
            // Uniform if no positive regrets
            return Eigen::VectorXd::Constant(num_actions, 1.0 / num_actions);
        }
    }

    // Average strategy (the Nash equilibrium approximation)
    Eigen::VectorXd average_strategy() const {
        double sum = cumulative_strategy.sum();
        if (sum > 0) {
            return cumulative_strategy / sum;
        } else {
            return Eigen::VectorXd::Constant(num_actions, 1.0 / num_actions);
        }
    }
};

// CFR iteration statistics
struct CFRStats {
    int iteration = 0;
    double exploitability = 0.0;
    double avg_regret = 0.0;
    double wall_time_ms = 0.0;
};

using CFRCallback = std::function<void(const CFRStats&)>;

// Counterfactual Regret Minimization solver
//
// CFR finds Nash equilibrium by iteratively:
// 1. Computing counterfactual values for each action
// 2. Accumulating regret for not playing each action
// 3. Using regret matching to update strategy
//
// After T iterations, average strategy converges to Nash at O(1/√T)
class CFR {
public:
    explicit CFR(const poker::PokerGame& game);

    // Run CFR for specified number of iterations
    void solve(int iterations);

    // Set callback for progress updates
    void set_callback(CFRCallback callback) { callback_ = callback; }

    // Get current strategy (regret matching)
    poker::Strategy current_strategy() const;

    // Get average strategy (Nash approximation)
    poker::Strategy average_strategy() const;

    // Get exploitability of current average strategy
    double exploitability() const;

    // Get iteration count
    int iterations() const { return iterations_; }

    // Access regret data (for analysis)
    const std::map<poker::InfoSetId, InfoSetData>& regret_data() const {
        return info_set_data_;
    }

protected:
    const poker::PokerGame& game_;
    poker::InfoSetIndex index_;
    std::map<poker::InfoSetId, InfoSetData> info_set_data_;
    int iterations_ = 0;
    std::optional<CFRCallback> callback_;

    // Initialize data structures
    void initialize();

    // Single CFR iteration for one player
    // Returns expected value for the traversing player
    double cfr_recursive(
        const poker::GameNode* node,
        poker::PlayerId traverser,
        double reach_p0,
        double reach_p1,
        double reach_chance
    );

    // Compute counterfactual reach probability
    double counterfactual_reach(poker::PlayerId player, double reach_p0, double reach_p1) const {
        return (player == poker::PLAYER_0) ? reach_p1 : reach_p0;
    }
};

// CFR+ variant with faster convergence
// Uses regret matching+ (floors negative regrets to 0 each iteration)
class CFRPlus : public CFR {
public:
    using CFR::CFR;

    void solve(int iterations);

private:
    double cfr_plus_recursive(
        const poker::GameNode* node,
        poker::PlayerId traverser,
        double reach_p0,
        double reach_p1,
        double reach_chance
    );
};

} // namespace quantnet::solver
