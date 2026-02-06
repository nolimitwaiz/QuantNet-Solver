#include "CFR.hpp"
#include "../poker/ExpectedValue.hpp"
#include <chrono>
#include <iostream>

namespace quantnet::solver {

CFR::CFR(const poker::PokerGame& game) : game_(game) {
    auto info_sets = game_.get_info_sets();
    index_.build(info_sets);
    initialize();
}

void CFR::initialize() {
    info_set_data_.clear();

    for (const auto& is : index_.all_info_sets()) {
        info_set_data_[is.id] = InfoSetData(static_cast<int>(is.legal_actions.size()));
    }
}

poker::Strategy CFR::current_strategy() const {
    Eigen::VectorXd w = Eigen::VectorXd::Zero(index_.total_dim());

    for (int i = 0; i < index_.num_info_sets(); ++i) {
        const auto& is = index_.info_set(i);
        const int start = index_.info_set_start(i);

        auto it = info_set_data_.find(is.id);
        if (it != info_set_data_.end()) {
            Eigen::VectorXd probs = it->second.regret_matching_strategy();
            // Convert probs to logits (inverse softmax)
            for (int a = 0; a < static_cast<int>(is.legal_actions.size()); ++a) {
                w(start + a) = std::log(std::max(probs(a), 1e-10));
            }
        }
    }

    return poker::Strategy::from_logits(w, index_);
}

poker::Strategy CFR::average_strategy() const {
    Eigen::VectorXd w = Eigen::VectorXd::Zero(index_.total_dim());

    for (int i = 0; i < index_.num_info_sets(); ++i) {
        const auto& is = index_.info_set(i);
        const int start = index_.info_set_start(i);

        auto it = info_set_data_.find(is.id);
        if (it != info_set_data_.end()) {
            Eigen::VectorXd probs = it->second.average_strategy();
            for (int a = 0; a < static_cast<int>(is.legal_actions.size()); ++a) {
                w(start + a) = std::log(std::max(probs(a), 1e-10));
            }
        }
    }

    return poker::Strategy::from_logits(w, index_);
}

double CFR::exploitability() const {
    poker::Strategy avg = average_strategy();
    return poker::compute_exploitability(game_.root(), avg);
}

void CFR::solve(int iterations) {
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        iterations_++;

        // Alternate which player we're traversing for
        // (or traverse for both each iteration)
        for (poker::PlayerId player : {poker::PLAYER_0, poker::PLAYER_1}) {
            cfr_recursive(game_.root(), player, 1.0, 1.0, 1.0);
        }

        // Report progress
        if (callback_ && (iter % 10 == 0 || iter == iterations - 1)) {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);

            CFRStats stats;
            stats.iteration = iterations_;
            stats.exploitability = exploitability();
            stats.wall_time_ms = static_cast<double>(duration.count());

            // Compute average absolute regret
            double total_regret = 0.0;
            int count = 0;
            for (const auto& [id, data] : info_set_data_) {
                total_regret += data.cumulative_regret.cwiseAbs().sum();
                count += data.num_actions;
            }
            stats.avg_regret = total_regret / std::max(count, 1);

            (*callback_)(stats);
        }
    }
}

double CFR::cfr_recursive(
    const poker::GameNode* node,
    poker::PlayerId traverser,
    double reach_p0,
    double reach_p1,
    double reach_chance
) {
    if (!node) return 0.0;

    switch (node->type) {
        case poker::NodeType::Terminal: {
            // Return payoff for traverser
            double payoff = node->payoff;  // Payoff to P0
            if (traverser == poker::PLAYER_1) {
                payoff = -payoff;
            }
            return payoff;
        }

        case poker::NodeType::Chance: {
            // Sum over chance outcomes
            double ev = 0.0;
            for (const auto& edge : node->children) {
                ev += edge.probability * cfr_recursive(
                    edge.child.get(), traverser,
                    reach_p0, reach_p1, reach_chance * edge.probability
                );
            }
            return ev;
        }

        case poker::NodeType::Player: {
            auto& data = info_set_data_[node->info_set_id];
            const int num_actions = static_cast<int>(node->legal_actions.size());

            // Get current strategy via regret matching
            Eigen::VectorXd strategy = data.regret_matching_strategy();

            // Compute counterfactual value for each action
            Eigen::VectorXd action_values(num_actions);

            for (int a = 0; a < num_actions; ++a) {
                double new_reach_p0 = reach_p0;
                double new_reach_p1 = reach_p1;

                if (node->player == poker::PLAYER_0) {
                    new_reach_p0 *= strategy(a);
                } else {
                    new_reach_p1 *= strategy(a);
                }

                action_values(a) = cfr_recursive(
                    node->children[a].child.get(), traverser,
                    new_reach_p0, new_reach_p1, reach_chance
                );
            }

            // Expected value under current strategy
            double node_value = strategy.dot(action_values);

            // Update regrets only for the traversing player
            if (node->player == traverser) {
                // Counterfactual reach: probability of reaching this node
                // due to opponent and chance (not traverser's actions)
                double cf_reach = counterfactual_reach(traverser, reach_p0, reach_p1) * reach_chance;

                for (int a = 0; a < num_actions; ++a) {
                    // Regret = counterfactual value of action - node value
                    double regret = cf_reach * (action_values(a) - node_value);
                    data.cumulative_regret(a) += regret;
                }
            }

            // Accumulate strategy for average (weighted by player's reach)
            double player_reach = (node->player == poker::PLAYER_0) ? reach_p0 : reach_p1;
            data.cumulative_strategy += player_reach * strategy;

            return node_value;
        }
    }

    return 0.0;
}

// ============================================================================
// CFR+ Implementation
// ============================================================================

void CFRPlus::solve(int iterations) {
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        iterations_++;

        for (poker::PlayerId player : {poker::PLAYER_0, poker::PLAYER_1}) {
            cfr_plus_recursive(game_.root(), player, 1.0, 1.0, 1.0);
        }

        // CFR+ modification: floor regrets to 0 after each iteration
        for (auto& [id, data] : info_set_data_) {
            data.cumulative_regret = data.cumulative_regret.cwiseMax(0.0);
        }

        if (callback_ && (iter % 10 == 0 || iter == iterations - 1)) {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);

            CFRStats stats;
            stats.iteration = iterations_;
            stats.exploitability = exploitability();
            stats.wall_time_ms = static_cast<double>(duration.count());

            (*callback_)(stats);
        }
    }
}

double CFRPlus::cfr_plus_recursive(
    const poker::GameNode* node,
    poker::PlayerId traverser,
    double reach_p0,
    double reach_p1,
    double reach_chance
) {
    // Same as vanilla CFR, but regret flooring happens in solve()
    return cfr_recursive(node, traverser, reach_p0, reach_p1, reach_chance);
}

} // namespace quantnet::solver
