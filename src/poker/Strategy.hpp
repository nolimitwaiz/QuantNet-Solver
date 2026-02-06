#pragma once

#include <Eigen/Dense>
#include <map>
#include <nlohmann/json.hpp>
#include "GameTypes.hpp"

namespace quantnet::poker {

// Stable softmax computation to avoid overflow
inline Eigen::VectorXd stable_softmax(const Eigen::VectorXd& logits) {
    const double max_logit = logits.maxCoeff();
    Eigen::VectorXd shifted = logits.array() - max_logit;
    Eigen::VectorXd exp_vals = shifted.array().exp();
    return exp_vals / exp_vals.sum();
}

// Strategy profile: maps information sets to action probability distributions
//
// Internally stores unconstrained logits w, converts to probabilities via softmax.
// This parameterization keeps strategies valid (probabilities sum to 1, all >= 0)
// while allowing unconstrained optimization.
class Strategy {
public:
    Strategy() = default;

    // Create strategy from flat logits vector using index mapping
    static Strategy from_logits(const Eigen::VectorXd& w, const InfoSetIndex& index);

    // Create uniform strategy (all logits = 0)
    static Strategy uniform(const InfoSetIndex& index);

    // Get probability distribution for an information set
    Eigen::VectorXd probs(const InfoSetId& info_set_id) const;

    // Get probability of specific action at information set
    double prob(const InfoSetId& info_set_id, Action action) const;

    // Get raw logits for an information set
    Eigen::VectorXd logits(const InfoSetId& info_set_id) const;

    // Convert back to flat logits vector
    Eigen::VectorXd to_flat_logits(const InfoSetIndex& index) const;

    // Serialize to JSON for telemetry
    nlohmann::json to_json() const;

    // Set logits for an information set directly
    void set_logits(const InfoSetId& info_set_id, const Eigen::VectorXd& logits);

    // Check if info set exists in strategy
    bool has_info_set(const InfoSetId& id) const {
        return logits_.find(id) != logits_.end();
    }

    // Get all info set IDs
    std::vector<InfoSetId> info_set_ids() const {
        std::vector<InfoSetId> ids;
        for (const auto& [id, _] : logits_) {
            ids.push_back(id);
        }
        return ids;
    }

    // Get number of info sets
    size_t size() const { return logits_.size(); }

private:
    // Map from info set ID to logits vector
    // logits_[id][i] is the unconstrained parameter for action i at info set id
    std::map<InfoSetId, Eigen::VectorXd> logits_;

    // Map from info set ID to action list (for prob() lookup)
    std::map<InfoSetId, std::vector<Action>> actions_;
};

} // namespace quantnet::poker
