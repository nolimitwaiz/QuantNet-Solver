#include "QRE.hpp"
#include <cmath>
#include <algorithm>

namespace quantnet::poker {

QREResidual::QREResidual(const PokerGame& game, double beta)
    : game_(game), beta_(beta)
{
    // Build info set index
    std::vector<InfoSet> info_sets = game_.get_info_sets();
    index_.build(info_sets);
}

std::map<InfoSetId, std::map<Action, double>> compute_all_expected_utilities(
    const PokerGame& game,
    const Strategy& sigma,
    const InfoSetIndex& index
) {
    std::map<InfoSetId, std::map<Action, double>> result;

    for (int i = 0; i < index.num_info_sets(); ++i) {
        const InfoSet& is = index.info_set(i);
        std::map<Action, double> action_eu;

        for (Action a : is.legal_actions) {
            double eu = expected_utility(game.root(), sigma, is.id, a, is.player);
            action_eu[a] = eu;
        }

        result[is.id] = action_eu;
    }

    return result;
}

Eigen::VectorXd QREResidual::logit_best_response(const Strategy& sigma) const {
    // Compute expected utilities for all actions at all info sets
    auto all_eu = compute_all_expected_utilities(game_, sigma, index_);

    // Convert to logit best response probabilities
    Eigen::VectorXd br(index_.total_dim());

    for (int i = 0; i < index_.num_info_sets(); ++i) {
        const InfoSet& is = index_.info_set(i);
        const int num_actions = static_cast<int>(is.legal_actions.size());
        const int start = index_.info_set_start(i);

        const auto& action_eu = all_eu[is.id];

        // Compute logit response: p(a) = exp(beta * EU(a)) / Z
        // Use stable softmax by subtracting max
        Eigen::VectorXd scaled_eu(num_actions);
        for (int a = 0; a < num_actions; ++a) {
            scaled_eu(a) = beta_ * action_eu.at(is.legal_actions[a]);
        }

        // Stable softmax
        double max_eu = scaled_eu.maxCoeff();
        Eigen::VectorXd exp_eu = (scaled_eu.array() - max_eu).exp();
        double Z = exp_eu.sum();

        for (int a = 0; a < num_actions; ++a) {
            br(start + a) = exp_eu(a) / Z;
        }
    }

    return br;
}

Eigen::VectorXd QREResidual::operator()(const Eigen::VectorXd& w) const {
    // Convert logits to strategy
    Strategy sigma = Strategy::from_logits(w, index_);

    // Compute logit best response
    Eigen::VectorXd br = logit_best_response(sigma);

    // Compute current strategy probabilities as flat vector
    Eigen::VectorXd sigma_flat(index_.total_dim());
    for (int i = 0; i < index_.num_info_sets(); ++i) {
        const InfoSet& is = index_.info_set(i);
        const int start = index_.info_set_start(i);
        Eigen::VectorXd probs = sigma.probs(is.id);

        for (int a = 0; a < static_cast<int>(is.legal_actions.size()); ++a) {
            sigma_flat(start + a) = probs(a);
        }
    }

    // Residual: sigma - BR
    return sigma_flat - br;
}

} // namespace quantnet::poker
