#include "Strategy.hpp"
#include <stdexcept>

namespace quantnet::poker {

Strategy Strategy::from_logits(const Eigen::VectorXd& w, const InfoSetIndex& index) {
    Strategy s;

    for (int i = 0; i < index.num_info_sets(); ++i) {
        const InfoSet& is = index.info_set(i);
        const int num_actions = static_cast<int>(is.legal_actions.size());
        const int start = index.info_set_start(i);

        Eigen::VectorXd is_logits(num_actions);
        for (int a = 0; a < num_actions; ++a) {
            is_logits(a) = w(start + a);
        }

        s.logits_[is.id] = is_logits;
        s.actions_[is.id] = is.legal_actions;
    }

    return s;
}

Strategy Strategy::uniform(const InfoSetIndex& index) {
    // Uniform strategy: all logits = 0, which gives equal probabilities via softmax
    Eigen::VectorXd w = Eigen::VectorXd::Zero(index.total_dim());
    return from_logits(w, index);
}

Eigen::VectorXd Strategy::probs(const InfoSetId& info_set_id) const {
    auto it = logits_.find(info_set_id);
    if (it == logits_.end()) {
        throw std::runtime_error("Unknown information set: " + info_set_id);
    }
    return stable_softmax(it->second);
}

double Strategy::prob(const InfoSetId& info_set_id, Action action) const {
    auto logits_it = logits_.find(info_set_id);
    auto actions_it = actions_.find(info_set_id);

    if (logits_it == logits_.end() || actions_it == actions_.end()) {
        throw std::runtime_error("Unknown information set: " + info_set_id);
    }

    const auto& actions = actions_it->second;
    for (size_t i = 0; i < actions.size(); ++i) {
        if (actions[i] == action) {
            Eigen::VectorXd p = stable_softmax(logits_it->second);
            return p(static_cast<int>(i));
        }
    }

    throw std::runtime_error("Action not legal at information set: " + info_set_id);
}

Eigen::VectorXd Strategy::logits(const InfoSetId& info_set_id) const {
    auto it = logits_.find(info_set_id);
    if (it == logits_.end()) {
        throw std::runtime_error("Unknown information set: " + info_set_id);
    }
    return it->second;
}

Eigen::VectorXd Strategy::to_flat_logits(const InfoSetIndex& index) const {
    Eigen::VectorXd w(index.total_dim());

    for (int i = 0; i < index.num_info_sets(); ++i) {
        const InfoSet& is = index.info_set(i);
        const int num_actions = static_cast<int>(is.legal_actions.size());
        const int start = index.info_set_start(i);

        auto it = logits_.find(is.id);
        if (it == logits_.end()) {
            // Default to uniform (zero logits)
            for (int a = 0; a < num_actions; ++a) {
                w(start + a) = 0.0;
            }
        } else {
            for (int a = 0; a < num_actions; ++a) {
                w(start + a) = it->second(a);
            }
        }
    }

    return w;
}

nlohmann::json Strategy::to_json() const {
    nlohmann::json j = nlohmann::json::object();

    for (const auto& [id, logits_vec] : logits_) {
        auto actions_it = actions_.find(id);
        if (actions_it == actions_.end()) continue;

        Eigen::VectorXd p = stable_softmax(logits_vec);
        const auto& actions = actions_it->second;

        nlohmann::json is_json = nlohmann::json::object();
        for (size_t i = 0; i < actions.size(); ++i) {
            is_json[action_to_string(actions[i])] = p(static_cast<int>(i));
        }
        j[id] = is_json;
    }

    return j;
}

void Strategy::set_logits(const InfoSetId& info_set_id, const Eigen::VectorXd& new_logits) {
    logits_[info_set_id] = new_logits;
}

} // namespace quantnet::poker
