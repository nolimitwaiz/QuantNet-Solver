#include "ExpectedValue.hpp"
#include <limits>
#include <cmath>
#include <map>

namespace quantnet::poker {

namespace detail {

double ev_recursive(
    const GameNode* node,
    const Strategy& sigma,
    double reach_p0,
    double reach_p1,
    double reach_chance,
    const std::optional<std::pair<InfoSetId, Action>>& override_opt
) {
    if (!node) return 0.0;

    switch (node->type) {
        case NodeType::Terminal: {
            // Terminal: return reach-weighted payoff to P0
            // Full reach probability = reach_p0 * reach_p1 * reach_chance
            return reach_p0 * reach_p1 * reach_chance * node->payoff;
        }

        case NodeType::Chance: {
            // Chance node: sum over outcomes weighted by probability
            double ev = 0.0;
            for (const auto& edge : node->children) {
                ev += ev_recursive(
                    edge.child.get(), sigma,
                    reach_p0, reach_p1, reach_chance * edge.probability,
                    override_opt
                );
            }
            return ev;
        }

        case NodeType::Player: {
            // Player node: sum over actions weighted by strategy
            double ev = 0.0;

            // Check if this info set has an override
            bool is_override = override_opt.has_value() &&
                              override_opt->first == node->info_set_id;

            Eigen::VectorXd action_probs;
            if (is_override) {
                // Play override action with probability 1
                action_probs = Eigen::VectorXd::Zero(node->legal_actions.size());
                for (size_t i = 0; i < node->legal_actions.size(); ++i) {
                    if (node->legal_actions[i] == override_opt->second) {
                        action_probs(static_cast<int>(i)) = 1.0;
                        break;
                    }
                }
            } else {
                // Use strategy profile
                action_probs = sigma.probs(node->info_set_id);
            }

            for (size_t i = 0; i < node->children.size(); ++i) {
                const auto& edge = node->children[i];
                double action_prob = action_probs(static_cast<int>(i));

                double new_reach_p0 = reach_p0;
                double new_reach_p1 = reach_p1;

                if (node->player == PLAYER_0) {
                    new_reach_p0 *= action_prob;
                } else {
                    new_reach_p1 *= action_prob;
                }

                ev += ev_recursive(
                    edge.child.get(), sigma,
                    new_reach_p0, new_reach_p1, reach_chance,
                    override_opt
                );
            }
            return ev;
        }
    }

    return 0.0;
}

// Best response computation
// For the br_player, we compute the value they can achieve by playing optimally
// against the fixed strategy sigma of the opponent.
double br_recursive(
    const GameNode* node,
    const Strategy& sigma,
    PlayerId br_player,
    double reach_opponent,  // Reach prob due to opponent (and chance)
    double reach_chance
) {
    if (!node) return 0.0;

    switch (node->type) {
        case NodeType::Terminal: {
            // Return payoff for br_player, weighted by opponent reach
            double payoff = node->payoff;  // Payoff to P0
            if (br_player == PLAYER_1) {
                payoff = -payoff;  // Convert to P1's payoff (zero-sum)
            }
            return reach_opponent * reach_chance * payoff;
        }

        case NodeType::Chance: {
            double ev = 0.0;
            for (const auto& edge : node->children) {
                ev += br_recursive(
                    edge.child.get(), sigma, br_player,
                    reach_opponent, reach_chance * edge.probability
                );
            }
            return ev;
        }

        case NodeType::Player: {
            if (node->player == br_player) {
                // BR player: maximize over actions
                double best_ev = -std::numeric_limits<double>::infinity();
                for (const auto& edge : node->children) {
                    double ev = br_recursive(
                        edge.child.get(), sigma, br_player,
                        reach_opponent, reach_chance
                    );
                    best_ev = std::max(best_ev, ev);
                }
                return best_ev;
            } else {
                // Opponent: weight by their strategy
                Eigen::VectorXd probs = sigma.probs(node->info_set_id);
                double ev = 0.0;
                for (size_t i = 0; i < node->children.size(); ++i) {
                    double p = probs(static_cast<int>(i));
                    ev += br_recursive(
                        node->children[i].child.get(), sigma, br_player,
                        reach_opponent * p, reach_chance
                    );
                }
                return ev;
            }
        }
    }

    return 0.0;
}

} // namespace detail

double compute_ev(const GameNode* root, const Strategy& sigma) {
    return detail::ev_recursive(root, sigma, 1.0, 1.0, 1.0, std::nullopt);
}

double compute_ev_with_override(
    const GameNode* root,
    const Strategy& sigma,
    const InfoSetId& override_info_set,
    Action override_action
) {
    return detail::ev_recursive(
        root, sigma, 1.0, 1.0, 1.0,
        std::make_pair(override_info_set, override_action)
    );
}

double expected_utility(
    const GameNode* root,
    const Strategy& sigma,
    const InfoSetId& info_set,
    Action action,
    PlayerId acting_player
) {
    // EU(I, a) for the acting player at info set I
    //
    // We compute this by:
    // 1. Find all nodes in the game tree belonging to this info set
    // 2. For each such node, compute the expected payoff when playing 'action'
    // 3. Weight by the probability of reaching that node (from opponent + chance)
    // 4. Normalize by the total reach probability to the info set

    // For a simpler but equivalent approach:
    // EU(I, a) = EV(override I->a) but only counting the portion from info set I
    //
    // Actually, the correct formula for QRE is:
    // EU(I, a) = sum over h in I of: π_{-i}(h) * u_i(h, a, sigma_{-i})
    // where π_{-i}(h) is the counterfactual reach probability (opponent + chance)
    // and u_i(h, a, sigma_{-i}) is the expected payoff from playing a at h

    // We implement this by traversing the tree and computing counterfactual values

    // Implementation: compute expected utility by traversing tree with override
    // The result is automatically weighted by reach probabilities in the traversal

    // For player 0 at info set I:
    //   EU(I, a) = EV_with_override(I, a) / reach(I)
    // But since we're computing for QRE comparison, we need the unnormalized value

    // Let's compute it properly:
    // We need the sum over histories h in I of:
    //   π_{-i}(h) * EV(starting from h, playing a, then following sigma)

    double ev = compute_ev_with_override(root, sigma, info_set, action);

    // For the acting player, we need to convert if they're P1
    if (acting_player == PLAYER_1) {
        ev = -ev;  // Convert P0 payoff to P1 payoff (zero-sum)
    }

    return ev;
}

double best_response_value(
    const GameNode* root,
    const Strategy& sigma,
    PlayerId br_player
) {
    return detail::br_recursive(root, sigma, br_player, 1.0, 1.0);
}

double compute_exploitability(const GameNode* root, const Strategy& sigma) {
    // Exploitability = (BR_value_p0 + BR_value_p1) / 2
    // where BR_value_p is the value player p can get by best-responding

    double br0 = best_response_value(root, sigma, PLAYER_0);
    double br1 = best_response_value(root, sigma, PLAYER_1);

    // At Nash equilibrium:
    // - BR_value_p0 should equal EV for P0 under sigma
    // - BR_value_p1 should equal EV for P1 under sigma = -EV for P0
    // - BR_value_p0 + BR_value_p1 = 0

    return (br0 + br1) / 2.0;
}

} // namespace quantnet::poker
