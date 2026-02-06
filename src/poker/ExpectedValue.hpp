#pragma once

#include <optional>
#include "GameTree.hpp"
#include "Strategy.hpp"

namespace quantnet::poker {

// Compute expected value for Player 0 under strategy profile sigma
// Uses tree traversal with reach probabilities
double compute_ev(const GameNode* root, const Strategy& sigma);

// Compute expected value for Player 0 when at info_set, the acting player
// plays action 'override_action' with probability 1 (deterministically),
// and all other decisions follow sigma.
//
// This is used to compute EU(I, a) for QRE:
//   EU(I, a) = EV when player at I plays action a deterministically
double compute_ev_with_override(
    const GameNode* root,
    const Strategy& sigma,
    const InfoSetId& override_info_set,
    Action override_action
);

// Compute expected utility of action 'a' at information set 'info_set'
// EU(I, a) = expected payoff to the acting player when they play 'a' at I
//            and all other decisions (including their own at other info sets)
//            follow sigma.
//
// For player P at info set I:
//   EU(I, a) = sum over histories h in I of:
//              P(reaching h) * EV(play a at h, then follow sigma)
//
// The reach probability accounts for opponent's strategy and chance.
double expected_utility(
    const GameNode* root,
    const Strategy& sigma,
    const InfoSetId& info_set,
    Action action,
    PlayerId acting_player
);

// Compute best response value for a player
// Returns the EV that the player can achieve by best-responding to opponent's sigma
double best_response_value(
    const GameNode* root,
    const Strategy& sigma,
    PlayerId br_player
);

// Compute exploitability: average of best response values for both players
// At Nash equilibrium, exploitability = 0
double compute_exploitability(const GameNode* root, const Strategy& sigma);

// ============================================================================
// Internal implementation details
// ============================================================================

namespace detail {

// Recursive EV computation with reach probabilities
double ev_recursive(
    const GameNode* node,
    const Strategy& sigma,
    double reach_p0,              // Reach probability contribution from P0
    double reach_p1,              // Reach probability contribution from P1
    double reach_chance,          // Reach probability contribution from chance
    const std::optional<std::pair<InfoSetId, Action>>& override_opt
);

// Best response recursive traversal
// Returns (EV for br_player, best action at each info set)
double br_recursive(
    const GameNode* node,
    const Strategy& sigma,
    PlayerId br_player,
    double reach_opponent,
    double reach_chance
);

} // namespace detail

} // namespace quantnet::poker
