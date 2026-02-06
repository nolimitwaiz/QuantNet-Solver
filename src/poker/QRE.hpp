#pragma once

#include <Eigen/Dense>
#include "GameTree.hpp"
#include "Strategy.hpp"
#include "ExpectedValue.hpp"
#include "GameTypes.hpp"

namespace quantnet::poker {

// Quantal Response Equilibrium (QRE) residual computation
//
// QRE is defined by the fixed-point equation:
//   sigma = LogitBR_beta(sigma)
//
// where LogitBR_beta(I, a) = exp(beta * EU(I, a)) / sum_b exp(beta * EU(I, b))
//
// The residual is:
//   R(sigma) = sigma - LogitBR_beta(sigma)
//
// Since sigma is parameterized by logits w via softmax:
//   sigma = softmax(w)
//
// We solve R(w) = 0 using Newton's method.
//
// At equilibrium, the current strategy equals the logit best response to itself.
class QREResidual {
public:
    // Construct QRE residual for a game
    QREResidual(const PokerGame& game, double beta = 1.0);

    // Compute residual R(w) given logits w
    // Returns vector of size total_dim() where:
    //   R[i] = sigma[i] - BR_beta[i]
    Eigen::VectorXd operator()(const Eigen::VectorXd& w) const;

    // Set temperature parameter
    void set_beta(double beta) { beta_ = beta; }
    double beta() const { return beta_; }

    // Get dimension (total number of strategy parameters)
    int dim() const { return index_.total_dim(); }

    // Get info set index
    const InfoSetIndex& index() const { return index_; }

    // Compute logit best response given current strategy
    // Returns vector of probabilities (not logits)
    Eigen::VectorXd logit_best_response(const Strategy& sigma) const;

    // Get the game
    const PokerGame& game() const { return game_; }

private:
    const PokerGame& game_;
    double beta_;
    InfoSetIndex index_;
};

// Compute expected utilities for all actions at all info sets
// Returns map: info_set_id -> (action -> EU)
std::map<InfoSetId, std::map<Action, double>> compute_all_expected_utilities(
    const PokerGame& game,
    const Strategy& sigma,
    const InfoSetIndex& index
);

} // namespace quantnet::poker
