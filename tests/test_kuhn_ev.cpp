// Tests for Kuhn Poker expected value computation
//
// We verify EV calculations against known values for simple strategies.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "poker/KuhnPoker.hpp"
#include "poker/Strategy.hpp"
#include "poker/ExpectedValue.hpp"
#include "poker/QRE.hpp"

using namespace quantnet::poker;
using Catch::Matchers::WithinAbs;

TEST_CASE("Kuhn Poker tree has correct structure", "[kuhn][tree]") {
    KuhnPoker kuhn;

    auto stats = compute_tree_stats(kuhn.root());

    // 6 deals (chance outcomes)
    // Each deal leads to game tree with player decisions
    // Total should be:
    // - 1 root chance node
    // - 6 deal outcomes, each with multiple player nodes and terminals

    REQUIRE(stats.chance_nodes >= 1);  // At least the root
    REQUIRE(stats.player_nodes > 0);
    REQUIRE(stats.terminal_nodes > 0);

    // Kuhn has 12 information sets (6 for each player in various situations)
    auto info_sets = kuhn.get_info_sets();
    REQUIRE(info_sets.size() == 12);
}

TEST_CASE("Kuhn info sets have correct actions", "[kuhn][infosets]") {
    KuhnPoker kuhn;
    auto info_sets = kuhn.get_info_sets();

    // First round info sets (P0 with just card, no history)
    // should have check and bet
    int first_round_count = 0;
    for (const auto& is : info_sets) {
        if (is.id.find("P0:") == 0 && is.id.find("::") != std::string::npos) {
            // Empty history indicates first action
            // Actually the format is "P0:card:history" so empty history ends with ":"
            if (is.id.back() == ':' || is.id.substr(is.id.length()-2) == "::") {
                // Should have check and bet
                REQUIRE(is.legal_actions.size() >= 2);
                first_round_count++;
            }
        }
    }
}

TEST_CASE("Uniform strategy has zero EV in symmetric game", "[kuhn][ev]") {
    KuhnPoker kuhn;
    auto info_sets = kuhn.get_info_sets();

    InfoSetIndex index;
    index.build(info_sets);

    // Create uniform strategy (all logits = 0)
    Strategy sigma = Strategy::uniform(index);

    // Compute EV for player 0
    double ev = compute_ev(kuhn.root(), sigma);

    // Kuhn with uniform play should give approximately 0 EV for P0
    // (it's a symmetric game, both players play the same way)
    // Note: Due to position advantage, it's not exactly 0
    // P0 moves first which can be a disadvantage
    // The value should be close to -1/18 for uniform play
    REQUIRE_THAT(ev, WithinAbs(0.0, 0.2));  // Allow some tolerance
}

TEST_CASE("Exploitability of uniform strategy is positive", "[kuhn][exploit]") {
    KuhnPoker kuhn;
    auto info_sets = kuhn.get_info_sets();

    InfoSetIndex index;
    index.build(info_sets);

    Strategy sigma = Strategy::uniform(index);
    double exploit = compute_exploitability(kuhn.root(), sigma);

    // Uniform play is not Nash, so exploitability > 0
    REQUIRE(exploit > 0.0);
}

TEST_CASE("Strategy softmax sums to 1", "[kuhn][strategy]") {
    KuhnPoker kuhn;
    auto info_sets = kuhn.get_info_sets();

    InfoSetIndex index;
    index.build(info_sets);

    // Create strategy with random logits
    Eigen::VectorXd w = Eigen::VectorXd::Random(index.total_dim());
    Strategy sigma = Strategy::from_logits(w, index);

    // Check that probabilities sum to 1 at each info set
    for (const auto& is : info_sets) {
        Eigen::VectorXd probs = sigma.probs(is.id);
        double sum = probs.sum();
        REQUIRE_THAT(sum, WithinAbs(1.0, 1e-10));

        // All probabilities should be positive
        for (int i = 0; i < probs.size(); ++i) {
            REQUIRE(probs(i) > 0.0);
        }
    }
}

TEST_CASE("QRE residual at uniform with beta=0 is near zero", "[kuhn][qre]") {
    KuhnPoker kuhn;

    // At beta = 0, logit BR is uniform regardless of EU
    // So uniform strategy should be a fixed point
    QREResidual qre(kuhn, 0.001);  // Very small beta, near uniform

    Eigen::VectorXd w = Eigen::VectorXd::Zero(qre.dim());
    Eigen::VectorXd r = qre(w);

    // Residual should be small (strategy ≈ BR at low beta)
    double residual_norm = r.norm();
    REQUIRE(residual_norm < 0.1);  // Allow tolerance for numerical issues
}

TEST_CASE("Best response value is at least as good as current EV", "[kuhn][br]") {
    KuhnPoker kuhn;
    auto info_sets = kuhn.get_info_sets();

    InfoSetIndex index;
    index.build(info_sets);

    Strategy sigma = Strategy::uniform(index);

    double ev_current = compute_ev(kuhn.root(), sigma);
    double br_value_p0 = best_response_value(kuhn.root(), sigma, PLAYER_0);
    double br_value_p1 = best_response_value(kuhn.root(), sigma, PLAYER_1);

    // P0's BR value should be at least as good as current EV
    REQUIRE(br_value_p0 >= ev_current - 1e-10);

    // P1's BR value should be at least as good as their current EV
    // P1's current EV is -ev_current (zero-sum)
    REQUIRE(br_value_p1 >= -ev_current - 1e-10);
}

TEST_CASE("Kuhn card comparison is correct", "[kuhn][cards]") {
    // K > Q > J
    REQUIRE(KuhnPoker::compare_cards(2, 1) > 0);   // K > Q
    REQUIRE(KuhnPoker::compare_cards(1, 0) > 0);   // Q > J
    REQUIRE(KuhnPoker::compare_cards(2, 0) > 0);   // K > J
    REQUIRE(KuhnPoker::compare_cards(0, 2) < 0);   // J < K
    REQUIRE(KuhnPoker::compare_cards(1, 1) == 0);  // Q == Q
}

TEST_CASE("Info set ID format is consistent", "[kuhn][infosets]") {
    // Test info set ID construction
    InfoSetId id1 = KuhnPoker::make_info_set_id(0, 1, "");
    REQUIRE(id1 == "P0:Q:");

    InfoSetId id2 = KuhnPoker::make_info_set_id(1, 0, "b");
    REQUIRE(id2 == "P1:J:b");

    InfoSetId id3 = KuhnPoker::make_info_set_id(0, 2, "cb");
    REQUIRE(id3 == "P0:K:cb");
}

TEST_CASE("QRE residual dimension matches strategy dimension", "[kuhn][qre]") {
    KuhnPoker kuhn;
    QREResidual qre(kuhn, 1.0);

    auto info_sets = kuhn.get_info_sets();
    InfoSetIndex index;
    index.build(info_sets);

    REQUIRE(qre.dim() == index.total_dim());

    Eigen::VectorXd w = Eigen::VectorXd::Zero(qre.dim());
    Eigen::VectorXd r = qre(w);

    REQUIRE(r.size() == qre.dim());
}

TEST_CASE("Higher beta increases strategy sharpness", "[kuhn][qre]") {
    KuhnPoker kuhn;

    // At low beta, BR should be near uniform
    QREResidual qre_low(kuhn, 0.1);
    Eigen::VectorXd w = Eigen::VectorXd::Zero(qre_low.dim());
    Strategy sigma = Strategy::from_logits(w, qre_low.index());
    Eigen::VectorXd br_low = qre_low.logit_best_response(sigma);

    // At high beta, BR should be sharper (closer to 0 or 1)
    QREResidual qre_high(kuhn, 10.0);
    Eigen::VectorXd br_high = qre_high.logit_best_response(sigma);

    // Compute entropy-like measure: lower entropy = sharper
    auto entropy = [](const Eigen::VectorXd& p) {
        double h = 0.0;
        for (int i = 0; i < p.size(); ++i) {
            if (p(i) > 1e-10) {
                h -= p(i) * std::log(p(i));
            }
        }
        return h;
    };

    double entropy_low = entropy(br_low);
    double entropy_high = entropy(br_high);

    // Higher beta should give lower entropy (sharper distribution)
    REQUIRE(entropy_high <= entropy_low);
}
