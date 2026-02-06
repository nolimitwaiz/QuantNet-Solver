#include "CardAbstraction.hpp"
#include <algorithm>
#include <cmath>
#include <set>

namespace quantnet::poker {

// ============================================================================
// CardAbstraction base class
// ============================================================================

HandFeatures CardAbstraction::compute_features(
    const std::array<int, 2>& hole,
    const std::vector<int>& board) const {

    HandFeatures features;

    // Use HandEvaluator for strength calculations
    features.hand_strength = HandEvaluator::hand_strength(hole, board);

    // Compute potential if not at river
    if (board.size() < 5) {
        auto [ppot, npot] = HandEvaluator::hand_potential(hole, board);
        features.positive_potential = ppot;
        features.negative_potential = npot;
    }

    return features;
}

// ============================================================================
// NullAbstraction
// ============================================================================

BucketId NullAbstraction::get_bucket(
    const std::array<int, 2>& hole,
    const std::vector<int>& board,
    BettingRound /*round*/) const {

    // Simple hash based on cards
    // This doesn't actually reduce state space much
    int h = (hole[0] * 52 + hole[1]) % 10000;
    for (int c : board) {
        h = (h * 52 + c) % 10000;
    }
    return static_cast<BucketId>(h);
}

int NullAbstraction::num_buckets(BettingRound /*round*/) const {
    return 10000;  // Arbitrary large number
}

int NullAbstraction::total_buckets() const {
    return 40000;  // 4 rounds * 10000
}

// ============================================================================
// PercentileAbstraction
// ============================================================================

PercentileAbstraction::PercentileAbstraction(int preflop, int flop, int turn, int river)
    : preflop_buckets_(preflop)
    , flop_buckets_(flop)
    , turn_buckets_(turn)
    , river_buckets_(river) {
    initialize_preflop_rankings();
}

void PercentileAbstraction::initialize_preflop_rankings() {
    // Pre-computed preflop hand rankings based on equity
    // Rankings are 0-168 where 0 is strongest (AA) and 168 is weakest

    // For simplicity, we use a basic ranking based on:
    // 1. Pairs (higher is better)
    // 2. Suited/unsuited connectors
    // 3. High cards

    // This is a simplified version - real implementations use
    // Monte Carlo simulations to compute exact rankings

    std::vector<std::pair<int, double>> hand_scores;

    // Enumerate all 169 canonical hands
    // Hand ID encoding: high_rank * 14 + low_rank + (suited ? 169 : 0) mapped to [0, 168]

    int hand_id = 0;
    // Pairs
    for (int r = 12; r >= 0; --r) {
        double score = 100.0 + r * 10;  // Pairs score high
        preflop_rankings_[r * 13 + r] = hand_id++;  // Pair encoding
    }

    // Suited hands (non-pairs), sorted by strength estimate
    std::vector<std::pair<double, std::pair<int, int>>> suited_hands;
    for (int r1 = 12; r1 >= 0; --r1) {
        for (int r2 = r1 - 1; r2 >= 0; --r2) {
            // Score based on high card + connectivity
            double score = r1 * 5 + r2 * 2 + ((r1 - r2 <= 4) ? 10 : 0) + 5;
            suited_hands.push_back({score, {r1, r2}});
        }
    }
    std::sort(suited_hands.rbegin(), suited_hands.rend());
    for (const auto& [score, hand] : suited_hands) {
        preflop_rankings_[hand.first * 100 + hand.second + 1000] = hand_id++;
    }

    // Unsuited hands (non-pairs)
    std::vector<std::pair<double, std::pair<int, int>>> unsuited_hands;
    for (int r1 = 12; r1 >= 0; --r1) {
        for (int r2 = r1 - 1; r2 >= 0; --r2) {
            double score = r1 * 5 + r2 * 2 + ((r1 - r2 <= 4) ? 5 : 0);
            unsuited_hands.push_back({score, {r1, r2}});
        }
    }
    std::sort(unsuited_hands.rbegin(), unsuited_hands.rend());
    for (const auto& [score, hand] : unsuited_hands) {
        preflop_rankings_[hand.first * 100 + hand.second] = hand_id++;
    }
}

int PercentileAbstraction::get_preflop_hand_id(int card1, int card2) const {
    int r1 = card_rank(card1);
    int r2 = card_rank(card2);
    int s1 = card_suit(card1);
    int s2 = card_suit(card2);

    int high = std::max(r1, r2);
    int low = std::min(r1, r2);
    bool suited = (s1 == s2);

    if (high == low) {
        // Pair
        return high * 13 + low;
    } else if (suited) {
        return high * 100 + low + 1000;
    } else {
        return high * 100 + low;
    }
}

BucketId PercentileAbstraction::get_bucket(
    const std::array<int, 2>& hole,
    const std::vector<int>& board,
    BettingRound round) const {

    if (round == BettingRound::Preflop) {
        // Use preflop rankings
        int hand_id = get_preflop_hand_id(hole[0], hole[1]);
        auto it = preflop_rankings_.find(hand_id);
        if (it != preflop_rankings_.end()) {
            // Map rank to bucket
            int rank = it->second;
            return static_cast<BucketId>(rank * preflop_buckets_ / 169);
        }
        return 0;
    }

    // Post-flop: use hand strength
    int num_buckets = 0;
    switch (round) {
        case BettingRound::Flop: num_buckets = flop_buckets_; break;
        case BettingRound::Turn: num_buckets = turn_buckets_; break;
        case BettingRound::River: num_buckets = river_buckets_; break;
        default: num_buckets = flop_buckets_;
    }

    double hs = HandEvaluator::hand_strength(hole, board);
    int bucket = static_cast<int>(hs * num_buckets);
    if (bucket >= num_buckets) bucket = num_buckets - 1;
    if (bucket < 0) bucket = 0;

    return static_cast<BucketId>(bucket);
}

int PercentileAbstraction::num_buckets(BettingRound round) const {
    switch (round) {
        case BettingRound::Preflop: return preflop_buckets_;
        case BettingRound::Flop: return flop_buckets_;
        case BettingRound::Turn: return turn_buckets_;
        case BettingRound::River: return river_buckets_;
    }
    return flop_buckets_;
}

int PercentileAbstraction::total_buckets() const {
    return preflop_buckets_ + flop_buckets_ + turn_buckets_ + river_buckets_;
}

// ============================================================================
// EHSAbstraction
// ============================================================================

EHSAbstraction::EHSAbstraction(int preflop, int flop, int turn, int river)
    : preflop_buckets_(preflop)
    , flop_buckets_(flop)
    , turn_buckets_(turn)
    , river_buckets_(river) {}

BucketId EHSAbstraction::discretize(double ehs, int num_buckets) const {
    int bucket = static_cast<int>(ehs * num_buckets);
    if (bucket >= num_buckets) bucket = num_buckets - 1;
    if (bucket < 0) bucket = 0;
    return static_cast<BucketId>(bucket);
}

BucketId EHSAbstraction::get_bucket(
    const std::array<int, 2>& hole,
    const std::vector<int>& board,
    BettingRound round) const {

    HandFeatures features = compute_features(hole, board);

    int num_buckets = 0;
    switch (round) {
        case BettingRound::Preflop: num_buckets = preflop_buckets_; break;
        case BettingRound::Flop: num_buckets = flop_buckets_; break;
        case BettingRound::Turn: num_buckets = turn_buckets_; break;
        case BettingRound::River: num_buckets = river_buckets_; break;
    }

    double ehs = features.effective_strength();
    return discretize(ehs, num_buckets);
}

int EHSAbstraction::num_buckets(BettingRound round) const {
    switch (round) {
        case BettingRound::Preflop: return preflop_buckets_;
        case BettingRound::Flop: return flop_buckets_;
        case BettingRound::Turn: return turn_buckets_;
        case BettingRound::River: return river_buckets_;
    }
    return flop_buckets_;
}

int EHSAbstraction::total_buckets() const {
    return preflop_buckets_ + flop_buckets_ + turn_buckets_ + river_buckets_;
}

// ============================================================================
// EMDAbstraction
// ============================================================================

EMDAbstraction::EMDAbstraction(int preflop, int flop, int turn, int river)
    : preflop_buckets_(preflop)
    , flop_buckets_(flop)
    , turn_buckets_(turn)
    , river_buckets_(river) {}

std::string EMDAbstraction::canonicalize(
    const std::array<int, 2>& hole,
    const std::vector<int>& board) const {

    // Create canonical string representation
    // Sort cards within each category
    std::vector<int> sorted_hole = {hole[0], hole[1]};
    std::sort(sorted_hole.begin(), sorted_hole.end());

    std::vector<int> sorted_board = board;
    std::sort(sorted_board.begin(), sorted_board.end());

    std::string result;
    for (int c : sorted_hole) result += card_to_string(c) + ":";
    for (int c : sorted_board) result += card_to_string(c) + ":";

    return result;
}

BucketId EMDAbstraction::get_bucket(
    const std::array<int, 2>& hole,
    const std::vector<int>& board,
    BettingRound round) const {

    // For preflop, use simple ranking
    if (round == BettingRound::Preflop) {
        // Use hand strength as proxy
        double hs = HandEvaluator::hand_strength(hole, {});
        int bucket = static_cast<int>(hs * preflop_buckets_);
        if (bucket >= preflop_buckets_) bucket = preflop_buckets_ - 1;
        return static_cast<BucketId>(bucket);
    }

    // Check pre-computed clusters
    std::string key = canonicalize(hole, board);
    int num_buckets = 0;

    const std::map<std::string, BucketId>* clusters = nullptr;
    switch (round) {
        case BettingRound::Flop:
            clusters = &flop_clusters_;
            num_buckets = flop_buckets_;
            break;
        case BettingRound::Turn:
            clusters = &turn_clusters_;
            num_buckets = turn_buckets_;
            break;
        case BettingRound::River:
            clusters = &river_clusters_;
            num_buckets = river_buckets_;
            break;
        default:
            return 0;
    }

    if (clusters) {
        auto it = clusters->find(key);
        if (it != clusters->end()) {
            return it->second;
        }
    }

    // Fallback to hand strength if no pre-computed bucket
    double hs = HandEvaluator::hand_strength(hole, board);
    int bucket = static_cast<int>(hs * num_buckets);
    if (bucket >= num_buckets) bucket = num_buckets - 1;
    return static_cast<BucketId>(bucket);
}

void EMDAbstraction::build_clusters(int /*samples_per_hand*/) {
    // Full EMD clustering is expensive
    // This is a placeholder - real implementation would:
    // 1. For each possible hand+board combo, compute equity distribution
    // 2. Use k-means or hierarchical clustering with EMD distance
    // 3. Store cluster assignments

    // For now, we skip pre-computation and fall back to hand strength
    // A real implementation would take hours to compute
}

int EMDAbstraction::num_buckets(BettingRound round) const {
    switch (round) {
        case BettingRound::Preflop: return preflop_buckets_;
        case BettingRound::Flop: return flop_buckets_;
        case BettingRound::Turn: return turn_buckets_;
        case BettingRound::River: return river_buckets_;
    }
    return flop_buckets_;
}

int EMDAbstraction::total_buckets() const {
    return preflop_buckets_ + flop_buckets_ + turn_buckets_ + river_buckets_;
}

// ============================================================================
// Factory and utilities
// ============================================================================

std::unique_ptr<CardAbstraction> create_abstraction(
    const std::string& name,
    int preflop, int flop, int turn, int river) {

    if (name == "null" || name == "Null") {
        return std::make_unique<NullAbstraction>();
    } else if (name == "percentile" || name == "Percentile") {
        return std::make_unique<PercentileAbstraction>(preflop, flop, turn, river);
    } else if (name == "ehs" || name == "EHS") {
        return std::make_unique<EHSAbstraction>(preflop, flop, turn, river);
    } else if (name == "emd" || name == "EMD") {
        return std::make_unique<EMDAbstraction>(preflop, flop, turn, river);
    }

    // Default to percentile
    return std::make_unique<PercentileAbstraction>(preflop, flop, turn, river);
}

int count_canonical_preflop_hands() {
    // In Hold'em:
    // - 13 pairs
    // - 78 suited non-pairs (13 choose 2)
    // - 78 unsuited non-pairs
    // Total: 169
    return 169;
}

std::tuple<int, int, bool> canonicalize_hole_cards(int card1, int card2) {
    int r1 = card_rank(card1);
    int r2 = card_rank(card2);
    int s1 = card_suit(card1);
    int s2 = card_suit(card2);

    bool suited = (s1 == s2);
    int high = std::max(r1, r2);
    int low = std::min(r1, r2);

    return {high, low, suited};
}

} // namespace quantnet::poker
