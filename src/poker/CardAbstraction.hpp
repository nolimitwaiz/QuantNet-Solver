#pragma once

#include <array>
#include <vector>
#include <map>
#include <cstdint>
#include <string>
#include <memory>
#include "HandEvaluator.hpp"

namespace quantnet::poker {

// Card abstraction maps hands to buckets to reduce state space
// This is essential for solving Texas Hold'em where the full game
// has ~10^14 information sets

// Abstraction bucket ID
using BucketId = uint16_t;

// Betting rounds
enum class BettingRound {
    Preflop,   // No community cards
    Flop,      // 3 community cards
    Turn,      // 4 community cards
    River      // 5 community cards
};

inline std::string round_to_string(BettingRound round) {
    switch (round) {
        case BettingRound::Preflop: return "Preflop";
        case BettingRound::Flop: return "Flop";
        case BettingRound::Turn: return "Turn";
        case BettingRound::River: return "River";
    }
    return "Unknown";
}

// Hand features used for abstraction
struct HandFeatures {
    double hand_strength = 0.0;     // Current strength [0, 1]
    double positive_potential = 0.0; // Probability of improving
    double negative_potential = 0.0; // Probability of being outdrawn

    // Derived feature: effective hand strength
    // EHS = HS + (1 - HS) * ppot - HS * npot
    double effective_strength() const {
        return hand_strength * (1.0 - negative_potential) +
               (1.0 - hand_strength) * positive_potential;
    }
};

// Base class for card abstraction
class CardAbstraction {
public:
    virtual ~CardAbstraction() = default;

    // Get bucket for a hole hand given the current board
    virtual BucketId get_bucket(const std::array<int, 2>& hole,
                                const std::vector<int>& board,
                                BettingRound round) const = 0;

    // Get number of buckets for a round
    virtual int num_buckets(BettingRound round) const = 0;

    // Get total buckets across all rounds
    virtual int total_buckets() const = 0;

    // Compute features for a hand
    virtual HandFeatures compute_features(const std::array<int, 2>& hole,
                                         const std::vector<int>& board) const;

    // Name of this abstraction
    virtual std::string name() const = 0;
};

// Null abstraction: no abstraction (one bucket per distinct hand)
// Only useful for very small games or debugging
class NullAbstraction : public CardAbstraction {
public:
    BucketId get_bucket(const std::array<int, 2>& hole,
                        const std::vector<int>& board,
                        BettingRound round) const override;

    int num_buckets(BettingRound round) const override;
    int total_buckets() const override;
    std::string name() const override { return "Null"; }
};

// Percentile abstraction: bucket by hand strength percentile
// Simple but effective for small-medium games
class PercentileAbstraction : public CardAbstraction {
public:
    // buckets_per_round: number of buckets for each round
    explicit PercentileAbstraction(int preflop_buckets = 169,
                                    int flop_buckets = 50,
                                    int turn_buckets = 50,
                                    int river_buckets = 50);

    BucketId get_bucket(const std::array<int, 2>& hole,
                        const std::vector<int>& board,
                        BettingRound round) const override;

    int num_buckets(BettingRound round) const override;
    int total_buckets() const override;
    std::string name() const override { return "Percentile"; }

private:
    int preflop_buckets_;
    int flop_buckets_;
    int turn_buckets_;
    int river_buckets_;

    // Preflop hand rankings (pre-computed)
    // Maps canonical hand ID to rank [0, 168]
    std::map<int, int> preflop_rankings_;

    void initialize_preflop_rankings();
    int get_preflop_hand_id(int card1, int card2) const;
};

// Earth Mover's Distance (EMD) abstraction
// Groups hands by equity distribution similarity
// This is the approach used in professional poker AIs
class EMDAbstraction : public CardAbstraction {
public:
    explicit EMDAbstraction(int preflop_buckets = 169,
                            int flop_buckets = 200,
                            int turn_buckets = 200,
                            int river_buckets = 200);

    BucketId get_bucket(const std::array<int, 2>& hole,
                        const std::vector<int>& board,
                        BettingRound round) const override;

    int num_buckets(BettingRound round) const override;
    int total_buckets() const override;
    std::string name() const override { return "EMD"; }

    // Build abstraction by clustering (can be expensive)
    // This should be called once to generate bucket assignments
    void build_clusters(int samples_per_hand = 100);

private:
    int preflop_buckets_;
    int flop_buckets_;
    int turn_buckets_;
    int river_buckets_;

    // Cluster assignments (computed by build_clusters)
    // Key: canonical representation, Value: bucket ID
    std::map<std::string, BucketId> flop_clusters_;
    std::map<std::string, BucketId> turn_clusters_;
    std::map<std::string, BucketId> river_clusters_;

    // Get canonical string for hand + board
    std::string canonicalize(const std::array<int, 2>& hole,
                            const std::vector<int>& board) const;
};

// Effective Hand Strength (EHS) abstraction
// Buckets by hand strength combined with potential
class EHSAbstraction : public CardAbstraction {
public:
    explicit EHSAbstraction(int preflop_buckets = 10,
                            int flop_buckets = 10,
                            int turn_buckets = 10,
                            int river_buckets = 10);

    BucketId get_bucket(const std::array<int, 2>& hole,
                        const std::vector<int>& board,
                        BettingRound round) const override;

    int num_buckets(BettingRound round) const override;
    int total_buckets() const override;
    std::string name() const override { return "EHS"; }

private:
    int preflop_buckets_;
    int flop_buckets_;
    int turn_buckets_;
    int river_buckets_;

    // Discretize EHS to bucket
    BucketId discretize(double ehs, int num_buckets) const;
};

// Factory function to create abstractions
std::unique_ptr<CardAbstraction> create_abstraction(
    const std::string& name,
    int preflop = 169,
    int flop = 50,
    int turn = 50,
    int river = 50);

// Utility: count canonical preflop hands (169 for Hold'em)
int count_canonical_preflop_hands();

// Utility: convert hole cards to canonical form
// Returns pair (high_rank, low_rank, suited)
std::tuple<int, int, bool> canonicalize_hole_cards(int card1, int card2);

} // namespace quantnet::poker
