#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <string>

namespace quantnet::poker {

// Card representation: 0-51
// Card = suit * 13 + rank
// Rank: 0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A
// Suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades

constexpr int NUM_RANKS = 13;
constexpr int NUM_SUITS = 4;
constexpr int DECK_SIZE = 52;

// Hand ranking categories (higher = better)
enum class HandRank : uint8_t {
    HighCard = 0,
    Pair = 1,
    TwoPair = 2,
    ThreeOfAKind = 3,
    Straight = 4,
    Flush = 5,
    FullHouse = 6,
    FourOfAKind = 7,
    StraightFlush = 8
};

// Convert hand rank to string
inline std::string hand_rank_to_string(HandRank rank) {
    switch (rank) {
        case HandRank::HighCard: return "High Card";
        case HandRank::Pair: return "Pair";
        case HandRank::TwoPair: return "Two Pair";
        case HandRank::ThreeOfAKind: return "Three of a Kind";
        case HandRank::Straight: return "Straight";
        case HandRank::Flush: return "Flush";
        case HandRank::FullHouse: return "Full House";
        case HandRank::FourOfAKind: return "Four of a Kind";
        case HandRank::StraightFlush: return "Straight Flush";
    }
    return "Unknown";
}

// Hand evaluation result
// Higher value = better hand
// Format: RRRRR_KKKKKKKKKKKKKK where R = rank category, K = kickers
struct HandValue {
    uint32_t value = 0;

    HandValue() = default;
    explicit HandValue(uint32_t v) : value(v) {}

    HandRank rank() const {
        return static_cast<HandRank>(value >> 20);
    }

    bool operator<(const HandValue& other) const { return value < other.value; }
    bool operator>(const HandValue& other) const { return value > other.value; }
    bool operator==(const HandValue& other) const { return value == other.value; }
    bool operator<=(const HandValue& other) const { return value <= other.value; }
    bool operator>=(const HandValue& other) const { return value >= other.value; }
};

// Card utilities
inline int card_rank(int card) { return card % 13; }
inline int card_suit(int card) { return card / 13; }
inline int make_card(int rank, int suit) { return suit * 13 + rank; }

inline char rank_char(int rank) {
    constexpr char chars[] = "23456789TJQKA";
    return chars[rank];
}

inline char suit_char(int suit) {
    constexpr char chars[] = "cdhs";
    return chars[suit];
}

inline std::string card_to_string(int card) {
    return std::string(1, rank_char(card_rank(card))) +
           std::string(1, suit_char(card_suit(card)));
}

// 7-card hand evaluator
// Evaluates the best 5-card hand from 7 cards (2 hole + 5 board)
class HandEvaluator {
public:
    // Evaluate 7 cards and return hand value
    // Cards are indices 0-51
    static HandValue evaluate(const std::array<int, 7>& cards);

    // Evaluate with separate hole cards and board
    static HandValue evaluate(const std::array<int, 2>& hole,
                             const std::array<int, 5>& board);

    // Evaluate any number of cards (finds best 5-card combination)
    static HandValue evaluate(const std::vector<int>& cards);

    // Compare two hands given shared board
    // Returns: positive if hand1 wins, negative if hand2 wins, 0 if tie
    static int compare(const std::array<int, 2>& hand1,
                      const std::array<int, 2>& hand2,
                      const std::array<int, 5>& board);

    // Compute hand strength (fraction of hands we beat or tie)
    // Enumerates all possible opponent hands
    static double hand_strength(const std::array<int, 2>& hole,
                               const std::vector<int>& board);

    // Compute positive/negative potential
    // Returns {ppot, npot} where:
    //   ppot = probability of improving to win
    //   npot = probability of being outdrawn
    static std::pair<double, double> hand_potential(
        const std::array<int, 2>& hole,
        const std::vector<int>& board);

private:
    // Count occurrences of each rank
    static void count_ranks(const int* cards, int n, int* rank_counts);

    // Check for flush (5+ cards of same suit)
    static int find_flush_suit(const int* cards, int n);

    // Find straight high card (returns -1 if no straight)
    static int find_straight_high(uint16_t rank_mask);

    // Build hand value from rank and kickers
    static HandValue make_value(HandRank rank, int k1 = 0, int k2 = 0,
                               int k3 = 0, int k4 = 0, int k5 = 0);
};

// Inline implementation of core methods for performance

inline void HandEvaluator::count_ranks(const int* cards, int n, int* rank_counts) {
    std::fill(rank_counts, rank_counts + NUM_RANKS, 0);
    for (int i = 0; i < n; ++i) {
        rank_counts[card_rank(cards[i])]++;
    }
}

inline int HandEvaluator::find_flush_suit(const int* cards, int n) {
    int suit_counts[NUM_SUITS] = {0, 0, 0, 0};
    for (int i = 0; i < n; ++i) {
        suit_counts[card_suit(cards[i])]++;
    }
    for (int s = 0; s < NUM_SUITS; ++s) {
        if (suit_counts[s] >= 5) return s;
    }
    return -1;
}

inline int HandEvaluator::find_straight_high(uint16_t rank_mask) {
    // Check for wheel (A-2-3-4-5)
    if ((rank_mask & 0x100F) == 0x100F) {
        return 3;  // 5-high straight
    }

    // Check for other straights
    for (int high = 12; high >= 4; --high) {
        uint16_t straight_mask = 0x1F << (high - 4);
        if ((rank_mask & straight_mask) == straight_mask) {
            return high;
        }
    }
    return -1;
}

inline HandValue HandEvaluator::make_value(HandRank rank, int k1, int k2,
                                          int k3, int k4, int k5) {
    uint32_t v = static_cast<uint32_t>(rank) << 20;
    v |= (k1 & 0xF) << 16;
    v |= (k2 & 0xF) << 12;
    v |= (k3 & 0xF) << 8;
    v |= (k4 & 0xF) << 4;
    v |= (k5 & 0xF);
    return HandValue(v);
}

} // namespace quantnet::poker
