#include "HandEvaluator.hpp"
#include <algorithm>
#include <stdexcept>
#include <random>

namespace quantnet::poker {

HandValue HandEvaluator::evaluate(const std::array<int, 7>& cards) {
    return evaluate(std::vector<int>(cards.begin(), cards.end()));
}

HandValue HandEvaluator::evaluate(const std::array<int, 2>& hole,
                                  const std::array<int, 5>& board) {
    std::array<int, 7> all_cards;
    all_cards[0] = hole[0];
    all_cards[1] = hole[1];
    for (int i = 0; i < 5; ++i) {
        all_cards[2 + i] = board[i];
    }
    return evaluate(all_cards);
}

HandValue HandEvaluator::evaluate(const std::vector<int>& cards) {
    if (cards.size() < 5) {
        throw std::invalid_argument("Need at least 5 cards to evaluate");
    }

    const int n = static_cast<int>(cards.size());
    int rank_counts[NUM_RANKS];
    count_ranks(cards.data(), n, rank_counts);

    // Build rank bitmask
    uint16_t rank_mask = 0;
    for (int r = 0; r < NUM_RANKS; ++r) {
        if (rank_counts[r] > 0) {
            rank_mask |= (1 << r);
        }
    }

    // Check for flush
    int flush_suit = find_flush_suit(cards.data(), n);

    if (flush_suit >= 0) {
        // Extract flush cards
        std::vector<int> flush_cards;
        for (int card : cards) {
            if (card_suit(card) == flush_suit) {
                flush_cards.push_back(card);
            }
        }

        // Build flush rank mask
        uint16_t flush_mask = 0;
        for (int card : flush_cards) {
            flush_mask |= (1 << card_rank(card));
        }

        // Check for straight flush
        int sf_high = find_straight_high(flush_mask);
        if (sf_high >= 0) {
            return make_value(HandRank::StraightFlush, sf_high);
        }

        // Regular flush - top 5 flush cards
        std::vector<int> flush_ranks;
        for (int card : flush_cards) {
            flush_ranks.push_back(card_rank(card));
        }
        std::sort(flush_ranks.begin(), flush_ranks.end(), std::greater<int>());

        return make_value(HandRank::Flush,
                         flush_ranks[0], flush_ranks[1], flush_ranks[2],
                         flush_ranks[3], flush_ranks[4]);
    }

    // Find quads, trips, pairs
    std::vector<int> quads, trips, pairs, singles;
    for (int r = NUM_RANKS - 1; r >= 0; --r) {
        switch (rank_counts[r]) {
            case 4: quads.push_back(r); break;
            case 3: trips.push_back(r); break;
            case 2: pairs.push_back(r); break;
            case 1: singles.push_back(r); break;
        }
    }

    // Four of a kind
    if (!quads.empty()) {
        int kicker = -1;
        if (!trips.empty()) kicker = trips[0];
        else if (!pairs.empty()) kicker = pairs[0];
        else if (!singles.empty()) kicker = singles[0];
        return make_value(HandRank::FourOfAKind, quads[0], kicker);
    }

    // Full house (trips + pair, or two trips)
    if (!trips.empty()) {
        if (trips.size() >= 2) {
            return make_value(HandRank::FullHouse, trips[0], trips[1]);
        }
        if (!pairs.empty()) {
            return make_value(HandRank::FullHouse, trips[0], pairs[0]);
        }
    }

    // Check for straight
    int straight_high = find_straight_high(rank_mask);
    if (straight_high >= 0) {
        return make_value(HandRank::Straight, straight_high);
    }

    // Three of a kind
    if (!trips.empty()) {
        // Get top 2 kickers
        std::vector<int> kickers;
        for (int r : pairs) kickers.push_back(r);
        for (int r : singles) kickers.push_back(r);
        std::sort(kickers.begin(), kickers.end(), std::greater<int>());

        return make_value(HandRank::ThreeOfAKind, trips[0],
                         kickers.size() > 0 ? kickers[0] : 0,
                         kickers.size() > 1 ? kickers[1] : 0);
    }

    // Two pair
    if (pairs.size() >= 2) {
        // Best kicker from remaining pairs or singles
        int kicker = 0;
        if (pairs.size() > 2) kicker = pairs[2];
        else if (!singles.empty()) kicker = singles[0];

        return make_value(HandRank::TwoPair, pairs[0], pairs[1], kicker);
    }

    // One pair
    if (pairs.size() == 1) {
        // Top 3 kickers from singles
        return make_value(HandRank::Pair, pairs[0],
                         singles.size() > 0 ? singles[0] : 0,
                         singles.size() > 1 ? singles[1] : 0,
                         singles.size() > 2 ? singles[2] : 0);
    }

    // High card
    return make_value(HandRank::HighCard,
                     singles[0], singles[1], singles[2],
                     singles[3], singles[4]);
}

int HandEvaluator::compare(const std::array<int, 2>& hand1,
                          const std::array<int, 2>& hand2,
                          const std::array<int, 5>& board) {
    HandValue v1 = evaluate(hand1, board);
    HandValue v2 = evaluate(hand2, board);

    if (v1.value > v2.value) return 1;
    if (v1.value < v2.value) return -1;
    return 0;
}

double HandEvaluator::hand_strength(const std::array<int, 2>& hole,
                                    const std::vector<int>& board) {
    // Mark cards that are used
    bool used[DECK_SIZE] = {false};
    used[hole[0]] = true;
    used[hole[1]] = true;
    for (int card : board) {
        used[card] = true;
    }

    // Build all cards for our hand
    std::vector<int> our_cards(hole.begin(), hole.end());
    for (int card : board) {
        our_cards.push_back(card);
    }

    // Need 5+ cards to evaluate
    if (our_cards.size() < 5) {
        return 0.5;  // Not enough cards, return 50%
    }

    HandValue our_value = evaluate(our_cards);

    // Enumerate opponent hands
    int wins = 0, losses = 0, ties = 0;

    for (int c1 = 0; c1 < DECK_SIZE; ++c1) {
        if (used[c1]) continue;
        for (int c2 = c1 + 1; c2 < DECK_SIZE; ++c2) {
            if (used[c2]) continue;

            // Build opponent hand
            std::vector<int> opp_cards = {c1, c2};
            for (int card : board) {
                opp_cards.push_back(card);
            }

            HandValue opp_value = evaluate(opp_cards);

            if (our_value > opp_value) ++wins;
            else if (our_value < opp_value) ++losses;
            else ++ties;
        }
    }

    int total = wins + losses + ties;
    if (total == 0) return 0.5;

    // Hand strength = (wins + 0.5 * ties) / total
    return (wins + 0.5 * ties) / total;
}

std::pair<double, double> HandEvaluator::hand_potential(
    const std::array<int, 2>& hole,
    const std::vector<int>& board) {

    // This is a simplified version - full implementation would enumerate
    // remaining cards and compute potential

    // Mark used cards
    bool used[DECK_SIZE] = {false};
    used[hole[0]] = true;
    used[hole[1]] = true;
    for (int card : board) {
        used[card] = true;
    }

    // Get unused cards
    std::vector<int> deck;
    for (int c = 0; c < DECK_SIZE; ++c) {
        if (!used[c]) deck.push_back(c);
    }

    if (board.size() >= 5) {
        // No more cards to come
        return {0.0, 0.0};
    }

    // Sample-based estimation for efficiency
    // Full enumeration would be O(C(45,2) * C(43,5-board_size) * C(remaining,2))

    // For simplicity, return basic estimates based on board size
    size_t cards_to_come = 5 - board.size();

    if (cards_to_come == 0) {
        return {0.0, 0.0};
    }

    // Build our current hand
    std::vector<int> our_cards(hole.begin(), hole.end());
    for (int card : board) {
        our_cards.push_back(card);
    }

    // Sample opponent hands and future boards
    const int SAMPLES = 500;
    int ahead_stays_ahead = 0, ahead_falls_behind = 0;
    int behind_catches_up = 0, behind_stays_behind = 0;
    int tied_wins = 0, tied_loses = 0;

    std::vector<int> shuffled_deck = deck;
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int sample = 0; sample < SAMPLES; ++sample) {
        // Shuffle deck for random sampling
        std::shuffle(shuffled_deck.begin(), shuffled_deck.end(), gen);

        // Pick opponent hole cards
        int opp1 = shuffled_deck[0];
        int opp2 = shuffled_deck[1];

        // Current hand comparison
        std::vector<int> opp_cards = {opp1, opp2};
        for (int card : board) {
            opp_cards.push_back(card);
        }

        // Fill to 5 cards minimum for evaluation
        std::vector<int> our_current = our_cards;
        std::vector<int> opp_current = opp_cards;

        size_t deck_idx = 2;  // Start after opponent cards

        while (our_current.size() < 5 && deck_idx < shuffled_deck.size()) {
            int next_card = shuffled_deck[deck_idx++];
            our_current.push_back(next_card);
            opp_current.push_back(next_card);
        }

        // Fill remaining board cards
        while (our_current.size() < 7 && deck_idx < shuffled_deck.size()) {
            int next_card = shuffled_deck[deck_idx++];
            our_current.push_back(next_card);
            opp_current.push_back(next_card);
        }

        // Evaluate with partial board (5 cards)
        std::vector<int> our_partial(our_cards);
        std::vector<int> opp_partial(opp_cards);

        // Pad to 5 if needed
        size_t fill_idx = 2;
        while (our_partial.size() < 5 && fill_idx < shuffled_deck.size()) {
            int c = shuffled_deck[fill_idx++];
            our_partial.push_back(c);
            opp_partial.push_back(c);
        }

        HandValue our_val_now = evaluate(our_partial);
        HandValue opp_val_now = evaluate(opp_partial);

        // Evaluate final
        HandValue our_val_final = evaluate(our_current);
        HandValue opp_val_final = evaluate(opp_current);

        // Categorize
        if (our_val_now > opp_val_now) {
            // Currently ahead
            if (our_val_final > opp_val_final) ahead_stays_ahead++;
            else ahead_falls_behind++;
        } else if (our_val_now < opp_val_now) {
            // Currently behind
            if (our_val_final > opp_val_final) behind_catches_up++;
            else behind_stays_behind++;
        } else {
            // Currently tied
            if (our_val_final > opp_val_final) tied_wins++;
            else tied_loses++;
        }
    }

    // Positive potential: P(win | currently behind or tied)
    int behind_total = behind_catches_up + behind_stays_behind;
    double ppot = behind_total > 0 ?
        static_cast<double>(behind_catches_up) / behind_total : 0.0;

    // Negative potential: P(lose | currently ahead or tied)
    int ahead_total = ahead_stays_ahead + ahead_falls_behind;
    double npot = ahead_total > 0 ?
        static_cast<double>(ahead_falls_behind) / ahead_total : 0.0;

    return {ppot, npot};
}

} // namespace quantnet::poker
