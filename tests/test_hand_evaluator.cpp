// Tests for Hand Evaluator
// Verifies correct evaluation of 5-7 card poker hands

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "poker/HandEvaluator.hpp"

using namespace quantnet::poker;
using Catch::Matchers::WithinAbs;

// Helper to make card from "Ah", "Kc", etc.
int make_card_from_string(const std::string& s) {
    int rank;
    switch (s[0]) {
        case '2': rank = 0; break;
        case '3': rank = 1; break;
        case '4': rank = 2; break;
        case '5': rank = 3; break;
        case '6': rank = 4; break;
        case '7': rank = 5; break;
        case '8': rank = 6; break;
        case '9': rank = 7; break;
        case 'T': rank = 8; break;
        case 'J': rank = 9; break;
        case 'Q': rank = 10; break;
        case 'K': rank = 11; break;
        case 'A': rank = 12; break;
        default: rank = 0;
    }

    int suit;
    switch (s[1]) {
        case 'c': suit = 0; break;
        case 'd': suit = 1; break;
        case 'h': suit = 2; break;
        case 's': suit = 3; break;
        default: suit = 0;
    }

    return make_card(rank, suit);
}

std::vector<int> make_hand(std::initializer_list<std::string> cards) {
    std::vector<int> result;
    for (const auto& c : cards) {
        result.push_back(make_card_from_string(c));
    }
    return result;
}

TEST_CASE("Card utilities work correctly", "[hand_eval]") {
    REQUIRE(card_rank(0) == 0);   // 2 of clubs
    REQUIRE(card_suit(0) == 0);   // clubs
    REQUIRE(card_rank(51) == 12); // Ace of spades
    REQUIRE(card_suit(51) == 3);  // spades

    REQUIRE(make_card(12, 3) == 51);  // Ace of spades
    REQUIRE(make_card(0, 0) == 0);    // 2 of clubs

    REQUIRE(card_to_string(0) == "2c");
    REQUIRE(card_to_string(51) == "As");
    REQUIRE(card_to_string(make_card(9, 2)) == "Jh");
}

TEST_CASE("High card hands rank correctly", "[hand_eval]") {
    auto high = make_hand({"Ah", "Kc", "Qd", "Js", "9h"});
    auto lower = make_hand({"Ah", "Kc", "Qd", "Js", "8h"});

    HandValue h = HandEvaluator::evaluate(high);
    HandValue l = HandEvaluator::evaluate(lower);

    REQUIRE(h.rank() == HandRank::HighCard);
    REQUIRE(l.rank() == HandRank::HighCard);
    REQUIRE(h > l);  // 9 kicker beats 8 kicker
}

TEST_CASE("Pair hands rank correctly", "[hand_eval]") {
    auto pair_aces = make_hand({"Ah", "Ac", "Kd", "Qs", "Jh"});
    auto pair_kings = make_hand({"Kh", "Kc", "Ad", "Qs", "Jh"});
    auto high_card = make_hand({"Ah", "Kc", "Qd", "Js", "9h"});

    HandValue pa = HandEvaluator::evaluate(pair_aces);
    HandValue pk = HandEvaluator::evaluate(pair_kings);
    HandValue hc = HandEvaluator::evaluate(high_card);

    REQUIRE(pa.rank() == HandRank::Pair);
    REQUIRE(pk.rank() == HandRank::Pair);
    REQUIRE(pa > pk);   // Pair of aces beats pair of kings
    REQUIRE(pk > hc);   // Any pair beats high card
}

TEST_CASE("Two pair hands rank correctly", "[hand_eval]") {
    auto aces_kings = make_hand({"Ah", "Ac", "Kd", "Ks", "Qh"});
    auto aces_queens = make_hand({"Ah", "Ac", "Qd", "Qs", "Kh"});
    auto pair = make_hand({"Ah", "Ac", "Kd", "Qs", "Jh"});

    HandValue ak = HandEvaluator::evaluate(aces_kings);
    HandValue aq = HandEvaluator::evaluate(aces_queens);
    HandValue p = HandEvaluator::evaluate(pair);

    REQUIRE(ak.rank() == HandRank::TwoPair);
    REQUIRE(aq.rank() == HandRank::TwoPair);
    REQUIRE(ak > aq);   // Aces and kings beats aces and queens
    REQUIRE(aq > p);    // Two pair beats one pair
}

TEST_CASE("Three of a kind ranks correctly", "[hand_eval]") {
    auto trips_aces = make_hand({"Ah", "Ac", "Ad", "Ks", "Qh"});
    auto trips_kings = make_hand({"Kh", "Kc", "Kd", "As", "Qh"});
    auto two_pair = make_hand({"Ah", "Ac", "Kd", "Ks", "Qh"});

    HandValue ta = HandEvaluator::evaluate(trips_aces);
    HandValue tk = HandEvaluator::evaluate(trips_kings);
    HandValue tp = HandEvaluator::evaluate(two_pair);

    REQUIRE(ta.rank() == HandRank::ThreeOfAKind);
    REQUIRE(tk.rank() == HandRank::ThreeOfAKind);
    REQUIRE(ta > tk);   // Trip aces beats trip kings
    REQUIRE(tk > tp);   // Trips beats two pair
}

TEST_CASE("Straight hands rank correctly", "[hand_eval]") {
    auto broadway = make_hand({"Ah", "Kc", "Qd", "Js", "Th"});
    auto six_high = make_hand({"6h", "5c", "4d", "3s", "2h"});
    auto wheel = make_hand({"Ah", "2c", "3d", "4s", "5h"});  // A-2-3-4-5

    HandValue bw = HandEvaluator::evaluate(broadway);
    HandValue sh = HandEvaluator::evaluate(six_high);
    HandValue wh = HandEvaluator::evaluate(wheel);

    REQUIRE(bw.rank() == HandRank::Straight);
    REQUIRE(sh.rank() == HandRank::Straight);
    REQUIRE(wh.rank() == HandRank::Straight);

    REQUIRE(bw > sh);   // Broadway beats 6-high straight
    REQUIRE(sh > wh);   // 6-high beats wheel (5-high)
}

TEST_CASE("Flush hands rank correctly", "[hand_eval]") {
    auto ace_flush = make_hand({"Ah", "Kh", "Qh", "Jh", "9h"});
    auto king_flush = make_hand({"Kh", "Qh", "Jh", "8h", "7h"});  // Not a straight
    auto straight = make_hand({"Ah", "Kc", "Qd", "Js", "Th"});

    HandValue af = HandEvaluator::evaluate(ace_flush);
    HandValue kf = HandEvaluator::evaluate(king_flush);
    HandValue st = HandEvaluator::evaluate(straight);

    REQUIRE(af.rank() == HandRank::Flush);
    REQUIRE(kf.rank() == HandRank::Flush);
    REQUIRE(af > kf);   // Ace-high flush beats king-high
    REQUIRE(kf > st);   // Flush beats straight
}

TEST_CASE("Full house hands rank correctly", "[hand_eval]") {
    auto aces_full = make_hand({"Ah", "Ac", "Ad", "Ks", "Kh"});
    auto kings_full = make_hand({"Kh", "Kc", "Kd", "As", "Ah"});
    auto flush = make_hand({"Ah", "Kh", "Qh", "Jh", "9h"});

    HandValue af = HandEvaluator::evaluate(aces_full);
    HandValue kf = HandEvaluator::evaluate(kings_full);
    HandValue fl = HandEvaluator::evaluate(flush);

    REQUIRE(af.rank() == HandRank::FullHouse);
    REQUIRE(kf.rank() == HandRank::FullHouse);
    REQUIRE(af > kf);   // Aces full beats kings full
    REQUIRE(kf > fl);   // Full house beats flush
}

TEST_CASE("Four of a kind ranks correctly", "[hand_eval]") {
    auto quad_aces = make_hand({"Ah", "Ac", "Ad", "As", "Kh"});
    auto quad_kings = make_hand({"Kh", "Kc", "Kd", "Ks", "Ah"});
    auto full_house = make_hand({"Ah", "Ac", "Ad", "Ks", "Kh"});

    HandValue qa = HandEvaluator::evaluate(quad_aces);
    HandValue qk = HandEvaluator::evaluate(quad_kings);
    HandValue fh = HandEvaluator::evaluate(full_house);

    REQUIRE(qa.rank() == HandRank::FourOfAKind);
    REQUIRE(qk.rank() == HandRank::FourOfAKind);
    REQUIRE(qa > qk);   // Quad aces beats quad kings
    REQUIRE(qk > fh);   // Quads beats full house
}

TEST_CASE("Straight flush ranks correctly", "[hand_eval]") {
    auto royal = make_hand({"Ah", "Kh", "Qh", "Jh", "Th"});  // Royal flush
    auto eight_high = make_hand({"8h", "7h", "6h", "5h", "4h"});
    auto steel_wheel = make_hand({"5h", "4h", "3h", "2h", "Ah"});  // A-2-3-4-5 suited
    auto quads = make_hand({"Ah", "Ac", "Ad", "As", "Kh"});

    HandValue rf = HandEvaluator::evaluate(royal);
    HandValue eh = HandEvaluator::evaluate(eight_high);
    HandValue sw = HandEvaluator::evaluate(steel_wheel);
    HandValue qd = HandEvaluator::evaluate(quads);

    REQUIRE(rf.rank() == HandRank::StraightFlush);
    REQUIRE(eh.rank() == HandRank::StraightFlush);
    REQUIRE(sw.rank() == HandRank::StraightFlush);

    REQUIRE(rf > eh);   // Royal beats 8-high straight flush
    REQUIRE(eh > sw);   // 8-high beats steel wheel (5-high)
    REQUIRE(sw > qd);   // Any straight flush beats quads
}

TEST_CASE("7-card evaluation finds best hand", "[hand_eval]") {
    // 7 cards with flush possible
    auto seven_with_flush = make_hand({
        "Ah", "Kh", "Qh", "Jh", "9h",  // Flush
        "2c", "3d"
    });

    HandValue v = HandEvaluator::evaluate(seven_with_flush);
    REQUIRE(v.rank() == HandRank::Flush);

    // 7 cards with full house
    auto seven_with_fh = make_hand({
        "Ah", "Ac", "Ad",  // Trip aces
        "Kh", "Kc",        // Pair kings
        "2c", "3d"
    });

    v = HandEvaluator::evaluate(seven_with_fh);
    REQUIRE(v.rank() == HandRank::FullHouse);
}

TEST_CASE("Hand comparison works", "[hand_eval]") {
    std::array<int, 2> hand1 = {make_card_from_string("Ah"), make_card_from_string("Kh")};
    std::array<int, 2> hand2 = {make_card_from_string("Qh"), make_card_from_string("2h")};
    std::array<int, 5> board = {
        make_card_from_string("Th"),
        make_card_from_string("9h"),
        make_card_from_string("4h"),  // Changed to avoid straight flush
        make_card_from_string("2c"),
        make_card_from_string("3d")
    };

    // Both have flush:
    // Hand1 has A-K-T-9-4 (ace high)
    // Hand2 has Q-T-9-4-2 (queen high)
    int cmp = HandEvaluator::compare(hand1, hand2, board);
    REQUIRE(cmp > 0);  // Hand1 wins with ace-high flush
}

TEST_CASE("Hand strength calculation works", "[hand_eval]") {
    std::array<int, 2> aces = {make_card_from_string("Ah"), make_card_from_string("As")};
    std::array<int, 2> sevens = {make_card_from_string("7h"), make_card_from_string("2c")};

    // On a board of K-Q-J-5-3 rainbow
    std::vector<int> board = {
        make_card_from_string("Kd"),
        make_card_from_string("Qc"),
        make_card_from_string("Js"),
        make_card_from_string("5h"),
        make_card_from_string("3d")
    };

    double aces_hs = HandEvaluator::hand_strength(aces, board);
    double sevens_hs = HandEvaluator::hand_strength(sevens, board);

    // Pocket aces should have very high hand strength
    REQUIRE(aces_hs > 0.8);

    // 7-2 offsuit should have low hand strength
    REQUIRE(sevens_hs < 0.3);

    // Aces should beat 7-2
    REQUIRE(aces_hs > sevens_hs);
}

TEST_CASE("Hand rank to string works", "[hand_eval]") {
    REQUIRE(hand_rank_to_string(HandRank::HighCard) == "High Card");
    REQUIRE(hand_rank_to_string(HandRank::Pair) == "Pair");
    REQUIRE(hand_rank_to_string(HandRank::TwoPair) == "Two Pair");
    REQUIRE(hand_rank_to_string(HandRank::StraightFlush) == "Straight Flush");
}
