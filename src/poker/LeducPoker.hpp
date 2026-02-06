#pragma once

#include <memory>
#include <vector>
#include <set>
#include "GameTree.hpp"
#include "GameTypes.hpp"

namespace quantnet::poker {

// Leduc Poker implementation
//
// Rules:
// - 6-card deck: Jack, Queen, King in two suits (Js, Jh, Qs, Qh, Ks, Kh)
// - Each player antes 1 chip
// - Round 1: Deal one private card to each player
//   - Betting round with small bet (2 chips), max 2 raises
// - Round 2: Deal one public card
//   - Betting round with big bet (4 chips), max 2 raises
// - Showdown: Pair beats high card, else higher card wins
//
// Betting actions:
// - After ante, first player can check or bet
// - After check, second player can check or bet
// - After bet, opponent can fold, call, or raise (up to max raises)
//
// Information sets:
// - Round 1: (player, private_card, history)
// - Round 2: (player, private_card, public_card, history)
class LeducPoker : public PokerGame {
public:
    // Configuration
    static constexpr int ANTE = 1;
    static constexpr int SMALL_BET = 2;
    static constexpr int BIG_BET = 4;
    static constexpr int MAX_RAISES = 2;
    static constexpr int NUM_CARDS = 6;  // 3 ranks x 2 suits

    LeducPoker();

    void build_tree() override;
    const GameNode* root() const override { return root_.get(); }
    std::vector<InfoSet> get_info_sets() const override;
    std::string name() const override { return "Leduc Poker"; }
    int deck_size() const override { return NUM_CARDS; }

    // Card rank (0=J, 1=Q, 2=K)
    static int card_rank(Card c) { return c / 2; }

    // Card suit (0=spade, 1=heart)
    static int card_suit(Card c) { return c % 2; }

    // Compare hands at showdown
    // Returns >0 if P0 wins, <0 if P1 wins, 0 if tie
    static int compare_hands(Card p0_card, Card p1_card, Card public_card);

    // Get card name
    static std::string card_name(Card c);

    // Build info set ID
    static InfoSetId make_info_set_id(
        PlayerId player, Card private_card, Card public_card,
        const std::string& history, int round
    );

private:
    std::unique_ptr<GameNode> root_;
    std::set<InfoSetId> info_set_ids_;

    // Build subtree for a betting round
    void build_betting_round(
        GameNode* node,
        PlayerId first_to_act,
        const std::string& history,
        Card p0_card,
        Card p1_card,
        Card public_card,  // -1 if round 1
        int pot,
        int to_call,       // Amount to call (0 if no outstanding bet)
        int raises_left,
        int round,         // 1 or 2
        int bet_size       // SMALL_BET or BIG_BET
    );

    // Continue to round 2 or showdown after round 1 completes
    void continue_after_round1(
        GameNode* node,
        Card p0_card,
        Card p1_card,
        int pot,
        const std::string& history
    );

    // Create showdown terminal node
    void make_showdown(GameNode* node, Card p0_card, Card p1_card, Card public_card, int pot);

    // Create fold terminal node
    void make_fold_terminal(GameNode* node, PlayerId folder, int pot);
};

} // namespace quantnet::poker
