#pragma once

#include <memory>
#include <vector>
#include <map>
#include <set>
#include "GameTree.hpp"
#include "GameTypes.hpp"

namespace quantnet::poker {

// Kuhn Poker implementation
//
// Rules:
// - 3-card deck: Jack (0), Queen (1), King (2)
// - Each player antes 1 chip
// - Each player is dealt one card
// - Player 0 acts first: check or bet (1 chip)
//   - If check: Player 1 can check (showdown) or bet
//     - If P1 bets: P0 can call (1 chip) or fold
//   - If bet: Player 1 can call (1 chip) or fold
// - Higher card wins at showdown
// - Payoffs: +/- ante (1) or +/- ante+bet (2)
//
// Information sets (12 total):
//   P0: J, Q, K (first action)
//   P1: J, Q, K after check (c)
//   P1: J, Q, K after bet (b)
//   P0: J, Q, K after check-bet (cb)
class KuhnPoker : public PokerGame {
public:
    KuhnPoker();

    void build_tree() override;
    const GameNode* root() const override { return root_.get(); }
    std::vector<InfoSet> get_info_sets() const override;
    std::string name() const override { return "Kuhn Poker"; }
    int deck_size() const override { return 3; }

    // Card comparison: King > Queen > Jack
    static int compare_cards(Card c1, Card c2);

    // Get card name
    static std::string card_name(Card c);

    // Build info set ID for a player node
    static InfoSetId make_info_set_id(PlayerId player, Card card, const std::string& history);

private:
    std::unique_ptr<GameNode> root_;
    std::set<InfoSetId> info_set_ids_;  // Track all info sets seen during build

    // Recursive tree building
    void build_subtree(
        GameNode* node,
        PlayerId to_act,
        const std::string& history,
        Card p0_card,
        Card p1_card,
        int pot,
        int p0_bet,   // How much P0 has put in beyond ante
        int p1_bet    // How much P1 has put in beyond ante
    );

    // Create terminal node with showdown
    void make_showdown(GameNode* node, Card p0_card, Card p1_card, int pot);

    // Create terminal node when a player folds
    void make_fold_terminal(GameNode* node, PlayerId folder, int pot);
};

} // namespace quantnet::poker
