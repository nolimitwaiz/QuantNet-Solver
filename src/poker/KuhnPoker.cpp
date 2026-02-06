#include "KuhnPoker.hpp"
#include <algorithm>
#include <stdexcept>

namespace quantnet::poker {

KuhnPoker::KuhnPoker() {
    build_tree();
}

int KuhnPoker::compare_cards(Card c1, Card c2) {
    // Higher card wins: K > Q > J
    if (c1 > c2) return 1;
    if (c1 < c2) return -1;
    return 0;  // Tie (shouldn't happen in Kuhn)
}

std::string KuhnPoker::card_name(Card c) {
    switch (c) {
        case 0: return "J";
        case 1: return "Q";
        case 2: return "K";
        default: return "?";
    }
}

InfoSetId KuhnPoker::make_info_set_id(PlayerId player, Card card, const std::string& history) {
    // Format: "P{player}:{card}:{history}"
    // Examples: "P0:Q:", "P1:K:b", "P0:J:cb"
    return "P" + std::to_string(player) + ":" + card_name(card) + ":" + history;
}

void KuhnPoker::build_tree() {
    info_set_ids_.clear();

    // Root is a chance node that deals cards
    root_ = std::make_unique<GameNode>();
    root_->type = NodeType::Chance;
    root_->player = CHANCE;
    root_->pot = 2;  // Both players ante 1
    root_->history = "";

    // Deal all 6 possible card combinations (3 choose 2, ordered)
    // Probability: 1/6 each
    const double deal_prob = 1.0 / 6.0;

    for (Card p0_card = 0; p0_card < 3; ++p0_card) {
        for (Card p1_card = 0; p1_card < 3; ++p1_card) {
            if (p0_card == p1_card) continue;  // Can't deal same card twice

            // Create child for this deal
            ChildEdge edge;
            edge.card = p0_card * 10 + p1_card;  // Encode both cards
            edge.probability = deal_prob;
            edge.child = std::make_unique<GameNode>();

            GameNode* child = edge.child.get();
            child->type = NodeType::Player;
            child->player = PLAYER_0;
            child->p0_card = p0_card;
            child->p1_card = p1_card;
            child->pot = 2;
            child->history = "";
            child->legal_actions = {Action::Check, Action::Bet};
            child->info_set_id = make_info_set_id(PLAYER_0, p0_card, "");
            info_set_ids_.insert(child->info_set_id);

            // Build subtree from P0's first decision
            build_subtree(child, PLAYER_0, "", p0_card, p1_card, 2, 0, 0);

            root_->children.push_back(std::move(edge));
        }
    }
}

void KuhnPoker::build_subtree(
    GameNode* node,
    PlayerId to_act,
    const std::string& history,
    Card p0_card,
    Card p1_card,
    int pot,
    int p0_bet,
    int p1_bet
) {
    // This function adds children to an existing player node

    for (Action action : node->legal_actions) {
        ChildEdge edge;
        edge.action = action;
        edge.child = std::make_unique<GameNode>();
        GameNode* child = edge.child.get();

        child->p0_card = p0_card;
        child->p1_card = p1_card;

        std::string new_history = history + action_to_char(action);
        child->history = new_history;

        // Process action
        if (to_act == PLAYER_0) {
            // Player 0 is acting
            if (action == Action::Check) {
                // P0 checks, P1 acts
                child->type = NodeType::Player;
                child->player = PLAYER_1;
                child->pot = pot;
                child->legal_actions = {Action::Check, Action::Bet};
                child->info_set_id = make_info_set_id(PLAYER_1, p1_card, new_history);
                info_set_ids_.insert(child->info_set_id);

                build_subtree(child, PLAYER_1, new_history, p0_card, p1_card, pot, p0_bet, p1_bet);
            }
            else if (action == Action::Bet) {
                // P0 bets 1, P1 must respond
                child->type = NodeType::Player;
                child->player = PLAYER_1;
                child->pot = pot + 1;
                child->legal_actions = {Action::Call, Action::Fold};
                child->info_set_id = make_info_set_id(PLAYER_1, p1_card, new_history);
                info_set_ids_.insert(child->info_set_id);

                build_subtree(child, PLAYER_1, new_history, p0_card, p1_card, pot + 1, p0_bet + 1, p1_bet);
            }
            else if (action == Action::Call) {
                // P0 calls P1's bet after cb
                child->pot = pot + 1;
                make_showdown(child, p0_card, p1_card, pot + 1);
            }
            else if (action == Action::Fold) {
                // P0 folds after cb
                make_fold_terminal(child, PLAYER_0, pot);
            }
        }
        else {
            // Player 1 is acting
            if (action == Action::Check) {
                // P1 checks after P0 check -> showdown
                child->pot = pot;
                make_showdown(child, p0_card, p1_card, pot);
            }
            else if (action == Action::Bet) {
                // P1 bets 1 after P0 check -> P0 must respond
                child->type = NodeType::Player;
                child->player = PLAYER_0;
                child->pot = pot + 1;
                child->legal_actions = {Action::Call, Action::Fold};
                child->info_set_id = make_info_set_id(PLAYER_0, p0_card, new_history);
                info_set_ids_.insert(child->info_set_id);

                build_subtree(child, PLAYER_0, new_history, p0_card, p1_card, pot + 1, p0_bet, p1_bet + 1);
            }
            else if (action == Action::Call) {
                // P1 calls P0's bet -> showdown
                child->pot = pot + 1;
                make_showdown(child, p0_card, p1_card, pot + 1);
            }
            else if (action == Action::Fold) {
                // P1 folds to P0's bet -> P0 wins
                make_fold_terminal(child, PLAYER_1, pot);
            }
        }

        node->children.push_back(std::move(edge));
    }
}

void KuhnPoker::make_showdown(GameNode* node, Card p0_card, Card p1_card, int pot) {
    node->type = NodeType::Terminal;
    node->player = -1;
    node->pot = pot;

    // Determine winner and payoff
    int cmp = compare_cards(p0_card, p1_card);
    if (cmp > 0) {
        // P0 wins
        // P0's payoff is what P1 put in (half the pot minus P0's contribution)
        // With antes of 1 each, P0 wins P1's contribution
        node->payoff = static_cast<double>(pot) / 2.0;
    }
    else if (cmp < 0) {
        // P1 wins
        node->payoff = -static_cast<double>(pot) / 2.0;
    }
    else {
        // Tie (shouldn't happen in Kuhn)
        node->payoff = 0.0;
    }
}

void KuhnPoker::make_fold_terminal(GameNode* node, PlayerId folder, int pot) {
    node->type = NodeType::Terminal;
    node->player = -1;
    node->pot = pot;

    if (folder == PLAYER_0) {
        // P0 folds, P1 wins what's in pot
        // P0 loses their contribution (ante + any bet)
        // Payoff is -(pot - P1's contribution) = -(P0's contribution)
        // But we express payoff as P0's gain/loss from the pot
        // If P0 folds, they lose their investment
        node->payoff = -1.0;  // P0 loses their ante (or ante + bet in cb->fold)
    }
    else {
        // P1 folds, P0 wins
        node->payoff = 1.0;  // P0 wins P1's ante
    }
}

std::vector<InfoSet> KuhnPoker::get_info_sets() const {
    std::vector<InfoSet> result;

    // Traverse tree to find all player nodes and extract info sets
    std::map<InfoSetId, InfoSet> info_set_map;

    traverse_tree(root_.get(), [&info_set_map](const GameNode* node, int) {
        if (node->type == NodeType::Player) {
            if (info_set_map.find(node->info_set_id) == info_set_map.end()) {
                InfoSet is;
                is.id = node->info_set_id;
                is.player = node->player;
                is.legal_actions = node->legal_actions;
                info_set_map[is.id] = is;
            }
        }
    });

    // Convert to vector, sorted by ID for deterministic ordering
    for (const auto& [id, is] : info_set_map) {
        result.push_back(is);
    }

    return result;
}

} // namespace quantnet::poker
