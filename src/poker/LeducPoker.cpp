#include "LeducPoker.hpp"
#include <algorithm>

namespace quantnet::poker {

LeducPoker::LeducPoker() {
    build_tree();
}

std::string LeducPoker::card_name(Card c) {
    int rank = card_rank(c);
    int suit = card_suit(c);
    std::string r;
    switch (rank) {
        case 0: r = "J"; break;
        case 1: r = "Q"; break;
        case 2: r = "K"; break;
        default: r = "?";
    }
    return r + (suit == 0 ? "s" : "h");
}

int LeducPoker::compare_hands(Card p0_card, Card p1_card, Card public_card) {
    int p0_rank = card_rank(p0_card);
    int p1_rank = card_rank(p1_card);
    int pub_rank = card_rank(public_card);

    bool p0_pair = (p0_rank == pub_rank);
    bool p1_pair = (p1_rank == pub_rank);

    // Pair beats no pair
    if (p0_pair && !p1_pair) return 1;
    if (!p0_pair && p1_pair) return -1;

    // Both have pair or neither has pair: higher card wins
    if (p0_rank > p1_rank) return 1;
    if (p0_rank < p1_rank) return -1;

    return 0;  // Tie
}

InfoSetId LeducPoker::make_info_set_id(
    PlayerId player, Card private_card, Card public_card,
    const std::string& history, int round
) {
    // Format: "P{player}:{private}:{public}:{history}"
    // Round 1: public = "-"
    std::string pub_str = (public_card < 0) ? "-" : card_name(public_card);
    // Use rank only for private card (suit doesn't matter for strategy)
    std::string priv_str = "";
    switch (card_rank(private_card)) {
        case 0: priv_str = "J"; break;
        case 1: priv_str = "Q"; break;
        case 2: priv_str = "K"; break;
    }
    // For public card in info set, also use rank only
    if (public_card >= 0) {
        switch (card_rank(public_card)) {
            case 0: pub_str = "J"; break;
            case 1: pub_str = "Q"; break;
            case 2: pub_str = "K"; break;
        }
    }
    return "P" + std::to_string(player) + ":" + priv_str + ":" + pub_str + ":R" +
           std::to_string(round) + ":" + history;
}

void LeducPoker::build_tree() {
    info_set_ids_.clear();

    // Root is a chance node that deals private cards
    root_ = std::make_unique<GameNode>();
    root_->type = NodeType::Chance;
    root_->player = CHANCE;
    root_->pot = 2 * ANTE;
    root_->history = "";

    // Deal all possible private card combinations
    // 6 cards, 2 to each player, order matters: 6 * 5 = 30 combinations
    const double deal_prob = 1.0 / 30.0;

    for (Card p0_card = 0; p0_card < NUM_CARDS; ++p0_card) {
        for (Card p1_card = 0; p1_card < NUM_CARDS; ++p1_card) {
            if (p0_card == p1_card) continue;

            ChildEdge edge;
            edge.card = p0_card * 10 + p1_card;  // Encode both cards
            edge.probability = deal_prob;
            edge.child = std::make_unique<GameNode>();

            GameNode* child = edge.child.get();
            child->type = NodeType::Player;
            child->player = PLAYER_0;
            child->p0_card = p0_card;
            child->p1_card = p1_card;
            child->public_card = -1;
            child->pot = 2 * ANTE;
            child->history = "";
            child->legal_actions = {Action::Check, Action::Bet};
            child->info_set_id = make_info_set_id(PLAYER_0, p0_card, -1, "", 1);
            info_set_ids_.insert(child->info_set_id);

            // Build round 1 betting
            build_betting_round(
                child, PLAYER_0, "", p0_card, p1_card, -1,
                2 * ANTE, 0, MAX_RAISES, 1, SMALL_BET
            );

            root_->children.push_back(std::move(edge));
        }
    }
}

void LeducPoker::build_betting_round(
    GameNode* node,
    PlayerId first_to_act,
    const std::string& history,
    Card p0_card,
    Card p1_card,
    Card public_card,
    int pot,
    int to_call,
    int raises_left,
    int round,
    int bet_size
) {
    // Build children for each legal action
    for (Action action : node->legal_actions) {
        ChildEdge edge;
        edge.action = action;
        edge.child = std::make_unique<GameNode>();
        GameNode* child = edge.child.get();

        child->p0_card = p0_card;
        child->p1_card = p1_card;
        child->public_card = public_card;

        std::string new_history = history + action_to_char(action);
        child->history = new_history;

        PlayerId current = node->player;
        PlayerId opponent = (current == PLAYER_0) ? PLAYER_1 : PLAYER_0;

        if (action == Action::Fold) {
            // Player folds, opponent wins
            make_fold_terminal(child, current, pot);
        }
        else if (action == Action::Check) {
            if (to_call == 0 && history.empty()) {
                // First check: opponent acts
                child->type = NodeType::Player;
                child->player = opponent;
                child->pot = pot;
                child->legal_actions = {Action::Check, Action::Bet};
                Card opp_card = (opponent == PLAYER_0) ? p0_card : p1_card;
                child->info_set_id = make_info_set_id(opponent, opp_card, public_card, new_history, round);
                info_set_ids_.insert(child->info_set_id);

                build_betting_round(
                    child, first_to_act, new_history, p0_card, p1_card, public_card,
                    pot, 0, raises_left, round, bet_size
                );
            }
            else if (to_call == 0) {
                // Second check: round ends
                if (round == 1) {
                    continue_after_round1(child, p0_card, p1_card, pot, new_history);
                } else {
                    make_showdown(child, p0_card, p1_card, public_card, pot);
                }
            }
        }
        else if (action == Action::Bet) {
            // Bet or raise
            int new_pot = pot + bet_size;
            child->type = NodeType::Player;
            child->player = opponent;
            child->pot = new_pot;

            // Opponent can fold, call, or raise (if raises remain)
            if (raises_left > 0) {
                child->legal_actions = {Action::Fold, Action::Call, Action::Raise};
            } else {
                child->legal_actions = {Action::Fold, Action::Call};
            }

            Card opp_card = (opponent == PLAYER_0) ? p0_card : p1_card;
            child->info_set_id = make_info_set_id(opponent, opp_card, public_card, new_history, round);
            info_set_ids_.insert(child->info_set_id);

            build_betting_round(
                child, first_to_act, new_history, p0_card, p1_card, public_card,
                new_pot, bet_size, raises_left, round, bet_size
            );
        }
        else if (action == Action::Call) {
            // Call the outstanding bet
            int new_pot = pot + to_call;
            child->pot = new_pot;

            if (round == 1) {
                continue_after_round1(child, p0_card, p1_card, new_pot, new_history);
            } else {
                make_showdown(child, p0_card, p1_card, public_card, new_pot);
            }
        }
        else if (action == Action::Raise) {
            // Raise: call + additional bet
            int new_pot = pot + to_call + bet_size;
            child->type = NodeType::Player;
            child->player = opponent;
            child->pot = new_pot;

            // After raise, opponent can fold, call, or raise (if raises remain)
            int new_raises = raises_left - 1;
            if (new_raises > 0) {
                child->legal_actions = {Action::Fold, Action::Call, Action::Raise};
            } else {
                child->legal_actions = {Action::Fold, Action::Call};
            }

            Card opp_card = (opponent == PLAYER_0) ? p0_card : p1_card;
            child->info_set_id = make_info_set_id(opponent, opp_card, public_card, new_history, round);
            info_set_ids_.insert(child->info_set_id);

            build_betting_round(
                child, first_to_act, new_history, p0_card, p1_card, public_card,
                new_pot, bet_size, new_raises, round, bet_size
            );
        }

        node->children.push_back(std::move(edge));
    }
}

void LeducPoker::continue_after_round1(
    GameNode* node,
    Card p0_card,
    Card p1_card,
    int pot,
    const std::string& history
) {
    // Deal public card: chance node
    node->type = NodeType::Chance;
    node->player = CHANCE;
    node->pot = pot;

    // 4 remaining cards can be dealt
    int cards_remaining = 0;
    for (Card c = 0; c < NUM_CARDS; ++c) {
        if (c != p0_card && c != p1_card) cards_remaining++;
    }
    double deal_prob = 1.0 / cards_remaining;

    for (Card pub = 0; pub < NUM_CARDS; ++pub) {
        if (pub == p0_card || pub == p1_card) continue;

        ChildEdge edge;
        edge.card = pub;
        edge.probability = deal_prob;
        edge.child = std::make_unique<GameNode>();

        GameNode* child = edge.child.get();
        child->type = NodeType::Player;
        child->player = PLAYER_0;  // P0 acts first in round 2
        child->p0_card = p0_card;
        child->p1_card = p1_card;
        child->public_card = pub;
        child->pot = pot;
        child->history = history + "|";  // | separates rounds
        child->legal_actions = {Action::Check, Action::Bet};
        child->info_set_id = make_info_set_id(PLAYER_0, p0_card, pub, child->history, 2);
        info_set_ids_.insert(child->info_set_id);

        build_betting_round(
            child, PLAYER_0, child->history, p0_card, p1_card, pub,
            pot, 0, MAX_RAISES, 2, BIG_BET
        );

        node->children.push_back(std::move(edge));
    }
}

void LeducPoker::make_showdown(GameNode* node, Card p0_card, Card p1_card, Card public_card, int pot) {
    node->type = NodeType::Terminal;
    node->player = -1;
    node->pot = pot;

    int cmp = compare_hands(p0_card, p1_card, public_card);
    if (cmp > 0) {
        // P0 wins
        node->payoff = static_cast<double>(pot) / 2.0;
    } else if (cmp < 0) {
        // P1 wins
        node->payoff = -static_cast<double>(pot) / 2.0;
    } else {
        // Tie: split pot
        node->payoff = 0.0;
    }
}

void LeducPoker::make_fold_terminal(GameNode* node, PlayerId folder, int pot) {
    node->type = NodeType::Terminal;
    node->player = -1;
    node->pot = pot;

    if (folder == PLAYER_0) {
        // P0 folds, loses their investment
        // Calculate P0's contribution and negate it
        node->payoff = -static_cast<double>(pot) / 2.0;
    } else {
        // P1 folds, P0 wins P1's investment
        node->payoff = static_cast<double>(pot) / 2.0;
    }
}

std::vector<InfoSet> LeducPoker::get_info_sets() const {
    std::vector<InfoSet> result;
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

    for (const auto& [id, is] : info_set_map) {
        result.push_back(is);
    }

    return result;
}

} // namespace quantnet::poker
