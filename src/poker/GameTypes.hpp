#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace quantnet::poker {

// Card representation: integer index
// For Kuhn: 0=Jack, 1=Queen, 2=King
// For Leduc: 0=J1, 1=J2, 2=Q1, 3=Q2, 4=K1, 5=K2 (rank * 2 + suit)
using Card = int;

// Player identifier
using PlayerId = int;
constexpr PlayerId PLAYER_0 = 0;
constexpr PlayerId PLAYER_1 = 1;
constexpr PlayerId CHANCE = -1;

// Actions available in poker
enum class Action : uint8_t {
    Check = 0,    // Pass, no bet
    Bet = 1,      // Add chips to pot
    Call = 2,     // Match opponent's bet
    Fold = 3,     // Give up hand
    Raise = 4     // Increase bet (for Leduc)
};

inline std::string action_to_string(Action a) {
    switch (a) {
        case Action::Check: return "check";
        case Action::Bet:   return "bet";
        case Action::Call:  return "call";
        case Action::Fold:  return "fold";
        case Action::Raise: return "raise";
    }
    return "unknown";
}

inline char action_to_char(Action a) {
    switch (a) {
        case Action::Check: return 'c';
        case Action::Bet:   return 'b';
        case Action::Call:  return 'k';  // 'k' for call to avoid confusion with 'c'heck
        case Action::Fold:  return 'f';
        case Action::Raise: return 'r';
    }
    return '?';
}

// Node types in the game tree
enum class NodeType : uint8_t {
    Chance,    // Nature deals cards
    Player,    // Player makes decision
    Terminal   // Game over, payoff determined
};

// Information set identifier
// Format depends on game:
//   Kuhn:  "P{player}:{card}:{history}"  e.g., "P0:Q:cb"
//   Leduc: "P{player}:{private}:{public}:{history}" e.g., "P0:J:Q:cbr"
using InfoSetId = std::string;

// Information set with legal actions
struct InfoSet {
    InfoSetId id;
    PlayerId player;
    std::vector<Action> legal_actions;

    bool operator<(const InfoSet& other) const { return id < other.id; }
};

// Index mapping between flat vector positions and (infoset, action) pairs
class InfoSetIndex {
public:
    // Build index from list of information sets
    void build(const std::vector<InfoSet>& info_sets) {
        info_sets_.clear();
        id_to_idx_.clear();
        flat_to_pair_.clear();
        pair_to_flat_.clear();

        int flat_idx = 0;
        for (size_t i = 0; i < info_sets.size(); ++i) {
            const auto& is = info_sets[i];
            info_sets_.push_back(is);
            id_to_idx_[is.id] = static_cast<int>(i);

            for (size_t a = 0; a < is.legal_actions.size(); ++a) {
                flat_to_pair_.push_back({static_cast<int>(i), static_cast<int>(a)});
                pair_to_flat_[{is.id, is.legal_actions[a]}] = flat_idx;
                flat_idx++;
            }
        }
        total_dim_ = flat_idx;
    }

    // Total dimension of the strategy vector
    int total_dim() const { return total_dim_; }

    // Number of information sets
    int num_info_sets() const { return static_cast<int>(info_sets_.size()); }

    // Get info set by index
    const InfoSet& info_set(int idx) const { return info_sets_[idx]; }

    // Get info set index by ID
    int info_set_idx(const InfoSetId& id) const {
        auto it = id_to_idx_.find(id);
        return (it != id_to_idx_.end()) ? it->second : -1;
    }

    // Get (info_set_idx, action_idx) from flat index
    std::pair<int, int> flat_to_pair(int flat_idx) const {
        return flat_to_pair_[flat_idx];
    }

    // Get flat index from (info_set_id, action)
    int pair_to_flat(const InfoSetId& id, Action action) const {
        auto it = pair_to_flat_.find({id, action});
        return (it != pair_to_flat_.end()) ? it->second : -1;
    }

    // Get start index in flat vector for an info set
    int info_set_start(int is_idx) const {
        int start = 0;
        for (int i = 0; i < is_idx; ++i) {
            start += static_cast<int>(info_sets_[i].legal_actions.size());
        }
        return start;
    }

    // Iterate over all info sets
    const std::vector<InfoSet>& all_info_sets() const { return info_sets_; }

private:
    std::vector<InfoSet> info_sets_;
    std::map<InfoSetId, int> id_to_idx_;
    std::vector<std::pair<int, int>> flat_to_pair_;  // flat_idx -> (is_idx, action_idx)
    std::map<std::pair<InfoSetId, Action>, int> pair_to_flat_;
    int total_dim_ = 0;
};

// Card names for display
inline std::string card_name_kuhn(Card c) {
    switch (c) {
        case 0: return "J";
        case 1: return "Q";
        case 2: return "K";
        default: return "?";
    }
}

inline std::string card_name_leduc(Card c) {
    // Leduc: 6 cards, 3 ranks x 2 suits
    // c = rank * 2 + suit
    int rank = c / 2;
    int suit = c % 2;
    std::string r;
    switch (rank) {
        case 0: r = "J"; break;
        case 1: r = "Q"; break;
        case 2: r = "K"; break;
        default: r = "?";
    }
    return r + (suit == 0 ? "s" : "h");  // spade or heart
}

} // namespace quantnet::poker
