#pragma once

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <functional>
#include "GameTypes.hpp"

namespace quantnet::poker {

// Forward declaration
struct GameNode;

// Child connection: either an action or a chance outcome
struct ChildEdge {
    Action action = Action::Check;  // For player nodes
    Card card = -1;                 // For chance nodes (card dealt)
    double probability = 1.0;       // For chance nodes
    std::unique_ptr<GameNode> child;
};

// Node in the game tree
struct GameNode {
    NodeType type = NodeType::Terminal;
    PlayerId player = PLAYER_0;          // For Player nodes
    InfoSetId info_set_id;               // For Player nodes
    std::vector<Action> legal_actions;   // For Player nodes
    std::vector<ChildEdge> children;     // Children (actions or chance outcomes)
    double payoff = 0.0;                 // For Terminal nodes (payoff to PLAYER_0)
    int pot = 0;                         // Current pot size
    std::string history;                 // Action history string
    Card p0_card = -1;                   // Player 0's private card
    Card p1_card = -1;                   // Player 1's private card
    Card public_card = -1;               // Public card (for Leduc)

    // Navigate to child by action
    GameNode* get_child(Action a) const {
        for (const auto& edge : children) {
            if (edge.action == a) {
                return edge.child.get();
            }
        }
        return nullptr;
    }

    // Navigate to child by dealt card
    GameNode* get_chance_child(Card c) const {
        for (const auto& edge : children) {
            if (edge.card == c) {
                return edge.child.get();
            }
        }
        return nullptr;
    }

    // Check if action is legal
    bool is_legal(Action a) const {
        for (Action legal : legal_actions) {
            if (legal == a) return true;
        }
        return false;
    }
};

// Visitor callback types for tree traversal
using NodeVisitor = std::function<void(const GameNode*, int depth)>;
using MutableNodeVisitor = std::function<void(GameNode*, int depth)>;

// Traverse game tree in pre-order
inline void traverse_tree(const GameNode* node, NodeVisitor visitor, int depth = 0) {
    if (!node) return;
    visitor(node, depth);
    for (const auto& edge : node->children) {
        traverse_tree(edge.child.get(), visitor, depth + 1);
    }
}

// Traverse and potentially modify tree
inline void traverse_tree_mut(GameNode* node, MutableNodeVisitor visitor, int depth = 0) {
    if (!node) return;
    visitor(node, depth);
    for (auto& edge : node->children) {
        traverse_tree_mut(edge.child.get(), visitor, depth + 1);
    }
}

// Count nodes of each type
struct TreeStats {
    int total_nodes = 0;
    int chance_nodes = 0;
    int player_nodes = 0;
    int terminal_nodes = 0;
    int max_depth = 0;
};

inline TreeStats compute_tree_stats(const GameNode* root) {
    TreeStats stats;
    traverse_tree(root, [&stats](const GameNode* node, int depth) {
        stats.total_nodes++;
        stats.max_depth = std::max(stats.max_depth, depth);
        switch (node->type) {
            case NodeType::Chance:   stats.chance_nodes++; break;
            case NodeType::Player:   stats.player_nodes++; break;
            case NodeType::Terminal: stats.terminal_nodes++; break;
        }
    });
    return stats;
}

// Abstract base class for poker games
class PokerGame {
public:
    virtual ~PokerGame() = default;

    // Build the complete game tree
    virtual void build_tree() = 0;

    // Get root node
    virtual const GameNode* root() const = 0;

    // Get all information sets with their legal actions
    virtual std::vector<InfoSet> get_info_sets() const = 0;

    // Get game name
    virtual std::string name() const = 0;

    // Get number of cards in deck
    virtual int deck_size() const = 0;
};

} // namespace quantnet::poker
