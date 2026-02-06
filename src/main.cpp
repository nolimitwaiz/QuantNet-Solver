// QuantNet-Solver: Newton-based Quantal Response Equilibrium solver for poker
//
// This solver finds equilibrium strategies for imperfect-information games
// using Newton's method on the QRE fixed-point equation:
//   σ = LogitBR_β(σ)
//
// Usage:
//   ./quantnet_solver [options]
//
// Options:
//   --game kuhn|leduc    Game to solve (default: kuhn)
//   --beta <value>       Target temperature (default: 10.0)
//   --tol <value>        Convergence tolerance (default: 1e-8)
//   --max-iters <n>      Max Newton iterations per beta (default: 50)
//   --output <path>      Output JSON file for visualization (default: viz/solver_output.json)
//   --verbose            Print iteration details

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>

#include "solver/NewtonSolver.hpp"
#include "poker/KuhnPoker.hpp"
#include "poker/LeducPoker.hpp"
#include "poker/Strategy.hpp"
#include "poker/QRE.hpp"
#include "poker/ExpectedValue.hpp"
#include "network/SimpleTelemetry.hpp"
#include "network/Telemetry.hpp"

using namespace quantnet;

// Command line arguments
struct Args {
    std::string game = "kuhn";
    double target_beta = 10.0;
    double tol = 1e-8;
    int max_iters = 50;
    std::string output_path = "viz/solver_output.json";
    bool verbose = false;
};

Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--game" && i + 1 < argc) {
            args.game = argv[++i];
        } else if (arg == "--beta" && i + 1 < argc) {
            args.target_beta = std::stod(argv[++i]);
        } else if (arg == "--tol" && i + 1 < argc) {
            args.tol = std::stod(argv[++i]);
        } else if (arg == "--max-iters" && i + 1 < argc) {
            args.max_iters = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            args.output_path = argv[++i];
        } else if (arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "QuantNet-Solver: Newton-based QRE solver for poker\n\n"
                      << "Usage: quantnet_solver [options]\n\n"
                      << "Options:\n"
                      << "  --game kuhn|leduc    Game to solve (default: kuhn)\n"
                      << "  --beta <value>       Target temperature (default: 10.0)\n"
                      << "  --tol <value>        Convergence tolerance (default: 1e-8)\n"
                      << "  --max-iters <n>      Max Newton iterations per beta (default: 50)\n"
                      << "  --output <path>      JSON file for visualization (default: viz/solver_output.json)\n"
                      << "  --verbose            Print iteration details\n"
                      << "  --help               Show this help\n";
            std::exit(0);
        }
    }
    return args;
}

// Beta continuation schedule: start low, increase to target
std::vector<double> make_beta_schedule(double target_beta) {
    std::vector<double> schedule;

    // Start with low beta (near uniform)
    schedule.push_back(0.01);

    // Geometric progression
    double beta = 0.05;
    while (beta < target_beta) {
        schedule.push_back(beta);
        beta *= 2.0;
    }

    // Always include target
    schedule.push_back(target_beta);

    return schedule;
}

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    std::cout << "======================================\n";
    std::cout << "  QuantNet-Solver v1.0\n";
    std::cout << "  Newton-based QRE Poker Solver\n";
    std::cout << "======================================\n\n";

    // Create game
    std::unique_ptr<poker::PokerGame> game;
    if (args.game == "kuhn") {
        game = std::make_unique<poker::KuhnPoker>();
    } else if (args.game == "leduc") {
        game = std::make_unique<poker::LeducPoker>();
    } else {
        std::cerr << "Unknown game: " << args.game << std::endl;
        return 1;
    }

    std::cout << "Game: " << game->name() << std::endl;

    // Get game stats
    auto tree_stats = poker::compute_tree_stats(game->root());
    auto info_sets = game->get_info_sets();

    std::cout << "Tree nodes: " << tree_stats.total_nodes << std::endl;
    std::cout << "  - Chance: " << tree_stats.chance_nodes << std::endl;
    std::cout << "  - Player: " << tree_stats.player_nodes << std::endl;
    std::cout << "  - Terminal: " << tree_stats.terminal_nodes << std::endl;
    std::cout << "Information sets: " << info_sets.size() << std::endl;

    // Build info set index
    poker::InfoSetIndex index;
    index.build(info_sets);
    std::cout << "Strategy dimensions: " << index.total_dim() << std::endl;
    std::cout << std::endl;

    // Create telemetry for visualization
    network::SimpleTelemetry telemetry(args.output_path);
    std::cout << "Writing telemetry to: " << args.output_path << std::endl;
    std::cout << "Open viz/index.html in a browser to see live visualization\n\n";

    // Configure Newton solver
    solver::NewtonConfig config;
    config.tol = args.tol;
    config.max_iters = args.max_iters;
    config.verbose = args.verbose;
    config.central_diff = true;
    config.fd_step = 1e-6;

    solver::NewtonSolver newton(config);

    // Initialize strategy to uniform (zero logits)
    Eigen::VectorXd w = Eigen::VectorXd::Zero(index.total_dim());

    // Create QRE residual
    poker::QREResidual qre(*game, 0.01);

    // Beta continuation schedule
    auto beta_schedule = make_beta_schedule(args.target_beta);

    std::cout << "Beta schedule: ";
    for (double b : beta_schedule) {
        std::cout << b << " ";
    }
    std::cout << "\n\n";

    int total_iters = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Solve for each beta in schedule
    for (double beta : beta_schedule) {
        qre.set_beta(beta);

        std::cout << "Solving for beta = " << std::fixed << std::setprecision(2) << beta << "...\n";

        // Set up callback for telemetry
        newton.set_callback([&](const solver::IterationStats& stats, const Eigen::VectorXd& current_x) {
            total_iters++;

            // Compute current strategy, exploitability, and expected values
            poker::Strategy sigma = poker::Strategy::from_logits(current_x, index);
            double exploit = poker::compute_exploitability(game->root(), sigma);
            double ev = poker::compute_ev(game->root(), sigma);

            // Compute per-action expected utilities (the "why" behind each strategy choice)
            auto all_eu = poker::compute_all_expected_utilities(*game, sigma, index);
            nlohmann::json action_evs_json;
            for (const auto& [is_id, action_map] : all_eu) {
                nlohmann::json is_evs;
                for (const auto& [action, eu_val] : action_map) {
                    is_evs[poker::action_to_string(action)] = eu_val;
                }
                action_evs_json[is_id] = is_evs;
            }

            if (args.verbose) {
                std::cout << "  Iter " << stats.iteration
                         << ": residual=" << std::scientific << stats.residual_norm
                         << ", exploit=" << exploit << std::endl;
            }

            // Write telemetry to JSON file
            auto snapshot = network::TelemetrySnapshot::from_solver_stats(
                stats, beta, sigma, game->name(), exploit, ev, action_evs_json
            );
            telemetry.log_iteration(snapshot.to_json());
        });

        // Define residual function for Newton
        auto residual_fn = [&qre](const Eigen::VectorXd& x) {
            return qre(x);
        };

        // Solve
        auto result = newton.solve(residual_fn, w);
        w = result.x;  // Warm start for next beta

        std::cout << "  " << (result.converged ? "Converged" : "Max iters")
                  << " in " << result.iterations << " iterations"
                  << ", residual = " << std::scientific << result.final_residual << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\n======================================\n";
    std::cout << "  Solver Complete\n";
    std::cout << "======================================\n\n";

    // Final strategy
    poker::Strategy final_sigma = poker::Strategy::from_logits(w, index);
    double final_exploit = poker::compute_exploitability(game->root(), final_sigma);
    double final_ev = poker::compute_ev(game->root(), final_sigma);

    std::cout << "Total iterations: " << total_iters << std::endl;
    std::cout << "Time: " << duration.count() << " ms\n";
    std::cout << "Final exploitability: " << std::scientific << final_exploit << std::endl;
    std::cout << "Expected value (P0): " << std::fixed << std::setprecision(6) << final_ev << std::endl;
    std::cout << std::endl;

    // Print final strategy
    std::cout << "Final Strategy:\n";
    std::cout << std::string(40, '-') << std::endl;

    for (const auto& is : info_sets) {
        Eigen::VectorXd probs = final_sigma.probs(is.id);
        std::cout << is.id << ":\n";
        for (size_t a = 0; a < is.legal_actions.size(); ++a) {
            std::cout << "  " << poker::action_to_string(is.legal_actions[a])
                     << ": " << std::fixed << std::setprecision(4) << probs(a) << std::endl;
        }
    }

    // Write completion to telemetry
    telemetry.finish(final_exploit, total_iters);
    std::cout << "\nVisualization data written to: " << args.output_path << std::endl;

    return 0;
}
