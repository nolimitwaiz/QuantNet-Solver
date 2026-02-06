# QuantNet-Solver

A high-performance C++20 numerical solver for computing Quantal Response Equilibrium (QRE) in imperfect-information poker games using Newton's method with Levenberg-Marquardt regularization.

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundations](#mathematical-foundations)
- [Algorithm Design](#algorithm-design)
- [Quantitative Finance Applications](#quantitative-finance-applications)
- [Supported Games](#supported-games)
- [Building](#building)
- [Usage](#usage)
- [Real-Time Visualization](#real-time-visualization)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [References](#references)
- [License](#license)

## Overview

QuantNet-Solver finds equilibrium strategies for poker variants by treating equilibrium computation as a root-finding problem and solving it directly with Newton's method. Unlike iterative approaches like CFR (Counterfactual Regret Minimization) that converge at O(1/sqrt(T)), Newton's method achieves quadratic convergence near the solution.

### Key Features

- **Newton-based optimization** with Levenberg-Marquardt regularization for numerical stability
- **Armijo backtracking line search** for robust convergence
- **Beta continuation** (homotopy method) for finding high-temperature equilibria
- **Real-time D3.js visualization** showing convergence and strategy evolution
- **Kuhn Poker** (12 information sets, 24 variables) and **Leduc Poker** (276 information sets, 690 variables)

## Mathematical Foundations

### Quantal Response Equilibrium (QRE)

QRE is a solution concept for games that models bounded rationality. Instead of always playing the best response (as in Nash equilibrium), players choose actions probabilistically based on their expected utilities.

The probability of choosing action `a` at information set `I` is given by the **logit response function**:

```
                exp(beta * EU(I, a))
sigma(I, a) = -------------------------
              sum_b exp(beta * EU(I, b))
```

Where:
- `EU(I, a)` is the expected utility of action `a` at information set `I`
- `beta` is the **rationality parameter** (temperature)
- The summation is over all legal actions `b` at information set `I`

#### Temperature Parameter Interpretation

The parameter `beta` controls how "rational" players are:

| Beta Value | Behavior |
|------------|----------|
| beta = 0 | Uniform random play (completely irrational) |
| beta = 1 | Standard softmax (Boltzmann distribution) |
| beta -> infinity | Pure best response (Nash equilibrium) |

This connects to **logit choice models** in economics and **softmax exploration** in reinforcement learning.

### Fixed-Point Formulation

A QRE is defined by the fixed-point equation:

```
sigma = LogitBR_beta(sigma)
```

Where `LogitBR_beta` computes the logit best response to the current strategy profile. We reformulate this as finding the root of the residual function:

```
R(sigma) = sigma - LogitBR_beta(sigma) = 0
```

### Logit Parameterization

To ensure strategies remain valid probability distributions during optimization, we parameterize them using **unconstrained logits** `w`:

```
                  exp(w[I,a])
sigma(I, a) = -------------------
              sum_b exp(w[I,b])
```

This transformation:
1. Guarantees `sigma(I, a) >= 0` for all actions
2. Guarantees `sum_a sigma(I, a) = 1` for each information set
3. Allows unconstrained optimization over `w`

The residual becomes:

```
R(w) = softmax(w) - LogitBR_beta(softmax(w))
```

This is an n-dimensional root-finding problem where `n` equals the total number of actions across all information sets.

## Algorithm Design

### Newton's Method with Levenberg-Marquardt Regularization

The standard Newton step for solving `R(w) = 0` is:

```
w_{k+1} = w_k - J^{-1}(w_k) * R(w_k)
```

Where `J` is the Jacobian matrix of `R`. However, `J` can be singular or ill-conditioned, causing numerical instability.

**Levenberg-Marquardt regularization** addresses this by solving the modified system:

```
(J^T * J + lambda * I) * d = -J^T * R
```

Where:
- `lambda` is the damping parameter
- `I` is the identity matrix
- `d` is the search direction

This interpolates between:
- **Newton's method** (lambda -> 0): Fast quadratic convergence near solution
- **Gradient descent** (lambda -> infinity): Robust but slow convergence far from solution

#### Adaptive Regularization

The solver automatically adjusts `lambda`:

```
if residual_decreased:
    lambda = max(lambda_min, lambda / factor)  # Trust Newton more
else:
    lambda = min(lambda_max, lambda * factor)  # Fall back to gradient descent
```

Default parameters:
- `lambda_init = 1e-6`
- `lambda_max = 1e6`
- `factor = 10`

### Armijo Backtracking Line Search

After computing the search direction `d`, we find a step size `alpha` that guarantees sufficient decrease in the merit function:

```
phi(w) = 0.5 * ||R(w)||^2
```

The **Armijo condition** requires:

```
phi(w + alpha*d) <= phi(w) + c * alpha * grad_phi^T * d
```

Where:
- `c = 1e-4` (small constant for "sufficient" decrease)
- `grad_phi = J^T * R` (gradient of merit function)

**Backtracking procedure:**
1. Start with `alpha = 1.0` (full Newton step)
2. If Armijo condition satisfied, accept step
3. Otherwise, `alpha = rho * alpha` (where `rho = 0.5`)
4. Repeat until condition met or minimum step reached

This guarantees monotonic decrease in the residual norm.

### Finite Difference Jacobian

The Jacobian is computed using **central finite differences**:

```
J_ij = [R_i(w + h*e_j) - R_i(w - h*e_j)] / (2*h)
```

Where:
- `e_j` is the j-th unit vector
- `h = 1e-7` is the step size

**Error analysis:**
- Central differences: O(h^2) truncation error
- Forward differences: O(h) truncation error (less accurate)
- Too small `h`: Roundoff error dominates
- Too large `h`: Truncation error dominates

The step size `h = 1e-7` balances these competing errors for IEEE 754 double precision.

**Computational cost:** 2n function evaluations per Jacobian computation (for n-dimensional problem).

### Beta Continuation (Homotopy Method)

Direct Newton solve at high `beta` often fails because:
1. The residual landscape becomes highly nonlinear
2. Good initial guesses are hard to find
3. The Jacobian becomes ill-conditioned

**Solution:** Trace a path through strategy space from low to high `beta`:

```
beta_schedule = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, target_beta]
```

For each `beta_k`:
1. Solve QRE using Newton's method
2. Use solution as warm start for `beta_{k+1}`

This **homotopy method** works because:
- At low `beta`, QRE is close to uniform (easy starting point)
- Solutions vary smoothly with `beta`
- Each warm start is close to the next solution

### Convergence Properties

Near the solution, Newton's method achieves **quadratic convergence**:

```
||e_{k+1}|| <= C * ||e_k||^2
```

Where `e_k` is the error at iteration `k`. This means:
- If error is 0.1, next error is ~0.01
- If error is 0.01, next error is ~0.0001
- Rapid convergence once close to solution

The solver terminates when `||R(w)|| < tolerance` (default: `1e-8`).

## Quantitative Finance Applications

### Game Theory in Market Microstructure

Modern financial markets are strategic environments where participants (market makers, traders, algorithms) interact. QRE provides a framework for modeling:

1. **Market Making**: Optimal bid-ask spreads when facing informed and uninformed traders
2. **Order Book Dynamics**: Strategic placement of limit orders
3. **Dark Pool Routing**: Deciding between lit and dark venues

### Opponent Modeling

In trading, understanding counterparty behavior is crucial:

- **Poker analogy**: Other players have private information (cards); you must infer their holdings from their actions
- **Trading analogy**: Other traders have private information (order flow, alpha signals); you must infer their intent from market activity

QRE naturally models opponents who are "approximately rational" - they generally make good decisions but not always optimally.

### Equilibrium Strategies in Auctions

Many financial mechanisms are auctions:
- IPO allocations
- Treasury auctions
- Spectrum auctions
- Ad exchanges

QRE extends to auction settings, providing strategies that account for bounded rationality and incomplete information.

### Risk-Adjusted Decision Making

The softmax formulation in QRE connects to:
- **Entropy-regularized optimization** in robust control
- **Information-theoretic bounded rationality**
- **Risk-sensitive decision making** where temperature represents risk aversion

## Supported Games

### Kuhn Poker

A simplified 3-card poker game ideal for testing and verification.

**Rules:**
- **Deck**: 3 cards (Jack, Queen, King)
- **Players**: 2
- **Ante**: Each player antes 1 chip
- **Deal**: One card to each player
- **Actions**: Check or Bet (1 chip)
- **Showdown**: Higher card wins the pot

**Game Tree:**
```
Player 0 acts first:
  - Check: Player 1 can Check (showdown) or Bet
    - If Bet: Player 0 can Call or Fold
  - Bet: Player 1 can Call or Fold
```

**Complexity:**
- Information sets: 12
- Strategy variables: 24
- Terminal nodes: 30
- Solve time: ~100ms

**Known Nash Equilibrium** (approximate):
- P0 with Jack: Check, fold to bet (never bluff-call)
- P0 with King: Mix between check-raise and bet
- P1 with Jack: Check behind, fold to bet
- P1 with King: Always call bets

**Theoretical EV for Player 0**: -1/18 (approximately -0.056)

This first-mover disadvantage arises because Player 1 acts with more information.

### Leduc Poker

A more complex 2-round poker game that tests scalability.

**Rules:**
- **Deck**: 6 cards (Jack, Queen, King x 2 suits)
- **Players**: 2
- **Ante**: Each player antes 1 chip
- **Round 1**: Private cards dealt, betting with 2-chip bets
- **Round 2**: Public card revealed, betting with 4-chip bets
- **Max Raises**: 2 per round
- **Hand Ranking**: Pair > High card

**Complexity:**
- Information sets: 276
- Strategy variables: 690
- Solve time: ~5 seconds

**Key Strategic Elements:**
- Information revelation through betting
- Pair potential affects round 1 play
- Position advantage in two-round structure

## Building

### Prerequisites

- **C++20 compiler**: GCC 10+, Clang 12+, or MSVC 2019+
- **CMake**: 3.16+

**macOS:**
```bash
brew install cmake
```

**Ubuntu/Debian:**
```bash
sudo apt-get install cmake build-essential
```

**Windows:**
Install Visual Studio 2019+ with C++ workload, or use MinGW-w64.

### Build Commands

```bash
git clone https://github.com/YOUR_USERNAME/QuantNet-Solver.git
cd QuantNet-Solver
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

### Dependencies

All dependencies are automatically fetched via CMake FetchContent:
- **Eigen 3.4**: Linear algebra (matrix operations, SVD, LU decomposition)
- **nlohmann_json 3.11**: JSON serialization for telemetry
- **Catch2 3.5**: Testing framework

No manual dependency installation required.

### Running Tests

```bash
cd build
ctest --output-on-failure
```

Test suites:
- `test_newton`: Newton solver convergence tests
- `test_kuhn_ev`: Kuhn Poker expected value verification
- `test_cfr`: Alternative CFR solver tests
- `test_hand_evaluator`: Card evaluation tests

## Usage

### Command Line Interface

```bash
./quantnet_solver [options]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--game` | `kuhn` | Game to solve: `kuhn` or `leduc` |
| `--beta` | `10.0` | Target rationality parameter |
| `--tol` | `1e-8` | Convergence tolerance |
| `--max-iters` | `50` | Max Newton iterations per beta step |
| `--output` | `viz/solver_output.json` | Output file for visualization |
| `--verbose` | off | Print iteration details |
| `--help` | - | Show help message |

### Examples

```bash
# Basic Kuhn Poker solve
./quantnet_solver --game kuhn

# Leduc Poker with higher precision
./quantnet_solver --game leduc --beta 20 --tol 1e-10

# Verbose output for debugging
./quantnet_solver --game kuhn --verbose

# Custom output location
./quantnet_solver --game kuhn --output /tmp/solver_output.json
```

### Output Format

The solver outputs JSON with the following structure:

```json
{
  "game": "Kuhn Poker",
  "iteration_count": 39,
  "iterations": [
    {
      "iteration": 0,
      "beta": 0.01,
      "residual_norm": 1.23e-6,
      "exploitability": 0.45,
      "expected_value": -0.052,
      "strategy": {
        "P0:J:": {"check": 0.7, "bet": 0.3},
        ...
      },
      "action_evs": {
        "P0:J:": {"check": -0.03, "bet": -0.11},
        ...
      }
    },
    ...
  ],
  "latest": {
    "type": "complete",
    "final_exploitability": 2.78e-7
  }
}
```

## Real-Time Visualization

The solver includes a D3.js-based dashboard for monitoring convergence in real-time.

### Starting the Visualization

1. **Run the solver** (generates `viz/solver_output.json`):
   ```bash
   ./quantnet_solver --game kuhn
   ```

2. **Start a local HTTP server**:
   ```bash
   cd viz
   python3 -m http.server 8080
   ```

3. **Open in browser**: http://localhost:8080

### Visualization Features

**Status Bar:**
- Connection status
- Current game
- Beta value
- Iteration count
- Residual norm
- Exploitability
- Expected value (P0)

**Strategy Profile:**
- Two-column layout (Player 0 / Player 1)
- Grouped by card (Jack, Queen, King)
- Decision context labels ("Opening action", "Facing a bet")
- Action probabilities with visual bars
- Per-action expected values (shows "why" behind each probability)

**Convergence Chart:**
- Dual-axis log plot (residual and exploitability)
- Beta transition markers (dashed vertical lines)
- Real-time updates every 200ms

### Development Server

For development with cache-busting:
```bash
cd viz
python3 serve.py 8080
```

This disables browser caching of HTML/JS/CSS files.

## Performance

### Benchmarks

| Game | Info Sets | Variables | Iterations | Time (beta=10) |
|------|-----------|-----------|------------|----------------|
| Kuhn | 12 | 24 | ~39 | ~100ms |
| Leduc | 276 | 690 | ~150 | ~5s |

### Complexity Analysis

**Per Newton iteration:**
- Jacobian computation: O(n^2) function evaluations (2n evaluations, each O(n) work)
- Linear system solve: O(n^3) via LU decomposition
- Line search: O(k*n) where k is backtracking iterations (typically 1-10)

**Total complexity:** O(T * n^3) where T is total Newton iterations across all beta steps.

### Memory Usage

- Strategy vector: O(n) doubles
- Jacobian matrix: O(n^2) doubles
- For Leduc: ~690^2 * 8 bytes = ~4 MB for Jacobian

## Project Structure

```
QuantNet-Solver/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── src/
│   ├── main.cpp               # Entry point
│   ├── solver/
│   │   ├── NewtonSolver.hpp   # Newton with LM regularization
│   │   ├── FiniteDiff.hpp     # Jacobian computation
│   │   ├── LineSearch.hpp     # Armijo backtracking
│   │   ├── Diagnostics.hpp    # Iteration tracking
│   │   └── CFR.hpp/cpp        # Alternative: CFR solver
│   ├── poker/
│   │   ├── GameTypes.hpp      # Enums and basic types
│   │   ├── GameTree.hpp       # Game tree structures
│   │   ├── KuhnPoker.hpp/cpp  # Kuhn Poker implementation
│   │   ├── LeducPoker.hpp/cpp # Leduc Poker implementation
│   │   ├── Strategy.hpp/cpp   # Strategy representation
│   │   ├── ExpectedValue.hpp/cpp
│   │   └── QRE.hpp/cpp        # QRE residual computation
│   └── network/
│       ├── SimpleTelemetry.hpp # JSON file output
│       └── Telemetry.hpp       # Snapshot formatting
├── tests/
│   ├── test_newton.cpp
│   ├── test_kuhn_ev.cpp
│   ├── test_cfr.cpp
│   └── test_hand_evaluator.cpp
└── viz/
    ├── index.html              # Dashboard HTML
    ├── app.js                  # D3.js visualization
    ├── styles.css              # Styling
    └── serve.py                # Development server
```

## Implementation Details

### Numerical Stability

**Stable Softmax:**
```cpp
VectorXd stable_softmax(const VectorXd& logits) {
    double max_logit = logits.maxCoeff();
    VectorXd shifted = logits.array() - max_logit;  // Prevent overflow
    VectorXd exp_vals = shifted.array().exp();
    return exp_vals / exp_vals.sum();
}
```

**Condition Number Monitoring:**
```cpp
JacobiSVD<MatrixXd> svd(J);
double cond = svd.singularValues()(0) / svd.singularValues()(n-1);
// High cond indicates ill-conditioning
```

### Expected Value Computation

EV is computed via tree traversal with **reach probabilities**:

```
EV(sigma) = sum over terminals z: pi(z) * u(z)
```

Where:
- `pi(z)` = probability of reaching terminal node z
- `u(z)` = payoff at terminal node z

Reach probability factors into:
- `pi_0(z)`: Player 0's contribution (product of P0's action probabilities)
- `pi_1(z)`: Player 1's contribution
- `pi_c(z)`: Chance contribution (card deals)

### Per-Action Expected Utility

For QRE, we need `EU(I, a)` - the expected utility of playing action `a` at information set `I`:

```
EU(I, a) = sum over h in I: pi_{-i}(h) * u_i(h, a, sigma_{-i})
```

This is computed by temporarily forcing action `a` at info set `I` and computing the resulting expected payoff.

### Exploitability

Exploitability measures strategy quality:

```
Exploitability = (BR_value_P0 + BR_value_P1) / 2
```

Where `BR_value_p` is the value player `p` can achieve by best-responding to the opponent's strategy.

At Nash equilibrium, exploitability = 0.

## References

1. **McKelvey, R. D., & Palfrey, T. R.** (1995). Quantal response equilibria for normal form games. *Games and Economic Behavior*, 10(1), 6-38.

2. **Kuhn, H. W.** (1950). A simplified two-person poker. *Contributions to the Theory of Games*, 1, 97-103.

3. **Southey, F., Bowling, M., Larson, B., Piccione, C., Burch, N., Billings, D., & Rayner, C.** (2005). Bayes' bluff: Opponent modelling in poker. *Proceedings of UAI*.

4. **Nocedal, J., & Wright, S. J.** (2006). *Numerical Optimization* (2nd ed.). Springer. [Newton methods, line search, trust regions]

5. **Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C.** (2008). Regret minimization in games with incomplete information. *Advances in Neural Information Processing Systems*.

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
