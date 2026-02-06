# soul.md - QuantNet-Solver

## Purpose
Make game theory computable. Find equilibrium strategies for poker by treating
equilibrium as a root-finding problem and solving it directly with Newton's method.

## Principles

### Correctness Over Cleverness
The solver must produce mathematically correct equilibria. Verified against known
Kuhn Poker solutions. Stable softmax, condition number monitoring, adaptive
regularization. No shortcuts that sacrifice correctness.

### Simplicity Over Generality
We solve poker. Each abstraction exists because it was needed, not because it might
be. SimpleTelemetry replaced WebSocket because polling a JSON file is simpler and
more reliable. When in doubt, choose the simpler approach.

### Debuggability
Every iteration logs residual norm, step size, line search alpha, regularization
lambda, and full strategy. The visualization exists so you can watch the solver
think. When something breaks, you can see exactly where and why.

### Mathematical Integrity
Strategy parameterization via logits ensures valid probability distributions by
construction. The QRE residual is clean: R(w) = σ(w) - BR_β(σ(w)). Newton's
method is principled for this smooth fixed-point equation.

## Priority Order
1. Correctness (equilibrium quality, verified against known solutions)
2. Convergence reliability (homotopy, regularization, line search)
3. Developer understanding (readable code, good diagnostics)
4. Performance (never at the cost of 1-3)

## What We Don't Do
- Over-engineer: No abstract factory patterns for one implementation
- Premature optimization: Profile first, optimize second
- Re-introduce complexity: WebSocket was removed for a reason
- Skip verification: Every change gets tested
