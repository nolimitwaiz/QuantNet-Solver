#pragma once

#include <Eigen/Dense>
#include <functional>
#include <thread>
#include <vector>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace quantnet::parallel {

// Performance metrics for benchmarking
struct JacobianMetrics {
    double wall_time_ms = 0.0;
    int function_evaluations = 0;
    int num_threads = 1;
    double speedup = 1.0;  // vs sequential
};

// Parallel Jacobian computation using OpenMP
// Falls back to sequential if OpenMP not available
template<typename Func>
Eigen::MatrixXd compute_jacobian_parallel(
    Func&& F,
    const Eigen::VectorXd& x,
    double h = 1e-7,
    JacobianMetrics* metrics = nullptr
) {
    auto start = std::chrono::high_resolution_clock::now();

    const int n = static_cast<int>(x.size());
    const Eigen::VectorXd f0 = F(x);
    const int m = static_cast<int>(f0.size());

    Eigen::MatrixXd J(m, n);
    int func_evals = 1;  // f0

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();

    #pragma omp parallel for reduction(+:func_evals)
    for (int j = 0; j < n; ++j) {
        Eigen::VectorXd x_plus = x;
        Eigen::VectorXd x_minus = x;
        x_plus(j) += h;
        x_minus(j) -= h;

        Eigen::VectorXd f_plus = F(x_plus);
        Eigen::VectorXd f_minus = F(x_minus);

        J.col(j) = (f_plus - f_minus) / (2.0 * h);
        func_evals += 2;
    }
#else
    int num_threads = 1;
    for (int j = 0; j < n; ++j) {
        Eigen::VectorXd x_plus = x;
        Eigen::VectorXd x_minus = x;
        x_plus(j) += h;
        x_minus(j) -= h;

        Eigen::VectorXd f_plus = F(x_plus);
        Eigen::VectorXd f_minus = F(x_minus);

        J.col(j) = (f_plus - f_minus) / (2.0 * h);
        func_evals += 2;
    }
#endif

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    if (metrics) {
        metrics->wall_time_ms = duration.count() / 1000.0;
        metrics->function_evaluations = func_evals;
        metrics->num_threads = num_threads;
    }

    return J;
}

// Thread pool based parallel Jacobian (C++11 compatible, no OpenMP needed)
template<typename Func>
Eigen::MatrixXd compute_jacobian_threadpool(
    Func&& F,
    const Eigen::VectorXd& x,
    double h = 1e-7,
    int num_threads = 0,  // 0 = hardware concurrency
    JacobianMetrics* metrics = nullptr
) {
    auto start = std::chrono::high_resolution_clock::now();

    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads == 0) num_threads = 4;  // fallback
    }

    const int n = static_cast<int>(x.size());
    const Eigen::VectorXd f0 = F(x);
    const int m = static_cast<int>(f0.size());

    Eigen::MatrixXd J(m, n);
    std::atomic<int> func_evals{1};

    // Divide columns among threads
    auto worker = [&](int start_col, int end_col) {
        for (int j = start_col; j < end_col; ++j) {
            Eigen::VectorXd x_plus = x;
            Eigen::VectorXd x_minus = x;
            x_plus(j) += h;
            x_minus(j) -= h;

            Eigen::VectorXd f_plus = F(x_plus);
            Eigen::VectorXd f_minus = F(x_minus);

            J.col(j) = (f_plus - f_minus) / (2.0 * h);
            func_evals += 2;
        }
    };

    std::vector<std::thread> threads;
    int cols_per_thread = (n + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        int start_col = t * cols_per_thread;
        int end_col = std::min(start_col + cols_per_thread, n);
        if (start_col < n) {
            threads.emplace_back(worker, start_col, end_col);
        }
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    if (metrics) {
        metrics->wall_time_ms = duration.count() / 1000.0;
        metrics->function_evaluations = func_evals.load();
        metrics->num_threads = num_threads;
    }

    return J;
}

// Benchmark parallel vs sequential
template<typename Func>
void benchmark_jacobian(
    Func&& F,
    const Eigen::VectorXd& x,
    int runs = 5
) {
    std::cout << "Jacobian Benchmark (n=" << x.size() << ", " << runs << " runs)\n";
    std::cout << std::string(50, '-') << "\n";

    double seq_total = 0.0;
    double par_total = 0.0;
    JacobianMetrics metrics;

    // Sequential
    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto J = quantnet::solver::compute_jacobian(F, x, 1e-7, true);
        auto end = std::chrono::high_resolution_clock::now();
        seq_total += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    double seq_avg = seq_total / runs / 1000.0;

    // Parallel
    for (int i = 0; i < runs; ++i) {
        compute_jacobian_parallel(F, x, 1e-7, &metrics);
        par_total += metrics.wall_time_ms;
    }
    double par_avg = par_total / runs;

    std::cout << "Sequential:  " << seq_avg << " ms\n";
    std::cout << "Parallel:    " << par_avg << " ms (" << metrics.num_threads << " threads)\n";
    std::cout << "Speedup:     " << seq_avg / par_avg << "x\n";
}

} // namespace quantnet::parallel
