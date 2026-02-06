#pragma once

#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace quantnet::network {

// Simple file-based telemetry that writes JSON to a file
// Browser polls this file instead of using WebSocket
class SimpleTelemetry {
public:
    explicit SimpleTelemetry(const std::string& output_path)
        : path_(output_path) {
        // Write initial empty state
        write_file();
    }

    // Log an iteration
    void log_iteration(const nlohmann::json& data) {
        history_.push_back(data);
        latest_ = data;
        write_file();
    }

    // Mark solver as complete
    void finish(double final_exploitability, int total_iterations) {
        nlohmann::json completion;
        completion["type"] = "complete";
        completion["final_exploitability"] = final_exploitability;
        completion["total_iterations"] = total_iterations;
        completion["status"] = "done";
        latest_ = completion;
        finished_ = true;
        write_file();
    }

    // Get output path
    const std::string& path() const { return path_; }

private:
    void write_file() {
        nlohmann::json output;
        output["status"] = finished_ ? "complete" : "running";
        output["iteration_count"] = history_.size();
        output["iterations"] = history_;
        output["latest"] = latest_;

        // Atomic write: temp file + rename prevents browser reading truncated JSON
        std::string tmp_path = path_ + ".tmp";
        std::ofstream f(tmp_path);
        if (f.is_open()) {
            f << output.dump(2);
            f.close();
            std::filesystem::rename(tmp_path, path_);
        }
    }

    std::string path_;
    std::vector<nlohmann::json> history_;
    nlohmann::json latest_;
    bool finished_ = false;
};

} // namespace quantnet::network
