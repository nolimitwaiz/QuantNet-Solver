cmake_minimum_required(VERSION 3.16)
project(QuantNet-Solver VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# OpenMP for parallel Jacobian computation
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - parallel Jacobian enabled")
else()
    message(STATUS "OpenMP not found - using sequential Jacobian")
endif()

# ============================================================================
# Dependencies
# ============================================================================

include(FetchContent)

# Eigen (linear algebra)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(eigen)

# nlohmann_json (JSON library)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(json)

# Catch2 (testing framework)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

# ============================================================================
# Core Library
# ============================================================================

add_library(quantnet_core STATIC
    src/solver/NewtonSolver.cpp
    src/solver/CFR.cpp
    src/poker/KuhnPoker.cpp
    src/poker/LeducPoker.cpp
    src/poker/Strategy.cpp
    src/poker/ExpectedValue.cpp
    src/poker/QRE.cpp
    src/poker/HandEvaluator.cpp
    src/poker/CardAbstraction.cpp
    src/exploit/OpponentModel.cpp
)

target_include_directories(quantnet_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(quantnet_core PUBLIC
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(quantnet_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ============================================================================
# Main Executable
# ============================================================================

add_executable(quantnet_solver src/main.cpp)
target_link_libraries(quantnet_solver PRIVATE quantnet_core)

# ============================================================================
# Tests
# ============================================================================

enable_testing()
add_subdirectory(tests)

# ============================================================================
# Installation
# ============================================================================

install(TARGETS quantnet_solver RUNTIME DESTINATION bin)
install(DIRECTORY viz/ DESTINATION share/quantnet/viz)
