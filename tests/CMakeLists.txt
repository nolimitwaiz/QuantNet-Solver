# Tests for QuantNet-Solver

add_executable(test_newton test_newton.cpp)
target_link_libraries(test_newton PRIVATE
    quantnet_core
    Catch2::Catch2WithMain
)

add_executable(test_kuhn_ev test_kuhn_ev.cpp)
target_link_libraries(test_kuhn_ev PRIVATE
    quantnet_core
    Catch2::Catch2WithMain
)

add_executable(test_cfr test_cfr.cpp)
target_link_libraries(test_cfr PRIVATE
    quantnet_core
    Catch2::Catch2WithMain
)

add_executable(test_hand_evaluator test_hand_evaluator.cpp)
target_link_libraries(test_hand_evaluator PRIVATE
    quantnet_core
    Catch2::Catch2WithMain
)

# Register tests with CTest
include(Catch)
catch_discover_tests(test_newton)
catch_discover_tests(test_kuhn_ev)
catch_discover_tests(test_cfr)
catch_discover_tests(test_hand_evaluator)
