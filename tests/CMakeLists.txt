include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TEST_FILES
    einsum/einsum_dispatcher_test.cpp
    einsum/einsum_node_test.cpp
    test.cpp
)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(sdfglib-einsum_test ${TEST_FILES})
target_include_directories(sdfglib-einsum_test PRIVATE ./)
target_link_libraries(sdfglib-einsum_test gtest_main sdfglib-einsum)

add_test(NAME sdfglib-einsum_test COMMAND sdfglib-einsum_test)

include(GoogleTest)
gtest_discover_tests(sdfglib-einsum_test)
