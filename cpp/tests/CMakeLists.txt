# C++ Tests CMakeLists.txt

# Find Google Test
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # Download and build Google Test if not found
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Test executable
add_executable(order_book_test test_order_book.cpp)
target_link_libraries(order_book_test 
    gtest 
    gtest_main
    ${Boost_LIBRARIES}
    Threads::Threads
)

# Add test
add_test(NAME OrderBookTest COMMAND order_book_test)

# Enable CTest
enable_testing() 