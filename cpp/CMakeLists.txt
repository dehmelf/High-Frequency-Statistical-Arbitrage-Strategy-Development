cmake_minimum_required(VERSION 3.16)
project(HighFrequencyArbitrage VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for low latency
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -funroll-loops")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra")

# Find required packages
find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread filesystem)

# Include directories
include_directories(include)
include_directories(${Boost_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/market_data_pipeline.cpp
    src/order_book.cpp
    src/tick_processor.cpp
    src/memory_pool.cpp
    src/ring_buffer.cpp
    src/timestamp.cpp
    src/udp_receiver.cpp
    src/market_data_writer.cpp
)

# Header files
set(HEADERS
    include/market_data_pipeline.h
    include/order_book.h
    include/tick_processor.h
    include/memory_pool.h
    include/ring_buffer.h
    include/timestamp.h
    include/udp_receiver.h
    include/market_data_writer.h
    include/types.h
    include/constants.h
)

# Create executable
add_executable(market_data_pipeline ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(market_data_pipeline 
    ${Boost_LIBRARIES}
    Threads::Threads
    rt  # For high-resolution timestamps
)

# Set compiler-specific flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(market_data_pipeline PRIVATE -fno-strict-aliasing)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(market_data_pipeline PRIVATE -fno-strict-aliasing)
endif()

# Installation
install(TARGETS market_data_pipeline DESTINATION bin)
install(FILES ${HEADERS} DESTINATION include/hf_arbitrage)

# Testing
enable_testing()
add_subdirectory(tests) 