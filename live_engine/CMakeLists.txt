cmake_minimum_required(VERSION 3.14)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

project(BinanceLiveEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

# ==========================================
# [Key settings: disable compression, enable OpenSSL]
# ==========================================
set(USE_ZLIB OFF CACHE BOOL "Disable ZLIB" FORCE)
set(USE_MBED_TLS OFF CACHE BOOL "Disable mbedtls" FORCE)
set(USE_OPEN_SSL ON CACHE BOOL "Enable OpenSSL" FORCE)

include(FetchContent)

message(STATUS "Fetching nlohmann/json...")
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

message(STATUS "Fetching IXWebSocket...")
set(USE_TLS ON CACHE BOOL "Enable TLS/SSL support" FORCE)
FetchContent_Declare(ixwebsocket
    GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
    GIT_TAG master
)
FetchContent_MakeAvailable(ixwebsocket)

message(STATUS "Fetching libzmq...")
set(BUILD_SHARED_LIBRARIES OFF CACHE BOOL "Disable shared libs" FORCE)
set(BUILD_STATIC_LIBRARIES ON CACHE BOOL "Enable static libs" FORCE)
set(ZMQ_BUILD_TESTS OFF CACHE BOOL "Disable ZMQ tests" FORCE)
set(WITH_PERF_TOOL OFF CACHE BOOL "Disable ZMQ perf tools" FORCE)

FetchContent_Declare(
    libzmq
    GIT_REPOSITORY https://github.com/zeromq/libzmq.git
    GIT_TAG        v4.3.5
)
FetchContent_MakeAvailable(libzmq)

message(STATUS "Fetching cppzmq...")
set(CPPZMQ_BUILD_TESTS OFF CACHE BOOL "Disable cppzmq tests" FORCE)

FetchContent_Declare(
    cppzmq
    GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
    GIT_TAG        v4.10.0
)
FetchContent_MakeAvailable(cppzmq)

FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.15.3
)
FetchContent_MakeAvailable(httplib)

find_package(OpenSSL REQUIRED)

# ==========================================
# Shared core (live + backtest)
# ==========================================
set(CORE_SOURCES
    src/core/risk_manager.cpp
    src/core/ipc_server.cpp
)

set(CORE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)

# ==========================================
# live_engine.exe  = core + live
# ==========================================
add_executable(live_engine
    ${CORE_SOURCES}
    src/live/main_live.cpp
    src/live/binance_ws.cpp
    src/live/binance_live_executor.cpp
)

target_include_directories(live_engine PRIVATE
    ${CORE_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/live
)

target_link_libraries(live_engine PRIVATE
    nlohmann_json::nlohmann_json
    ixwebsocket
    cppzmq
    httplib::httplib
    OpenSSL::SSL
    OpenSSL::Crypto
)
target_compile_definitions(live_engine PRIVATE ZMQ_STATIC)

# ==========================================
# backtest_engine.exe  = core + backtest
# (no WebSocket / httplib / OpenSSL required)
# ==========================================
add_executable(backtest_engine
    ${CORE_SOURCES}
    src/backtest/main_backtest.cpp
    src/backtest/mock_executor.cpp
    src/backtest/data_replayer.cpp
)

target_include_directories(backtest_engine PRIVATE
    ${CORE_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backtest
)

target_link_libraries(backtest_engine PRIVATE
    nlohmann_json::nlohmann_json
    cppzmq
)
target_compile_definitions(backtest_engine PRIVATE ZMQ_STATIC)
