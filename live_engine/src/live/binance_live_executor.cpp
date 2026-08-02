#include "binance_live_executor.h"
#include <httplib.h>
#include <openssl/hmac.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

BinanceLiveExecutor::BinanceLiveExecutor(const std::string& api_key, const std::string& secret_key)
    : api_key_(api_key), secret_key_(secret_key) {
    std::cout << "🛡️ [Execution] BinanceLiveExecutor initialized (bound to Binance Testnet)." << std::endl;
}

std::string BinanceLiveExecutor::generate_signature(const std::string& query_string) {
    unsigned char hash[32];
    unsigned int length = 32;

    HMAC_CTX* hmac = HMAC_CTX_new();
    HMAC_Init_ex(hmac, secret_key_.c_str(), static_cast<int>(secret_key_.length()), EVP_sha256(), NULL);
    HMAC_Update(hmac, reinterpret_cast<const unsigned char*>(query_string.c_str()), query_string.length());
    HMAC_Final(hmac, hash, &length);
    HMAC_CTX_free(hmac);

    std::stringstream ss;
    for (unsigned int i = 0; i < length; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    return ss.str();
}

void BinanceLiveExecutor::send_order(const std::string& symbol,
                                     const std::string& side,
                                     double quantity,
                                     double price,
                                     bool reduce_only) {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    std::stringstream query_ss;
    query_ss << "symbol=" << symbol
             << "&side=" << side
             << "&type=LIMIT"
             << "&timeInForce=GTC"
             << "&quantity=" << quantity
             << "&price=" << price
             << "&timestamp=" << ms;

    if (reduce_only) {
        query_ss << "&reduceOnly=true";
    }
    std::string query_string = query_ss.str();
    std::string signature = generate_signature(query_string);
    std::string payload = query_string + "&signature=" + signature;

    httplib::Headers headers = {
        {"X-MBX-APIKEY", api_key_}
    };

    httplib::Client cli("https://testnet.binancefuture.com");
    cli.set_connection_timeout(5);

    std::cout << "\n🔫 [BinanceLiveExecutor] Preparing to send order: " << side << " " << symbol
              << " @ " << price << " (qty: " << quantity << ")" << std::endl;

    auto res = cli.Post("/fapi/v1/order", headers, payload, "application/x-www-form-urlencoded");

    if (res) {
        if (res->status == 200) {
            std::cout << "✅ [BinanceLiveExecutor] Order delivered successfully! Server response: "
                      << res->body << std::endl;
        } else {
            std::cout << "❌ [BinanceLiveExecutor] Order rejected! Status: " << res->status
                      << " | Response: " << res->body << std::endl;
        }
    } else {
        auto err = res.error();
        std::cout << "🔥 [BinanceLiveExecutor] Network error! Error: "
                  << httplib::to_string(err) << std::endl;
    }
}

std::string BinanceLiveExecutor::get_listen_key() {
    httplib::Client cli("https://testnet.binancefuture.com");
    cli.set_connection_timeout(5);

    httplib::Headers headers = {
        {"X-MBX-APIKEY", api_key_}
    };

    std::cout << "🔑 [BinanceLiveExecutor] Requesting private listenKey from Binance..." << std::endl;

    auto res = cli.Post("/fapi/v1/listenKey", headers, "", "application/x-www-form-urlencoded");

    if (res && res->status == 200) {
        try {
            json j = json::parse(res->body);
            std::string listen_key = j["listenKey"];
            std::cout << "✅ [BinanceLiveExecutor] Successfully obtained listenKey!" << std::endl;
            return listen_key;
        } catch (const std::exception& e) {
            std::cerr << "🔥 Failed to parse listenKey: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "❌ Failed to obtain listenKey! Status: "
                  << (res ? std::to_string(res->status) : "connection error") << std::endl;
    }
    return "";
}

bool BinanceLiveExecutor::get_initial_state(double& out_usdt_balance, double& out_btcusdt_position) {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    std::string query_string = "timestamp=" + std::to_string(ms);
    std::string signature = generate_signature(query_string);
    std::string path = "/fapi/v2/account?" + query_string + "&signature=" + signature;

    httplib::Headers headers = {
        {"X-MBX-APIKEY", api_key_}
    };

    httplib::Client cli("https://testnet.binancefuture.com");
    cli.set_connection_timeout(5);

    std::cout << "🔍 [BinanceLiveExecutor] Querying initial account state..." << std::endl;

    auto res = cli.Get(path.c_str(), headers);

    if (res && res->status == 200) {
        try {
            json j = json::parse(res->body);
            out_usdt_balance = 0.0;
            out_btcusdt_position = 0.0;

            for (auto& asset : j["assets"]) {
                if (asset["asset"] == "USDT") {
                    out_usdt_balance = std::stod(asset["walletBalance"].get<std::string>());
                }
            }

            for (auto& pos : j["positions"]) {
                if (pos["symbol"] == "BTCUSDT") {
                    out_btcusdt_position = std::stod(pos["positionAmt"].get<std::string>());
                }
            }
            return true;
        } catch (const std::exception& e) {
            std::cerr << "🔥 Failed to parse initial account state: " << e.what() << std::endl;
            return false;
        }
    }

    std::cerr << "❌ Initial state query failed! Status: "
              << (res ? std::to_string(res->status) : "connection error") << std::endl;
    if (res) std::cerr << "Response: " << res->body << std::endl;
    return false;
}
