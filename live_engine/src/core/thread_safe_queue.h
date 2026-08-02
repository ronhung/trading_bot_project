#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class ThreadSafeQueue {
public:
    void push(const T& item) {
        std::lock_guard<std::mutex> lock(mtx);
        q.push(item);
        cv.notify_one();
    }

    void wait_and_pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]() { return !q.empty(); });
        item = q.front();
        q.pop();
    }

    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mtx);
        if (q.empty()) return false;
        item = q.front();
        q.pop();
        return true;
    }

private:
    std::queue<T> q;
    std::mutex mtx;
    std::condition_variable cv;
};
