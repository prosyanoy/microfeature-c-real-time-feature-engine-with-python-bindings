#pragma once
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "microfeature/engine.hpp"

namespace mftest {

// Reference implementation: recompute every feature from the raw event list.
inline microfeature::Features brute_force(const microfeature::EventBatch& b, uint64_t key,
                                          int64_t now, int64_t window_ms, int64_t halflife_ms) {
  microfeature::Features f;
  double num = 0, den = 0;
  int64_t newest = INT64_MIN;
  bool seen = false;
  for (size_t i = 0; i < b.size(); ++i) {
    if (b.keys[i] != key) continue;
    seen = true;
    newest = std::max(newest, b.ts[i]);
  }
  const double tau = halflife_ms / std::log(2.0);
  for (size_t i = 0; i < b.size(); ++i) {
    if (b.keys[i] != key) continue;
    double w = std::exp(-static_cast<double>(newest - b.ts[i]) / tau);
    num += w * b.values[i];
    den += w;
    if (b.ts[i] > now - window_ms && b.ts[i] <= now) {
      f.count += 1;
      f.sum += b.values[i];
    }
  }
  f.mean = f.count > 0 ? f.sum / f.count : std::nan("");
  f.rate_per_sec = f.count / (window_ms / 1000.0);
  f.ewma = seen ? num / den : std::nan("");
  return f;
}

inline microfeature::EventBatch random_events(size_t n, uint64_t keys, int64_t span_ms, uint32_t seed) {
  std::mt19937_64 rng(seed);
  microfeature::EventBatch b;
  for (size_t i = 0; i < n; ++i)
    b.push(rng() % keys, static_cast<int64_t>(rng() % static_cast<uint64_t>(span_ms)),
           static_cast<double>(rng() % 1000) / 10.0);
  return b;
}

}  // namespace mftest
