// MicroFeature engine: sharded in-memory store of per-key sliding-window features.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "microfeature/ring_buffer.hpp"

namespace microfeature {

struct Config {
  int64_t window_ms = 60'000;       // sliding window length for count/sum/mean/rate
  int64_t halflife_ms = 30'000;     // half-life of the exponentially weighted mean
  size_t num_shards = 64;           // rounded up to a power of two
  size_t max_events_per_key = 1u << 20;  // hard cap on ring size per key
};

constexpr size_t kNumFeatures = 5;
// Column order of every feature row produced by the engine.
inline const char* const* feature_names() {
  static const char* const names[kNumFeatures] = {"count", "sum", "mean", "rate_per_sec", "ewma"};
  return names;
}

struct Features {
  double count = 0.0;
  double sum = 0.0;
  double mean = 0.0;          // NaN when the window is empty
  double rate_per_sec = 0.0;  // count / window length in seconds
  double ewma = 0.0;          // NaN when the key has never been seen
  void write(double* out) const {
    out[0] = count; out[1] = sum; out[2] = mean; out[3] = rate_per_sec; out[4] = ewma;
  }
};

struct Stats {
  uint64_t accepted = 0;         // events inserted into a window
  uint64_t late_dropped = 0;     // older than the key's window watermark (EWMA still updated)
  uint64_t overflow_dropped = 0; // oldest sample evicted because of max_events_per_key
  uint64_t rejected = 0;         // non-finite values
  uint64_t keys = 0;
};

struct Event {
  uint64_t key;
  int64_t ts;  // event time, milliseconds
  double value;
};

// Non-owning structure-of-arrays view (maps 1:1 onto NumPy buffers).
struct EventView {
  const uint64_t* keys;
  const int64_t* ts;
  const double* values;
  size_t n;
};

// Owning helper, mostly for tests and benchmarks.
struct EventBatch {
  std::vector<uint64_t> keys;
  std::vector<int64_t> ts;
  std::vector<double> values;
  void push(uint64_t k, int64_t t, double v) { keys.push_back(k); ts.push_back(t); values.push_back(v); }
  size_t size() const { return keys.size(); }
  EventView view() const { return {keys.data(), ts.data(), values.data(), keys.size()}; }
};

// State for a single key. Not thread-safe; the owning shard serialises access.
class KeyState {
 public:
  enum class Result { kAccepted, kLate, kOverflow };

  Result insert(int64_t ts, double value, const Config& cfg);
  Features query(int64_t now, const Config& cfg) const;
  const RingBuffer& ring() const { return ring_; }

 private:
  void update_ewma(int64_t ts, double value, double tau_ms);
  void evict(int64_t watermark);

  RingBuffer ring_;
  double total_ = 0.0;  // running sum of everything in ring_
  size_t evictions_since_resync_ = 0;
  bool has_events_ = false;
  int64_t max_ts_ = 0;
  bool has_ewma_ = false;
  int64_t ewma_t_ = 0;
  double ewma_num_ = 0.0;
  double ewma_den_ = 0.0;
};

class Engine {
 public:
  explicit Engine(const Config& cfg = Config());

  const Config& config() const { return cfg_; }

  // ---- ingestion (thread-safe) ----
  bool ingest(uint64_t key, int64_t ts, double value);
  size_t ingest_batch(const EventView& ev);
  // Splits the batch across `threads` worker threads (events interleaved by index).
  size_t ingest_parallel(const EventView& ev, unsigned threads);

  // ---- deterministic replay (single-threaded, canonical order) ----
  // Events are applied sorted by (ts, input index) regardless of input order.
  size_t replay(const EventView& ev);
  // Same ordering; writes one kNumFeatures row per input event (in input order).
  // inclusive=false: features exclude the event itself (point-in-time safe for ML).
  void replay_features(const EventView& ev, bool inclusive, double* out);

  // ---- lookups (thread-safe) ----
  Features query(uint64_t key, int64_t now) const;
  void query_batch(const uint64_t* keys, size_t n, int64_t now, double* out) const;
  void query_batch_at(const uint64_t* keys, const int64_t* nows, size_t n, double* out) const;

  // ---- introspection ----
  Stats stats() const;
  std::vector<Sample> window_samples(uint64_t key) const;  // ring contents, oldest first
  void clear();

 private:
  struct alignas(64) Shard {
    mutable std::mutex mu;
    std::unordered_map<uint64_t, KeyState> map;
    uint64_t accepted = 0, late = 0, overflow = 0;
  };

  Shard& shard_for(uint64_t key) { return shards_[mix(key) & shard_mask_]; }
  const Shard& shard_for(uint64_t key) const { return shards_[mix(key) & shard_mask_]; }
  static uint64_t mix(uint64_t x);

  Config cfg_;
  size_t shard_mask_;
  std::vector<Shard> shards_;
  std::atomic<uint64_t> rejected_{0};
};

}  // namespace microfeature
