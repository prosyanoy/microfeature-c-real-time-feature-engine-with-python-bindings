#include "microfeature/engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace microfeature {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kLn2 = 0.693147180559945309417;
constexpr size_t kResyncEvery = 4096;

int64_t sat_sub(int64_t a, int64_t b) {
  return a < std::numeric_limits<int64_t>::min() + b ? std::numeric_limits<int64_t>::min() : a - b;
}
}  // namespace

// ---------------------------------------------------------------- KeyState

void KeyState::update_ewma(int64_t ts, double value, double tau_ms) {
  // Time-decayed mean: sum_i v_i * exp(-(T - t_i)/tau) / sum_i exp(-(T - t_i)/tau), where T is
  // the newest timestamp. Each event is weighted relative to T, so the result does not depend
  // on arrival order (up to floating-point rounding).
  if (!has_ewma_) {
    ewma_num_ = value; ewma_den_ = 1.0; ewma_t_ = ts; has_ewma_ = true;
  } else if (ts >= ewma_t_) {
    double d = std::exp(-static_cast<double>(ts - ewma_t_) / tau_ms);
    ewma_num_ = ewma_num_ * d + value;
    ewma_den_ = ewma_den_ * d + 1.0;
    ewma_t_ = ts;
  } else {
    double d = std::exp(-static_cast<double>(ewma_t_ - ts) / tau_ms);
    ewma_num_ += value * d;
    ewma_den_ += d;
  }
}

void KeyState::evict(int64_t watermark) {
  // Drops everything with ts <= watermark (outside every future window).
  while (!ring_.empty() && ring_[0].ts <= watermark) {
    total_ -= ring_[0].value;
    ring_.pop_front();
    ++evictions_since_resync_;
  }
  if (ring_.empty()) {
    total_ = 0.0;
    evictions_since_resync_ = 0;
  } else if (evictions_since_resync_ >= kResyncEvery) {
    // Bound floating-point drift of the running sum.
    total_ = ring_.sum_range(0, ring_.size());
    evictions_since_resync_ = 0;
  }
}

KeyState::Result KeyState::insert(int64_t ts, double value, const Config& cfg) {
  update_ewma(ts, value, static_cast<double>(cfg.halflife_ms) / kLn2);
  if (has_events_ && ts <= sat_sub(max_ts_, cfg.window_ms)) return Result::kLate;

  Result result = Result::kAccepted;
  if (ring_.size() >= cfg.max_events_per_key) {
    total_ -= ring_[0].value;
    ring_.pop_front();
    result = Result::kOverflow;
  }
  ring_.insert_sorted({ts, value});
  total_ += value;
  if (!has_events_ || ts > max_ts_) max_ts_ = ts;
  has_events_ = true;
  evict(sat_sub(max_ts_, cfg.window_ms));
  return result;
}

Features KeyState::query(int64_t now, const Config& cfg) const {
  Features f;
  f.mean = kNaN;
  f.ewma = has_ewma_ ? ewma_num_ / ewma_den_ : kNaN;
  // Window is (now - W, now], matching pandas rolling('W', closed='right').
  const size_t n = ring_.size();
  const size_t lo = ring_.upper_bound_ts(sat_sub(now, cfg.window_ms));
  const size_t hi = ring_.upper_bound_ts(now);
  const size_t count = hi > lo ? hi - lo : 0;
  f.count = static_cast<double>(count);
  f.rate_per_sec = f.count / (static_cast<double>(cfg.window_ms) / 1000.0);
  if (count == 0) return f;
  // Sum whichever side is cheaper: the window itself or the (typically expired) complement.
  if (count <= lo + (n - hi)) {
    f.sum = ring_.sum_range(lo, hi);
  } else {
    f.sum = total_ - ring_.sum_range(0, lo) - ring_.sum_range(hi, n);
  }
  f.mean = f.sum / f.count;
  return f;
}

// ------------------------------------------------------------------ Engine

uint64_t Engine::mix(uint64_t x) {  // splitmix64 finaliser
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

namespace {
size_t round_shards(size_t n) {
  size_t shards = 1;
  while (shards < std::max<size_t>(n, 1)) shards <<= 1;
  return shards;
}
}  // namespace

Engine::Engine(const Config& cfg)
    : cfg_(cfg), shard_mask_(round_shards(cfg.num_shards) - 1), shards_(round_shards(cfg.num_shards)) {
  if (cfg_.window_ms <= 0) throw std::invalid_argument("window_ms must be positive");
  if (cfg_.halflife_ms <= 0) throw std::invalid_argument("halflife_ms must be positive");
  if (cfg_.max_events_per_key == 0) throw std::invalid_argument("max_events_per_key must be positive");
  cfg_.num_shards = shard_mask_ + 1;
}

bool Engine::ingest(uint64_t key, int64_t ts, double value) {
  if (!std::isfinite(value)) {
    rejected_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  Shard& sh = shard_for(key);
  std::lock_guard<std::mutex> lock(sh.mu);
  switch (sh.map[key].insert(ts, value, cfg_)) {
    case KeyState::Result::kAccepted: ++sh.accepted; return true;
    case KeyState::Result::kOverflow: ++sh.accepted; ++sh.overflow; return true;
    case KeyState::Result::kLate: ++sh.late; return false;
  }
  return false;
}

size_t Engine::ingest_batch(const EventView& ev) {
  size_t ok = 0;
  for (size_t i = 0; i < ev.n; ++i) ok += ingest(ev.keys[i], ev.ts[i], ev.values[i]) ? 1 : 0;
  return ok;
}

size_t Engine::ingest_parallel(const EventView& ev, unsigned threads) {
  if (threads <= 1) return ingest_batch(ev);
  std::atomic<size_t> ok{0};
  std::vector<std::thread> pool;
  pool.reserve(threads);
  for (unsigned t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      size_t local = 0;
      for (size_t i = t; i < ev.n; i += threads)
        local += ingest(ev.keys[i], ev.ts[i], ev.values[i]) ? 1 : 0;
      ok.fetch_add(local, std::memory_order_relaxed);
    });
  }
  for (auto& th : pool) th.join();
  return ok.load();
}

namespace {
std::vector<size_t> canonical_order(const EventView& ev) {
  std::vector<size_t> idx(ev.n);
  std::iota(idx.begin(), idx.end(), size_t{0});
  std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return ev.ts[a] < ev.ts[b]; });
  return idx;
}
}  // namespace

size_t Engine::replay(const EventView& ev) {
  size_t ok = 0;
  for (size_t i : canonical_order(ev)) ok += ingest(ev.keys[i], ev.ts[i], ev.values[i]) ? 1 : 0;
  return ok;
}

void Engine::replay_features(const EventView& ev, bool inclusive, double* out) {
  for (size_t i : canonical_order(ev)) {
    if (inclusive) {
      ingest(ev.keys[i], ev.ts[i], ev.values[i]);
      query(ev.keys[i], ev.ts[i]).write(out + i * kNumFeatures);
    } else {
      query(ev.keys[i], ev.ts[i]).write(out + i * kNumFeatures);
      ingest(ev.keys[i], ev.ts[i], ev.values[i]);
    }
  }
}

Features Engine::query(uint64_t key, int64_t now) const {
  const Shard& sh = shard_for(key);
  std::lock_guard<std::mutex> lock(sh.mu);
  auto it = sh.map.find(key);
  if (it == sh.map.end()) {
    Features f;
    f.mean = kNaN;
    f.ewma = kNaN;
    return f;
  }
  return it->second.query(now, cfg_);
}

void Engine::query_batch(const uint64_t* keys, size_t n, int64_t now, double* out) const {
  for (size_t i = 0; i < n; ++i) query(keys[i], now).write(out + i * kNumFeatures);
}

void Engine::query_batch_at(const uint64_t* keys, const int64_t* nows, size_t n, double* out) const {
  for (size_t i = 0; i < n; ++i) query(keys[i], nows[i]).write(out + i * kNumFeatures);
}

Stats Engine::stats() const {
  Stats s;
  for (const Shard& sh : shards_) {
    std::lock_guard<std::mutex> lock(sh.mu);
    s.accepted += sh.accepted;
    s.late_dropped += sh.late;
    s.overflow_dropped += sh.overflow;
    s.keys += sh.map.size();
  }
  s.rejected = rejected_.load();
  return s;
}

std::vector<Sample> Engine::window_samples(uint64_t key) const {
  const Shard& sh = shard_for(key);
  std::lock_guard<std::mutex> lock(sh.mu);
  std::vector<Sample> out;
  auto it = sh.map.find(key);
  if (it == sh.map.end()) return out;
  const RingBuffer& r = it->second.ring();
  for (size_t i = 0; i < r.size(); ++i) out.push_back(r[i]);
  return out;
}

void Engine::clear() {
  for (Shard& sh : shards_) {
    std::lock_guard<std::mutex> lock(sh.mu);
    sh.map.clear();
    sh.accepted = sh.late = sh.overflow = 0;
  }
  rejected_ = 0;
}

}  // namespace microfeature
