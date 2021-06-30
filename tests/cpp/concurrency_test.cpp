// Concurrency suite: designed to run clean under ThreadSanitizer (make tsan).
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>

#include "test_util.hpp"

using namespace microfeature;

namespace {
Config cfg() {
  Config c;
  c.window_ms = 2000;
  c.halflife_ms = 1000;
  c.num_shards = 8;
  return c;
}

void expect_same_state(const Engine& a, const Engine& b, uint64_t keys, int64_t now) {
  for (uint64_t k = 0; k < keys; ++k) {
    auto sa = a.window_samples(k), sb = b.window_samples(k);
    ASSERT_EQ(sa.size(), sb.size()) << "key " << k;
    for (size_t i = 0; i < sa.size(); ++i) {
      ASSERT_EQ(sa[i].ts, sb[i].ts);
      ASSERT_EQ(sa[i].value, sb[i].value);
    }
    Features fa = a.query(k, now), fb = b.query(k, now);
    EXPECT_EQ(fa.count, fb.count);
    EXPECT_NEAR(fa.sum, fb.sum, 1e-6);
    if (!std::isnan(fa.ewma)) EXPECT_NEAR(fa.ewma, fb.ewma, 1e-9);
  }
}
}  // namespace

TEST(Concurrency, ParallelIngestMatchesDeterministicReplay) {
  const uint64_t keys = 64;
  EventBatch b = mftest::random_events(100000, keys, 20000, 11);
  Engine replayed(cfg()), parallel(cfg());
  replayed.replay(b.view());
  parallel.ingest_parallel(b.view(), 4);
  expect_same_state(replayed, parallel, keys, 20000);
  EXPECT_EQ(replayed.stats().keys, parallel.stats().keys);
}

TEST(Concurrency, ManyThreadsHammeringOneKey) {
  EventBatch b = mftest::random_events(40000, 1, 10000, 21);
  Engine replayed(cfg()), parallel(cfg());
  replayed.replay(b.view());
  parallel.ingest_parallel(b.view(), 8);
  expect_same_state(replayed, parallel, 1, 10000);
}

TEST(Concurrency, ReadersRunAlongsideWriters) {
  const uint64_t keys = 32;
  EventBatch b = mftest::random_events(60000, keys, 30000, 31);
  Engine e(cfg());
  std::atomic<bool> done{false};
  std::atomic<uint64_t> reads{0};
  std::vector<std::thread> readers;
  for (int r = 0; r < 3; ++r) {
    readers.emplace_back([&, r] {
      uint64_t k = r;
      while (!done.load(std::memory_order_acquire)) {
        Features f = e.query(k % keys, 30000);
        // Invariants that must hold at every instant, whatever the writers are doing.
        EXPECT_GE(f.count, 0);
        if (f.count > 0) EXPECT_FALSE(std::isnan(f.mean));
        (void)e.stats();
        ++k;
        reads.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  e.ingest_parallel(b.view(), 4);
  done.store(true, std::memory_order_release);
  for (auto& t : readers) t.join();
  EXPECT_GT(reads.load(), 0u);
  Engine ref(cfg());
  ref.replay(b.view());
  expect_same_state(ref, e, keys, 30000);
}

TEST(Concurrency, StatsAddUp) {
  EventBatch b = mftest::random_events(50000, 16, 20000, 41);
  Engine e(cfg());
  size_t accepted = e.ingest_parallel(b.view(), 4);
  Stats s = e.stats();
  EXPECT_EQ(s.accepted, accepted);
  EXPECT_EQ(s.accepted + s.late_dropped + s.rejected, b.size());
}

TEST(Concurrency, ConcurrentClearAndIngestIsSafe) {
  Engine e(cfg());
  EventBatch b = mftest::random_events(20000, 8, 5000, 51);
  std::thread w([&] { e.ingest_parallel(b.view(), 2); });
  for (int i = 0; i < 20; ++i) e.clear();
  w.join();
  SUCCEED();
}
