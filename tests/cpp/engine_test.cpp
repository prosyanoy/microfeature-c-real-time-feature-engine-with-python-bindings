#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "test_util.hpp"

using namespace microfeature;

namespace {
Config small_cfg() {
  Config c;
  c.window_ms = 1000;
  c.halflife_ms = 500;
  c.num_shards = 4;
  return c;
}
void expect_features_near(const Features& a, const Features& b) {
  EXPECT_DOUBLE_EQ(a.count, b.count);
  EXPECT_NEAR(a.sum, b.sum, 1e-8);
  EXPECT_DOUBLE_EQ(a.rate_per_sec, b.rate_per_sec);
  if (std::isnan(b.mean)) EXPECT_TRUE(std::isnan(a.mean)); else EXPECT_NEAR(a.mean, b.mean, 1e-8);
  if (std::isnan(b.ewma)) EXPECT_TRUE(std::isnan(a.ewma)); else EXPECT_NEAR(a.ewma, b.ewma, 1e-9);
}
}  // namespace

TEST(Engine, UnknownKeyIsEmpty) {
  Engine e(small_cfg());
  Features f = e.query(42, 0);
  EXPECT_EQ(f.count, 0);
  EXPECT_TRUE(std::isnan(f.mean));
  EXPECT_TRUE(std::isnan(f.ewma));
}

TEST(Engine, WindowBoundaryIsHalfOpen) {
  Engine e(small_cfg());
  e.ingest(1, 0, 10);
  e.ingest(1, 500, 20);
  e.ingest(1, 1000, 30);
  Features f = e.query(1, 1000);  // window (0, 1000]: the ts=0 sample is out
  EXPECT_EQ(f.count, 2);
  EXPECT_DOUBLE_EQ(f.sum, 50);
  EXPECT_DOUBLE_EQ(f.mean, 25);
  EXPECT_DOUBLE_EQ(f.rate_per_sec, 2.0);
  // ts=0 was evicted when the watermark reached 0, so history before the retained window is gone.
  EXPECT_EQ(e.query(1, 999).count, 1);
}

TEST(Engine, ExpiryAsTimeAdvancesWithoutNewEvents) {
  Engine e(small_cfg());
  e.ingest(1, 100, 5);
  EXPECT_EQ(e.query(1, 1000).count, 1);
  EXPECT_EQ(e.query(1, 1100).count, 0);  // expired purely by query time
  EXPECT_TRUE(std::isnan(e.query(1, 1100).mean));
  EXPECT_FALSE(std::isnan(e.query(1, 1100).ewma));  // EWMA has no hard window
}

TEST(Engine, MatchesBruteForceOnRandomStream) {
  Config cfg = small_cfg();
  Engine e(cfg);
  EventBatch b = mftest::random_events(20000, 17, 10000, 1);
  e.replay(b.view());
  // All keys' newest events define the watermark; probe at and after the newest event.
  for (uint64_t k = 0; k < 17; ++k) {
    int64_t newest = 0;
    for (size_t i = 0; i < b.size(); ++i) if (b.keys[i] == k) newest = std::max(newest, b.ts[i]);
    for (int64_t now : {newest, newest + 250, newest + 999, newest + 5000})
      expect_features_near(e.query(k, now), mftest::brute_force(b, k, now, cfg.window_ms, cfg.halflife_ms));
  }
}

TEST(Engine, QueryInThePastWithinBufferedWindow) {
  Config cfg = small_cfg();
  Engine e(cfg);
  EventBatch b;
  for (int i = 0; i < 100; ++i) b.push(3, i * 10, double(i));
  e.replay(b.view());
  // Newest ts is 990; the buffer still holds (-10, 990] so an earlier query is exact for ts>0.
  expect_features_near(e.query(3, 600), mftest::brute_force(b, 3, 600, cfg.window_ms, cfg.halflife_ms));
}

TEST(Engine, EwmaMatchesClosedFormAndIsOrderIndependent) {
  Config cfg = small_cfg();
  EventBatch b = mftest::random_events(500, 1, 3000, 9);
  Engine fwd(cfg), rev(cfg);
  for (size_t i = 0; i < b.size(); ++i) fwd.ingest(b.keys[i], b.ts[i], b.values[i]);
  for (size_t i = b.size(); i-- > 0;) rev.ingest(b.keys[i], b.ts[i], b.values[i]);
  double expect = mftest::brute_force(b, 0, 3000, cfg.window_ms, cfg.halflife_ms).ewma;
  EXPECT_NEAR(fwd.query(0, 3000).ewma, expect, 1e-9);
  EXPECT_NEAR(rev.query(0, 3000).ewma, expect, 1e-9);
}

TEST(Engine, EwmaHalfLife) {
  Engine e(small_cfg());  // half-life 500 ms
  e.ingest(1, 0, 0.0);
  e.ingest(1, 500, 3.0);  // old sample has weight 0.5 -> (0*0.5 + 3)/(1.5) = 2
  EXPECT_NEAR(e.query(1, 500).ewma, 2.0, 1e-12);
}

TEST(Engine, LateEventsAreDroppedFromWindowButCounted) {
  Engine e(small_cfg());
  EXPECT_TRUE(e.ingest(1, 5000, 1));
  EXPECT_FALSE(e.ingest(1, 3000, 1));  // 3000 <= 5000 - 1000
  EXPECT_TRUE(e.ingest(1, 4500, 1));   // late but still inside the window
  Stats s = e.stats();
  EXPECT_EQ(s.accepted, 2u);
  EXPECT_EQ(s.late_dropped, 1u);
  EXPECT_EQ(e.query(1, 5000).count, 2);
}

TEST(Engine, NonFiniteValuesRejected) {
  Engine e(small_cfg());
  EXPECT_FALSE(e.ingest(1, 0, std::nan("")));
  EXPECT_FALSE(e.ingest(1, 0, std::numeric_limits<double>::infinity()));
  EXPECT_EQ(e.stats().rejected, 2u);
  EXPECT_EQ(e.stats().keys, 0u);
}

TEST(Engine, PerKeyCapDropsOldest) {
  Config c = small_cfg();
  c.window_ms = 1'000'000;
  c.max_events_per_key = 10;
  Engine e(c);
  for (int i = 0; i < 25; ++i) e.ingest(1, i, 1.0);
  EXPECT_EQ(e.query(1, 100).count, 10);
  EXPECT_EQ(e.stats().overflow_dropped, 15u);
  EXPECT_NEAR(e.query(1, 100).sum, 10.0, 1e-12);
}

TEST(Engine, ReplayFeaturesPointInTimeVsInclusive) {
  Config cfg = small_cfg();
  EventBatch b;
  b.push(1, 0, 10); b.push(1, 100, 20); b.push(1, 200, 30);
  Engine ex(cfg), in(cfg);
  std::vector<double> a(b.size() * kNumFeatures), c(b.size() * kNumFeatures);
  ex.replay_features(b.view(), false, a.data());
  in.replay_features(b.view(), true, c.data());
  EXPECT_EQ(a[0 * kNumFeatures], 0);  // nothing before the first event
  EXPECT_TRUE(std::isnan(a[0 * kNumFeatures + 2]));
  EXPECT_EQ(a[2 * kNumFeatures], 2);
  EXPECT_DOUBLE_EQ(a[2 * kNumFeatures + 1], 30);
  EXPECT_EQ(c[2 * kNumFeatures], 3);
  EXPECT_DOUBLE_EQ(c[2 * kNumFeatures + 2], 20);
}

TEST(Engine, ReplayIsDeterministicAcrossInputOrder) {
  Config cfg = small_cfg();
  EventBatch b = mftest::random_events(5000, 8, 5000, 3);
  EventBatch shuffled;
  std::vector<size_t> idx(b.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
  std::mt19937 rng(5);
  std::shuffle(idx.begin(), idx.end(), rng);
  for (size_t i : idx) shuffled.push(b.keys[i], b.ts[i], b.values[i]);
  Engine x(cfg), y(cfg);
  x.replay(b.view());
  y.replay(shuffled.view());
  for (uint64_t k = 0; k < 8; ++k) {
    auto sx = x.window_samples(k), sy = y.window_samples(k);
    ASSERT_EQ(sx.size(), sy.size());
    for (size_t i = 0; i < sx.size(); ++i) {
      EXPECT_EQ(sx[i].ts, sy[i].ts);
      EXPECT_EQ(sx[i].value, sy[i].value);
    }
    expect_features_near(x.query(k, 5000), y.query(k, 5000));
  }
}

TEST(Engine, ClearResets) {
  Engine e(small_cfg());
  e.ingest(1, 0, 1);
  e.clear();
  EXPECT_EQ(e.stats().keys, 0u);
  EXPECT_EQ(e.query(1, 0).count, 0);
}

TEST(Engine, InvalidConfigThrows) {
  Config c;
  c.window_ms = 0;
  EXPECT_THROW(Engine{c}, std::invalid_argument);
}

TEST(Engine, ShardCountRoundedToPowerOfTwo) {
  Config c;
  c.num_shards = 5;
  EXPECT_EQ(Engine(c).config().num_shards, 8u);
}
