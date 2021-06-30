#include <gtest/gtest.h>

#include <algorithm>
#include <random>

#include "microfeature/ring_buffer.hpp"

using microfeature::RingBuffer;
using microfeature::Sample;

TEST(RingBuffer, AppendInOrderAndWrapAround) {
  RingBuffer r(4);
  for (int i = 0; i < 100; ++i) {
    r.insert_sorted({i, double(i)});
    if (r.size() > 3) r.pop_front();  // forces head to wrap many times
  }
  ASSERT_EQ(r.size(), 3u);
  EXPECT_EQ(r[0].ts, 97);
  EXPECT_EQ(r[2].ts, 99);
}

TEST(RingBuffer, GrowsAndKeepsOrder) {
  RingBuffer r(2);
  for (int i = 0; i < 1000; ++i) r.insert_sorted({i, 1.0});
  EXPECT_EQ(r.size(), 1000u);
  EXPECT_GE(r.capacity(), 1000u);
  for (size_t i = 1; i < r.size(); ++i) EXPECT_LE(r[i - 1].ts, r[i].ts);
}

TEST(RingBuffer, OutOfOrderInsertIsSorted) {
  std::mt19937 rng(7);
  std::vector<Sample> in;
  for (int i = 0; i < 500; ++i) in.push_back({int64_t(rng() % 100), double(rng() % 10)});
  RingBuffer r;
  for (auto& s : in) r.insert_sorted(s);
  std::sort(in.begin(), in.end(), microfeature::sample_less);
  ASSERT_EQ(r.size(), in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    EXPECT_EQ(r[i].ts, in[i].ts);
    EXPECT_EQ(r[i].value, in[i].value);
  }
}

TEST(RingBuffer, UpperBoundAndSumRange) {
  RingBuffer r;
  for (int i = 0; i < 10; ++i) r.insert_sorted({i * 10, double(i)});
  EXPECT_EQ(r.upper_bound_ts(-1), 0u);
  EXPECT_EQ(r.upper_bound_ts(0), 1u);
  EXPECT_EQ(r.upper_bound_ts(45), 5u);
  EXPECT_EQ(r.upper_bound_ts(90), 10u);
  EXPECT_DOUBLE_EQ(r.sum_range(2, 5), 2 + 3 + 4);
}
