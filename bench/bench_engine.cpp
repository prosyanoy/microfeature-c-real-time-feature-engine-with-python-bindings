// Native (no Python) micro-benchmarks. Items/s = events/s or lookups/s.
#include <benchmark/benchmark.h>

#include <random>

#include "microfeature/engine.hpp"

using namespace microfeature;

namespace {
constexpr uint64_t kKeys = 10'000;

EventBatch make_stream(size_t n, uint32_t seed = 1) {
  std::mt19937_64 rng(seed);
  EventBatch b;
  int64_t t = 0;
  for (size_t i = 0; i < n; ++i) {
    t += static_cast<int64_t>(rng() % 3);  // ~0.7 events/ms overall, time-ordered
    b.push(rng() % kKeys, t, static_cast<double>(rng() % 1000) / 10.0);
  }
  return b;
}
}  // namespace

static void BM_IngestSingleThread(benchmark::State& st) {
  EventBatch b = make_stream(1'000'000);
  for (auto _ : st) {
    Engine e;
    benchmark::DoNotOptimize(e.ingest_batch(b.view()));
  }
  st.SetItemsProcessed(st.iterations() * static_cast<int64_t>(b.size()));
}
BENCHMARK(BM_IngestSingleThread)->Unit(benchmark::kMillisecond);

static void BM_IngestParallel(benchmark::State& st) {
  EventBatch b = make_stream(1'000'000);
  const unsigned threads = static_cast<unsigned>(st.range(0));
  for (auto _ : st) {
    Engine e;
    benchmark::DoNotOptimize(e.ingest_parallel(b.view(), threads));
  }
  st.SetItemsProcessed(st.iterations() * static_cast<int64_t>(b.size()));
}
BENCHMARK(BM_IngestParallel)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseRealTime()->Unit(benchmark::kMillisecond);

static void BM_ReplayFeatures(benchmark::State& st) {
  EventBatch b = make_stream(500'000);
  std::vector<double> out(b.size() * kNumFeatures);
  for (auto _ : st) {
    Engine e;
    e.replay_features(b.view(), true, out.data());
    benchmark::DoNotOptimize(out.data());
  }
  st.SetItemsProcessed(st.iterations() * static_cast<int64_t>(b.size()));
}
BENCHMARK(BM_ReplayFeatures)->Unit(benchmark::kMillisecond);

static void BM_Lookup(benchmark::State& st) {
  static Engine* engine = [] {
    auto* e = new Engine();
    EventBatch b = make_stream(1'000'000);
    e->ingest_batch(b.view());
    return e;
  }();
  std::mt19937_64 rng(3);
  const int64_t now = 700'000;
  uint64_t k = 0;
  for (auto _ : st) {
    k = rng() % kKeys;
    benchmark::DoNotOptimize(engine->query(k, now));
  }
  st.SetItemsProcessed(st.iterations());
}
BENCHMARK(BM_Lookup);

static void BM_LookupContended(benchmark::State& st) {
  static Engine* engine = [] {
    auto* e = new Engine();
    EventBatch b = make_stream(1'000'000);
    e->ingest_batch(b.view());
    return e;
  }();
  std::mt19937_64 rng(st.thread_index() + 1);
  for (auto _ : st) {
    uint64_t k = rng() % kKeys;
    benchmark::DoNotOptimize(engine->query(k, 700'000));
    if ((rng() & 7) == 0) engine->ingest(k, 700'000 + static_cast<int64_t>(rng() % 100), 1.0);
  }
  st.SetItemsProcessed(st.iterations());
}
BENCHMARK(BM_LookupContended)->Threads(1)->Threads(4);

BENCHMARK_MAIN();
