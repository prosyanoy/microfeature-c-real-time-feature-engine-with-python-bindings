// Time-ordered ring buffer of (timestamp, value) samples.
//
// Samples are kept sorted by (ts, value). Appending in timestamp order is O(1);
// a late (out-of-order) sample is inserted by shifting the few newer samples.
// Keeping the contents sorted makes the buffer state independent of arrival
// order, which is what lets multi-threaded ingestion match deterministic replay.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace microfeature {

struct Sample {
  int64_t ts;
  double value;
};

inline bool sample_less(const Sample& a, const Sample& b) {
  return a.ts < b.ts || (a.ts == b.ts && a.value < b.value);
}

class RingBuffer {
 public:
  explicit RingBuffer(size_t initial_capacity = 8) {
    size_t cap = 2;
    while (cap < initial_capacity) cap <<= 1;
    buf_.resize(cap);
    mask_ = cap - 1;
  }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  size_t capacity() const { return buf_.size(); }

  // Logical index: 0 is the oldest sample.
  const Sample& operator[](size_t i) const { return buf_[(head_ + i) & mask_]; }

  void pop_front() {
    head_ = (head_ + 1) & mask_;
    --size_;
  }

  void clear() {
    head_ = 0;
    size_ = 0;
  }

  void insert_sorted(const Sample& s) {
    if (size_ == buf_.size()) grow();
    size_t pos = size_;
    while (pos > 0 && sample_less(s, (*this)[pos - 1])) --pos;
    ++size_;
    for (size_t i = size_ - 1; i > pos; --i) slot(i) = slot(i - 1);
    slot(pos) = s;
  }

  // Index of the first sample with ts > t (== size() if none).
  size_t upper_bound_ts(int64_t t) const {
    size_t lo = 0, hi = size_;
    while (lo < hi) {
      size_t mid = lo + (hi - lo) / 2;
      if ((*this)[mid].ts <= t) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  double sum_range(size_t begin, size_t end) const {
    double s = 0.0;
    for (size_t i = begin; i < end; ++i) s += (*this)[i].value;
    return s;
  }

 private:
  Sample& slot(size_t i) { return buf_[(head_ + i) & mask_]; }

  void grow() {
    std::vector<Sample> next(buf_.size() * 2);
    for (size_t i = 0; i < size_; ++i) next[i] = (*this)[i];
    buf_.swap(next);
    head_ = 0;
    mask_ = buf_.size() - 1;
  }

  std::vector<Sample> buf_;
  size_t head_ = 0;
  size_t size_ = 0;
  size_t mask_ = 0;
};

}  // namespace microfeature
