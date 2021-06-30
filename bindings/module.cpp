// pybind11 bindings: NumPy in, NumPy out. Heavy loops run with the GIL released.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

#include "microfeature/engine.hpp"

namespace py = pybind11;
using namespace microfeature;

namespace {

using KeyArray = py::array_t<uint64_t, py::array::c_style | py::array::forcecast>;
using TsArray = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using ValArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

EventView make_view(const KeyArray& k, const TsArray& t, const ValArray& v) {
  if (k.ndim() != 1 || t.ndim() != 1 || v.ndim() != 1)
    throw std::invalid_argument("keys, ts and values must be 1-D arrays");
  if (k.size() != t.size() || k.size() != v.size())
    throw std::invalid_argument("keys, ts and values must have equal length");
  return {k.data(), t.data(), v.data(), static_cast<size_t>(k.size())};
}

py::array_t<double> feature_matrix(size_t n) {
  return py::array_t<double>({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(kNumFeatures)});
}

}  // namespace

PYBIND11_MODULE(microfeature, m) {
  m.doc() = "Low-latency sliding-window feature engine (C++17) with NumPy bindings";

  py::list names;
  for (size_t i = 0; i < kNumFeatures; ++i) names.append(feature_names()[i]);
  m.attr("FEATURE_NAMES") = py::tuple(names);

  py::class_<Engine>(m, "Engine")
      .def(py::init([](int64_t window_ms, int64_t halflife_ms, size_t num_shards,
                       size_t max_events_per_key) {
             Config c;
             c.window_ms = window_ms;
             c.halflife_ms = halflife_ms;
             c.num_shards = num_shards;
             c.max_events_per_key = max_events_per_key;
             return std::make_unique<Engine>(c);
           }),
           py::arg("window_ms") = 60000, py::arg("halflife_ms") = 30000,
           py::arg("num_shards") = 64, py::arg("max_events_per_key") = (1u << 20))
      .def_property_readonly("window_ms", [](const Engine& e) { return e.config().window_ms; })
      .def_property_readonly("halflife_ms", [](const Engine& e) { return e.config().halflife_ms; })
      .def_property_readonly("num_shards", [](const Engine& e) { return e.config().num_shards; })
      .def("ingest_one",
           [](Engine& e, uint64_t key, int64_t ts, double value) { return e.ingest(key, ts, value); },
           py::arg("key"), py::arg("ts"), py::arg("value"),
           "Ingest one event; returns False if it was rejected or arrived too late for its window.")
      .def("ingest",
           [](Engine& e, const KeyArray& k, const TsArray& t, const ValArray& v) {
             EventView ev = make_view(k, t, v);
             py::gil_scoped_release nogil;
             return e.ingest_batch(ev);
           },
           py::arg("keys"), py::arg("ts"), py::arg("values"),
           "Ingest arrays of events on the calling thread; returns the number accepted.")
      .def("ingest_parallel",
           [](Engine& e, const KeyArray& k, const TsArray& t, const ValArray& v, unsigned threads) {
             EventView ev = make_view(k, t, v);
             py::gil_scoped_release nogil;
             return e.ingest_parallel(ev, threads);
           },
           py::arg("keys"), py::arg("ts"), py::arg("values"), py::arg("threads") = 4,
           "Ingest arrays of events from several C++ threads; returns the number accepted.")
      .def("replay",
           [](Engine& e, const KeyArray& k, const TsArray& t, const ValArray& v) {
             EventView ev = make_view(k, t, v);
             py::gil_scoped_release nogil;
             return e.replay(ev);
           },
           py::arg("keys"), py::arg("ts"), py::arg("values"),
           "Deterministic replay: apply events sorted by (ts, input index) on one thread.")
      .def("replay_features",
           [](Engine& e, const KeyArray& k, const TsArray& t, const ValArray& v, bool inclusive) {
             EventView ev = make_view(k, t, v);
             auto out = feature_matrix(ev.n);
             double* p = out.mutable_data();
             {
               py::gil_scoped_release nogil;
               e.replay_features(ev, inclusive, p);
             }
             return out;
           },
           py::arg("keys"), py::arg("ts"), py::arg("values"), py::arg("inclusive") = false,
           "Replay events in canonical order and return the (n, 5) feature matrix observed at each "
           "event. inclusive=False (default) excludes the event itself, i.e. point-in-time safe.")
      .def("features",
           [](const Engine& e, const KeyArray& k, int64_t now) {
             if (k.ndim() != 1) throw std::invalid_argument("keys must be a 1-D array");
             auto out = feature_matrix(static_cast<size_t>(k.size()));
             const uint64_t* kp = k.data();
             double* op = out.mutable_data();
             size_t n = static_cast<size_t>(k.size());
             py::gil_scoped_release nogil;
             e.query_batch(kp, n, now, op);
             return out;
           },
           py::arg("keys"), py::arg("now"),
           "Features for each key over the window (now - window_ms, now]; returns an (n, 5) float64 "
           "array with columns FEATURE_NAMES. Unknown keys give count=0 and NaN mean/ewma.")
      .def("features_at",
           [](const Engine& e, const KeyArray& k, const TsArray& nows) {
             if (k.ndim() != 1 || nows.ndim() != 1 || k.size() != nows.size())
               throw std::invalid_argument("keys and nows must be 1-D arrays of equal length");
             auto out = feature_matrix(static_cast<size_t>(k.size()));
             const uint64_t* kp = k.data();
             const int64_t* np_ = nows.data();
             double* op = out.mutable_data();
             size_t n = static_cast<size_t>(k.size());
             py::gil_scoped_release nogil;
             e.query_batch_at(kp, np_, n, op);
             return out;
           },
           py::arg("keys"), py::arg("nows"), "Like features() but with a separate query time per key.")
      .def("lookup",
           [](const Engine& e, uint64_t key, int64_t now) {
             auto out = py::array_t<double>(static_cast<py::ssize_t>(kNumFeatures));
             e.query(key, now).write(out.mutable_data());
             return out;
           },
           py::arg("key"), py::arg("now"), "Single-key feature lookup; returns a length-5 array.")
      .def("window_samples",
           [](const Engine& e, uint64_t key) {
             auto s = e.window_samples(key);
             py::array_t<double> out({static_cast<py::ssize_t>(s.size()), static_cast<py::ssize_t>(2)});
             double* p = out.mutable_data();
             for (size_t i = 0; i < s.size(); ++i) {
               p[2 * i] = static_cast<double>(s[i].ts);
               p[2 * i + 1] = s[i].value;
             }
             return out;
           },
           py::arg("key"), "Raw (ts, value) samples currently buffered for a key.")
      .def("stats",
           [](const Engine& e) {
             Stats s = e.stats();
             py::dict d;
             d["accepted"] = s.accepted;
             d["late_dropped"] = s.late_dropped;
             d["overflow_dropped"] = s.overflow_dropped;
             d["rejected"] = s.rejected;
             d["keys"] = s.keys;
             return d;
           })
      .def("clear", &Engine::clear);
}
