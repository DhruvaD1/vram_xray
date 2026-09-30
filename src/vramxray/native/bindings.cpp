#include <pybind11/numpy.h>
#include <torch/extension.h>

#include "common.h"
#include "cupti_attr.h"
#include "mirror.h"
#include "nvml_shim.h"
#include "streams.h"
#include "torch_hooks.h"

namespace {

template <class T>
py::array_t<T> column(const std::vector<vramxray::Event>& ev, T vramxray::Event::*field) {
  py::array_t<T> arr(ev.size());
  T* out = arr.mutable_data();
  for (size_t i = 0; i < ev.size(); i++) out[i] = ev[i].*field;
  return arr;
}

// numeric columns come back as numpy arrays, so a long run does not become a million Python ints
py::dict drain() {
  auto ev = vramxray::take_events();
  auto ctxs = vramxray::take_contexts();
  const std::vector<std::string> names = vramxray::lib_names();
  std::vector<std::string> stacks = vramxray::symbolize_contexts(ctxs);
  py::list lib, stack;
  for (auto& e : ev) {
    lib.append(names[e.lib]);
    stack.append(e.ctx >= 0 && (size_t)e.ctx < stacks.size() ? stacks[e.ctx] : std::string());
  }
  py::dict d;
  d["ts"] = column(ev, &vramxray::Event::ts);
  d["action"] = column(ev, &vramxray::Event::action);
  d["device"] = column(ev, &vramxray::Event::device);
  d["addr"] = column(ev, &vramxray::Event::addr);
  d["size"] = column(ev, &vramxray::Event::size);
  d["stream"] = column(ev, &vramxray::Event::stream);
  d["lib"] = lib;
  d["stack"] = stack;
  return d;
}

py::list stream_findings() {
  py::list out;
  for (auto& f : vramxray::streams_findings()) {
    py::dict d;
    d["kind"] = f.kind;
    d["lib"] = f.lib;
    d["detail"] = f.detail;
    d["count"] = f.count;
    out.append(d);
  }
  return out;
}

py::dict mirror_stats(int device, double old_seconds) {
  auto m = vramxray::mirror_stats(device, old_seconds);
  py::dict d;
  d["reserved"] = m.reserved;
  d["live"] = m.live;
  d["free_in_segments"] = m.free_in_segments;
  d["largest_free"] = m.largest_free;
  d["old_bytes"] = m.old_bytes;
  d["segments"] = m.segments;
  d["blocks"] = m.blocks;
  d["old_blocks"] = m.old_blocks;
  return d;
}

py::list mirror_top_sites(int device, size_t top, size_t scan) {
  py::list out;
  for (const auto& s : vramxray::mirror_top_sites(device, top, scan)) {
    py::dict d;
    d["where"] = s.where;
    d["bytes"] = s.bytes;
    d["blocks"] = s.blocks;
    out.append(d);
  }
  return out;
}

py::list mirror_site_history() {
  py::list out;
  for (const auto& s : vramxray::mirror_site_history()) {
    py::list sites;
    for (const auto& b : s.sites) {
      py::dict d;
      d["where"] = b.where;
      d["bytes"] = b.bytes;
      d["blocks"] = b.blocks;
      sites.append(d);
    }
    out.append(py::make_tuple(s.t, s.device, sites));
  }
  return out;
}

py::list mirror_pools(int device) {
  py::list out;
  for (const auto& p : vramxray::mirror_pools(device)) {
    py::dict d;
    d["id"] = py::make_tuple(p.id0, p.id1);
    d["bytes"] = p.bytes;
    d["blocks"] = p.blocks;
    out.append(d);
  }
  return out;
}

py::dict stats() {
  py::dict d;
  d["events"] = vramxray::event_count();
  d["oom_calls"] = vramxray::oom_calls();
  d["subscribed"] = vramxray::cupti_active();
  d["nvml"] = vramxray::nvml().ok;
  d["live_ptrs"] = vramxray::live_ptrs();
  d["live_handles"] = vramxray::live_handles();
  return d;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("install", &vramxray::install, "attach to torch's allocator (call after CUDA init)");
  m.def("cupti_start", &vramxray::cupti_start, "subscribe to driver allocation callbacks; returns CUPTI result");
  m.def("cupti_stop", &vramxray::cupti_stop);
  m.def("libs", &vramxray::libs, "live device bytes per calling library");
  m.def("kernel_images", &vramxray::kernel_images, "module/library image bytes per calling library");
  m.def("drain", &drain, "take all buffered events");
  m.def("streams_start", &vramxray::streams_start, py::arg("channels") = false,
        py::arg("check_legacy") = false,
        "enable stream checks. channels=True also maps streams to hardware channels");
  m.def("streams_stop", &vramxray::streams_stop);
  m.def("streams_findings", &stream_findings, "stream hygiene findings so far");
  m.def("stats", &stats);
  m.def("mirror_stats", &mirror_stats, py::arg("device") = 0, py::arg("old_seconds") = 60.0,
        "allocator layout numbers kept live from the trace events");
  m.def("mirror_devices", &vramxray::mirror_devices);
  m.def("mirror_top_sites", &mirror_top_sites, py::arg("device") = 0, py::arg("top") = 8,
        py::arg("scan") = 512, "live bytes by call site, without taking a snapshot");
  m.def("mirror_sample_sites", &vramxray::mirror_sample_sites, py::arg("device") = 0,
        py::arg("top") = 4096, "record one sample of the live blocks, names unresolved");
  m.def("mirror_site_history", &mirror_site_history,
        "resolve the recorded samples. Runs Python, so call it from the main thread");
  m.def("mirror_pools", &mirror_pools, py::arg("device") = 0,
        "live bytes held by each CUDA graph private pool");
  // seeding only needs the shape of what is already there, never a stack
  m.def(
      "mirror_event",
      [](int32_t action, int32_t device, uint64_t addr, uint64_t size) {
        vramxray::mirror_on_trace(action, device, addr, size);
      },
      py::arg("action"), py::arg("device"), py::arg("addr"), py::arg("size"),
      "feed one event in by hand, used to seed the mirror with what existed before we attached");
  m.def("mirror_reset", &vramxray::mirror_reset);
  m.def("stalls", &vramxray::stalls, "nanoseconds and calls inside driver allocation APIs, per library");
}
