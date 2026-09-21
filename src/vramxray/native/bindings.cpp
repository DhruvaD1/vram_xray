#include <pybind11/numpy.h>
#include <torch/extension.h>

#include "common.h"
#include "cupti_attr.h"
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
  std::vector<std::string> names;
  {
    std::lock_guard<std::mutex> g(vramxray::g_mu);
    names = vramxray::lib_names_locked();
  }
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
}
