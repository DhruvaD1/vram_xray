#include <torch/extension.h>

#include "common.h"
#include "cupti_attr.h"
#include "nvml_shim.h"
#include "torch_hooks.h"

namespace {

py::dict drain() {
  auto ev = vramxray::take_events();
  std::vector<std::string> names;
  {
    std::lock_guard<std::mutex> g(vramxray::g_mu);
    names = vramxray::lib_names_locked();
  }
  py::list ts, action, device, addr, size, stream, lib;
  for (auto& e : ev) {
    ts.append(e.ts);
    action.append(e.action);
    device.append(e.device);
    addr.append(e.addr);
    size.append(e.size);
    stream.append(e.stream);
    lib.append(names[e.lib]);
  }
  py::dict d;
  d["ts"] = ts;
  d["action"] = action;
  d["device"] = device;
  d["addr"] = addr;
  d["size"] = size;
  d["stream"] = stream;
  d["lib"] = lib;
  return d;
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
  m.def("stats", &stats);
}
