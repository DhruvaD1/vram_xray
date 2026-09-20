#include "nvml_shim.h"

#include <dlfcn.h>

namespace vramxray {

namespace {
struct Mem2 {
  unsigned version;
  unsigned long long total, reserved, free, used;
};
}  // namespace

void Nvml::open() {
  for (const char* n : {"libnvidia-ml.so.1", "/usr/lib/wsl/lib/libnvidia-ml.so.1"}) {
    h_ = dlopen(n, RTLD_NOW);
    if (h_) break;
  }
  if (!h_) return;
  init_ = (int (*)())dlsym(h_, "nvmlInit_v2");
  by_index_ = (int (*)(unsigned, void**))dlsym(h_, "nvmlDeviceGetHandleByIndex_v2");
  meminfo_ = (int (*)(void*, void*))dlsym(h_, "nvmlDeviceGetMemoryInfo_v2");
  ok = init_ && by_index_ && meminfo_ && init_() == 0;
}

long long Nvml::used(unsigned dev) {
  if (!ok) return -1;
  void*& hd = handles_[dev];
  if (!hd && by_index_(dev, &hd) != 0) return -1;
  Mem2 m{};
  m.version = (2u << 24) | (unsigned)sizeof(Mem2);
  if (meminfo_(hd, &m) != 0) return -1;
  return (long long)m.used;
}

Nvml& nvml() {
  static Nvml n;
  return n;
}

}  // namespace vramxray
