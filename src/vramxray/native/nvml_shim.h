#pragma once

#include <unordered_map>

namespace vramxray {

struct Nvml {
  bool ok = false;
  void open();
  long long used(unsigned device);  // -1 when unavailable

 private:
  void* h_ = nullptr;
  int (*init_)() = nullptr;
  int (*by_index_)(unsigned, void**) = nullptr;
  int (*meminfo_)(void*, void*) = nullptr;
  std::unordered_map<unsigned, void*> handles_;
};

Nvml& nvml();

}  // namespace vramxray
