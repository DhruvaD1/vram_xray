#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace c10 {
struct GatheredContext;
}

namespace vramxray {

struct Event {
  double ts;        // seconds, monotonic
  int32_t action;   // TraceEntry::Action for torch events, DriverKind for driver events
  int32_t device;
  uint64_t addr;
  uint64_t size;
  uint64_t stream;
  int32_t lib;      // index into lib_names(), 0 = torch allocator
  int32_t ctx = -1; // index into the kept contexts (allocation stacks), -1 for none
};

enum DriverKind : int32_t {
  DRV_ALLOC = 100,
  DRV_FREE = 101,
  DRV_CREATE = 102,
  DRV_RELEASE = 103,
  DRV_MODULE = 104,
};

extern std::mutex g_mu;  // guards everything below plus the per-library counters in cupti_attr

double now_s();
void push(Event e);                       // drops events once the buffer is full
std::vector<Event> take_events();         // swap out everything buffered
size_t event_count();

int32_t lib_id_locked(const std::string& name);  // caller holds g_mu
std::vector<std::string> lib_names_locked();

// allocation stacks torch gathered, kept until the next drain and symbolized then
int32_t keep_context_locked(std::shared_ptr<c10::GatheredContext> ctx);
std::vector<std::shared_ptr<c10::GatheredContext>> take_contexts();

}  // namespace vramxray
