// Shared state for the native core: the event buffer, the library name table, and the
// allocation stacks torch hands us.
//
// Locking. Each subsystem owns one mutex and no code path holds two at once, which is what
// keeps the allocator hot path from serialising on a single global lock. If a future change
// really does need two, take them in the order they are declared here and say so at the call
// site. Functions whose name ends in _locked expect the matching mutex to be held already.
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
  double ts;         // seconds, monotonic
  int32_t action;    // TraceEntry::Action for torch events, DriverKind for driver events
  int32_t device;
  uint64_t addr;
  uint64_t size;
  uint64_t stream;
  int32_t lib;       // index into lib_names(), 0 = torch allocator
  int32_t ctx = -1;  // index into the kept stacks, -1 when none was recorded
};

enum DriverKind : int32_t {
  DRV_ALLOC = 100,
  DRV_FREE = 101,
  DRV_CREATE = 102,
  DRV_RELEASE = 103,
  DRV_MODULE = 104,
};

// declared in lock order, see the note at the top of this file
std::mutex& libs_mu();    // the library name table
std::mutex& events_mu();  // the event buffer and the kept stacks

double now_s() noexcept;

// Both buffers are bounded. Dropping the newest event is the only safe answer when a program
// allocates faster than anything drains us, and a run that never asks for the timeline must
// not grow without limit.
void push(const Event& e) noexcept;
std::vector<Event> take_events();
size_t event_count();

int32_t lib_id_locked(const std::string& name);  // caller holds libs_mu()
const std::vector<std::string>& lib_names_locked();
std::vector<std::string> lib_names();  // takes libs_mu() itself

// Returns the index to store in Event::ctx, or -1 once the buffer is full.
int32_t keep_context_locked(std::shared_ptr<c10::GatheredContext> ctx);
std::vector<std::shared_ptr<c10::GatheredContext>> take_contexts();

}  // namespace vramxray
