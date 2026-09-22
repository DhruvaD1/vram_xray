// A running copy of the allocator's layout, rebuilt from the trace events as they arrive.
//
// Why bother when torch can hand over a snapshot: a snapshot walks every block and allocates a
// large Python structure, so it is far too slow to sample often. Keeping our own copy means the
// numbers below are available at any moment for the cost of a map lookup per allocation, which
// is what makes a memory history possible at all.
//
// One caveat. The ALLOC event carries the size the caller asked for, while the block torch hands
// back is rounded up, and in the large pool it can be up to 1 MiB bigger when the remainder was
// too small to split off. So a gap here is either real free space or that rounding slack, and
// holes smaller than kSlackBytes are ignored for that reason. Reserved is exact.
#pragma once

#include <cstdint>
#include <vector>

namespace vramxray {

// same values as c10's TraceEntry::Action, checked with a static_assert where they are used
enum TraceAction : int32_t {
  TA_ALLOC = 0,
  TA_FREE_REQUESTED = 1,
  TA_FREE_COMPLETED = 2,
  TA_SEGMENT_ALLOC = 3,
  TA_SEGMENT_FREE = 4,
  TA_SEGMENT_MAP = 5,
  TA_SEGMENT_UNMAP = 6,
};

struct MirrorStats {
  uint64_t reserved = 0;          // bytes torch holds from the driver, exact
  uint64_t live = 0;              // bytes handed out, as requested by the caller
  uint64_t free_in_segments = 0;  // reserved minus live, so it includes rounding slack
  uint64_t largest_free = 0;      // biggest usable gap, ignoring slack
  uint64_t old_bytes = 0;         // live bytes allocated more than old_seconds ago
  uint32_t segments = 0;
  uint32_t blocks = 0;
  uint32_t old_blocks = 0;
};

void mirror_on_trace(int32_t action, int32_t device, uint64_t addr, uint64_t size);

// old_seconds decides what counts as long lived. Memory still held from early in a run is what
// a leak looks like, and it costs nothing to measure because the blocks are already walked.
MirrorStats mirror_stats(int32_t device, double old_seconds = 60.0);
std::vector<int32_t> mirror_devices();
void mirror_reset();

}  // namespace vramxray
