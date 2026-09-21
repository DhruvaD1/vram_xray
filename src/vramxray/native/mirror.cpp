#include "mirror.h"

#include <map>
#include <mutex>

#include "common.h"

namespace vramxray {

namespace {

// gaps at or below this are rounding slack rather than usable free space, see mirror.h
constexpr uint64_t kSlackBytes = 1u << 20;

struct Seg {
  uint64_t base = 0;
  uint64_t size = 0;
  std::map<uint64_t, uint64_t> live;  // block address -> size the caller asked for
  uint64_t live_bytes = 0;
};

// device -> segment base -> segment. Ordered so we can find the segment holding an address.
std::map<int32_t, std::map<uint64_t, Seg>> g_devices;

Seg* segment_holding(std::map<uint64_t, Seg>& segs, uint64_t addr) {
  auto it = segs.upper_bound(addr);
  if (it == segs.begin()) return nullptr;
  --it;
  Seg& s = it->second;
  return addr < s.base + s.size ? &s : nullptr;
}

// the largest run of free space in one segment, skipping gaps that are only rounding slack
uint64_t largest_gap(const Seg& s) {
  uint64_t best = 0;
  uint64_t cursor = s.base;
  for (const auto& [addr, size] : s.live) {
    if (addr > cursor) best = std::max(best, addr - cursor);
    cursor = addr + size;
  }
  if (s.base + s.size > cursor) best = std::max(best, s.base + s.size - cursor);
  return best > kSlackBytes ? best : 0;
}

}  // namespace

void mirror_on_trace(int32_t action, int32_t device, uint64_t addr, uint64_t size) {
  std::lock_guard<std::mutex> g(g_mu);
  auto& segs = g_devices[device];
  switch (action) {
    case TA_SEGMENT_ALLOC:
    case TA_SEGMENT_MAP: {
      // a map into a range we already track is part of that segment, not a new one
      if (segment_holding(segs, addr)) return;
      Seg& s = segs[addr];
      s.base = addr;
      s.size = size;
      return;
    }
    case TA_SEGMENT_FREE:
    case TA_SEGMENT_UNMAP: {
      auto it = segs.find(addr);
      if (it != segs.end()) segs.erase(it);
      return;
    }
    case TA_ALLOC: {
      Seg* s = segment_holding(segs, addr);
      if (!s) return;  // an allocation we never saw the segment for, so skip it
      auto [it, fresh] = s->live.emplace(addr, size);
      if (fresh) {
        s->live_bytes += size;
      } else if (it->second != size) {
        s->live_bytes += size - it->second;
        it->second = size;
      }
      return;
    }
    case TA_FREE_COMPLETED: {
      Seg* s = segment_holding(segs, addr);
      if (!s) return;
      auto it = s->live.find(addr);
      if (it == s->live.end()) return;
      s->live_bytes -= std::min(s->live_bytes, it->second);
      s->live.erase(it);
      return;
    }
    default:
      return;
  }
}

MirrorStats mirror_stats(int32_t device) {
  std::lock_guard<std::mutex> g(g_mu);
  MirrorStats out;
  auto it = g_devices.find(device);
  if (it == g_devices.end()) return out;
  for (const auto& [base, s] : it->second) {
    out.reserved += s.size;
    out.live += s.live_bytes;
    out.segments++;
    out.blocks += (uint32_t)s.live.size();
    out.largest_free = std::max(out.largest_free, largest_gap(s));
  }
  out.free_in_segments = out.reserved - std::min(out.reserved, out.live);
  return out;
}

std::vector<int32_t> mirror_devices() {
  std::lock_guard<std::mutex> g(g_mu);
  std::vector<int32_t> out;
  out.reserve(g_devices.size());
  for (const auto& [dev, _] : g_devices) out.push_back(dev);
  return out;
}

void mirror_reset() {
  std::lock_guard<std::mutex> g(g_mu);
  g_devices.clear();
}

}  // namespace vramxray
