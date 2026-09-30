#include "mirror.h"

#include <algorithm>
#include <deque>
#include <map>
#include <mutex>
#include <unordered_map>

#include "common.h"
#include "torch_hooks.h"

namespace {
// guards every container in this file, see the lock note in common.h
std::mutex g_mirror_mu;
}  // namespace

namespace vramxray {

namespace {

// gaps at or below this are rounding slack rather than usable free space, see mirror.h
constexpr uint64_t kSlackBytes = 1u << 20;

// one live allocation: what the caller asked for, when it was handed out, which graph pool it
// belongs to, and the stack torch already gathered for it
struct Live {
  uint64_t size = 0;
  double t = 0;
  uint64_t pool0 = 0;
  uint64_t pool1 = 0;
  std::shared_ptr<c10::GatheredContext> ctx;
};

struct Seg {
  uint64_t base = 0;
  uint64_t size = 0;
  std::map<uint64_t, Live> live;  // block address -> allocation
  uint64_t live_bytes = 0;
};

// device -> segment base -> segment. Ordered so we can find the segment holding an address.
std::map<int32_t, std::map<uint64_t, Seg>> g_devices;

// raw samples, names not yet resolved. Bounded so a long run cannot grow without limit.
struct RawSample {
  double t;
  int32_t device;
  std::vector<std::pair<uint64_t, std::shared_ptr<c10::GatheredContext>>> blocks;
  uint64_t unknown_bytes = 0;
  uint32_t unknown_blocks = 0;
};
std::deque<RawSample> g_site_samples;
// samples whose names have already been resolved. Holding these instead of the raw ones frees
// a reference per live block, and it means asking twice costs nothing the second time.
std::deque<SiteSample> g_resolved;
constexpr size_t kMaxResolved = 512;
// bounded by total blocks kept rather than by sample count, because a leak made of many small
// blocks is invisible if each sample only keeps the biggest few
constexpr size_t kMaxSiteBlocks = 20000;
size_t g_site_blocks = 0;

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
  for (const auto& [addr, blk] : s.live) {
    if (addr > cursor) best = std::max(best, addr - cursor);
    cursor = addr + blk.size;
  }
  if (s.base + s.size > cursor) best = std::max(best, s.base + s.size - cursor);
  return best > kSlackBytes ? best : 0;
}

}  // namespace

// called from torch's allocator by way of on_trace, so it must not throw
void mirror_on_trace(
    int32_t action,
    int32_t device,
    uint64_t addr,
    uint64_t size,
    uint64_t pool0,
    uint64_t pool1,
    std::shared_ptr<c10::GatheredContext> context) {
  std::lock_guard<std::mutex> g(g_mirror_mu);
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
      Live blk{size, now_s(), pool0, pool1, std::move(context)};
      auto [it, fresh] = s->live.emplace(addr, blk);
      if (fresh) {
        s->live_bytes += size;
      } else {
        s->live_bytes += size - it->second.size;
        it->second = std::move(blk);
      }
      return;
    }
    case TA_FREE_COMPLETED: {
      Seg* s = segment_holding(segs, addr);
      if (!s) return;
      auto it = s->live.find(addr);
      if (it == s->live.end()) return;
      s->live_bytes -= std::min(s->live_bytes, it->second.size);
      s->live.erase(it);
      return;
    }
    default:
      return;
  }
}

MirrorStats mirror_stats(int32_t device, double old_seconds) {
  std::lock_guard<std::mutex> g(g_mirror_mu);
  MirrorStats out;
  auto it = g_devices.find(device);
  if (it == g_devices.end()) return out;
  const double cutoff = now_s() - old_seconds;
  for (const auto& [base, s] : it->second) {
    out.reserved += s.size;
    out.live += s.live_bytes;
    out.segments++;
    out.blocks += (uint32_t)s.live.size();
    out.largest_free = std::max(out.largest_free, largest_gap(s));
    for (const auto& [addr, blk] : s.live) {
      if (blk.t < cutoff) {
        out.old_bytes += blk.size;
        out.old_blocks++;
      }
    }
  }
  out.free_in_segments = out.reserved - std::min(out.reserved, out.live);
  return out;
}

std::vector<SiteBytes> mirror_top_sites(int32_t device, size_t top, size_t scan) {
  std::vector<std::pair<uint64_t, std::shared_ptr<c10::GatheredContext>>> blocks;
  SiteBytes unknown{"(no Python stack)", 0, 0};
  {
    std::lock_guard<std::mutex> g(g_mirror_mu);
    auto it = g_devices.find(device);
    if (it == g_devices.end()) return {};
    for (const auto& [base, s] : it->second)
      for (const auto& [addr, blk] : s.live) {
        if (blk.ctx) {
          blocks.emplace_back(blk.size, blk.ctx);
        } else {
          // history is off, or the allocation came from a thread with no Python on it
          unknown.bytes += blk.size;
          unknown.blocks++;
        }
      }
  }
  if (blocks.empty() && !unknown.blocks) return {};
  // biggest first, because those are the only ones worth the cost of naming
  if (blocks.size() > scan) {
    std::partial_sort(
        blocks.begin(), blocks.begin() + scan, blocks.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    blocks.resize(scan);
  }

  std::vector<std::shared_ptr<c10::GatheredContext>> ctxs;
  ctxs.reserve(blocks.size());
  for (const auto& b : blocks) ctxs.push_back(b.second);
  std::vector<std::string> names = symbolize_contexts(ctxs);

  std::unordered_map<std::string, SiteBytes> by_site;
  for (size_t i = 0; i < blocks.size(); i++) {
    if (names[i].empty()) {
      unknown.bytes += blocks[i].first;
      unknown.blocks++;
      continue;
    }
    auto& e = by_site[names[i]];
    e.where = names[i];
    e.bytes += blocks[i].first;
    e.blocks++;
  }
  std::vector<SiteBytes> out;
  out.reserve(by_site.size() + 1);
  for (auto& [k, v] : by_site) out.push_back(v);
  if (unknown.blocks) out.push_back(unknown);
  std::sort(out.begin(), out.end(), [](const SiteBytes& a, const SiteBytes& b) {
    return a.bytes > b.bytes;
  });
  if (out.size() > top) out.resize(top);
  return out;
}

void mirror_sample_sites(int32_t device, size_t top) {
  RawSample sample;
  sample.t = now_s();
  sample.device = device;
  {
    std::lock_guard<std::mutex> g(g_mirror_mu);
    auto it = g_devices.find(device);
    if (it == g_devices.end()) return;
    for (const auto& [base, s] : it->second)
      for (const auto& [addr, blk] : s.live) {
        if (blk.ctx) {
          sample.blocks.emplace_back(blk.size, blk.ctx);
        } else {
          sample.unknown_bytes += blk.size;
          sample.unknown_blocks++;
        }
      }
  }
  if (sample.blocks.size() > top) {
    std::partial_sort(
        sample.blocks.begin(), sample.blocks.begin() + top, sample.blocks.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    sample.blocks.resize(top);
  }
  std::lock_guard<std::mutex> g(g_mirror_mu);
  g_site_blocks += sample.blocks.size();
  g_site_samples.push_back(std::move(sample));
  while (g_site_blocks > kMaxSiteBlocks && g_site_samples.size() > 2) {
    g_site_blocks -= g_site_samples.front().blocks.size();
    g_site_samples.pop_front();
  }
}

std::vector<SiteSample> mirror_site_history() {
  // take only what has not been named yet, so repeated calls stay cheap
  std::deque<RawSample> raw;
  {
    std::lock_guard<std::mutex> g(g_mirror_mu);
    raw.swap(g_site_samples);
    g_site_blocks = 0;
  }
  // symbolize every sample in one batch, then split the answers back out per sample
  std::vector<std::shared_ptr<c10::GatheredContext>> all;
  for (const auto& s : raw)
    for (const auto& b : s.blocks) all.push_back(b.second);
  std::vector<std::string> names = symbolize_contexts(all);

  std::vector<SiteSample> out;
  out.reserve(raw.size());
  size_t cursor = 0;
  for (const auto& s : raw) {
    std::unordered_map<std::string, SiteBytes> by_site;
    SiteBytes unknown{"(no Python stack)", s.unknown_bytes, s.unknown_blocks};
    for (const auto& b : s.blocks) {
      const std::string& where = names[cursor++];
      if (where.empty()) {
        unknown.bytes += b.first;
        unknown.blocks++;
        continue;
      }
      auto& e = by_site[where];
      e.where = where;
      e.bytes += b.first;
      e.blocks++;
    }
    SiteSample sample{s.t, s.device, {}};
    for (auto& [k, v] : by_site) sample.sites.push_back(v);
    if (unknown.blocks) sample.sites.push_back(unknown);
    std::sort(sample.sites.begin(), sample.sites.end(), [](const SiteBytes& a, const SiteBytes& b) {
      return a.bytes > b.bytes;
    });
    out.push_back(std::move(sample));
  }

  std::lock_guard<std::mutex> g(g_mirror_mu);
  for (auto& s : out) g_resolved.push_back(s);
  while (g_resolved.size() > kMaxResolved) g_resolved.pop_front();
  return {g_resolved.begin(), g_resolved.end()};
}

std::vector<PoolBytes> mirror_pools(int32_t device) {
  std::lock_guard<std::mutex> g(g_mirror_mu);
  std::map<std::pair<uint64_t, uint64_t>, PoolBytes> pools;
  auto it = g_devices.find(device);
  if (it == g_devices.end()) return {};
  for (const auto& [base, s] : it->second) {
    for (const auto& [addr, blk] : s.live) {
      if (!blk.pool0 && !blk.pool1) continue;  // the ordinary pool, not a graph's own
      auto& e = pools[{blk.pool0, blk.pool1}];
      e.id0 = blk.pool0;
      e.id1 = blk.pool1;
      e.bytes += blk.size;
      e.blocks++;
    }
  }
  std::vector<PoolBytes> out;
  out.reserve(pools.size());
  for (auto& [k, v] : pools) out.push_back(v);
  return out;
}

std::vector<int32_t> mirror_devices() {
  std::lock_guard<std::mutex> g(g_mirror_mu);
  std::vector<int32_t> out;
  out.reserve(g_devices.size());
  for (const auto& [dev, _] : g_devices) out.push_back(dev);
  return out;
}

void mirror_reset() {
  std::lock_guard<std::mutex> g(g_mirror_mu);
  g_devices.clear();
  g_site_samples.clear();
  g_resolved.clear();
  g_site_blocks = 0;
}

}  // namespace vramxray
