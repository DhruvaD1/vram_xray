#include "streams.h"

#include <cuda.h>
#include <cupti_activity.h>
#include <cupti_driver_cbid.h>
#include <cupti_result.h>

#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "common.h"
#include "cupti_attr.h"

namespace vramxray {

namespace {

// exact layouts from CUPTI's generated meta header, declared here so we do not need it
struct P_cuStreamCreate { CUstream* phStream; unsigned Flags; };
struct P_cuStreamCreateWithPriority { CUstream* phStream; unsigned flags; int priority; };
struct P_cuStreamBeginCapture_v2 { CUstream hStream; CUstreamCaptureMode mode; };
struct P_cuStreamBeginCaptureToGraph { CUstream hStream; CUgraph hGraph; const CUgraphNode* deps; const void* depData; size_t n; CUstreamCaptureMode mode; };
struct P_cuStreamEndCapture { CUstream hStream; CUgraph* phGraph; };
struct P_cuEventRecord { CUevent hEvent; CUstream hStream; };
struct P_cuStreamWaitEvent { CUstream hStream; CUevent hEvent; unsigned Flags; };
struct P_cuLaunchKernel { CUfunction f; unsigned gx, gy, gz, bx, by, bz, shm; CUstream hStream; void** params; void** extra; };
struct P_cuLaunchKernelEx { const CUlaunchConfig* config; CUfunction f; void** params; void** extra; };
struct P_cuGraphLaunch { CUgraphExec hGraph; CUstream hStream; };
struct P_StreamLast3 { uint64_t a, b, c; CUstream hStream; };  // memcpy/memset async: stream is the 4th word

bool g_enabled = false;
bool g_channels = false;
bool g_check_legacy = false;

// capture state. Global mode captures apply to every thread, thread local ones only to theirs.
std::unordered_set<CUstream> g_global_capturing;
thread_local std::unordered_set<CUstream> t_capturing;
std::unordered_map<CUevent, bool> g_capture_events;  // events recorded inside a capture
std::atomic<int> g_capture_depth{0};                 // read on every launch, so keep it lock free

enum Kind : int { K_BLOCKING = 0, K_OUTSIDE_CAPTURE = 1, K_LEGACY = 2 };
const char* kKindNames[] = {"blocking_stream", "launch_outside_capture", "legacy_stream_launch"};

// Resolving the calling library means walking the stack, which costs microseconds. Doing that on
// every launch made a training step five times slower, so we count every occurrence but only
// sample the library from the first few of each kind.
const int kSampleLibs = 16;

struct Counter {
  uint64_t count = 0;
  int sampled = 0;
  std::string api;
  std::set<int32_t> libs;
};

std::map<std::pair<int, uint32_t>, Counter> g_findings;  // (kind, callback id) -> counter
std::map<uint32_t, std::set<uint32_t>> g_channel_streams;  // channelID -> streamIds seen on it

void note(int kind, CUpti_CallbackId cbid, const char* api) {
  bool sample;
  {
    std::lock_guard<std::mutex> g(g_mu);
    auto& c = g_findings[{kind, (uint32_t)cbid}];
    if (c.api.empty()) c.api = api ? api : "?";
    c.count++;
    sample = c.sampled < kSampleLibs;
    if (sample) c.sampled++;
  }
  if (!sample) return;
  int32_t lib = caller_lib();
  std::lock_guard<std::mutex> g(g_mu);
  g_findings[{kind, (uint32_t)cbid}].libs.insert(lib);
}

bool any_capture() { return g_capture_depth.load(std::memory_order_relaxed) > 0; }

bool is_capturing(CUstream s) {
  if (t_capturing.count(s)) return true;
  std::lock_guard<std::mutex> g(g_mu);
  return g_global_capturing.count(s) > 0;
}

void begin_capture(CUstream s, CUstreamCaptureMode mode) {
  if (mode == CU_STREAM_CAPTURE_MODE_GLOBAL) {
    std::lock_guard<std::mutex> g(g_mu);
    g_global_capturing.insert(s);
  } else {
    t_capturing.insert(s);
  }
  g_capture_depth++;
}

void end_capture(CUstream s) {
  bool had = t_capturing.erase(s) > 0;
  {
    std::lock_guard<std::mutex> g(g_mu);
    had = g_global_capturing.erase(s) > 0 || had;
    if (g_global_capturing.empty()) g_capture_events.clear();
  }
  if (had) g_capture_depth--;
}

// a stream that waits on an event recorded during a capture joins that capture
void join_capture(CUstream s, CUevent e) {
  bool joined;
  {
    std::lock_guard<std::mutex> g(g_mu);
    auto it = g_capture_events.find(e);
    if (it == g_capture_events.end()) return;
    joined = true;
    if (!g_global_capturing.empty()) g_global_capturing.insert(s);
  }
  if (joined) t_capturing.insert(s);
}

CUstream launch_stream(CUpti_CallbackId id, const void* P) {
  switch (id) {
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz:
      return ((const P_cuLaunchKernel*)P)->hStream;
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz:
      return ((const P_cuLaunchKernelEx*)P)->config->hStream;
    case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch:
    case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz:
      return ((const P_cuGraphLaunch*)P)->hStream;
    default:
      return ((const P_StreamLast3*)P)->hStream;  // the async memcpy and memset family
  }
}

bool is_launch(CUpti_CallbackId id) {
  switch (id) {
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch:
    case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async_ptsz:
      return true;
    default:
      return false;
  }
}

// ---- hardware channels, through the activity API --------------------------------------------

void CUPTIAPI buf_request(uint8_t** buf, size_t* size, size_t* max_records) {
  *size = 1 << 20;
  *buf = (uint8_t*)malloc(*size);
  *max_records = 0;
}

void CUPTIAPI buf_complete(CUcontext, uint32_t, uint8_t* buf, size_t, size_t valid) {
  CUpti_Activity* rec = nullptr;
  while (cuptiActivityGetNextRecord(buf, valid, &rec) == CUPTI_SUCCESS) {
    if (rec->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL && rec->kind != CUPTI_ACTIVITY_KIND_KERNEL) continue;
    auto* k = (const CUpti_ActivityKernel9*)rec;
    std::lock_guard<std::mutex> g(g_mu);
    g_channel_streams[k->channelID].insert(k->streamId);
  }
  free(buf);
}

const CUpti_CallbackId kStreamCbids[] = {
    CUPTI_DRIVER_TRACE_CBID_cuStreamCreate, CUPTI_DRIVER_TRACE_CBID_cuStreamCreateWithPriority,
    CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2, CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCaptureToGraph, CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCaptureToGraph_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture, CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuEventRecord, CUPTI_DRIVER_TRACE_CBID_cuEventRecord_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuEventRecordWithFlags, CUPTI_DRIVER_TRACE_CBID_cuEventRecordWithFlags_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent, CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel, CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch, CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync, CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2, CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2, CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2, CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async, CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async, CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async_ptsz,
};

}  // namespace

int streams_start(bool channels, bool check_legacy) {
  if (!cupti_active()) return -1;
  g_check_legacy = check_legacy;
  if (g_enabled) return 0;
  for (auto id : kStreamCbids) cuptiEnableCallback(1, cupti_subscriber(), CUPTI_CB_DOMAIN_DRIVER_API, id);
  if (channels) {
    CUptiResult r = cuptiActivityRegisterCallbacks(buf_request, buf_complete);
    if (r == CUPTI_SUCCESS) r = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (r != CUPTI_SUCCESS) return (int)r;
    g_channels = true;
  }
  g_enabled = true;
  return 0;
}

void streams_stop() {
  if (!g_enabled) return;
  if (cupti_active())
    for (auto id : kStreamCbids) cuptiEnableCallback(0, cupti_subscriber(), CUPTI_CB_DOMAIN_DRIVER_API, id);
  if (g_channels) cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  g_enabled = false;
}

bool streams_enabled() { return g_enabled; }

void streams_on_callback(CUpti_CallbackId id, const CUpti_CallbackData* d, bool exit, bool failed) {
  if (!g_enabled || !exit || failed) return;
  const void* P = d->functionParams;
  switch (id) {
    case CUPTI_DRIVER_TRACE_CBID_cuStreamCreate: {
      auto* p = (const P_cuStreamCreate*)P;
      if (!(p->Flags & CU_STREAM_NON_BLOCKING)) note(K_BLOCKING, id, d->functionName);
      return;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuStreamCreateWithPriority: {
      auto* p = (const P_cuStreamCreateWithPriority*)P;
      if (!(p->flags & CU_STREAM_NON_BLOCKING)) note(K_BLOCKING, id, d->functionName);
      return;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2_ptsz: {
      auto* p = (const P_cuStreamBeginCapture_v2*)P;
      begin_capture(p->hStream, p->mode);
      return;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCaptureToGraph:
    case CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCaptureToGraph_ptsz: {
      auto* p = (const P_cuStreamBeginCaptureToGraph*)P;
      begin_capture(p->hStream, p->mode);
      return;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture:
    case CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture_ptsz:
      end_capture(((const P_cuStreamEndCapture*)P)->hStream);
      return;
    case CUPTI_DRIVER_TRACE_CBID_cuEventRecord:
    case CUPTI_DRIVER_TRACE_CBID_cuEventRecord_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuEventRecordWithFlags:
    case CUPTI_DRIVER_TRACE_CBID_cuEventRecordWithFlags_ptsz: {
      auto* p = (const P_cuEventRecord*)P;  // WithFlags has the same first two fields
      if (is_capturing(p->hStream)) {
        std::lock_guard<std::mutex> g(g_mu);
        g_capture_events[p->hEvent] = true;
      }
      return;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent:
    case CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent_ptsz: {
      auto* p = (const P_cuStreamWaitEvent*)P;
      join_capture(p->hStream, p->hEvent);
      return;
    }
    default:
      break;
  }
  // the hot path: this runs on every kernel launch, so it must stay cheap
  if (!is_launch(id)) return;
  const bool capturing = any_capture();
  if (!capturing && !g_check_legacy) return;
  CUstream s = launch_stream(id, P);
  if (g_check_legacy && (s == nullptr || s == CU_STREAM_LEGACY)) note(K_LEGACY, id, d->functionName);
  if (capturing && !is_capturing(s)) note(K_OUTSIDE_CAPTURE, id, d->functionName);
}

std::vector<Finding> streams_findings() {
  if (g_channels) cuptiActivityFlushAll(0);
  std::vector<Finding> out;
  std::lock_guard<std::mutex> g(g_mu);
  auto names = lib_names_locked();
  for (auto& [key, c] : g_findings) {
    std::string libs;
    for (int32_t id : c.libs) {
      if (!libs.empty()) libs += ", ";
      libs += (size_t)id < names.size() ? names[id] : "?";
    }
    if (c.count > (uint64_t)kSampleLibs && !libs.empty()) libs += " (sampled)";
    out.push_back(Finding{kKindNames[key.first], libs, c.api, c.count});
  }
  for (auto& [chan, streams] : g_channel_streams) {
    if (streams.size() < 2) continue;
    std::string ids;
    for (auto sid : streams) ids += (ids.empty() ? "" : ",") + std::to_string(sid);
    out.push_back(Finding{"shared_channel", "", "channel " + std::to_string(chan) + ": streams " + ids, streams.size()});
  }
  return out;
}

}  // namespace vramxray
