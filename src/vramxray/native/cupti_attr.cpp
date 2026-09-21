#include "cupti_attr.h"

#include <cuda.h>
#include <cupti_activity.h>
#include <cupti_callbacks.h>
#include <cupti_driver_cbid.h>
#include <cupti_result.h>
#include <dlfcn.h>
#include <execinfo.h>

#include <algorithm>
#include <cstring>
#include <unordered_map>

#include "common.h"
#include "nvml_shim.h"
#include "streams.h"

namespace vramxray {

namespace {

// hand-declared so we do not depend on CUPTI's generated meta headers, which need matching CUDA
struct P_cuMemAlloc_v2 { CUdeviceptr* dptr; size_t bytesize; };
struct P_cuMemAllocPitch_v2 { CUdeviceptr* dptr; size_t* pPitch; size_t WidthInBytes; size_t Height; unsigned ElementSizeBytes; };
struct P_cuMemAllocManaged { CUdeviceptr* dptr; size_t bytesize; unsigned flags; };
struct P_cuMemAllocAsync { CUdeviceptr* dptr; size_t bytesize; CUstream hStream; };
struct P_cuMemAllocFromPoolAsync { CUdeviceptr* dptr; size_t bytesize; CUmemoryPool pool; CUstream hStream; };
struct P_cuMemFree_v2 { CUdeviceptr dptr; };
struct P_cuMemFreeAsync { CUdeviceptr dptr; CUstream hStream; };
struct P_cuMemCreate { CUmemGenericAllocationHandle* handle; size_t size; const CUmemAllocationProp* prop; unsigned long long flags; };
struct P_cuMemRelease { CUmemGenericAllocationHandle handle; };

struct Live { uint64_t size; int32_t lib; };

CUpti_SubscriberHandle g_sub = nullptr;
bool g_subscribed = false;
std::string g_self;  // basename of this .so, skipped when walking the stack

// all guarded by g_mu from common.h
std::unordered_map<uint64_t, Live> g_live_ptrs;     // device pointer -> allocation
std::unordered_map<uint64_t, Live> g_live_handles;  // cuMemCreate handle -> allocation
std::vector<uint64_t> g_bytes_by_lib{0};
std::vector<uint64_t> g_images_by_lib{0};

const char* base(const char* p) {
  const char* s = strrchr(p, '/');
  return s ? s + 1 : p;
}

bool is_plumbing(const char* b) {
  return strstr(b, "libcuda.") || strstr(b, "libcupti") || strstr(b, "libcudart") ||
         (!g_self.empty() && strstr(b, g_self.c_str()));
}

}  // namespace

// the library that called into the driver: first frame above libcuda/libcudart/ourselves
int32_t caller_lib() {
  void* frames[64];
  int n = backtrace(frames, 64);
  bool past = false;
  for (int i = 0; i < n; i++) {
    Dl_info di;
    if (!dladdr(frames[i], &di) || !di.dli_fname) continue;
    const char* b = base(di.dli_fname);
    if (is_plumbing(b)) {
      past = true;
      continue;
    }
    if (past) {
      std::lock_guard<std::mutex> g(g_mu);
      return lib_id_locked(b);
    }
  }
  std::lock_guard<std::mutex> g(g_mu);
  return lib_id_locked("unknown");
}

namespace {

void grow(std::vector<uint64_t>& v, int32_t id) {
  if ((size_t)id >= v.size()) v.resize(id + 1, 0);
}

void account_alloc(uint64_t key, uint64_t size, int32_t lib, bool handle) {
  std::lock_guard<std::mutex> g(g_mu);
  grow(g_bytes_by_lib, lib);
  g_bytes_by_lib[lib] += size;
  (handle ? g_live_handles : g_live_ptrs)[key] = Live{size, lib};
}

void account_free(uint64_t key, bool handle) {
  std::lock_guard<std::mutex> g(g_mu);
  auto& m = handle ? g_live_handles : g_live_ptrs;
  auto it = m.find(key);
  if (it == m.end()) return;
  auto& total = g_bytes_by_lib[it->second.lib];
  total -= std::min(total, it->second.size);
  m.erase(it);
}

bool is_module_load(CUpti_CallbackId id) {
  switch (id) {
    case CUPTI_DRIVER_TRACE_CBID_cuModuleLoad:
    case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData:
    case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx:
    case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary:
    case CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData:
    case CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile:
      return true;
    default:
      return false;
  }
}

// per-thread scratch so the ENTER site can hand values to the EXIT site
thread_local long long t_used_before = -1;
thread_local unsigned t_dev = 0;

// kernel images never go through a malloc, so measure them as an NVML delta around the load
void on_module_load(const CUpti_CallbackData* d, bool exit, bool failed) {
  uint32_t dev = 0;
  cuptiGetDeviceId(d->context, &dev);
  if (!exit) {
    t_dev = dev;
    t_used_before = nvml().used(dev);
    return;
  }
  if (failed || t_used_before < 0) return;
  long long after = nvml().used(t_dev);
  if (after <= t_used_before) return;
  uint64_t grew = (uint64_t)(after - t_used_before);
  int32_t lib = caller_lib();
  {
    std::lock_guard<std::mutex> g(g_mu);
    grow(g_images_by_lib, lib);
    g_images_by_lib[lib] += grew;
  }
  push(Event{now_s(), DRV_MODULE, (int32_t)t_dev, 0, grew, 0, lib});
}

void CUPTIAPI on_cupti(void*, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void* info) {
  if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
  auto* d = (const CUpti_CallbackData*)info;
  const bool exit = d->callbackSite == CUPTI_API_EXIT;
  const bool failed = exit && d->functionReturnValue &&
                      *(const CUresult*)d->functionReturnValue != CUDA_SUCCESS;

  if (streams_enabled()) streams_on_callback(id, d, exit, failed);
  if (is_module_load(id)) {
    on_module_load(d, exit, failed);
    return;
  }
  if (!exit || failed) return;

  const void* P = d->functionParams;
  uint64_t key = 0, size = 0;
  bool handle = false;
  int32_t kind = DRV_ALLOC;
  switch (id) {
    case CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2: {
      auto* p = (const P_cuMemAlloc_v2*)P; key = *p->dptr; size = p->bytesize; break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch_v2: {
      auto* p = (const P_cuMemAllocPitch_v2*)P; key = *p->dptr; size = (uint64_t)(*p->pPitch) * p->Height; break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocManaged: {
      auto* p = (const P_cuMemAllocManaged*)P; key = *p->dptr; size = p->bytesize; break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync:
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync_ptsz: {
      auto* p = (const P_cuMemAllocAsync*)P; key = *p->dptr; size = p->bytesize; break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync:
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync_ptsz: {
      auto* p = (const P_cuMemAllocFromPoolAsync*)P; key = *p->dptr; size = p->bytesize; break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemCreate: {
      auto* p = (const P_cuMemCreate*)P; key = (uint64_t)*p->handle; size = p->size; handle = true; kind = DRV_CREATE; break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2: {
      key = ((const P_cuMemFree_v2*)P)->dptr;
      account_free(key, false);
      push(Event{now_s(), DRV_FREE, 0, key, 0, 0, 0});
      return;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync:
    case CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz: {
      key = ((const P_cuMemFreeAsync*)P)->dptr;
      account_free(key, false);
      push(Event{now_s(), DRV_FREE, 0, key, 0, 0, 0});
      return;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemRelease: {
      key = ((const P_cuMemRelease*)P)->handle;
      account_free(key, true);
      push(Event{now_s(), DRV_RELEASE, 0, key, 0, 0, 0});
      return;
    }
    default:
      return;
  }
  int32_t lib = caller_lib();
  account_alloc(key, size, lib, handle);
  uint32_t dev = 0;
  cuptiGetDeviceId(d->context, &dev);
  push(Event{now_s(), kind, (int32_t)dev, key, size, 0, lib});
}

std::map<std::string, uint64_t> by_lib(const std::vector<uint64_t>& v) {
  std::lock_guard<std::mutex> g(g_mu);
  auto names = lib_names_locked();
  std::map<std::string, uint64_t> out;
  for (size_t i = 0; i < v.size() && i < names.size(); i++)
    if (v[i]) out[names[i]] = v[i];
  return out;
}

const CUpti_CallbackId kWatched[] = {
    CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch_v2,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocManaged,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemCreate,
    CUPTI_DRIVER_TRACE_CBID_cuMemRelease,
    CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2,
    CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync,
    CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuModuleLoad,
    CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData,
    CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx,
    CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary,
    CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData,
    CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile,
};

}  // namespace

int cupti_start() {
  if (g_subscribed) return 0;
  if (g_self.empty()) {
    Dl_info di;
    if (dladdr((void*)&cupti_start, &di) && di.dli_fname) g_self = base(di.dli_fname);
  }
  if (!nvml().ok) nvml().open();
  CUptiResult r = cuptiSubscribe(&g_sub, (CUpti_CallbackFunc)on_cupti, nullptr);
  if (r != CUPTI_SUCCESS) return (int)r;
  for (auto id : kWatched) cuptiEnableCallback(1, g_sub, CUPTI_CB_DOMAIN_DRIVER_API, id);
  g_subscribed = true;
  return 0;
}

void cupti_stop() {
  if (!g_subscribed) return;
  cuptiUnsubscribe(g_sub);
  g_subscribed = false;
}

bool cupti_active() { return g_subscribed; }
CUpti_SubscriberHandle cupti_subscriber() { return g_sub; }
std::map<std::string, uint64_t> libs() { return by_lib(g_bytes_by_lib); }
std::map<std::string, uint64_t> kernel_images() { return by_lib(g_images_by_lib); }
size_t live_ptrs() { std::lock_guard<std::mutex> g(g_mu); return g_live_ptrs.size(); }
size_t live_handles() { std::lock_guard<std::mutex> g(g_mu); return g_live_handles.size(); }

}  // namespace vramxray
