#include "torch_hooks.h"

#include <ATen/Context.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <atomic>
#include <functional>
#include <type_traits>

#include "common.h"

namespace vramxray {

namespace cca = c10::cuda::CUDACachingAllocator;

// TraceEntry moved namespaces between torch releases; pull the type out of the tracker signature
template <class F> struct arg0;
template <class R, class A> struct arg0<std::function<R(A)>> {
  using type = std::remove_cv_t<std::remove_reference_t<A>>;
};
using TraceEntry = arg0<cca::AllocatorTraceTracker>::type;

namespace {
std::atomic<long> g_oom_calls{0};
bool g_installed = false;

void on_trace(const TraceEntry& e) {
  push(Event{now_s(), (int32_t)e.action_, (int32_t)e.device_, (uint64_t)e.addr_,
             (uint64_t)e.size_, (uint64_t)(uintptr_t)e.stream_, 0});
}

void on_oom(int64_t, size_t, size_t, size_t) { g_oom_calls++; }
}  // namespace

void install() {
  if (g_installed) return;
  // device allocators are created lazily; attaching before init attaches to nothing
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  cca::attachAllocatorTraceTracker(on_trace);
  cca::attachOutOfMemoryObserver(on_oom);
  g_installed = true;
}

long oom_calls() { return g_oom_calls.load(); }

}  // namespace vramxray
