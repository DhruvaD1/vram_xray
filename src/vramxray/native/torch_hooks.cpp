#include "torch_hooks.h"

#include <ATen/Context.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/profiler/combined_traceback.h>

#include <atomic>
#include <functional>
#include <type_traits>

#include "common.h"
#include "mirror.h"

namespace vramxray {

namespace cca = c10::cuda::CUDACachingAllocator;

// TraceEntry moved namespaces between torch releases, so pull the type out of the tracker signature
template <class F> struct arg0;
template <class R, class A> struct arg0<std::function<R(A)>> {
  using type = std::remove_cv_t<std::remove_reference_t<A>>;
};
using TraceEntry = arg0<cca::AllocatorTraceTracker>::type;

namespace {
std::atomic<long> g_oom_calls{0};
bool g_installed = false;

// our enum has to line up with torch's, or the mirror would read the wrong events
static_assert((int)TraceEntry::Action::ALLOC == TA_ALLOC);
static_assert((int)TraceEntry::Action::FREE_COMPLETED == TA_FREE_COMPLETED);
static_assert((int)TraceEntry::Action::SEGMENT_ALLOC == TA_SEGMENT_ALLOC);
static_assert((int)TraceEntry::Action::SEGMENT_FREE == TA_SEGMENT_FREE);
static_assert((int)TraceEntry::Action::SEGMENT_MAP == TA_SEGMENT_MAP);
static_assert((int)TraceEntry::Action::SEGMENT_UNMAP == TA_SEGMENT_UNMAP);

void on_trace(const TraceEntry& e) {
  mirror_on_trace((int32_t)e.action_, (int32_t)e.device_, (uint64_t)e.addr_, (uint64_t)e.size_);
  int32_t ctx = -1;
  if (e.context_) {
    // torch already gathered the stack (record_memory_history is on). Keep it, symbolize later
    std::lock_guard<std::mutex> g(g_mu);
    ctx = keep_context_locked(e.context_);
  }
  push(Event{now_s(), (int32_t)e.action_, (int32_t)e.device_, (uint64_t)e.addr_,
             (uint64_t)e.size_, (uint64_t)(uintptr_t)e.stream_, 0, ctx});
}

void on_oom(int64_t, size_t, size_t, size_t) { g_oom_calls++; }
}  // namespace

void install() {
  if (g_installed) return;
  // device allocators are created lazily, so attaching before init attaches to nothing
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  cca::attachAllocatorTraceTracker(on_trace);
  cca::attachOutOfMemoryObserver(on_oom);
  g_installed = true;
}

long oom_calls() { return g_oom_calls.load(); }

std::vector<std::string> symbolize_contexts(
    const std::vector<std::shared_ptr<c10::GatheredContext>>& ctxs) {
  std::vector<std::string> out(ctxs.size());
  std::vector<torch::CapturedTraceback*> tbs;
  std::vector<size_t> where;
  for (size_t i = 0; i < ctxs.size(); i++) {
    auto* tb = dynamic_cast<torch::CapturedTraceback*>(ctxs[i].get());
    if (tb) {
      tbs.push_back(tb);
      where.push_back(i);
    }
  }
  if (tbs.empty()) return out;
  torch::SymbolizedTracebacks st = torch::symbolize(tbs);
  for (size_t k = 0; k < tbs.size(); k++) {
    std::string text;
    int kept = 0;
    for (uint64_t fi : st.tracebacks[k]) {
      const auto& f = st.all_frames[fi];
      if (f.filename.find("/torch/") != std::string::npos) continue;  // user frames first
      auto slash = f.filename.rfind('/');
      std::string base = slash == std::string::npos ? f.filename : f.filename.substr(slash + 1);
      if (!text.empty()) text += " from ";
      text += base + ":" + std::to_string(f.lineno) + " " + f.funcname;
      if (++kept == 2) break;
    }
    out[where[k]] = text;
  }
  return out;
}

}  // namespace vramxray
