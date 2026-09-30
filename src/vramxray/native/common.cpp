#include "common.h"

#include <chrono>
#include <unordered_map>

namespace vramxray {

namespace {

constexpr size_t kMaxEvents = 1u << 20;
// Stacks cost far more than events, because each one keeps a whole traceback object alive.
// Sixty thousand is already more than anyone reads, and it bounds the cost at a few megabytes.
constexpr size_t kMaxContexts = 1u << 16;

std::vector<Event> g_events;
std::vector<std::shared_ptr<c10::GatheredContext>> g_contexts;
std::vector<std::string> g_libnames{"torch"};
std::unordered_map<std::string, int32_t> g_libindex{{"torch", 0}};

}  // namespace

std::mutex& libs_mu() {
  static std::mutex m;
  return m;
}

std::mutex& events_mu() {
  static std::mutex m;
  return m;
}

double now_s() noexcept {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

void push(const Event& e) noexcept {
  // Called straight from torch's allocator and from CUPTI callbacks, so nothing may escape.
  // An allocation failure here means we lose an event, which is the right thing to lose.
  try {
    std::lock_guard<std::mutex> g(events_mu());
    if (g_events.size() < kMaxEvents) g_events.push_back(e);
  } catch (...) {
  }
}

std::vector<Event> take_events() {
  std::lock_guard<std::mutex> g(events_mu());
  std::vector<Event> out;
  out.swap(g_events);
  return out;
}

size_t event_count() {
  std::lock_guard<std::mutex> g(events_mu());
  return g_events.size();
}

int32_t lib_id_locked(const std::string& name) {
  auto it = g_libindex.find(name);
  if (it != g_libindex.end()) return it->second;
  const auto id = static_cast<int32_t>(g_libnames.size());
  g_libnames.push_back(name);
  g_libindex.emplace(name, id);
  return id;
}

const std::vector<std::string>& lib_names_locked() { return g_libnames; }

std::vector<std::string> lib_names() {
  std::lock_guard<std::mutex> g(libs_mu());
  return g_libnames;
}

int32_t keep_context_locked(std::shared_ptr<c10::GatheredContext> ctx) {
  if (g_contexts.size() >= kMaxContexts) return -1;
  g_contexts.push_back(std::move(ctx));
  return static_cast<int32_t>(g_contexts.size()) - 1;
}

std::vector<std::shared_ptr<c10::GatheredContext>> take_contexts() {
  std::lock_guard<std::mutex> g(events_mu());
  std::vector<std::shared_ptr<c10::GatheredContext>> out;
  out.swap(g_contexts);
  return out;
}

}  // namespace vramxray
