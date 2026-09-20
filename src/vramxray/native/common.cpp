#include "common.h"

#include <chrono>
#include <unordered_map>

namespace vramxray {

std::mutex g_mu;

namespace {
std::vector<Event> g_events;
const size_t g_cap = 1u << 20;
std::vector<std::string> g_libnames{"torch"};
std::unordered_map<std::string, int32_t> g_libindex{{"torch", 0}};
}  // namespace

double now_s() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

void push(Event e) {
  std::lock_guard<std::mutex> g(g_mu);
  if (g_events.size() < g_cap) g_events.push_back(e);
}

std::vector<Event> take_events() {
  std::lock_guard<std::mutex> g(g_mu);
  std::vector<Event> out;
  out.swap(g_events);
  return out;
}

size_t event_count() {
  std::lock_guard<std::mutex> g(g_mu);
  return g_events.size();
}

int32_t lib_id_locked(const std::string& name) {
  auto it = g_libindex.find(name);
  if (it != g_libindex.end()) return it->second;
  int32_t id = (int32_t)g_libnames.size();
  g_libnames.push_back(name);
  g_libindex.emplace(name, id);
  return id;
}

std::vector<std::string> lib_names_locked() { return g_libnames; }

}  // namespace vramxray
