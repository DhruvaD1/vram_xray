#pragma once

#include <memory>
#include <string>
#include <vector>

namespace c10 {
struct GatheredContext;
}

namespace vramxray {
void install();       // attach to torch's caching allocator, call after CUDA init
long oom_calls();

// "file:line func from file:line func" for each context, user frames first, "" when unknown
std::vector<std::string> symbolize_contexts(
    const std::vector<std::shared_ptr<c10::GatheredContext>>& ctxs);
}  // namespace vramxray
