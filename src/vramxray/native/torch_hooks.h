#pragma once

namespace vramxray {
void install();       // attach to torch's caching allocator; call after CUDA init
long oom_calls();
}  // namespace vramxray
