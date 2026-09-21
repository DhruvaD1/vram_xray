// Stream hygiene checks on the same CUPTI subscriber: blocking streams, work launched outside a
// graph capture, launches on the legacy null stream, and streams that share a hardware channel.
#pragma once

#include <cupti_callbacks.h>

#include <cstdint>
#include <string>
#include <vector>

namespace vramxray {

struct Finding {
  std::string kind;    // blocking_stream, launch_outside_capture, legacy_stream_launch, shared_channel
  std::string lib;     // who did it, when that makes sense
  std::string detail;  // api name, stream ids, capture mode
  uint64_t count;
};

// needs cupti_start() first, returns a CUPTI result. check_legacy is off by default because
// torch launches everything on the legacy stream, so it fires on every kernel.
int streams_start(bool channels, bool check_legacy);
void streams_stop();
bool streams_enabled();
void streams_on_callback(CUpti_CallbackId id, const CUpti_CallbackData* d, bool exit, bool failed);
std::vector<Finding> streams_findings();

}  // namespace vramxray
