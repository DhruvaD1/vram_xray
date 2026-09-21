#pragma once

#include <cupti_callbacks.h>

#include <cstdint>
#include <map>
#include <string>
#include <utility>

namespace vramxray {
int cupti_start();
CUpti_SubscriberHandle cupti_subscriber();
int32_t caller_lib();  // library id of whoever called into the driver, from the current stack
void cupti_stop();
bool cupti_active();
std::map<std::string, uint64_t> libs();           // live device bytes per calling library
std::map<std::string, uint64_t> kernel_images();  // module/library image bytes per library
// nanoseconds and call count spent inside driver allocation calls, per library
std::map<std::string, std::pair<uint64_t, uint64_t>> stalls();
size_t live_ptrs();
size_t live_handles();
}  // namespace vramxray
