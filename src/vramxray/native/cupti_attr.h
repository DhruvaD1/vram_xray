#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace vramxray {
int cupti_start();   // returns the CUPTI result; 39 means someone else holds the subscription
void cupti_stop();
bool cupti_active();
std::map<std::string, uint64_t> libs();           // live device bytes per calling library
std::map<std::string, uint64_t> kernel_images();  // module/library image bytes per library
size_t live_ptrs();
size_t live_handles();
}  // namespace vramxray
