#pragma once

namespace nexus {

constexpr size_t logN = 16;
constexpr size_t N = 1ULL << 16;
constexpr size_t slot_count = N / 2;
constexpr size_t L = 22, boot_level = 14, total_level = L + boot_level;
constexpr size_t logp = 40, logq = 51, log_special_prime = 51;
constexpr double scale = 1ULL << logp;

}  // namespace nexus
