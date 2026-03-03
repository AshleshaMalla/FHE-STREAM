#include "common/BenchmarkUtils.h"

#include <cerrno>
#include <cstdlib>
#include <cstdint>
#include <vector>

void RaiderSTREAM_Arguments(benchmark::internal::Benchmark* b) {
  const std::vector<std::int64_t> RingDims = {16384, 32768, 65536, 131072};
  const std::vector<std::int64_t> Depths = {1, 5, 20, 40};

  std::int64_t BatchSize = 100;
  if (const char* env = std::getenv("RS_BATCH_SIZE")) {
    errno = 0;
    char* end = nullptr;
    const long long parsed = std::strtoll(env, &end, 10);
    if (errno == 0 && end != env && parsed > 0) {
      BatchSize = static_cast<std::int64_t>(parsed);
    }
  }

  for (const auto RingDim : RingDims) {
    for (const auto Depth : Depths) {
      b->Args({RingDim, Depth, BatchSize});
    }
  }
}
