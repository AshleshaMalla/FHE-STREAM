#include "common/BenchmarkUtils.h"

#include <cerrno>
#include <cstdlib>
#include <cstdint>
#include <vector>

void RaiderSTREAM_Arguments(benchmark::internal::Benchmark* b) {
  const std::vector<std::int64_t> RingDims = {16384, 32768, 65536, 131072};
  const std::vector<std::int64_t> Depths = {1, 5, 20, 40};
  const std::vector<std::int64_t> ShuffleModes = {0, 1, 2};  // None=0, Poly=1, Coeff=2

  for (const auto RingDim : RingDims) {
    for (const auto Depth : Depths) {
      for (const auto Mode : ShuffleModes) {
        b->Args({RingDim, Depth, Mode});
      }
    }
  }
}

void RaiderSTREAM_Arguments_Sequential(benchmark::internal::Benchmark* b) {
  const std::vector<std::int64_t> RingDims = {16384, 32768, 65536, 131072};
  const std::vector<std::int64_t> Depths = {1, 5, 20, 40};
  const std::int64_t Mode = 0;  // ShuffleMode::None only

  for (const auto RingDim : RingDims) {
    for (const auto Depth : Depths) {
      b->Args({RingDim, Depth, Mode});
    }
  }
}

void RaiderSTREAM_Arguments_Irregular(benchmark::internal::Benchmark* b) {
  const std::vector<std::int64_t> RingDims = {16384, 32768, 65536, 131072};
  const std::vector<std::int64_t> Depths = {1, 5, 20, 40};
  const std::vector<std::int64_t> ShuffleModes = {1, 2};  // Poly=1, Coeff=2 only

  for (const auto RingDim : RingDims) {
    for (const auto Depth : Depths) {
      for (const auto Mode : ShuffleModes) {
        b->Args({RingDim, Depth, Mode});
      }
    }
  }
}
