/*
  FHE-RaiderSTREAM Benchmark: Fixture Implementation
  
  Implements the SetUp and TearDown lifecycle methods for the benchmark fixture.
  Handles FHE context initialization, memory allocation with NUMA awareness,
  and resource cleanup between benchmark iterations.
*/

#include "FHERaiderSTREAM.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <tuple>

#ifdef __GLIBC__
#include <malloc.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* Global toggle for optional setup printing (defined in main.cpp) */
extern bool RS_PrintSetupConfig;

namespace {

/* Minimum total data footprint for benchmark working set: 4 GiB */
constexpr std::uint64_t kMinFootprintBytes = 4ULL << 30;  // 4 GiB

/* 
  Rounds up the result of integer division without overflow.
  Computes ceil(num / den) for unsigned 64-bit integers.
*/
inline std::size_t CeilDivU64(std::uint64_t num, std::uint64_t den) {
  return static_cast<std::size_t>((num + den - 1) / den);
}

/* 
  Calculates the deep byte size of a single DCRTPoly in the RNS representation.
  Each polynomial has ringDim coefficients per tower, with numTowers in total,
  and each coefficient is a 64-bit NativeInteger (8 bytes).
*/
inline std::int64_t DeepBytesPerPoly(std::int64_t ringDim, std::int64_t numTowers) {
  return ringDim * numTowers * 8;  // RingDim * NumTowers * 8 bytes per NativeInteger
}

}  // namespace

/*
  Benchmark fixture setup: creates the FHE data structures and initializes arrays A, B, C.
  Called before each benchmark iteration to prepare the working set.
*/
void FHERaiderSTREAM::SetUp(const benchmark::State& state) {
  /* 
    Disable OpenFHE's internal parallelism while preserving benchmark-level OpenMP threads.
    OpenFHE's SetNumThreads(1) calls omp_set_num_threads(1) internally, which would
    override OMP_NUM_THREADS. We save the current thread count and restore it after.
  */
#ifdef _OPENMP
  const int savedThreads = omp_get_max_threads();
#endif
  lbcrypto::OpenFHEParallelControls.SetNumThreads(1);
#ifdef _OPENMP
  omp_set_num_threads(savedThreads);  // Restore thread count for benchmark parallel regions
#endif

  /* Extract benchmark parameters: ring dimension, multiplicative depth, and batch size */
  const std::int64_t ringDim = state.range(0);
  const std::int64_t multDepth = state.range(1);
  const std::int64_t numPolysArg = state.range(2);
  const std::size_t nPolys = numPolysArg > 0 ? static_cast<std::size_t>(numPolysArg) : 1;

  /* 
    Construct cyclotomic parameters for the requested configuration.
    Cyclotomic order = 2 * ringDim for power-of-two rings; depth = numTowers RNS moduli.
  */
  const std::uint32_t cyclOrder = static_cast<std::uint32_t>(2 * ringDim);
  const std::uint32_t depth = static_cast<std::uint32_t>(multDepth);
  constexpr std::uint32_t bitsPerTower = 60;  // 60-bit moduli for each RNS tower
  params = std::make_shared<lbcrypto::ILDCRTParams<lbcrypto::BigInteger>>(cyclOrder, depth, bitsPerTower);

  /* 
    Precompute per-tower moduli and Barrett reduction constants (mu) for efficient
    modular arithmetic operations in the benchmark loops.
  */
  towerModuli.clear();
  towerMu.clear();
  towerModuli.reserve(static_cast<std::size_t>(multDepth));
  towerMu.reserve(static_cast<std::size_t>(multDepth));
  const auto& nativeParams = params->GetParams();
  for (std::size_t t = 0; t < static_cast<std::size_t>(multDepth) && t < nativeParams.size(); ++t) {
    const auto& q = nativeParams[t]->GetModulus();
    towerModuli.push_back(q);
    towerMu.push_back(q.ComputeMu());  // Pre-compute Barrett mu for ModMulFast
  }

  /* 
    Create a minimal CryptoContext (BFV flavor) for compatibility with OpenFHE ecosystem.
    Focus is on memory bandwidth measurement, not cryptographic operations.
  */
  lbcrypto::CCParams<lbcrypto::CryptoContextBFVRNS> ccParams;
  ccParams.SetSecurityLevel(lbcrypto::HEStd_NotSet);   // No security requirement for benchmarking
  ccParams.SetPlaintextModulus(65537);                  // Small plaintext space
  ccParams.SetMultiplicativeDepth(static_cast<uint32_t>(multDepth));
  ccParams.SetRingDim(static_cast<uint32_t>(ringDim));  // Match requested ring dimension
  cc = lbcrypto::GenCryptoContext(ccParams);

  /* Compute per-array footprint based on the user batch size. */
  const std::uint64_t bytesPerPoly = static_cast<std::uint64_t>(DeepBytesPerPoly(ringDim, multDepth));

  if (RS_PrintSetupConfig) {
    static std::set<std::tuple<std::int64_t, std::int64_t, std::size_t>> printedConfigs;
    const auto configKey = std::make_tuple(ringDim, multDepth, nPolys);
    const bool shouldPrint = printedConfigs.insert(configKey).second;
    if (shouldPrint) {
      const double totalFootprintGB = (bytesPerPoly * nPolys) / 1e9;
      std::cout << "\n" << std::string(70, '=') << std::endl;
      std::cout << "  FHE-RaiderSTREAM Setup Configuration" << std::endl;
      std::cout << std::string(70, '=') << std::endl;
      std::cout << "  Ring Dimension:        " << ringDim << std::endl;
      std::cout << "  Multiplicative Depth:  " << multDepth << std::endl;
      std::cout << "  Number of Polys:       " << nPolys << std::endl;
      std::cout << "  Per-Array Footprint:   " << std::fixed << std::setprecision(5) << totalFootprintGB << " GB" << std::endl;
      std::cout << "  Total Footprint (A+B+C): " << std::fixed << std::setprecision(5) << (3.0 * totalFootprintGB) << " GB" << std::endl;
      std::cout << std::string(70, '=') << "\n" << std::endl;
    }
  }

  /* Allocate polynomial vectors */
  A.resize(nPolys);
  B.resize(nPolys);
  C.resize(nPolys);

  /* 
    Initialize polynomials in parallel with NUMA first-touch to bind memory
    to the thread's local NUMA node (improves cache locality).
  */
  const auto q0 = params->GetParams().at(0)->GetModulus();

#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < nPolys; ++i) {
    lbcrypto::DCRTPoly::DugType dug(q0);  // Seeded random generator for reproducibility

    A[i] = lbcrypto::DCRTPoly(dug, params, ::Format::EVALUATION);  // Random coefficients
    B[i] = lbcrypto::DCRTPoly(dug, params, ::Format::EVALUATION);
    C[i] = lbcrypto::DCRTPoly(dug, params, ::Format::EVALUATION);
  }

  /*
    Initialize the index vector for irregular (gather/scatter) access patterns.
    Sequential fill first to establish a baseline; enable std::shuffle for
    randomized access once the control case has been verified.
  */
  IDX.resize(nPolys);
  std::iota(IDX.begin(), IDX.end(), std::size_t{0});  // Fill with 0, 1, 2, ..., nPolys-1
  std::mt19937 rng(42);  // Fixed seed for reproducibility
  std::shuffle(IDX.begin(), IDX.end(), rng);  // Randomized access pattern

  IDX_WRITE.resize(nPolys);
  std::iota(IDX_WRITE.begin(), IDX_WRITE.end(), std::size_t{0});  // Fill with 0, 1, 2, ..., nPolys-1
  std::mt19937 rng_write(99);  // Distinct seed for write-side shuffling
  std::shuffle(IDX_WRITE.begin(), IDX_WRITE.end(), rng_write);

  /*
    Initialize coefficient index vector for inner-loop gather.
    Sequential fill first to establish a baseline; enable std::shuffle for
    randomized access once the control case has been verified.
  */
  COEFF_IDX.resize(static_cast<std::size_t>(ringDim));
  std::iota(COEFF_IDX.begin(), COEFF_IDX.end(), std::size_t{0});  // 0, 1, 2, ..., ringDim-1
  std::mt19937 coeff_rng(43);  // Fixed seed for reproducibility
  std::shuffle(COEFF_IDX.begin(), COEFF_IDX.end(), coeff_rng);  // Randomized access pattern

  COEFF_IDX_WRITE.resize(static_cast<std::size_t>(ringDim));
  std::iota(COEFF_IDX_WRITE.begin(), COEFF_IDX_WRITE.end(), std::size_t{0});  // 0, 1, 2, ..., ringDim-1
  std::mt19937 coeff_rng_write(99);  // Distinct seed for write-side shuffling
  std::shuffle(COEFF_IDX_WRITE.begin(), COEFF_IDX_WRITE.end(), coeff_rng_write);
}

/*
  Benchmark fixture teardown: releases all dynamically allocated resources.
  Called after each benchmark iteration to clean up the working set.
*/
void FHERaiderSTREAM::TearDown(const benchmark::State&) {
  /* Release OpenFHE objects */
  cc.reset();
  params.reset();

  /* Clear precomputed modular arithmetic constants */
  towerModuli.clear();
  towerMu.clear();

  /* Force immediate deallocation of polynomial vectors by swapping with empty vectors */
  std::vector<lbcrypto::DCRTPoly>().swap(A);
  std::vector<lbcrypto::DCRTPoly>().swap(B);
  std::vector<lbcrypto::DCRTPoly>().swap(C);
  std::vector<std::size_t>().swap(IDX);
  std::vector<std::size_t>().swap(IDX_WRITE);
  std::vector<std::size_t>().swap(COEFF_IDX);
  std::vector<std::size_t>().swap(COEFF_IDX_WRITE);

#ifdef __GLIBC__
  /* Return freed memory pages to the OS to avoid cross-benchmark contamination */
  malloc_trim(0);
#endif
}
