/* Main benchmark header defining the FHERaiderSTREAM fixture class */
#include "FHERaiderSTREAM.h"

/* Standard library includes for algorithm operations, type definitions, and formatting */
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>

/* Conditional includes for memory management and OpenMP parallelization */
#ifdef __GLIBC__
#include <malloc.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* Anonymous namespace for internal helper functions and constants */
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

  /* Extract benchmark parameters: ring dimension and number of RNS towers */
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);

  /* 
    Construct cyclotomic parameters for the requested configuration.
    Cyclotomic order = 2 * ringDim for power-of-two rings; depth = numTowers RNS moduli.
  */
  const std::uint32_t cyclOrder = static_cast<std::uint32_t>(2 * ringDim);
  const std::uint32_t depth = static_cast<std::uint32_t>(numTowers);
  constexpr std::uint32_t bitsPerTower = 60;  // 60-bit moduli for each RNS tower
  params = std::make_shared<lbcrypto::ILDCRTParams<lbcrypto::BigInteger>>(cyclOrder, depth, bitsPerTower);

  /* 
    Precompute per-tower moduli and Barrett reduction constants (mu) for efficient
    modular arithmetic operations in the benchmark loops.
  */
  towerModuli.clear();
  towerMu.clear();
  towerModuli.reserve(static_cast<std::size_t>(numTowers));
  towerMu.reserve(static_cast<std::size_t>(numTowers));
  const auto& nativeParams = params->GetParams();
  for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers) && t < nativeParams.size(); ++t) {
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
  ccParams.SetMultiplicativeDepth(1);                   // Minimal multiplicative depth
  ccParams.SetRingDim(static_cast<uint32_t>(ringDim));  // Match requested ring dimension
  cc = lbcrypto::GenCryptoContext(ccParams);

  /* 
    Calculate the number of polynomials needed to meet minimum 4 GiB working set size
    across all three arrays (A, B, C). Each array has identical per-element size.
  */
  const std::uint64_t bytesPerPoly = static_cast<std::uint64_t>(DeepBytesPerPoly(ringDim, numTowers));
  const std::uint64_t bytesPerIndexAllArrays = bytesPerPoly * 3ULL;  // Three arrays: A, B, C
  const std::size_t nPolys = std::max<std::size_t>(1, CeilDivU64(kMinFootprintBytes, bytesPerIndexAllArrays));

  /* 
    Print setup configuration once per unique parameter set.
    Uses static variables to avoid redundant output across benchmark iterations.
  */
  const double totalFootprintGB = (bytesPerPoly * nPolys) / 1e9;
  static std::int64_t lastRingDim = -1;
  static std::int64_t lastNumTowers = -1;
  if (lastRingDim != ringDim || lastNumTowers != numTowers) {
    lastRingDim = ringDim;
    lastNumTowers = numTowers;
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  FHE-RaiderSTREAM Setup Configuration" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "  Ring Dimension:        " << ringDim << std::endl;
    std::cout << "  Number of Towers:      " << numTowers << std::endl;
    std::cout << "  Number of Polys:       " << nPolys << std::endl;
    std::cout << "  Per-Array Footprint:   " << std::fixed << std::setprecision(5) << totalFootprintGB << " GB" << std::endl;
    std::cout << "  Total Footprint (A+B+C): " << std::fixed << std::setprecision(5) << (3.0 * totalFootprintGB) << " GB" << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;
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

#ifdef __GLIBC__
  /* Return freed memory pages to the OS to avoid cross-benchmark contamination */
  malloc_trim(0);
#endif
}

/* 
  COPY kernel: C[i] = A[i] for all polynomials.
  Simple memory read and write pattern; fundamental bandwidth benchmark.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SEQ_COPY)(benchmark::State& state) {
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();

  for (auto _ : state) {
#pragma omp parallel for schedule(static)  // Distribute polynomials across threads
    for (std::size_t i = 0; i < nPolys; ++i) {
      auto& cTowers = C[i].GetAllElements();
      const auto& aTowers = A[i].GetAllElements();
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& cTower = cTowers[t];
        const auto& aTower = aTowers[t];
        for (std::size_t j = 0; j < static_cast<std::size_t>(ringDim); ++j) {
          cTower[j] = aTower[j];  // Per-coefficient copy
        }
        benchmark::DoNotOptimize(cTower);  // Prevent dead code elimination
      }
    }
    benchmark::ClobberMemory();  // Memory fence between iterations
  }

  /* Report aggregate bytes read/written: 2 arrays (A and C) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 2;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/* 
  SCALE kernel: B[i] = scalar * C[i] for all polynomials.
  Memory read/write with modular multiplication operation; tests compute+memory overlap.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SEQ_SCALE)(benchmark::State& state) {
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = C.size();
  const lbcrypto::NativeInteger scalarNI(static_cast<uint64_t>(scalar));  // Pre-computed scalar

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < nPolys; ++i) {
      auto& bTowers = B[i].GetAllElements();
      const auto& cTowers = C[i].GetAllElements();
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& bTower = bTowers[t];
        const auto& cTower = cTowers[t];
        const auto& mod = towerModuli[t];
        const auto& mu = towerMu[t];
        for (std::size_t j = 0; j < static_cast<std::size_t>(ringDim); ++j) {
          bTower[j] = cTower[j].ModMulFast(scalarNI, mod, mu);  // Fast modular multiplication
        }
        benchmark::DoNotOptimize(bTower);
      }
    }
    benchmark::ClobberMemory();
  }

  /* Report aggregate bytes: 2 arrays (C and B) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 2;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/* 
  ADD kernel: C[i] = A[i] + B[i] for all polynomials.
  Ternary operation (read two, write one); tests memory and addition throughput.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SEQ_ADD)(benchmark::State& state) {
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < nPolys; ++i) {
      auto& cTowers = C[i].GetAllElements();
      const auto& aTowers = A[i].GetAllElements();
      const auto& bTowers = B[i].GetAllElements();
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& cTower = cTowers[t];
        const auto& aTower = aTowers[t];
        const auto& bTower = bTowers[t];
        const auto& mod = towerModuli[t];
        for (std::size_t j = 0; j < static_cast<std::size_t>(ringDim); ++j) {
          cTower[j] = aTower[j].ModAddFast(bTower[j], mod);  // Modular addition
        }
        benchmark::DoNotOptimize(cTower);
      }
    }
    benchmark::ClobberMemory();
  }

  /* Report aggregate bytes: 3 arrays (A, B, C) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 3;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/* 
  TRIAD kernel: A[i] = B[i] + scalar * C[i] for all polynomials.
  Classic stream triad pattern; combines multiplication and addition with three-array access.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SEQ_TRIAD)(benchmark::State& state) {
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();
  const lbcrypto::NativeInteger scalarNI(static_cast<uint64_t>(scalar));  // Pre-computed scalar

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < nPolys; ++i) {
      auto& aTowers = A[i].GetAllElements();
      const auto& bTowers = B[i].GetAllElements();
      const auto& cTowers = C[i].GetAllElements();
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& aTower = aTowers[t];
        const auto& bTower = bTowers[t];
        const auto& cTower = cTowers[t];
        const auto& mod = towerModuli[t];
        const auto& mu = towerMu[t];
        for (std::size_t j = 0; j < static_cast<std::size_t>(ringDim); ++j) {
          const auto scaled = cTower[j].ModMulFast(scalarNI, mod, mu);  // Modular multiply
          aTower[j] = bTower[j].ModAddFast(scaled, mod);               // Modular add
        }
        benchmark::DoNotOptimize(aTower);
      }
    }
    benchmark::ClobberMemory();
  }

  /* Report aggregate bytes: 3 arrays (A, B, C) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 3;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/* Anonymous namespace for benchmark parameter configuration */
namespace {

/* 
  Registers benchmark parameter sets representing three distinct FHE use cases.
  Each set has a unique ring dimension and RNS tower count.
*/
void SchemeArgs(benchmark::internal::Benchmark* b) {
  /* CKKS scheme: Large ring dimension (2^16) with deep RNS tower stack (32) */
  b->Args({1 << 16, 32});

  /* BFV scheme: Medium ring dimension (2^15) with moderate tower count (16) */
  b->Args({1 << 15, 16});

  /* TFHE-style: Small ring dimension (2^11) with minimal towers (2) for fast gate evaluation */
  b->Args({1 << 11, 2});
}

}  // namespace

/* Register each benchmark kernel with all FHE parameter sets */
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SEQ_COPY)->Apply(SchemeArgs);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SEQ_SCALE)->Apply(SchemeArgs);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SEQ_ADD)->Apply(SchemeArgs);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SEQ_TRIAD)->Apply(SchemeArgs);
