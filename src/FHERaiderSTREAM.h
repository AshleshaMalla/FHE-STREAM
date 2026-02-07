/*
  FHE-RaiderSTREAM Benchmark: Class Definition
  
  Defines the benchmark fixture for memory bandwidth measurement of OpenFHE
  homomorphic polynomial operations. Implements the standard three-array
  structure (A, B, C) with RNS-tower decomposed DCRTPoly elements.
*/

#pragma once

#include <benchmark/benchmark.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "openfhe.h"

/*
  FHERaiderSTREAM Benchmark Fixture
  
  Inherits from Google Benchmark's Fixture class to support parameterized
  benchmarking with automatic SetUp/TearDown lifecycle management.
  
  Template Parameters:
  - Ring Dimension (RingDim): Polynomial ring size (2^11 to 2^16)
  - Number of Towers: RNS modulus count (2 to 32 60-bit primes)
  
  Workload: 4GB minimum working set across three arrays (A, B, C)
*/
class FHERaiderSTREAM : public benchmark::Fixture {
public:
  /* DCRTPoly arrays for the standard memory bandwidth pattern */
  std::vector<lbcrypto::DCRTPoly> A;  // Input array 1
  std::vector<lbcrypto::DCRTPoly> B;  // Input array 2
  std::vector<lbcrypto::DCRTPoly> C;  // Output array

  /* Index vector for irregular (gather/scatter) access patterns */
  std::vector<std::size_t> IDX;

  /* Index vector for coefficient shuffling */
  std::vector<std::size_t> COEFF_IDX;

  /* Precomputed per-tower moduli and Barrett reduction constants */
  std::vector<lbcrypto::NativeInteger> towerModuli;  // Modulus for each RNS tower
  std::vector<lbcrypto::NativeInteger> towerMu;      // Barrett mu for ModMulFast

  /* Scalar multiplier for SCALE and TRIAD kernels */
  int64_t scalar = 3;

  /* OpenFHE cryptographic context and parameter object */
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;
  std::shared_ptr<lbcrypto::ILDCRTParams<lbcrypto::BigInteger>> params;

  /*
    Fixture lifecycle: called before each benchmark iteration.
    Initializes FHE context, allocates working arrays with NUMA first-touch,
    and configures OpenMP thread management.
  */
  void SetUp(const benchmark::State& state) override;

  /*
    Fixture lifecycle: called after each benchmark iteration.
    Releases all OpenFHE resources and returns memory to the OS.
  */
  void TearDown(const benchmark::State& state) override;
};
