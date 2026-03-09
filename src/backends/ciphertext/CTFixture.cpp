/*
  FHE-RaiderSTREAM Benchmark: Ciphertext Fixture Implementation

  Implements the SetUp and TearDown lifecycle methods for the ciphertext-level
  benchmark fixture.  SetUp creates a BFVrns CryptoContext, generates keys,
  encrypts plaintext vectors of 1s and 2s into ct_A / ct_B, and clones ct_A
  into ct_C for pre-allocation.  TearDown releases all resources.
*/

#include "backends/ciphertext/CTFixture.h"

#include <cstdint>
#include <iostream>
#include <set>
#include <tuple>
#include <vector>

#ifdef __GLIBC__
#include <malloc.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* Global toggle for optional setup printing (defined in main.cpp) */
extern bool RS_PrintSetupConfig;

/*
  Benchmark fixture setup: creates a BFVrns CryptoContext, generates key
  material, and encrypts the working-set ciphertexts A, B, C.
*/
void CTFixture::SetUp(const benchmark::State& state) {
  /*
    Disable OpenFHE's internal parallelism while preserving benchmark-level
    OpenMP threads.  OpenFHE's SetNumThreads(1) calls omp_set_num_threads(1)
    internally which would override OMP_NUM_THREADS.
  */
#ifdef _OPENMP
  const int savedThreads = omp_get_max_threads();
#endif
  lbcrypto::OpenFHEParallelControls.SetNumThreads(1);
#ifdef _OPENMP
  omp_set_num_threads(savedThreads);
#endif

  /* Extract benchmark parameters */
  const std::int64_t RingDim   = state.range(0);
  const std::int64_t Depth     = state.range(1);
  const std::int64_t BatchSize = state.range(2);
  const std::size_t  batchSz   = BatchSize > 0 ? static_cast<std::size_t>(BatchSize) : 1;

  /* ---- CryptoContext creation ---- */
  CCParams<CryptoContextBFVRNS> parameters;
  parameters.SetSecurityLevel(HEStd_NotSet);  // Benchmarking only — no security requirement
  parameters.SetMultiplicativeDepth(static_cast<uint32_t>(Depth));
  parameters.SetRingDim(static_cast<uint32_t>(RingDim));
  /*
    Packed encoding requires (ptMod - 1) % (2 * RingDim) == 0.
    65537 works for RingDim <= 32768; 786433 (= 3*2^18 + 1, prime) works for all.
  */
  const PlaintextModulus ptMod = (RingDim <= 32768) ? 65537 : 786433;
  parameters.SetPlaintextModulus(ptMod);
  cc = GenCryptoContext(parameters);

  /* Enable required features */
  cc->Enable(PKE);
  cc->Enable(LEVELEDSHE);

  /* Key generation */
  keyPair = cc->KeyGen();
  cc->EvalMultKeyGen(keyPair.secretKey);

  /* Optional setup print (once per unique config) */
  if (RS_PrintSetupConfig) {
    static std::set<std::tuple<std::int64_t, std::int64_t, std::size_t>> printedConfigs;
    const auto configKey = std::make_tuple(RingDim, Depth, batchSz);
    if (printedConfigs.insert(configKey).second) {
      std::cout << "\n" << std::string(70, '=') << "\n"
                << "  CTFixture Setup Configuration\n"
                << std::string(70, '=') << "\n"
                << "  Ring Dimension:        " << RingDim   << "\n"
                << "  Multiplicative Depth:  " << Depth     << "\n"
                << "  Batch Size:            " << batchSz   << "\n"
                << std::string(70, '=') << "\n\n";
    }
  }

  /* ---- Plaintext vectors ---- */
  const std::size_t slots = static_cast<std::size_t>(RingDim);
  std::vector<int64_t> ones(slots, 1);
  std::vector<int64_t> twos(slots, 2);

  Plaintext pt_ones = cc->MakePackedPlaintext(ones);
  Plaintext pt_twos = cc->MakePackedPlaintext(twos);

  /* ---- Allocate ciphertext vectors ---- */
  ct_A.resize(batchSz);
  ct_B.resize(batchSz);
  ct_C.resize(batchSz);
  ct_A_deg2.resize(batchSz);

  /* Encrypt in parallel; each call is independent */
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < batchSz; ++i) {
    ct_A[i] = cc->Encrypt(keyPair.publicKey, pt_ones);
    ct_B[i] = cc->Encrypt(keyPair.publicKey, pt_twos);
    ct_C[i] = ct_A[i]->Clone();
  }

  /* Precompute degree-2 ciphertexts for relinearization benchmarking */
#pragma omp parallel for
  for (std::size_t i = 0; i < batchSz; ++i) {
    ct_A_deg2[i] = cc->EvalMultNoRelin(ct_A[i], ct_B[i]);
  }
}

/*
  Benchmark fixture teardown: releases all dynamically allocated resources.
*/
void CTFixture::TearDown(const benchmark::State&) {
  /* Release ciphertext vectors */
  std::vector<Ciphertext<DCRTPoly>>().swap(ct_A);
  std::vector<Ciphertext<DCRTPoly>>().swap(ct_B);
  std::vector<Ciphertext<DCRTPoly>>().swap(ct_C);
  std::vector<Ciphertext<DCRTPoly>>().swap(ct_A_deg2);

  /* Release key material and context */
  keyPair = KeyPair<DCRTPoly>();
  cc.reset();

#ifdef __GLIBC__
  malloc_trim(0);
#endif
}
