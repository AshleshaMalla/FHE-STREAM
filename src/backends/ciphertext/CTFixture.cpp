/*
  FHE-RaiderSTREAM Benchmark: Ciphertext Fixture Implementation

  Implements the SetUp and TearDown lifecycle methods for the ciphertext-level
  benchmark fixture.  SetUp creates a BFVrns CryptoContext, generates keys,
  encrypts plaintext vectors of 1s and 2s into ct_A / ct_B, and clones ct_A
  into ct_C for pre-allocation.  TearDown releases all resources.
*/

#include "backends/ciphertext/CTFixture.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <tuple>
#include <vector>

#include <unistd.h>

#ifdef __GLIBC__
#include <malloc.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* Global toggle for optional setup printing (defined in main.cpp) */
extern bool RS_PrintSetupConfig;

namespace {

std::uint64_t CurrentRSSBytes() {
  std::ifstream statm("/proc/self/statm");
  long totalPages = 0;
  long residentPages = 0;
  if (!(statm >> totalPages >> residentPages) || residentPages <= 0) {
    return 0;
  }

  const long pageSize = sysconf(_SC_PAGESIZE);
  if (pageSize <= 0) {
    return 0;
  }

  return static_cast<std::uint64_t>(residentPages) * static_cast<std::uint64_t>(pageSize);
}

std::size_t SerializedCiphertextBytes(const Ciphertext<DCRTPoly>& ciphertext) {
  if (!ciphertext) {
    return 0;
  }

  std::ostringstream os;
  lbcrypto::Serial::Serialize(ciphertext, os, lbcrypto::SerType::BINARY);
  return os.str().size();
}

}  // namespace

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
  const std::int64_t RingDim = state.range(0);
  const std::int64_t Depth = state.range(1);

  /* Read batch size from RS_BATCH_SIZE environment variable (default: 100) */
  std::int64_t batchArg = 100;
  if (const char* env = std::getenv("RS_BATCH_SIZE")) {
    errno = 0;
    char* end = nullptr;
    const long long parsed = std::strtoll(env, &end, 10);
    if (errno == 0 && end != env && parsed > 0) {
      batchArg = static_cast<std::int64_t>(parsed);
    }
  }
  const std::size_t batchSz = static_cast<std::size_t>(batchArg);

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

  /* ---- Allocate ciphertext vector storage (the shared_ptr vectors
     themselves; the ciphertext payloads are populated below) ---- */
  ct_A.resize(batchSz);
  ct_B.resize(batchSz);
  ct_C.resize(batchSz);
  ct_A_deg2.resize(batchSz);

  /* ---- Isolated degree-1 RSS measurement ----
     Encrypt ct_A ALONE (batchSz degree-1 ciphertexts) between two snapshots,
     so the marginal resident cost is measured for exactly one batch of one
     object type. This mirrors the degree-2 measurement below (one batch,
     divided by batchSz) so the two numbers are directly comparable. */
  const std::uint64_t rssBeforeDeg1 = CurrentRSSBytes();
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < batchSz; ++i) {
    ct_A[i] = cc->Encrypt(keyPair.publicKey, pt_ones);
  }
  const std::uint64_t rssAfterDeg1 = CurrentRSSBytes();

  /* Allocate the remaining working buffers the kernels need (ct_B, ct_C).
     These are NOT included in the degree-1 RSS delta measured above. */
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < batchSz; ++i) {
    ct_B[i] = cc->Encrypt(keyPair.publicKey, pt_twos);
    ct_C[i] = ct_A[i]->Clone();
  }

  /* ---- Isolated degree-2 RSS measurement ----
     Marginal delta of the degree-2 batch alone, divided by batchSz. */
  const std::uint64_t rssBeforeDeg2 = CurrentRSSBytes();
#pragma omp parallel for
  for (std::size_t i = 0; i < batchSz; ++i) {
    ct_A_deg2[i] = cc->EvalMultNoRelin(ct_A[i], ct_B[i]);
  }
  const std::uint64_t rssAfterDeg2 = CurrentRSSBytes();

  if (!ct_A.empty()) {
    ctSerializedBytes = SerializedCiphertextBytes(ct_A.front());
    ctDeg2SerializedBytes = SerializedCiphertextBytes(ct_A_deg2.front());
  } else {
    ctSerializedBytes = 0;
    ctDeg2SerializedBytes = 0;
  }

  const std::uint64_t deg1Baseline = (rssAfterDeg1 > rssBeforeDeg1) ? (rssAfterDeg1 - rssBeforeDeg1) : 0;
  const std::uint64_t deg2Baseline = (rssAfterDeg2 > rssBeforeDeg2) ? (rssAfterDeg2 - rssBeforeDeg2) : 0;
  rssDeltaDeg1Bytes = deg1Baseline;
  rssDeltaDeg2Bytes = deg2Baseline;

  if (batchSz > 0) {
    rssDeg1BytesPerCt = rssDeltaDeg1Bytes / static_cast<std::uint64_t>(batchSz);
    rssDeg2BytesPerCt = rssDeltaDeg2Bytes / static_cast<std::uint64_t>(batchSz);
  } else {
    rssDeg1BytesPerCt = 0;
    rssDeg2BytesPerCt = 0;
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
  ctSerializedBytes = 0;
  ctDeg2SerializedBytes = 0;
  rssDeltaDeg1Bytes = 0;
  rssDeltaDeg2Bytes = 0;
  rssDeg1BytesPerCt = 0;
  rssDeg2BytesPerCt = 0;

  /* Release key material and context */
  keyPair = KeyPair<DCRTPoly>();
  cc.reset();

#ifdef __GLIBC__
  malloc_trim(0);
#endif
}
