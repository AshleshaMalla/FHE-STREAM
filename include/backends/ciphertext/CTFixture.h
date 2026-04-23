#pragma once

#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

#include "openfhe.h"

using namespace lbcrypto;

class CTFixture : public benchmark::Fixture {
public:
  CryptoContext<DCRTPoly> cc;
  KeyPair<DCRTPoly> keyPair;
  std::vector<Ciphertext<DCRTPoly>> ct_A;
  std::vector<Ciphertext<DCRTPoly>> ct_B;
  std::vector<Ciphertext<DCRTPoly>> ct_C;
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ct_A_deg2;
  std::size_t ctSerializedBytes = 0;
  std::size_t ctDeg2SerializedBytes = 0;
  std::uint64_t rssDeltaDeg1Bytes = 0;
  std::uint64_t rssDeltaDeg2Bytes = 0;
  std::uint64_t rssDeg1BytesPerCt = 0;
  std::uint64_t rssDeg2BytesPerCt = 0;

  void SetUp(const benchmark::State& state) override;
  void TearDown(const benchmark::State& state) override;
};
