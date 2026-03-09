#pragma once

#include <benchmark/benchmark.h>

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

  void SetUp(const benchmark::State& state) override;
  void TearDown(const benchmark::State& state) override;
};
