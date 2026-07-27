/*
  FHE-RaiderSTREAM Benchmark: Ciphertext-level Kernel Implementations

  Measures the throughput of high-level OpenFHE ciphertext operations
  (Clone, EvalAdd, EvalMult, EvalAddInPlace) using the CTFixture working set.
*/

#include "backends/ciphertext/CTFixture.h"
#include "common/BenchmarkUtils.h"
#include "common/MPIUtils.h"
#include "ciphertext-ser.h"
#include "scheme/bfvrns/bfvrns-ser.h"
#include "utils/serial.h"

#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef LIKWID_PERFMON
#include <likwid-marker.h>
#endif

namespace {

inline void ReportCiphertextMetrics(benchmark::State& state,
                                    const CTFixture& fixture,
                                    std::int64_t goodputBytesPerIter,
                                    std::int64_t serializedBytesPerIter) {
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * goodputBytesPerIter);
  state.counters["PayloadBandwidth"] = benchmark::Counter(
    static_cast<double>(state.iterations()) * static_cast<double>(goodputBytesPerIter),
    benchmark::Counter::kIsRate);
  state.counters["SerializedBandwidth"] = benchmark::Counter(
    static_cast<double>(state.iterations()) * static_cast<double>(serializedBytesPerIter),
    benchmark::Counter::kIsRate);
  state.counters["ObjectToPayloadRatio"] = benchmark::Counter(
    (goodputBytesPerIter > 0)
      ? static_cast<double>(serializedBytesPerIter) / static_cast<double>(goodputBytesPerIter)
      : 0.0);
  state.counters["SerializedBytesPerCt"] = benchmark::Counter(static_cast<double>(fixture.ctSerializedBytes));
  state.counters["SerializedDeg2BytesPerCt"] = benchmark::Counter(static_cast<double>(fixture.ctDeg2SerializedBytes));
  state.counters["RSSDeg1BytesPerCt"] = benchmark::Counter(static_cast<double>(fixture.rssDeg1BytesPerCt));
  state.counters["RSSDeg2BytesPerCt"] = benchmark::Counter(static_cast<double>(fixture.rssDeg2BytesPerCt));
}

}  // namespace

/* ------------------------------------------------------------------ */
/*  CT_SEQ_COPY — C[i] = Clone(A[i])                                  */
/* ------------------------------------------------------------------ */
BENCHMARK_DEFINE_F(CTFixture, CT_SEQ_COPY)(benchmark::State& state) {
  const std::size_t batchSz = ct_A.size();

  RS_BARRIER();

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < batchSz; ++i) {
      ct_C[i] = ct_A[i]->Clone();
    }
  }

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::int64_t batch = static_cast<std::int64_t>(batchSz);
  const std::int64_t bytesPerPoly = ringDim * numTowers * 8;
  const std::int64_t bytesPerCt = bytesPerPoly * 2;
  ReportCiphertextMetrics(state,
                          *this,
                          batch * (2 * bytesPerCt),
                          batch * (2 * static_cast<std::int64_t>(ctSerializedBytes)));
  AggregateBandwidth(state);
}

/* ------------------------------------------------------------------ */
/*  CT_SEQ_ADD — C[i] = EvalAdd(A[i], B[i])                           */
/* ------------------------------------------------------------------ */
BENCHMARK_DEFINE_F(CTFixture, CT_SEQ_ADD)(benchmark::State& state) {
  const std::size_t batchSz = ct_A.size();

  RS_BARRIER();

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < batchSz; ++i) {
      ct_C[i] = cc->EvalAdd(ct_A[i], ct_B[i]);
    }
  }

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::int64_t batch = static_cast<std::int64_t>(batchSz);
  const std::int64_t bytesPerPoly = ringDim * numTowers * 8;
  const std::int64_t bytesPerCt = bytesPerPoly * 2;
  ReportCiphertextMetrics(state,
                          *this,
                          batch * (3 * bytesPerCt),
                          batch * (3 * static_cast<std::int64_t>(ctSerializedBytes)));
  AggregateBandwidth(state);
}

/* ------------------------------------------------------------------ */
/*  CT_SEQ_SCALE — B[i] = EvalMult(C[i], scalar)                      */
/* ------------------------------------------------------------------ */
BENCHMARK_DEFINE_F(CTFixture, CT_SEQ_SCALE)(benchmark::State& state) {
  const std::size_t batchSz = ct_C.size();
  const std::size_t slots = static_cast<std::size_t>(state.range(0));
  std::vector<int64_t> scalarVec(slots, 3);
  Plaintext pt_scalar = cc->MakePackedPlaintext(scalarVec);

  RS_BARRIER();

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < batchSz; ++i) {
      ct_B[i] = cc->EvalMult(ct_C[i], pt_scalar);
    }
  }

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::int64_t batch = static_cast<std::int64_t>(batchSz);
  const std::int64_t bytesPerPoly = ringDim * numTowers * 8;
  const std::int64_t bytesPerCt = bytesPerPoly * 2;
  ReportCiphertextMetrics(state,
                          *this,
                          batch * (2 * bytesPerCt),
                          batch * (2 * static_cast<std::int64_t>(ctSerializedBytes)));
  AggregateBandwidth(state);
}

/* ------------------------------------------------------------------ */
/*  CT_SEQ_TRIAD — A[i] = EvalAdd(B[i], EvalMult(C[i], scalar))       */
/* ------------------------------------------------------------------ */
BENCHMARK_DEFINE_F(CTFixture, CT_SEQ_TRIAD)(benchmark::State& state) {
  const std::size_t batchSz = ct_A.size();
  const std::size_t slots = static_cast<std::size_t>(state.range(0));
  std::vector<int64_t> scalarVec(slots, 3);
  Plaintext pt_scalar = cc->MakePackedPlaintext(scalarVec);

  RS_BARRIER();

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < batchSz; ++i) {
      auto scaled = cc->EvalMult(ct_C[i], pt_scalar);
      ct_A[i] = cc->EvalAdd(ct_B[i], scaled);
    }
  }

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::int64_t batch = static_cast<std::int64_t>(batchSz);
  const std::int64_t bytesPerPoly = ringDim * numTowers * 8;
  const std::int64_t bytesPerCt = bytesPerPoly * 2;
  ReportCiphertextMetrics(state,
                          *this,
                          batch * (3 * bytesPerCt),
                          batch * (3 * static_cast<std::int64_t>(ctSerializedBytes)));
  AggregateBandwidth(state);
}

/* ------------------------------------------------------------------ */
/*  CT_SEQ_MULT_NO_RELIN — C[i] = EvalMultNoRelin(A[i], B[i])         */
/* ------------------------------------------------------------------ */
BENCHMARK_DEFINE_F(CTFixture, CT_SEQ_MULT_NO_RELIN)(benchmark::State& state) {
  const std::size_t batchSz = ct_A.size();

  RS_BARRIER();

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < batchSz; ++i) {
      ct_C[i] = cc->EvalMultNoRelin(ct_A[i], ct_B[i]);
    }
  }

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::int64_t batch = static_cast<std::int64_t>(batchSz);
  const std::int64_t bytesPerPoly = ringDim * numTowers * 8;
  const std::int64_t bytesPerCt = bytesPerPoly * 2;
  const std::int64_t bytesPerDeg2Ct = static_cast<std::int64_t>(ctDeg2SerializedBytes);
  ReportCiphertextMetrics(state,
                          *this,
                          batch * (7 * bytesPerPoly),
                          batch * (2 * static_cast<std::int64_t>(ctSerializedBytes) + bytesPerDeg2Ct));
  AggregateBandwidth(state);
}

/* ------------------------------------------------------------------ */
/*  CT_SEQ_RELIN — C[i] = Relinearize(A_deg2[i])                      */
/* ------------------------------------------------------------------ */
BENCHMARK_DEFINE_F(CTFixture, CT_SEQ_RELIN)(benchmark::State& state) {
  const std::size_t batchSz = ct_A_deg2.size();

  RS_BARRIER();

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < batchSz; ++i) {
      ct_C[i] = cc->Relinearize(ct_A_deg2[i]);
    }
  }

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::int64_t batch = static_cast<std::int64_t>(batchSz);
  const std::int64_t bytesPerPoly = ringDim * numTowers * 8;
  const std::int64_t bytesPerCt = bytesPerPoly * 2;
  const std::int64_t bytesPerDeg2Ct = static_cast<std::int64_t>(ctDeg2SerializedBytes);
  ReportCiphertextMetrics(state,
                          *this,
                          batch * (5 * bytesPerPoly),
                          batch * (bytesPerDeg2Ct + static_cast<std::int64_t>(ctSerializedBytes)));
  AggregateBandwidth(state);
}

/* ------------------------------------------------------------------ */
/*  CT_SEQ_ADD_INPLACE — EvalAddInPlace(C[i], A[i])                    */
/* ------------------------------------------------------------------ */
BENCHMARK_DEFINE_F(CTFixture, CT_SEQ_ADD_INPLACE)(benchmark::State& state) {
  const std::size_t batchSz = ct_A.size();

  RS_BARRIER();

#ifdef LIKWID_PERFMON
#pragma omp parallel
  {
    LIKWID_MARKER_THREADINIT;
    LIKWID_MARKER_REGISTER("RS_CT_SEQ_ADD_INPLACE");
  }
#endif

  for (auto _ : state) {
#pragma omp parallel
    {
#ifdef LIKWID_PERFMON
      LIKWID_MARKER_THREADINIT;
      LIKWID_MARKER_REGISTER("RS_CT_SEQ_ADD_INPLACE");
      LIKWID_MARKER_START("RS_CT_SEQ_ADD_INPLACE");
#endif
#pragma omp for schedule(static)
      for (std::size_t i = 0; i < batchSz; ++i) {
        cc->EvalAddInPlace(ct_C[i], ct_A[i]);
      }
#ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP("RS_CT_SEQ_ADD_INPLACE");
#endif
    }
  }

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::int64_t batch = static_cast<std::int64_t>(batchSz);
  const std::int64_t bytesPerPoly = ringDim * numTowers * 8;
  const std::int64_t bytesPerCt = bytesPerPoly * 2;
  ReportCiphertextMetrics(state,
                          *this,
                          batch * (2 * bytesPerCt),
                          batch * (2 * static_cast<std::int64_t>(ctSerializedBytes)));
  AggregateBandwidth(state);
}

/* ------------------------------------------------------------------ */
/*  CT_MPI_SENDRECV — Serialize ct on rank 0, send, recv, deserialize  */
/*  on rank 1.  Measures serialization + MPI transfer overhead.        */
/* ------------------------------------------------------------------ */
BENCHMARK_DEFINE_F(CTFixture, CT_MPI_SENDRECV)(benchmark::State& state) {
#ifdef RAIDERSTREAM_MPI
  if (RS_MPI_Size < 2) {
    state.SkipWithError("Requires at least 2 MPI ranks");
    return;
  }

  /*
   * NOTE: This kernel uses ->Iterations(N) in registration so that
   * Google Benchmark runs exactly the same iteration count on every
   * MPI rank.  Without this, each rank auto-detects its own count,
   * and the paired Send/Recv calls deadlock.
   */
  for (auto _ : state) {
    MPI_Barrier(MPI_COMM_WORLD);          // keep ranks in lock-step
    for (std::size_t i = 0; i < ct_A.size(); ++i) {
      if (RS_MPI_Rank == 0) {
        std::ostringstream os;
        lbcrypto::Serial::Serialize(ct_A[i], os, lbcrypto::SerType::BINARY);
        std::string buf = os.str();
        uint64_t sz = static_cast<uint64_t>(buf.size());
        MPI_Send(&sz,         1, MPI_UINT64_T, 1, 0, MPI_COMM_WORLD);
        MPI_Send(buf.c_str(), static_cast<int>(sz), MPI_CHAR, 1, 1, MPI_COMM_WORLD);
      } else if (RS_MPI_Rank == 1) {
        uint64_t sz = 0;
        MPI_Recv(&sz, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::string buf(sz, '\0');
        MPI_Recv(&buf[0], static_cast<int>(sz), MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::istringstream is(buf);
        lbcrypto::Serial::Deserialize(ct_C[i], is, lbcrypto::SerType::BINARY);
      }
    }
  }

  RS_BARRIER();

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::int64_t bytesPerPoly = ringDim * numTowers * 8;
  const std::int64_t payloadBytes = bytesPerPoly * 2;
  const std::int64_t serializedBytes = static_cast<std::int64_t>(ctSerializedBytes);
  /* Only rank 0 reports bytes so AggregateBandwidth doesn't double-count. */
  if (RS_MPI_Rank == 0)
    state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(ct_A.size()) * payloadBytes);
  else
    state.SetBytesProcessed(0);

  state.counters["SerializedBandwidth"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * static_cast<double>(ct_A.size()) * static_cast<double>(serializedBytes),
      benchmark::Counter::kIsRate);
    state.counters["PayloadBandwidth"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * static_cast<double>(ct_A.size()) * static_cast<double>(payloadBytes),
      benchmark::Counter::kIsRate);
    state.counters["ObjectToPayloadRatio"] = benchmark::Counter(
      (payloadBytes > 0) ? static_cast<double>(serializedBytes) / static_cast<double>(payloadBytes) : 0.0);
    state.counters["SerializedBytesPerCt"] = benchmark::Counter(static_cast<double>(ctSerializedBytes));
    state.counters["SerializedDeg2BytesPerCt"] = benchmark::Counter(static_cast<double>(ctDeg2SerializedBytes));
    state.counters["RSSDeg1BytesPerCt"] = benchmark::Counter(static_cast<double>(rssDeg1BytesPerCt));
    state.counters["RSSDeg2BytesPerCt"] = benchmark::Counter(static_cast<double>(rssDeg2BytesPerCt));

  AggregateBandwidth(state);
#else
  state.SkipWithError("Built without MPI support");
#endif
}

/* ------------------------------------------------------------------ */
/*  Registration                                                       */
/* ------------------------------------------------------------------ */
BENCHMARK_REGISTER_F(CTFixture, CT_SEQ_COPY)
  ->Apply(RaiderSTREAM_Arguments)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(CTFixture, CT_SEQ_ADD)
  ->Apply(RaiderSTREAM_Arguments)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(CTFixture, CT_SEQ_SCALE)
  ->Apply(RaiderSTREAM_Arguments)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(CTFixture, CT_SEQ_TRIAD)
  ->Apply(RaiderSTREAM_Arguments)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(CTFixture, CT_SEQ_MULT_NO_RELIN)
  ->Apply(RaiderSTREAM_Arguments)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(CTFixture, CT_SEQ_RELIN)
  ->Apply(RaiderSTREAM_Arguments)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(CTFixture, CT_SEQ_ADD_INPLACE)
  ->Apply(RaiderSTREAM_Arguments)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(CTFixture, CT_MPI_SENDRECV)
  ->Apply(RaiderSTREAM_Arguments)
  ->Iterations(10)
  ->Unit(benchmark::kMillisecond);
