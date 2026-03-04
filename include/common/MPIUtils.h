#pragma once

#include <benchmark/benchmark.h>

#ifdef RAIDERSTREAM_MPI
#include <mpi.h>
#endif

extern int RS_MPI_Rank;
extern int RS_MPI_Size;

inline void RS_BARRIER() {
#ifdef RAIDERSTREAM_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

inline void AggregateBandwidth(benchmark::State& state) {
#ifdef RAIDERSTREAM_MPI
  RS_BARRIER();
  double local_bytes = static_cast<double>(state.bytes_processed());
  double total_bytes = 0;
  MPI_Reduce(&local_bytes, &total_bytes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (RS_MPI_Rank == 0) {
    state.counters["AggregateBandwidth"] = benchmark::Counter(total_bytes, benchmark::Counter::kIsRate);
  }
#endif
}
