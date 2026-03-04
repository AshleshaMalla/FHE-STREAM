# FHE-RaiderSTREAM

Standalone C++ benchmark that approximates **sustainable memory bandwidth** for OpenFHE data structures (`lbcrypto::DCRTPoly`) using access patterns inspired by RaiderSTREAM:

- Copy
- Add
- Gather
- Scatter

## Dependencies

- OpenFHE (built + installed with CMake package config)
- Google Benchmark
- OpenMP (optional, used if found)
- MPI (optional, e.g., OpenMPI or MPICH, for distributed benchmarking)

## Build

```bash
cd fhe-raiderstream
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If Google Benchmark is not installed on your system, CMake will fetch and build it automatically. Use `-DENABLE_MPI=ON/OFF` to explicitly control MPI support (detected automatically if found).

## Run

### Local Execution
```bash
./build/fhe_raiderstream --benchmark_min_time=1
```

### Distributed Execution (MPI)
```bash
mpirun -n <num_ranks> ./build/fhe_raiderstream --benchmark_min_time=1
```
Distributed runs report an `AggregateBandwidth` counter representing the total throughput across all ranks.

Useful options:

- `--benchmark_filter=Copy|Add|Gather|Scatter`
- `--benchmark_repetitions=5 --benchmark_report_aggregates_only=true`

## Notes on bandwidth numbers

`DCRTPoly` is a rich object with internal allocations; the benchmark uses `sizeof(DCRTPoly)` as a **conservative lower-bound** estimate of bytes moved. For a more accurate bandwidth model, you can extend the code to estimate internal storage (towers, coefficient arrays) based on the active OpenFHE parameters.
