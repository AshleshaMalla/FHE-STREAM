# FHE-RaiderSTREAM

FHE-RaiderSTREAM is a C++ micro-benchmark suite for OpenFHE memory behavior.
It measures bandwidth limits across three layers:

- `DCRTPoly` backend: hardware and irregular-access limits
- `Ciphertext` backend: software/allocator overhead and capacity limits
- `MPI` extension: distributed aggregation and serialization overhead

For full kernel descriptions and bytes models, see `BENCHMARKS.md`.

## Dependencies

- OpenFHE (CMake package)
- Google Benchmark (auto-fetched if missing)
- OpenMP
- MPI (optional, for distributed runs)

## Build

### DCRT backend only (default)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### DCRT + Ciphertext + MPI

```bash
cmake -S . -B build \
	-DCMAKE_BUILD_TYPE=Release \
	-DENABLE_CIPHERTEXT=ON \
	-DENABLE_MPI=ON
cmake --build build -j
```

## Run

### Local DCRT example

```bash
RS_BATCH_SIZE=40 ./build/fhe_raiderstream \
	--benchmark_filter="RS_SEQ_ADD/32768/5"
```

### Local Ciphertext example

```bash
RS_BATCH_SIZE=2 ./build/fhe_raiderstream \
	--benchmark_filter="CTFixture/CT_SEQ_ADD/16384/5" \
	--benchmark_min_time=0.05s
```

### MPI example

```bash
RS_BATCH_SIZE=2 mpirun -np 2 ./build/fhe_raiderstream \
	--benchmark_filter="CTFixture/CT_MPI_SENDRECV/16384/5"
```

MPI runs report `AggregateBandwidth` on rank 0.

## Notes

- `RS_BATCH_SIZE` controls working-set size (default `100`).
- Large RingDim + Depth combinations can exceed memory quickly; start with `RS_BATCH_SIZE=2` for ciphertext benchmarks.
