# FHE-RaiderSTREAM

FHE-RaiderSTREAM is a C++ micro-benchmark suite for OpenFHE memory behavior.
It measures bandwidth limits across three layers:

- `DCRTPoly` backend: hardware and irregular-access limits
- `Ciphertext` backend: payload goodput plus serialized object footprint
- `MPI` extension: distributed aggregation and serialization overhead

For full kernel descriptions and bytes models, see `BENCHMARKS.md`.

**To reproduce the paper results**, see [`reproducibility/README.md`](reproducibility/README.md).

## Dependencies

- GCC 11.5.0 (or any C++17 compiler)
- CMake >= 3.5
- OpenFHE 1.5.0
- Google Benchmark 1.8.5 (auto-fetched if missing)
- OpenMP
- MPI (optional, for distributed runs)
- Intel HEXL 1.2.6 (optional, Intel node only)
- LIKWID 5.4.1-daemon (optional, for hardware counter validation)

All dependencies except the compiler and CMake are built automatically
by `setup_env.sh`.

## Setup

```bash
./setup_env.sh          # one-time: builds OpenFHE, Google Benchmark, HEXL
source env/activate     # required before every build or run session
```

Add `--hexl` to `setup_env.sh` to build Intel HEXL support (requires
an Intel node with AVX-512 IFMA).

## Build

### DCRT backend only (default)

```bash
source env/activate
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### DCRT + Ciphertext + MPI

```bash
source env/activate
cmake -S . -B build \
	-DCMAKE_BUILD_TYPE=Release \
	-DENABLE_CIPHERTEXT=ON \
	-DENABLE_MPI=ON
cmake --build build -j
```

### CMake Options

| Option | Default | Description |
|---|---|---|
| `ENABLE_DCRT` | ON | DCRTPoly backend |
| `ENABLE_CIPHERTEXT` | OFF | Ciphertext backend |
| `ENABLE_HEXL` | ON | Intel HEXL acceleration |
| `ENABLE_MPI` | ON | MPI distributed support |
| `LIKWID_PERFMON` | OFF | LIKWID Marker API instrumentation |

## Run

### Local DCRT example

```bash
RS_BATCH_SIZE=512 ./build/fhe_raiderstream \
	--benchmark_filter="RS_SEQ_ADD/131072/40"
```

### Local Ciphertext example

```bash
RS_BATCH_SIZE=256 ./build/fhe_raiderstream \
	--benchmark_filter="CTFixture/CT_SEQ_ADD/131072/40" \
	--benchmark_min_time=1s
```

### MPI example

```bash
RS_BATCH_SIZE=512 srun -n 4 ./build/fhe_raiderstream \
	--benchmark_filter="RS_SEQ_ADD/131072/40"
```

MPI runs report `AggregateBandwidth` on rank 0. Ciphertext kernels expose
dual metrics for fair comparisons against DCRT and object-level realism:

- `BytesProcessed` and `PayloadBandwidth`: payload-model goodput (same model family as DCRT).
- `SerializedBandwidth`: wire-format proxy for full ciphertext object traffic.
- `ObjectToPayloadRatio`: serialized bytes divided by payload bytes for the kernel.
- `RSSDeg1BytesPerCt` and `RSSDeg2BytesPerCt`: setup-time resident-memory delta
	per ciphertext (degree-1 and degree-2 pools, measured from `/proc/self/statm`).

## Notes

- `RS_BATCH_SIZE` controls working-set size (default `100`). Paper results
  use `512` for the DCRTPoly backend and `256` for the Ciphertext backend;
  always set this explicitly.
- Large RingDim + Depth combinations can exceed memory quickly; start with
  `RS_BATCH_SIZE=2` for initial ciphertext testing.
