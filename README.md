# FHE-STREAM

A memory microbenchmark suite for RNS-based fully homomorphic encryption (FHE),
built on [OpenFHE](https://github.com/openfheorg/openfhe-development) 1.5.0.
It runs the access patterns of [RaiderSTREAM](https://github.com/michael-beebe/RaiderSTREAM)
(Sequential, Gather, Scatter, Scatter-Gather) over OpenFHE's real data layouts. It then
compares the bytes each kernel needs to move with the DRAM traffic measured
by LIKWID.

**Paper:** *FHE-STREAM: Characterizing Memory Access Behavior in Fully Homomorphic
Encryption Workloads*, PMBS Workshop at SC26 ([PDF](FHE-STREAM_SC26.pdf)).

## Key result

Amplification is measured DRAM bytes ÷ the bytes the algorithm needs to move
(1.00× = no wasted traffic).

| Kernel | Read | Write |
|---|:---:|:---:|
| Gather, whole-polynomial shuffle | 1.00× | 1.00× |
| Sequential add | 1.51× | 1.00× |
| Gather, coefficient shuffle | 1.87× | 1.01× |
| NTT round trip | 2.78× | 2.20× |
| Scatter-Gather, coefficient shuffle | 3.84× | 2.44× |

AMD EPYC 9754, 256 threads, N = 131072, L = 40. The main driver is random
8-byte access, which wastes most of each 64-byte cache line.
Details: [`results/validation_summary.md`](results/validation_summary.md).

## Quick start

Requires a C++17 compiler, CMake ≥ 3.5 and OpenMP. `setup_env.sh` builds
OpenFHE and Google Benchmark for you.

```bash
./setup_env.sh                 # one time (add --hexl for Intel HEXL)
source env/activate            # in every new shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CIPHERTEXT=ON
cmake --build build -j

export RS_BATCH_SIZE=512       # always set; 256 for Ciphertext kernels
OMP_NUM_THREADS=256 OMP_PROC_BIND=true \
  ./build/fhe_raiderstream --benchmark_filter='RS_SEQ_ADD/131072/40/0'
```

Benchmark names are `KERNEL/N/L/shuffle`, where shuffle is 0 (none),
1 (whole polynomial) or 2 (coefficient). Kernel definitions are in
[`BENCHMARKS.md`](BENCHMARKS.md).

To reproduce the paper's figures, see
[`reproducibility/README.md`](reproducibility/README.md).

## License

MIT. See [`LICENSE`](LICENSE).
