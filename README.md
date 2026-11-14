# FHE-STREAM

**Characterizing memory access behavior in fully homomorphic encryption workloads.**

FHE-STREAM is a memory microbenchmark suite for RNS-based fully homomorphic
encryption (FHE), built on [OpenFHE](https://github.com/openfheorg/openfhe-development) 1.5.0.
It asks one question: on CPUs, how much DRAM traffic do FHE workloads really
generate, compared with the bytes the algorithm needs to move?

The code name, used for the repository and the binary, is FHE-RaiderSTREAM.

**Contents:** [Paper](#paper) · [Key result](#key-result) · [What it measures](#what-it-measures) ·
[Quick start](#quick-start) · [Reproducing the paper](#reproducing-the-paper) ·
[Repository layout](#repository-layout) · [License](#license)

## Paper

> Ashlesha Malla, Yash Kumar, Michael Beebe, Yong Chen (Texas Tech University),
> Wendy Poole, Steve Poole (Los Alamos National Laboratory).
> **FHE-STREAM: Characterizing Memory Access Behavior in Fully Homomorphic
> Encryption Workloads.** PMBS Workshop at SC26, November 2026.

- PDF: [`FHE-STREAM_SC26.pdf`](FHE-STREAM_SC26.pdf)
- Talk slides: [`FHE-STREAM_SC26.pptx`](FHE-STREAM_SC26.pptx)

```bibtex
@inproceedings{malla2026fhestream,
  title     = {{FHE-STREAM}: Characterizing Memory Access Behavior in
               Fully Homomorphic Encryption Workloads},
  author    = {Malla, Ashlesha and Kumar, Yash and Beebe, Michael and
               Chen, Yong and Poole, Wendy and Poole, Steve},
  booktitle = {Workshop on Performance Modeling, Benchmarking and Simulation
               of High Performance Computer Systems (PMBS), held with SC26},
  year      = {2026}
}
```

## Key result

The *amplification factor* is the DRAM bytes measured with hardware counters
divided by the bytes the algorithm logically needs to move. A value of 1.00×
means no wasted traffic.

| Kernel | Access pattern | Read amplification | Write amplification |
|---|---|:---:|:---:|
| GATHER_ADD, whole-polynomial shuffle | Gather | 1.00× | 1.00× |
| SEQ_ADD | Sequential | 1.51× | 1.00× |
| GATHER_ADD, coefficient shuffle | Gather | 1.87× | 1.01× |
| NTT round trip | Strided (butterfly) | 2.78× | 2.20× |
| SCATTER_GATHER_TRIAD, coefficient shuffle | Scatter-Gather | 3.84× | 2.44× |

Setting: DCRTPoly backend, AMD EPYC 9754 (2 × 128 cores), 256 threads, batch of
512 polynomials, ring dimension N = 131072, multiplicative depth L = 40.
In three repeated runs, measured traffic varied by at most 1.5%.

The main factor is granularity, not irregularity on its own. Shuffling the order
of whole polynomials (each a block of 1 MiB or more read in sequence) wastes no
traffic. Random 8-byte coefficient accesses waste most of each 64-byte cache line
they touch. Full numbers and their interpretation are in
[`results/validation_summary.md`](results/validation_summary.md).

## What it measures

FHE-STREAM takes the access patterns of
[RaiderSTREAM](https://github.com/michael-beebe/RaiderSTREAM) (STREAM's
COPY / SCALE / ADD / TRIAD kernels under Sequential, Gather, Scatter and
Scatter-Gather access) and runs them over OpenFHE's real data layouts. In OpenFHE
a polynomial in RNS form is a list of pointers to L separately allocated arrays
("limbs") of N 64-bit coefficients, so each access goes through a pointer first.

There are two backends, one for each level of the software stack:

- **DCRTPoly backend.** STREAM-style loops run directly over OpenFHE polynomial
  arrays. Gather and Scatter shuffle either whole polynomials (*Poly* granularity)
  or individual coefficients (*Coeff* granularity). It also has an NTT round trip
  (an in-place pass whose access distance halves at each stage, like an FFT) and a
  key-switching mock.
- **Ciphertext backend.** Real encrypted BFV ciphertexts processed through the
  OpenFHE API (add, scalar multiply, multiply, relinearize). It reports the logical
  payload bandwidth together with the size of the serialized object, so you can see
  how much the library's object overhead adds.

Each kernel reports **modeled bandwidth**: logical bytes (Eq. 1–3 in the paper)
divided by time, as reported by Google Benchmark. Modeled bandwidth is not DRAM
traffic. To measure DRAM traffic, the same binary runs under LIKWID, which counts
DRAM bytes for the whole process. Each kernel runs at two iteration counts and the
two totals are subtracted, so one-time setup traffic cancels out. LIKWID's
per-region marker mode was tried first. It turned out to be unreliable on this
platform, and its results were retracted
([`results/validation_summary_MARKER_MODE_RETRACTED.md`](results/validation_summary_MARKER_MODE_RETRACTED.md)
is kept for the record).

Kernel definitions and byte models are in [`BENCHMARKS.md`](BENCHMARKS.md).

## Quick start

### Dependencies

| Dependency | Version | Notes |
|---|---|---|
| GCC | 11.5.0 (any C++17 compiler) | |
| CMake | ≥ 3.5 | |
| OpenFHE | 1.5.0 | built by `setup_env.sh` |
| Google Benchmark | 1.8.5 | built by `setup_env.sh` |
| OpenMP | | |
| MPI | | optional |
| Intel HEXL | 1.2.6 | optional, Intel CPUs with AVX-512 IFMA |
| LIKWID | 5.4.1 | optional, for DRAM counter measurements |

### Set up and build

```bash
./setup_env.sh            # one time: builds OpenFHE and Google Benchmark into env/
                          # add --hexl to also build Intel HEXL
source env/activate       # in every new shell

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CIPHERTEXT=ON
cmake --build build -j    # produces build/fhe_raiderstream
```

| CMake option | Default | Meaning |
|---|---|---|
| `ENABLE_DCRT` | ON | DCRTPoly backend |
| `ENABLE_CIPHERTEXT` | OFF | Ciphertext backend |
| `ENABLE_MPI` | ON | MPI support; skipped if MPI is not found |
| `ENABLE_HEXL` | ON | Intel HEXL; skipped if HEXL is not found |
| `LIKWID_PERFMON` | OFF | LIKWID marker regions (not used for the paper's numbers) |

### Run

Benchmark names have the form `KERNEL/N/L/shuffle`, where `N` is the ring
dimension, `L` is the number of limbs, and `shuffle` is 0 (none), 1 (Poly) or
2 (Coeff). The NTT and Ciphertext kernels ignore the shuffle, so use `/0` for them.

```bash
# Always set the batch size. The default (100) is not what the paper used.
export RS_BATCH_SIZE=512          # DCRTPoly backend; use 256 for Ciphertext kernels

OMP_NUM_THREADS=256 OMP_PROC_BIND=true OMP_PLACES="{0:256}" \
  ./build/fhe_raiderstream --benchmark_filter='FHERaiderSTREAM/RS_SEQ_ADD/131072/40/0'
```

Use `OMP_NUM_THREADS` and `OMP_PLACES` values that match your machine's core count.
The paper's settings (N = 131072, L = 40) need a large amount of memory. Start
with a small `RS_BATCH_SIZE` (for example, 2) when trying Ciphertext kernels on a
new machine.

Ciphertext kernels report extra counters: `PayloadBandwidth` (modeled, same byte
model as the DCRTPoly backend), `SerializedBandwidth` (based on the serialized
object size), `ObjectToPayloadRatio`, and the resident memory per ciphertext.
MPI runs (`srun -n <ranks> ...`) report the aggregate modeled bandwidth on rank 0.

## Reproducing the paper

[`reproducibility/README.md`](reproducibility/README.md) covers hardware,
software versions and one script per figure. Use an exclusive node: other jobs
on a shared node add DRAM traffic that the counters cannot tell apart from yours.

## Repository layout

```
include/, src/
  backends/dcrt/          DCRTPoly backend: setup, kernels, loop and NTT code
  backends/ciphertext/    Ciphertext backend: setup and kernels
  common/                 argument grids (N, L, shuffle)
  main.cpp                MPI, threading and LIKWID initialization
reproducibility/          build and run scripts for every figure
python_plot/              plotting scripts  →  plots/
results/                  raw and summarized results cited by the paper
BENCHMARKS.md             kernel definitions and byte models
setup_env.sh, build.sh    environment setup and development build
```

## License

MIT. See [`LICENSE`](LICENSE).

## Acknowledgments

FHE-STREAM builds on [OpenFHE](https://github.com/openfheorg/openfhe-development)
and the access patterns of [RaiderSTREAM](https://github.com/michael-beebe/RaiderSTREAM).
