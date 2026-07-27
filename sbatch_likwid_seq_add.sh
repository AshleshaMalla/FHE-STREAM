export RS_BATCH_SIZE=512
export OMP_NUM_THREADS=4
likwid-perfctr -C 0-3 -g MEMREAD -m -o results/seq_add_debug_4threads.csv \
  ./build/fhe_raiderstream --benchmark_filter='RS_SEQ_ADD/131072/40' \
  --benchmark_min_time=0.1s 2>/dev/null

grep -i "ACTUAL_CPU_CLOCK\|Runtime (RDTSC)\|call count" results/seq_add_debug_4threads.csv