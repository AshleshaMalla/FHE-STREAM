#!/usr/bin/env python3
"""
Parse LIKWID marker-mode CSV output (-o file.csv).

Simple mode (single CSV):
    python3 scripts/parse_likwid_csv.py <csv_file> <MEMREAD|MEMWRITE>
    Prints: nonzero HWThread count, total data volume [GBytes].

Amplification mode (4 CSVs: both groups × both sockets):
    python3 scripts/parse_likwid_csv.py --amplification \\
        --filter <benchmark_filter> --arrays <2|3> \\
        --memread-s0 <csv> --memread-s1 <csv> \\
        --memwrite-s0 <csv> --memwrite-s1 <csv>
    Prints: amplification summary table (logical vs measured bytes, ratio).

NOTE on bandwidth: NEVER use LIKWID's derived bandwidth columns from marker-mode
runs — the ~55x instrumentation overhead inflates wall time, corrupting those values.
Use the data volume here together with timing from a separate, unmarked run.
"""
import sys
import csv
import re
import argparse

# Paper measurement constants (hardcoded for N=131072, L=40, B=512)
_N = 131072
_L = 40
_B = 512
_EXPECTED_THREADS = 128  # threads per socket (half of 256-thread zen4 node)


def _parse_csv(path):
    """
    Parse one LIKWID marker CSV.
    Returns (call_counts, data_volume_gb) as lists of per-HWThread floats.
    Stops parsing each row at the first non-numeric value (skips Sum/Min/Max/Avg).
    """
    call_counts = []
    data_volume = []

    with open(path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            label = row[0].strip()

            def extract_floats(row):
                vals = []
                for v in row[1:]:
                    try:
                        vals.append(float(v))
                    except ValueError:
                        break
                return vals

            if label.lower() == 'call count':
                parsed = extract_floats(row)
                if parsed:
                    call_counts = parsed

            if re.search(r'memory.*data volume.*gbytes', label, re.IGNORECASE):
                parsed = extract_floats(row)
                if parsed:
                    data_volume = parsed

    return call_counts, data_volume


def _sum_nonzero(call_counts, data_volume):
    """Sum data_volume entries where the corresponding call_count is nonzero."""
    nonzero_idx = [i for i, c in enumerate(call_counts) if c > 0]
    total = sum(data_volume[i] for i in nonzero_idx if i < len(data_volume))
    return len(nonzero_idx), total


def _sanity(path, label, call_counts):
    nonzero = sum(1 for c in call_counts if c > 0)
    if nonzero < _EXPECTED_THREADS:
        print(
            f"WARNING {label} ({path}):\n"
            f"  only {nonzero}/{len(call_counts)} HWThreads have nonzero call count\n"
            f"  — registration bug may have recurred; data from this socket may be incomplete",
            file=sys.stderr,
        )
    return nonzero


# ---------------------------------------------------------------------------
# Simple mode
# ---------------------------------------------------------------------------

def _simple(path, group):
    cc, dv = _parse_csv(path)
    if not cc:
        sys.exit(f"ERROR: 'call count' row not found in {path}")
    if not dv:
        sys.exit(f"ERROR: 'Memory data volume [GBytes]' row not found in {path}")
    nonzero, total = _sum_nonzero(cc, dv)
    print(f"Group             : {group}")
    print(f"Nonzero HWThreads : {nonzero} / {len(cc)}")
    print(f"Total data volume : {total:.4f} GBytes")


# ---------------------------------------------------------------------------
# Amplification mode
# ---------------------------------------------------------------------------

def _amplification(args):
    paths = {
        'read_s0':  args.memread_s0,
        'read_s1':  args.memread_s1,
        'write_s0': args.memwrite_s0,
        'write_s1': args.memwrite_s1,
    }

    data = {}
    for key, path in paths.items():
        cc, dv = _parse_csv(path)
        if not cc:
            sys.exit(f"ERROR: 'call count' not found in {path}")
        if not dv:
            sys.exit(f"ERROR: 'Memory data volume [GBytes]' not found in {path}")
        data[key] = (cc, dv)

    print()

    # Sanity checks — warn to stderr if any socket has incomplete registration
    _sanity(args.memread_s0,  "MEMREAD  socket0", data['read_s0'][0])
    _sanity(args.memread_s1,  "MEMREAD  socket1", data['read_s1'][0])
    _sanity(args.memwrite_s0, "MEMWRITE socket0", data['write_s0'][0])
    _sanity(args.memwrite_s1, "MEMWRITE socket1", data['write_s1'][0])

    # Sum measured volumes across both sockets
    _, read_s0_gb  = _sum_nonzero(*data['read_s0'])
    _, read_s1_gb  = _sum_nonzero(*data['read_s1'])
    _, write_s0_gb = _sum_nonzero(*data['write_s0'])
    _, write_s1_gb = _sum_nonzero(*data['write_s1'])

    measured_read_gb  = read_s0_gb  + read_s1_gb
    measured_write_gb = write_s0_gb + write_s1_gb

    # Iteration counts from max call count per socket CSV
    # (all threads should agree; use max in case of partial registration)
    def max_cc(cc): return max(cc) if cc else 0
    read_iters  = max(max_cc(data['read_s0'][0]),  max_cc(data['read_s1'][0]))
    write_iters = max(max_cc(data['write_s0'][0]), max_cc(data['write_s1'][0]))

    if read_iters != write_iters:
        print(
            f"WARNING: iteration count mismatch: MEMREAD={int(read_iters)}, "
            f"MEMWRITE={int(write_iters)} — amplification ratios may not be comparable",
            file=sys.stderr,
        )

    # Logical byte computation (paper params: N, L, B hardcoded above)
    arrays      = int(args.arrays)
    read_arrays = arrays - 1   # 2-array → 1 read;  3-array → 2 reads
    vpoly_gb    = _N * _L * 8 / 1e9

    logical_read_gb  = read_iters  * _B * read_arrays * vpoly_gb
    logical_write_gb = write_iters * _B * 1            * vpoly_gb

    read_amp  = measured_read_gb  / logical_read_gb  if logical_read_gb  > 0 else float('nan')
    write_amp = measured_write_gb / logical_write_gb if logical_write_gb > 0 else float('nan')

    # Summary table
    print()
    print("=== Amplification Summary ===")
    print(f"Kernel    : {args.filter}")
    print(f"Params    : N={_N}, L={_L}, B={_B}, arrays={arrays}")
    print(f"Iterations: MEMREAD={int(read_iters)}, MEMWRITE={int(write_iters)}")
    print()
    print(f"{'':20s} {'Logical (GB)':>14} {'Measured (GB)':>14} {'Amplification':>14}")
    print(f"{'Read':20s} {logical_read_gb:>14.3f} {measured_read_gb:>14.3f} {read_amp:>13.2f}x")
    print(f"{'Write':20s} {logical_write_gb:>14.3f} {measured_write_gb:>14.3f} {write_amp:>13.2f}x")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--amplification':
        p = argparse.ArgumentParser()
        p.add_argument('--amplification', action='store_true')
        p.add_argument('--filter',      required=True)
        p.add_argument('--arrays',      required=True, choices=['2', '3'])
        p.add_argument('--memread-s0',  required=True, dest='memread_s0')
        p.add_argument('--memread-s1',  required=True, dest='memread_s1')
        p.add_argument('--memwrite-s0', required=True, dest='memwrite_s0')
        p.add_argument('--memwrite-s1', required=True, dest='memwrite_s1')
        _amplification(p.parse_args())
    else:
        if len(sys.argv) != 3:
            sys.exit(f"Usage: {sys.argv[0]} <csv_file> <MEMREAD|MEMWRITE>")
        _simple(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
