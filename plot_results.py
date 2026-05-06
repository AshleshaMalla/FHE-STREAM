import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fhe_plot_style import configure_matplotlib, save_plot, FIGSIZE_SINGLE, HATCH_PATTERNS

def parse_data(csv_file):
    with open(csv_file, 'r') as f:
        skip = next(i for i, line in enumerate(f) if line.startswith('name'))
    df = pd.read_csv(csv_file, skiprows=skip)
    df = df[df['name'].str.startswith('FHERaiderSTREAM')]
    cols = df['name'].str.split('/', expand=True)
    df['Kernel'] = cols[1]
    df['RingDim'] = cols[2].astype(int)
    df['Depth'] = cols[3].astype(int)
    df['GiB_s'] = df['bytes_per_second'] / (1024.0**3)
    return df

def generate_plots():
    csv_file = "results.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return

    df = parse_data(csv_file)
    configure_matplotlib()

    print("Generating FHE-RaiderSTREAM Plots...")

    # ==========================================
    # PLOT 1: The Access Pattern Penalty
    # ==========================================
    subset1 = df[(df['RingDim'] == 65536) & (df['Depth'] == 20)].copy()
    if not subset1.empty:
        plt.figure(figsize=FIGSIZE_SINGLE)
        order1 = ['RS_SEQ_COPY', 'RS_GATHER_COPY', 'RS_GATHER_ADD', 'RS_GATHER_TRIAD', 'RS_NTT_ROUNDTRIP']
        actual_order1 = [k for k in order1 if k in subset1['Kernel'].values]
        ax = sns.barplot(data=subset1, x='Kernel', y='GiB_s', order=actual_order1, palette='magma', width=0.5, edgecolor='black')
        for i, patch in enumerate(ax.patches):
            patch.set_hatch(HATCH_PATTERNS[i % len(HATCH_PATTERNS)])
        plt.title('Plot 1: Memory Access Patterns', fontweight='bold')
        plt.ylabel('Bandwidth (GiB/s)', fontweight='bold')
        plt.xlabel('Operation Kernel', fontweight='bold')
        plt.xticks(rotation=25, ha='right')
        save_plot('plots/plot1_access_pattern.pdf')

    # ==========================================
    # PLOT 2: Multi-Stream Degradation
    # ==========================================
    subset2 = df[(df['RingDim'] == 65536) & (df['Kernel'] == 'RS_SEQ_ADD')].copy()
    if not subset2.empty:
        plt.figure(figsize=FIGSIZE_SINGLE)
        sns.lineplot(data=subset2, x='Depth', y='GiB_s', marker='o', color='#e74c3c')
        plt.title('Plot 2: Multi-Stream Degradation', fontweight='bold')
        plt.ylabel('Bandwidth (GiB/s)', fontweight='bold')
        plt.xlabel('Multiplicative Depth (RNS Towers)', fontweight='bold')
        plt.xticks([1, 5, 20, 40])
        save_plot('plots/plot2_multi_stream.pdf')

    # ==========================================
    # PLOT 3: The Cache Cliff (Ring Scaling)
    # ==========================================
    subset3 = df[(df['Depth'] == 1) & (df['Kernel'] == 'RS_SEQ_TRIAD')].copy()
    if not subset3.empty:
        plt.figure(figsize=FIGSIZE_SINGLE)
        sns.lineplot(data=subset3, x='RingDim', y='GiB_s', marker='s', color='#3498db')
        plt.xscale('log', base=2)
        plt.xticks([32768, 65536, 131072], ['32k', '65k', '131k'])
        plt.title('Plot 3: Ring Dimension Scaling', fontweight='bold')
        plt.ylabel('Bandwidth (GiB/s)', fontweight='bold')
        plt.xlabel('Ring Dimension', fontweight='bold')
        save_plot('plots/plot3_cache_cliff.pdf')

    # ==========================================
    # PLOT 4: Arithmetic Intensity (Classic STREAM)
    # ==========================================
    subset4 = df[(df['RingDim'] == 65536) & (df['Depth'] == 20) & (df['Kernel'].str.startswith('RS_SEQ'))].copy()
    if not subset4.empty:
        plt.figure(figsize=FIGSIZE_SINGLE)
        ax = sns.barplot(data=subset4, x='Kernel', y='GiB_s', order=['RS_SEQ_COPY', 'RS_SEQ_SCALE', 'RS_SEQ_ADD', 'RS_SEQ_TRIAD'], palette='crest', width=0.5, edgecolor='black')
        for i, patch in enumerate(ax.patches):
            patch.set_hatch(HATCH_PATTERNS[i % len(HATCH_PATTERNS)])
        plt.title('Plot 4: Arithmetic Intensity', fontweight='bold')
        plt.ylabel('Bandwidth (GiB/s)', fontweight='bold')
        plt.xlabel('Kernel Type', fontweight='bold')
        plt.xticks(rotation=20, ha='right')
        save_plot('plots/plot4_arithmetic.pdf')

    # ==========================================
    # PLOT 5: The Indirection Penalty (Seq vs Gather)
    # ==========================================
    subset5 = df[(df['RingDim'] == 65536) & (df['Depth'] == 20)].copy()
    if not subset5.empty:
        subset5 = subset5[subset5['Kernel'] != 'RS_NTT_ROUNDTRIP']
        subset5['Access'] = subset5['Kernel'].apply(lambda x: 'Sequential' if 'SEQ' in x else 'Gather')
        subset5['Operation'] = subset5['Kernel'].apply(lambda x: x.split('_')[-1])
        plt.figure(figsize=FIGSIZE_SINGLE)
        ax = sns.barplot(data=subset5, x='Operation', y='GiB_s', hue='Access', palette='Set2', width=0.5, edgecolor='black')
        # In grouped barplots, the number of patches corresponds to categories * groups
        num_cats = len(subset5['Operation'].unique())
        for i, patch in enumerate(ax.patches):
            hatch_idx = i // num_cats
            patch.set_hatch(HATCH_PATTERNS[hatch_idx % len(HATCH_PATTERNS)])
        plt.title('Plot 5: The Indirection Penalty', fontweight='bold')
        plt.ylabel('Bandwidth (GiB/s)', fontweight='bold')
        plt.xlabel('Operation', fontweight='bold')
        plt.legend(title='Access Pattern')
        save_plot('plots/plot5_indirection.pdf')

    # ==========================================
    # PLOT 6: The Capacity Wall (Footprint vs Bandwidth)
    # ==========================================
    subset6 = df[df['Kernel'] == 'RS_SEQ_COPY'].copy()
    if not subset6.empty:
        # Footprint = Ring * Depth * Batch * 8 bytes * 2 arrays
        subset6['Footprint_MB'] = (subset6['RingDim'] * subset6['Depth'] * 40 * 8 * 2) / (1024**2)
        plt.figure(figsize=FIGSIZE_SINGLE)
        sns.lineplot(data=subset6, x='Footprint_MB', y='GiB_s', marker='D', color='purple')
        plt.xscale('log', base=10)
        plt.title('Plot 6: The Capacity Wall', fontweight='bold')
        plt.ylabel('Bandwidth (GiB/s)', fontweight='bold')
        plt.xlabel('Working Set Size (MB, Log Scale)', fontweight='bold')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        save_plot('plots/plot6_capacity.pdf')

if __name__ == "__main__":
    generate_plots()