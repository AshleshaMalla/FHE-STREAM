import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    print("Generating FHE-RaiderSTREAM Plots...")

    # ==========================================
    # PLOT 1: The Access Pattern Penalty
    # ==========================================
    subset1 = df[(df['RingDim'] == 65536) & (df['Depth'] == 20)].copy()
    if not subset1.empty:
        plt.figure(figsize=(10, 6))
        order1 = ['RS_SEQ_COPY', 'RS_GATHER_COPY', 'RS_GATHER_ADD', 'RS_GATHER_TRIAD', 'RS_NTT_ROUNDTRIP']
        actual_order1 = [k for k in order1 if k in subset1['Kernel'].values]
        sns.barplot(data=subset1, x='Kernel', y='GiB_s', order=actual_order1, palette='magma', width=0.5)
        plt.title('Plot 1: Memory Bandwidth by Access Pattern\n(Ring=65k, Depth=20)', fontweight='bold')
        plt.ylabel('Utilizable Bandwidth (GiB/s)', fontweight='bold')
        plt.xlabel('Operation Kernel', fontweight='bold')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig('plot1_access_pattern.png', dpi=300)
        print(" -> Saved 'plot1_access_pattern.png'")

    # ==========================================
    # PLOT 2: Multi-Stream Degradation
    # ==========================================
    subset2 = df[(df['RingDim'] == 65536) & (df['Kernel'] == 'RS_SEQ_ADD')].copy()
    if not subset2.empty:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=subset2, x='Depth', y='GiB_s', marker='o', markersize=10, linewidth=3, color='#e74c3c')
        plt.title('Plot 2: Multi-Stream Memory Degradation\n(Ring=65k, RS_SEQ_ADD)', fontweight='bold')
        plt.ylabel('Bandwidth (GiB/s)', fontweight='bold')
        plt.xlabel('Multiplicative Depth (Simultaneous RNS Towers)', fontweight='bold')
        plt.xticks([1, 5, 20, 40])
        plt.ylim(subset2['GB_s'].min() - 5, subset2['GB_s'].max() + 5)
        plt.tight_layout()
        plt.savefig('plot2_multi_stream.png', dpi=300)
        print(" -> Saved 'plot2_multi_stream.png'")

    # ==========================================
    # PLOT 3: The Cache Cliff (Ring Scaling)
    # ==========================================
    subset3 = df[(df['Depth'] == 1) & (df['Kernel'] == 'RS_SEQ_TRIAD')].copy()
    if not subset3.empty:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=subset3, x='RingDim', y='GiB_s', marker='s', markersize=10, linewidth=3, color='#3498db')
        plt.xscale('log', base=2)
        plt.xticks([32768, 65536, 131072], ['32k', '65k', '131k'])
        plt.title('Plot 3: Ring Dimension Scaling\n(Depth=1, RS_SEQ_TRIAD)', fontweight='bold')
        plt.ylabel('Bandwidth (GiB/s)', fontweight='bold')
        plt.xlabel('Ring Dimension', fontweight='bold')
        plt.ylim(0, subset3['GB_s'].max() * 1.1)
        plt.tight_layout()
        plt.savefig('plot3_cache_cliff.png', dpi=300)
        print(" -> Saved 'plot3_cache_cliff.png'")

    # ==========================================
    # PLOT 4: Arithmetic Intensity (Classic STREAM)
    # ==========================================
    subset4 = df[(df['RingDim'] == 65536) & (df['Depth'] == 20) & (df['Kernel'].str.startswith('RS_SEQ'))].copy()
    if not subset4.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=subset4, x='Kernel', y='GiB_s', order=['RS_SEQ_COPY', 'RS_SEQ_SCALE', 'RS_SEQ_ADD', 'RS_SEQ_TRIAD'], palette='crest', width=0.5)
        plt.title('Plot 4: Bandwidth by Arithmetic Intensity\n(Ring=65k, Depth=20)', fontweight='bold')
        plt.ylabel('Bandwidth (GiB/s)', fontweight='bold')
        plt.xlabel('Kernel Type', fontweight='bold')
        plt.tight_layout()
        plt.savefig('plot4_arithmetic.png', dpi=300)
        print(" -> Saved 'plot4_arithmetic.png'")

    # ==========================================
    # PLOT 5: The Indirection Penalty (Seq vs Gather)
    # ==========================================
    subset5 = df[(df['RingDim'] == 65536) & (df['Depth'] == 20)].copy()
    if not subset5.empty:
        subset5 = subset5[subset5['Kernel'] != 'RS_NTT_ROUNDTRIP']
        subset5['Access'] = subset5['Kernel'].apply(lambda x: 'Sequential' if 'SEQ' in x else 'Gather')
        subset5['Operation'] = subset5['Kernel'].apply(lambda x: x.split('_')[-1])
        plt.figure(figsize=(8, 5))
        sns.barplot(data=subset5, x='Operation', y='GiB_s', hue='Access', palette='Set2', width=0.5)
        plt.title('Plot 5: The Indirection Penalty\n(Ring=65k, Depth=20)', fontweight='bold')
        plt.ylabel('Bandwidth (GiB/s)', fontweight='bold')
        plt.xlabel('Operation', fontweight='bold')
        plt.legend(title='Access Pattern')
        plt.tight_layout()
        plt.savefig('plot5_indirection.png', dpi=300)
        print(" -> Saved 'plot5_indirection.png'")

    # ==========================================
    # PLOT 6: The Capacity Wall (Footprint vs Bandwidth)
    # ==========================================
    subset6 = df[df['Kernel'] == 'RS_SEQ_COPY'].copy()
    if not subset6.empty:
        # Footprint = Ring * Depth * Batch * 8 bytes * 2 arrays
        subset6['Footprint_MB'] = (subset6['RingDim'] * subset6['Depth'] * 40 * 8 * 2) / (1024**2)
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=subset6, x='Footprint_MB', y='GiB_s', marker='D', markersize=8, color='purple', linewidth=2.5)
        plt.xscale('log', base=10)
        plt.title('Plot 6: The Capacity Wall\nBandwidth vs. Memory Footprint (RS_SEQ_COPY)', fontweight='bold')
        plt.ylabel('Bandwidth (GiB/s)', fontweight='bold')
        plt.xlabel('Working Set Size (MB, Log Scale)', fontweight='bold')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig('plot6_capacity.png', dpi=300)
        print(" -> Saved 'plot6_capacity.png'")

if __name__ == "__main__":
    generate_plots()