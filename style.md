# FHE-RaiderSTREAM Visualization Style Guide  
This document establishes the definitive visual standards for all performance benchmarking plots within the FHE-RaiderSTREAM project. Adherence to these guidelines ensures publication-quality consistency, technical accuracy, and professional aesthetic across all generated figures.  
---  
## 1. Core Philosophy  
Visualizations must prioritize **clarity**, **reproducibility**, and **academic rigor**. Every chart must explicitly state its experimental context and maintain a strict hierarchical layering of data over auxiliary elements.  
---  
## 2. Typography & Color  
### 2.1 Font Selection & Styling  
- **Academic Standard:** Use a serif font stack to ensure professional legibility in papers and reports.  
- **Priority Stack:** `Times New Roman` -> `Times` -> `DejaVu Serif` -> `Liberation Serif` -> `serif`.  
- **Colors:** All text, axis lines, and tick marks must be **pure black** (`#000000`).  
- **Sizing:** Base font size should be increased by **3 points** relative to standard defaults (e.g., use `13pt` for labels if `10pt` was baseline) to improve legibility in complex layouts.
**Implementation:**  
```python  
def configure_matplotlib() -> None:  
"""Configure matplotlib with serif font and academic color styling."""  
available_fonts = {font.name for font in font_manager.fontManager.ttflist}  
serif_fonts = ["Times New Roman"] if "Times New Roman" in available_fonts else []  
serif_fonts.extend(["Times", "DejaVu Serif", "Liberation Serif", "serif"])  
plt.rcParams.update(  
{  
"font.family": "serif",  
"font.serif": serif_fonts,  
"font.size": 13, # Increased by 3pts from default 10
"text.color": "black",  
"axes.edgecolor": "black",  
"axes.labelcolor": "black",  
"xtick.color": "black",  
"ytick.color": "black",  
}  
)  
```  
### 2.2 Text Weights  
- **Titles:** The primary title should be rendered in **bold** to establish clear hierarchy.  
- **Axis Labels:** Standard weight (Normal).  
**Implementation:**  
```python  
plt.title(f'MPI Scaling: Aggregate System Bandwidth\n(..., BatchSize={batch})', fontweight='bold')  
```  
---  
## 3. Layout & Structure  
### 3.1 Dual X-Axes (Scaling Plots)  
MPI and distributed performance charts MUST utilize a dual x-axis configuration to show both physical ranks and total core counts.  
**Implementation:**  
```python  
# Create primary axis (MPI Ranks)  
fig, ax1 = plt.subplots(figsize=(10, 7))  
ax1.set_xlabel('Number of MPI Ranks')  
# Add secondary x-axis for Cores  
ax2 = ax1.twiny()  
ax2.set_xlim(ax1.get_xlim())  
ax2.set_xticks(range(len(ranks)))  
ax2.set_xticklabels(cores)  
ax2.set_xlabel('Total CPU Cores')  
ax2.grid(False) # Disable redundant grid  
```  
### 3.2 Y-Axis & Headroom  
Always provide **15% headroom** above the maximum data point to prevent labels from clipping.  
**Implementation:**  
```python  
ax1.set_ylim(0, max(agg_bw) * 1.15)  
```  
---  
## 4. Rendering & Layering (Z-Order)  
Strict control of the rendering stack ensures that background elements never obscure data.  
### 4.1 Strict Layering Order  
- **Z-Order 0:** Background grid.  
- **Z-Order 3:** Primary data bars.  
- **Z-Order 4+:** Data labels.  
**Implementation:**  
```python  
# Force grid behind EVERYTHING  
ax1.set_axisbelow(True)  
# Plot bars with explicit zorder  
bars = ax1.bar(range(len(ranks)), agg_bw, zorder=3, color='midnightblue', ...)  
```  
---  
## 5. Annotations & Units  
### 5.1 Unit Adaptive Logic  
Convert to **TB/s** automatically for high-bandwidth results to keep labels concise.  
**Implementation:**  
```python  
for bar in bars:  
height = bar.get_height()  
label = f'{height/1000:.1f} TB/s' if height > 1000 else f'{height:.0f}'  
# Vertical offset (approx 0.18lh - 0.2lh)  
ax1.text(bar.get_x() + bar.get_width()/2., height + max(agg_bw)*0.018,  
label, ha='center', va='bottom')  
```  
---  
## 6. Export Standards  
Export to **PDF** using vector-based settings and a tight bounding box.  
**Implementation:**  
```python  
plt.tight_layout()  
output_path = 'results/mpi_scaling_analysis.pdf'  
plt.savefig(output_path, dpi=300, bbox_inches='tight')  
```