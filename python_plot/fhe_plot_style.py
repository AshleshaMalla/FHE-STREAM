import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

# Standard colors for consistency across plots
COLOR_DCRT = "#377eb8"    # Blue
COLOR_CT = "#ff7f00"      # Orange
COLOR_HEXL = "#4daf4a"    # Green
COLOR_REGULAR = "#e41a1c" # Red
COLOR_MPI = "midnightblue"

# Standard hatching patterns for B&W legibility
HATCH_PATTERNS = ["", "//", "\\\\", "xx", "..", "++", "//\\\\"]

# Standard figure sizes (width, height) in inches
# IEEE Column Width is ~3.3 to 3.5 inches.
FIGSIZE_SINGLE = (3.5, 2.6)  # Single column
FIGSIZE_WIDE = (7.0, 3.8)    # Double column or detailed plots

def configure_matplotlib() -> None:
    """Configure matplotlib with serif font and academic color styling optimized for IEEE column width."""
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    serif_fonts = ["Times New Roman"] if "Times New Roman" in available_fonts else []
    serif_fonts.extend(["Times", "DejaVu Serif", "Liberation Serif", "serif"])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": serif_fonts,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "text.color": "black",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": "gray",
            "axes.axisbelow": True,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.figsize": FIGSIZE_SINGLE,
            "pdf.fonttype": 42, # Ensure fonts are embedded
            "ps.fonttype": 42,
            "hatch.linewidth": 0.5,
            "hatch.color": "white",
        }
    )

def add_value_labels(ax, bars, max_val=None, vertical=True):
    """Add value labels on top of bars with adaptive unit logic (GB/s to TB/s)."""
    if max_val is None:
        # Try to find max height from bars
        try:
            max_val = max(bar.get_height() for bar in bars)
        except:
            max_val = 1.0

    for bar in bars:
        height = bar.get_height()
        if vertical:
            label = f'{height/1000:.1f}T' if height > 1000 else f'{height:.1f}'
            ax.text(bar.get_x() + bar.get_width()/2., height + max_val * 0.03,
                    label, ha='center', va='bottom', fontsize=6.5, fontweight='bold')
        else:
            width = bar.get_width()
            label = f'{width/1000:.1f}T' if width > 1000 else f'{width:.1f}'
            ax.text(width + max_val * 0.03, bar.get_y() + bar.get_height()/2.,
                    label, ha='left', va='center', fontsize=6.5, fontweight='bold')

def save_plot(filename: str, dpi: int = 300):
    """Save the plot to the results directory as a PDF with tight layout."""
    # Ensure results directory exists
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f" -> Saved '{filename}'")

def setup_dual_xaxis(ax1, ranks, cores, xlabel_ranks='Number of MPI Ranks', xlabel_cores='Total CPU Cores'):
    """Setup dual x-axis for MPI scaling plots."""
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(range(len(ranks)))
    ax2.set_xticklabels(cores)
    ax2.set_xlabel(xlabel_cores)
    ax2.grid(False)
    return ax2
