import matplotlib as mpl

# colorblind-friendly palette
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow  — avoid on white background
    "#000000",  # black
]

mpl.rcParams.update({
    # Font — must match your LaTeX document
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,

    # Layout
    "axes.spines.top": False,       # remove top spine
    "axes.spines.right": False,     # remove right spine
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",

    # Output
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
})