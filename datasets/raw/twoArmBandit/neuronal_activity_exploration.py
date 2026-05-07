# %% [markdown]
# # Two-Arm Bandit: Neuronal Activity Exploration
# 
# A deep visual exploration of the NeuroPixels neuronal activity data — raw spikes, quality metrics, probe anatomy, firing rates, population dynamics, trial-aligned responses, correlation structure, and dimensionality.
# 
# **Primary session:** `JPAS_0023_20230922`  
# **Multi-session comparisons** where lightweight.  
# Behavioral context is used as an overlay only — this notebook is about the **neurons**.

# %%
# %load_ext autoreload  # IPython magic — not needed when running as a script
# %autoreload 2

from pathlib import Path
import json
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.ndimage import gaussian_filter1d
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ncmcm.data_loaders.bandit_task import BanditTaskNeuroPixelsDataset
from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural_plotly

# ── Session discovery ──────────────────────────────────────────────────────────
DATA_ROOT = Path("/home/kerim/Projects/Neural Algorithms/NC-MCM/datasets/raw/twoArmBandit")
SESSION_DIRS = sorted({
    p.parent
    for p in DATA_ROOT.glob("**/metrics.json")
    if (p.parent / "cluster_info.tsv").exists()
})
PRIMARY = SESSION_DIRS[0]

print(f"Found {len(SESSION_DIRS)} sessions:")
for s in SESSION_DIRS:
    print(f"  {s.name}")
print(f"\nPrimary session: {PRIMARY.name}")

# %%
# ── HTML export directory ────────────────────────────────────────────────────
OUT_DIR = DATA_ROOT / "neuronal_activity_exploration"
OUT_DIR.mkdir(exist_ok=True)
print(f"HTML exports → {OUT_DIR}")

# ── Per-figure caption text ───────────────────────────────────────────────────
_FIGURE_CAPTIONS = {
    "01_session_summary.html": (
        "Cross-session comparison of three key metrics: total good-quality neurons (Kilosort 'good' label), "
        "mean population firing rate, and total recording duration. Each bar group is one session. "
        "Built from cluster_info.tsv and spike_times_milliseconds_sync_to_behav.npy."
    ),
    "02_probe_anatomy.html": (
        "Physical placement of neurons on the Neuropixels probe. Grey dots = all 384 electrode channels; "
        "colored dots = good neurons at their recorded depth (um) and lateral position on the probe. "
        "Color encodes firing rate (Hz); dot size encodes SNR. Hover for cluster ID, rate, SNR, depth. "
        "Source: channel_positions.npy and cluster_info.tsv."
    ),
    "03_neuron_quality_metrics.html": (
        "Six spike-sorting quality metrics for all good neurons, each as a violin + box + outlier plot: "
        "SNR, firing rate (Hz), ISI violation ratio, contamination %, waveform amplitude, and isolation distance. "
        "Wider violin = more neurons at that value. Source: cluster_info.tsv quality columns."
    ),
    "04_depth_vs_firing_rate.html": (
        "Scatter plot of good neurons: X = firing rate (Hz), Y = probe depth (um), color = SNR, "
        "dot size = spike count (clipped at 95th percentile). "
        "Reveals whether deeper or shallower neurons fire faster or are better isolated. Source: cluster_info.tsv."
    ),
    "05_firing_rate_distribution.html": (
        "Three views of the firing-rate distribution across all good neurons: "
        "left = log-scale histogram (most neurons fire below 10 Hz), "
        "middle = neurons ranked from highest to lowest rate, "
        "right = CDF with 50th/80th/95th percentile markers. Source: cluster_info.tsv."
    ),
    "06_multiresolution_sampling.html": (
        "The most active neuron shown in a 10-second window, binned at four temporal resolutions: "
        "5, 30, 100, and 300 Hz. Higher frequency = finer temporal structure but noisier spike counts per bin. "
        "Shows how sampling frequency trades temporal precision against signal smoothness."
    ),
    "07_binning_methods.html": (
        "The same neuron in a 20-second window at 30 Hz, processed with four strategies: "
        "binary (1 if any spike), raw count, rate (spikes/s), and Gaussian-smoothed rate. "
        "Shows how the preprocessing choice shapes the neural signal fed to decoders or BunDLe-Net."
    ),
    "08_spike_raster.html": (
        "Classic spike raster of the first 2 minutes. Each tick = one spike; rows = neurons sorted by probe depth "
        "(deepest at bottom). Colored background bands show behavioral states "
        "(hold, choosing left/right, reward, intertrial) from metrics.json. "
        "Source: spike_times_milliseconds_sync_to_behav.npy and spike_clusters.npy."
    ),
    "09_population_activity.html": (
        "Two stacked panels sharing a time axis: (top) population-mean firing rate across all neurons smoothed "
        "with a 2-second Gaussian kernel; (bottom) per-neuron normalized activity heatmap, neurons sorted by mean rate. "
        "Vertical lines mark reward-probability block transitions. Built from the 30 Hz spike-count matrix."
    ),
    "10_psth_heatmap.html": (
        "Trial-averaged population response aligned to the moment of arm choice (t = 0). "
        "Top: all neurons sorted by peak-response latency (pre-choice ramps at top, post-choice responders at bottom), "
        "row-normalized to [0, 1]. Bottom: population mean +/- SEM (Hz). Averaged over all valid trials."
    ),
    "11_psth_correct_vs_incorrect.html": (
        "Population-mean PSTH aligned to choice time, split by trial outcome: rewarded (green) vs not rewarded (red). "
        "Each trace = mean firing rate (Hz) across all neurons, Gaussian-smoothed. "
        "Shows whether the population fires at a different level or timing on correct vs incorrect trials."
    ),
    "12_correlation_heatmap.html": (
        "Pearson correlation matrix of all neuron pairs, reordered by hierarchical Ward clustering so that "
        "functionally similar (co-active) neurons appear adjacent. Red = positive, blue = negative; "
        "diagonal masked. Computed on z-scored 30 Hz activity subsampled 3x in time."
    ),
    "13_correlation_distribution.html": (
        "Left: histogram of all off-diagonal pairwise Pearson correlations. "
        "Right: bar chart of the 20 most correlated neuron pairs. "
        "The bulk of pairs cluster near 0; a small fraction of tightly coupled cells deviate significantly above."
    ),
    "14_top_correlated_pairs.html": (
        "Activity traces of the 5 most correlated neuron pairs over the first 60 seconds, "
        "smoothed with a 0.5-second Gaussian kernel. Blue = neuron A, red = neuron B. "
        "Provides a concrete look at what functional coupling looks like in the continuous signal."
    ),
    "15_pca_scree.html": (
        "Left: scree plot of variance explained by each of the first 30 PCs. "
        "Right: cumulative variance curve with markers at the number of PCs needed for 80/90/95% coverage. "
        "PCA is run on the z-scored 30 Hz population matrix (n_neurons x T/3). "
        "Shows how concentrated or distributed the population's activity structure is."
    ),
    "16_pca_pc1_vs_pc2.html": (
        "Each subsampled timepoint projected onto the first two principal components, "
        "colored by behavioral state (hold, choosing left/right, reward, intertrial). "
        "Reveals whether distinct behavioral contexts occupy geometrically separate regions "
        "of the low-dimensional neural state space."
    ),
    "17_pca_3d.html": (
        "Same as the PC1 vs PC2 scatter extended to three dimensions (PC1 x PC2 x PC3). "
        "The interactive 3D plot can be rotated. Each point = one subsampled timepoint, colored by behavioral state. "
        "PC3 captures variance not visible in 2D; total variance explained is shown in the title."
    ),
    "18_laminar_profile.html": (
        "Left: mean population firing rate per depth bin (10 equal-width windows, shallowest to deepest) "
        "as a horizontal bar chart. Right: neuron count per depth bin. "
        "Reveals whether spiking activity is concentrated at specific probe depths or uniformly distributed."
    ),
    "19_depth_time_heatmap.html": (
        "2D heatmap with depth bins as rows and time as columns (downsampled 10x, Gaussian-smoothed). "
        "Color = mean activity per depth bin. Block transitions are overlaid as vertical lines. "
        "Shows whether behavioral-context modulation of activity differs across depth (layer-specific effects)."
    ),
    "20_autocorrelation.html": (
        "Autocorrelation function (0-500 ms lag, computed via FFT) of the 8 most active neurons. "
        "A sharp peak at lag=0 with fast decay = Poisson-like. Oscillatory side-peaks = bursting or rhythmic firing. "
        "Helps classify neurons as tonic (regular rate) vs bursty."
    ),
    "21_burstiness.html": (
        "Left: distribution of the burstiness index B = (sigma - mu) / (sigma + mu) of inter-spike intervals "
        "across all good neurons (B > 0 = bursty, B = 0 = Poisson, B < 0 = regular). "
        "Right: distribution of CV of ISIs (CV = sigma/mu; CV = 1 is Poisson). "
        "Both computed from raw spike times per neuron."
    ),
    "22_isi_distribution.html": (
        "Log-scale histogram of all inter-spike intervals (ISIs) from all good neurons pooled "
        "(log-spaced bins 0.1-10000 ms). The red band at t < 2 ms marks the absolute refractory period; "
        "spikes here indicate spike-sorting contamination. "
        "The peak and shape reflect typical firing statistics of this population."
    ),
    "23_umap.html": (
        "UMAP (n_neighbors=30, min_dist=0.1) of the z-scored population activity matrix, "
        "subsampled to 8000 timepoints, colored by behavioral state. "
        "Unlike PCA, UMAP preserves local neighborhood structure and reveals cluster geometry, "
        "manifold topology, and state boundaries that linear projections collapse."
    ),
    "24_neural_trajectory.html": (
        "The population state vector traced through PC1 x PC2 space in chronological order. "
        "Each dot = one timepoint; color = elapsed time (dark = session start, bright = end). "
        "Diamonds mark reward-probability block transitions; start and end points are labeled. "
        "Reveals drifts, loops, and resets in population dynamics across the session."
    ),
    "25_block_psth.html": (
        "Population mean firing rate aligned to each reward-probability block transition (t = 0), "
        "covering 30 s before to 60 s after the switch. "
        "Thin semi-transparent lines = individual transition snippets; thick lines = condition means. "
        "With only 2 transitions per direction in this session, individual traces are shown to avoid misleading averages."
    ),
    "26_fano_factor.html": (
        "Population Fano factor (variance / mean spike count per neuron) computed in 5-second sliding windows "
        "with 1-second steps across the full session. Mean (blue) and median (orange dashed) across neurons are shown. "
        "Dashed reference at F = 1 is the Poisson expectation. Block transitions marked. "
        "F > 1 = overdispersed / bursty; F < 1 = underdispersed / quenched variability."
    ),
    "27_singletrial_psth.html": (
        "Every valid trial as one row in a heatmap (population-mean activity per bin, smoothed, row-normalized). "
        "Rows sorted by reaction time (fast at top, slow at bottom). "
        "Right panel: RT distribution with outcome color (green = rewarded, red = not). "
        "Unlike the trial-averaged PSTH, this reveals trial-to-trial variability and whether fast vs slow choices "
        "differ in their pre-choice population state."
    ),
    "28_decoder.html": (
        "Normalized inter-trial decoder: can the population predict left vs right choice across the full "
        "interval between trials? The x-axis runs from the previous choice (\u03c4=0) through the current "
        "choice (\u03c4=0.5, always) to the next choice (\u03c4=1). The left half [0, 0.5] is independently "
        "normalized to the prev\u2192curr ITI; the right half [0.5, 1] to the curr\u2192next ITI. "
        "At each of 100 grid points, a logistic regression decodes left/right from the full population "
        "vector (z-scored, 5-fold stratified CV). The curve reveals when (relative to neighboring choices) "
        "the choice signal first becomes decodable above the 50% chance baseline."
    ),
}

_BANNER_STYLE = (
    "background:#f0f4ff;border-left:5px solid #4a7fdc;padding:14px 20px 12px;"
    "margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
    "font-size:0.9rem;color:#1a202c;line-height:1.55"
)


def write_html_with_caption(fig, path):
    """Write a Plotly figure to HTML, injecting a descriptive caption banner at the top of the page."""
    fname = Path(path).name
    caption = _FIGURE_CAPTIONS.get(fname, "")
    html = fig.to_html(include_plotlyjs=True, full_html=True)
    if caption:
        banner = (
            f'<div style="{_BANNER_STYLE}">'
            f"<strong>What this shows:</strong> {caption}"
            f"</div>"
        )
        html = html.replace("<body>", "<body>\n" + banner, 1)
    Path(path).write_text(html, encoding="utf-8")


# %% [markdown]
# ---
# ## 0 · Session Overview
# A quick cross-session summary before we dive in.

# %%
rows = []
for sdir in SESSION_DIRS:
    ci = pd.read_csv(sdir / "cluster_info.tsv", sep="\t")
    good = ci[ci["group"] == "good"]
    st = np.load(sdir / "spike_times_milliseconds_sync_to_behav.npy")
    with open(sdir / "metrics.json") as f:
        m = json.load(f)
    n_trials = len(m.get("metrics", {}).get("trials", []))
    recording_min = (st.max() - st.min()) / 60_000
    rows.append({
        "Session": sdir.name,
        "Good neurons": len(good),
        "Total neurons": len(ci),
        "Trials": n_trials,
        "Recording (min)": round(recording_min, 1),
        "Mean FR (Hz)": round(good["firing_rate"].mean(), 2),
        "Median FR (Hz)": round(good["firing_rate"].median(), 2),
        "Mean SNR": round(good["snr"].mean(), 2) if "snr" in good.columns else "–",
    })

overview_df = pd.DataFrame(rows)
print(overview_df.to_string(index=False))

# %%
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=["Good neurons per session", "Mean firing rate (Hz)", "Recording duration (min)"],
)
colors = px.colors.qualitative.Set2[:len(overview_df)]

for col_idx, (col, unit) in enumerate([
    ("Good neurons", ""),
    ("Mean FR (Hz)", " Hz"),
    ("Recording (min)", " min"),
], start=1):
    fig.add_trace(
        go.Bar(
            x=overview_df["Session"].str[-8:],
            y=overview_df[col],
            marker_color=colors,
            text=[f"{v}{unit}" for v in overview_df[col]],
            textposition="outside",
            showlegend=False,
        ),
        row=1, col=col_idx,
    )

fig.update_layout(height=380, title_text="Session Summary", template="plotly_white")
write_html_with_caption(fig, OUT_DIR / "01_session_summary.html")

# %% [markdown]
# ---
# ## 1 · Probe Anatomy & Neuron Quality
# 
# Where are the recorded neurons on the probe, and how good are they?

# %%
ci = pd.read_csv(PRIMARY / "cluster_info.tsv", sep="\t")
good = ci[ci["group"] == "good"].copy().reset_index(drop=True)

channel_positions = np.load(PRIMARY / "channel_positions.npy")

# Map each good neuron's channel index to a physical y-position
ch_to_y = {i: channel_positions[i, 1] for i in range(len(channel_positions))}
good["probe_y"] = good["ch"].map(ch_to_y)
good["probe_x"] = good["ch"].map({i: channel_positions[i, 0] for i in range(len(channel_positions))})

print(f"Good neurons: {len(good)}  |  Probe depth range: {channel_positions[:,1].min():.0f} – {channel_positions[:,1].max():.0f} µm")

# %%
# ── All 384 channels (grey) + good neurons (colored by firing rate) ─────────
fig = go.Figure()

# All channels as grey dots
fig.add_trace(go.Scatter(
    x=channel_positions[:, 0],
    y=channel_positions[:, 1],
    mode="markers",
    marker=dict(size=4, color="lightgrey", line=dict(width=0)),
    name="All channels",
    hoverinfo="skip",
))

# Good neurons colored by firing rate, size by SNR
snr_vals = good["snr"].fillna(0).clip(0, 20)  # fill NaN before clipping for sizing
fig.add_trace(go.Scatter(
    x=good["probe_x"],
    y=good["probe_y"],
    mode="markers",
    marker=dict(
        size=4 + 10 * (snr_vals / snr_vals.max()) if snr_vals.max() > 0 else 4,
        color=good["firing_rate"],
        colorscale="Plasma",
        colorbar=dict(title="Firing rate (Hz)", thickness=14),
        cmin=0, cmax=good["firing_rate"].quantile(0.97),
        line=dict(width=0.4, color="black"),
    ),
    customdata=np.column_stack([good["cluster_id"], good["firing_rate"].round(2), snr_vals.round(2), good["depth"].round(0)]),
    hovertemplate="Cluster %{customdata[0]}<br>FR: %{customdata[1]} Hz<br>SNR: %{customdata[2]}<br>Depth: %{customdata[3]} µm<extra></extra>",
    name="Good neurons",
))

fig.update_layout(
    title="Probe Anatomy — Good Neurons (color=FR, size=SNR)",
    xaxis_title="Probe x (µm)",
    yaxis_title="Probe depth (µm)",
    width=420,
    height=780,
    template="plotly_white",
    legend=dict(x=1.15, y=1),
)
write_html_with_caption(fig, OUT_DIR / "02_probe_anatomy.html")

# %%
metrics_to_plot = [
    ("snr", "SNR"),
    ("firing_rate", "Firing rate (Hz)"),
    ("isi_violations_ratio", "ISI violation ratio"),
    ("ContamPct", "Contamination %"),
    ("amplitude_median", "Amplitude (µV)"),
    ("isolation_distance", "Isolation distance"),
]
# Drop metrics that don't exist in this file
metrics_to_plot = [(col, lbl) for col, lbl in metrics_to_plot if col in good.columns]

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=[lbl for _, lbl in metrics_to_plot],
    vertical_spacing=0.14, horizontal_spacing=0.08,
)
palette = px.colors.qualitative.Set2

for i, (col, lbl) in enumerate(metrics_to_plot):
    r, c = divmod(i, 3)
    vals = good[col].dropna()
    fig.add_trace(
        go.Violin(
            y=vals,
            box_visible=True,
            meanline_visible=True,
            line_color=palette[i % len(palette)],
            fillcolor=palette[i % len(palette)],
            opacity=0.6,
            name=lbl,
            showlegend=False,
            points="outliers",
        ),
        row=r + 1, col=c + 1,
    )

fig.update_layout(
    height=560,
    title_text=f"Neuron Quality Metrics — {PRIMARY.name}",
    template="plotly_white",
)
write_html_with_caption(fig, OUT_DIR / "03_neuron_quality_metrics.html")

# %%
# Depth vs firing rate scatter, color = SNR
fig = px.scatter(
    good,
    x="firing_rate",
    y="probe_y",
    color="snr",
    size=good["n_spikes"].clip(upper=good["n_spikes"].quantile(0.95)) / good["n_spikes"].max() * 18 + 4,
    color_continuous_scale="Viridis",
    hover_data=["cluster_id", "firing_rate", "snr", "isi_violations_ratio"],
    labels={"firing_rate": "Firing rate (Hz)", "probe_y": "Probe depth (µm)", "snr": "SNR"},
    title="Depth vs Firing Rate (color=SNR, size=n_spikes)",
    template="plotly_white",
)
fig.update_layout(height=540, width=620)
write_html_with_caption(fig, OUT_DIR / "04_depth_vs_firing_rate.html")

# %% [markdown]
# ---
# ## 2 · Firing Rate Distribution
# 
# How are firing rates spread across the population? Most neurons are sparse.

# %%
fr = good["firing_rate"].dropna().values

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[
        "Histogram (log x-axis)",
        "Neurons ranked by firing rate",
        "CDF of firing rates",
    ],
)

# Histogram (log scale on x)
fig.add_trace(
    go.Histogram(
        x=np.log10(fr + 1e-3),
        nbinsx=50,
        marker_color="steelblue",
        name="FR distribution",
        showlegend=False,
    ),
    row=1, col=1,
)
fig.update_xaxes(title_text="log₁₀(FR + 0.001) Hz", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=1)

# Ranked bar (sorted)
fr_sorted = np.sort(fr)[::-1]
fig.add_trace(
    go.Bar(
        x=np.arange(len(fr_sorted)),
        y=fr_sorted,
        marker=dict(
            color=fr_sorted,
            colorscale="Plasma",
            showscale=False,
        ),
        name="Neurons ranked",
        showlegend=False,
    ),
    row=1, col=2,
)
fig.update_xaxes(title_text="Neuron rank", row=1, col=2)
fig.update_yaxes(title_text="Firing rate (Hz)", row=1, col=2)

# CDF
fr_cdf = np.sort(fr)
cdf = np.arange(1, len(fr_cdf) + 1) / len(fr_cdf)
fig.add_trace(
    go.Scatter(
        x=fr_cdf, y=cdf,
        mode="lines",
        line=dict(color="darkorange", width=2),
        name="CDF",
        showlegend=False,
    ),
    row=1, col=3,
)
# Add percentile markers
for p in [50, 80, 95]:
    val = np.percentile(fr_cdf, p)
    fig.add_vline(x=val, line_dash="dot", line_color="grey", line_width=1, row=1, col=3)
    fig.add_annotation(
        x=val, y=p / 100 + 0.04, xref="x3", yref="y3",
        text=f"{p}th\n{val:.1f} Hz", showarrow=False,
        font=dict(size=10, color="grey"),
    )
fig.update_xaxes(title_text="Firing rate (Hz)", row=1, col=3)
fig.update_yaxes(title_text="Cumulative fraction", row=1, col=3)

fig.update_layout(
    height=380,
    title_text=f"Firing Rate Distribution — {PRIMARY.name}",
    template="plotly_white",
)
write_html_with_caption(fig, OUT_DIR / "05_firing_rate_distribution.html")

print(f"Firing rate stats:  mean={fr.mean():.2f} Hz  median={np.median(fr):.2f} Hz  p95={np.percentile(fr,95):.2f} Hz  max={fr.max():.2f} Hz")

# %% [markdown]
# ---
# ## 3 · Multi-Resolution Sampling
# 
# The same spikes can be binned at different temporal resolutions. Here we compare 5, 30, 100, and 300 Hz  
# and show how the same 10-second window looks at each resolution.

# %%
FREQS = [5, 30, 100, 300]
datasets_by_fs = {}

for fs in FREQS:
    print(f"Loading {fs} Hz ...", end=" ")
    ds = BanditTaskNeuroPixelsDataset(
        data_path=PRIMARY,
        downsample_fs=fs,
        downsample_method="count",
        good_neurons_only=True,
        normalize_method=None,
        choosing_state_mode="correctness",
    )
    datasets_by_fs[fs] = ds
    x_dense = ds.x.toarray()
    print(f"shape={x_dense.shape}  sparsity={100*(x_dense==0).mean():.1f}%  mean_val={x_dense.mean():.4f}")

# %%
rows_fs = []
for fs, ds in datasets_by_fs.items():
    x = ds.x.toarray()
    rows_fs.append({
        "Sampling freq (Hz)": fs,
        "Timepoints (T)": x.shape[1],
        "Shape (neurons × T)": f"{x.shape[0]} × {x.shape[1]}",
        "Sparsity (%)": f"{100*(x==0).mean():.1f}",
        "Mean spikes/bin": f"{x.mean():.4f}",
        "Max spikes/bin": f"{x.max():.0f}",
        "Memory (MB)": f"{x.nbytes / 1e6:.1f}",
    })
fs_df = pd.DataFrame(rows_fs)
print(fs_df.to_string(index=False))

# %%
# ── Same neuron, same 10-second window at 4 different resolutions ───────────
WINDOW_START_S = 60      # start of zoom window (seconds into recording)
WINDOW_LEN_S   = 10      # window length in seconds

# Pick the most active neuron at 30 Hz as the reference neuron
x30 = datasets_by_fs[30].x.toarray()  # (neurons, T)
ref_neuron_idx = int(np.argmax(x30.sum(axis=1)))

fig = make_subplots(
    rows=len(FREQS), cols=1,
    subplot_titles=[f"{fs} Hz" for fs in FREQS],
    shared_xaxes=False,
    vertical_spacing=0.09,
)
colors_fs      = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"]
fill_colors_fs = ["rgba(230,57,70,0.3)", "rgba(69,123,157,0.3)", "rgba(42,157,143,0.3)", "rgba(233,196,106,0.3)"]

for row_idx, (fs, color, fill_color) in enumerate(zip(FREQS, colors_fs, fill_colors_fs), start=1):
    ds = datasets_by_fs[fs]
    x = ds.x.toarray()
    actual_fs = ds.fs

    t_start = int(WINDOW_START_S * actual_fs)
    t_end   = int((WINDOW_START_S + WINDOW_LEN_S) * actual_fs)
    t_end   = min(t_end, x.shape[1])

    neuron_trace = x[ref_neuron_idx, t_start:t_end]
    time_axis_s  = np.arange(len(neuron_trace)) / actual_fs + WINDOW_START_S

    # Filled-area traces render correctly at any bin width (no sub-pixel bar washout)
    fig.add_trace(
        go.Scatter(
            x=time_axis_s,
            y=neuron_trace,
            mode="lines",
            line=dict(color=color, width=1.2),
            fill="tozeroy",
            fillcolor=fill_color,
            name=f"{fs} Hz",
            showlegend=False,
        ),
        row=row_idx, col=1,
    )
    fig.update_yaxes(title_text="Spikes/bin", row=row_idx, col=1)

fig.update_xaxes(title_text="Time (s)", row=len(FREQS), col=1)
fig.update_layout(
    height=80 + 160 * len(FREQS),
    title_text=f"Same neuron (#{ref_neuron_idx}) — {WINDOW_LEN_S}s window at 4 sampling frequencies",
    template="plotly_white",
)
write_html_with_caption(fig, OUT_DIR / "06_multiresolution_sampling.html")

# %%
# ── Compare binning methods at 30 Hz ────────────────────────────────────────
METHODS = ["binary", "count", "rate", "gaussian"]
method_labels = {"binary": "Binary (spike/no-spike)", "count": "Count (spikes/bin)", "rate": "Rate (spikes/s)", "gaussian": "Gaussian-smoothed"}
colors_m = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"]

datasets_by_method = {}
for method in METHODS:
    ds = BanditTaskNeuroPixelsDataset(
        data_path=PRIMARY,
        downsample_fs=30,
        downsample_method=method,
        good_neurons_only=True,
        normalize_method=None,
        choosing_state_mode="correctness",
    )
    datasets_by_method[method] = ds

fig = make_subplots(
    rows=len(METHODS), cols=1,
    subplot_titles=[method_labels[m] for m in METHODS],
    shared_xaxes=True,
    vertical_spacing=0.07,
)
WIN_START_S = 120
WIN_LEN_S   = 20

for row_idx, (method, color) in enumerate(zip(METHODS, colors_m), start=1):
    ds = datasets_by_method[method]
    x = ds.x.toarray()
    actual_fs = ds.fs
    t_s = int(WIN_START_S * actual_fs)
    t_e = min(int((WIN_START_S + WIN_LEN_S) * actual_fs), x.shape[1])
    trace = x[ref_neuron_idx, t_s:t_e]
    t_ax  = np.arange(len(trace)) / actual_fs + WIN_START_S
    fig.add_trace(
        go.Scatter(
            x=t_ax, y=trace,
            mode="lines",
            line=dict(color=color, width=1.5),
            name=method_labels[method],
            showlegend=False,
        ),
        row=row_idx, col=1,
    )
    fig.update_yaxes(title_text="Value", row=row_idx, col=1)

fig.update_xaxes(title_text="Time (s)", row=len(METHODS), col=1)
fig.update_layout(
    height=80 + 160 * len(METHODS),
    title_text=f"Binning method comparison — neuron #{ref_neuron_idx} @ 30 Hz, {WIN_LEN_S}s window",
    template="plotly_white",
)
write_html_with_caption(fig, OUT_DIR / "07_binning_methods.html")

# %% [markdown]
# ---
# ## 4 · Spike Raster
# 
# Classic spike raster using raw spike times. Neurons are sorted by probe depth (deepest at bottom).  
# Colored bands show the current behavioral state.

# %%
# ── Load raw data ───────────────────────────────────────────────────────────
spike_times_ms = np.load(PRIMARY / "spike_times_milliseconds_sync_to_behav.npy").flatten()
spike_clusters = np.load(PRIMARY / "spike_clusters.npy").flatten()

with open(PRIMARY / "metrics.json") as f:
    metrics = json.load(f)

# ── Get good neuron cluster IDs, sort by depth ──────────────────────────────
ci = pd.read_csv(PRIMARY / "cluster_info.tsv", sep="\t")
good_ci = ci[ci["group"] == "good"].copy()
good_ci = good_ci.sort_values("depth").reset_index(drop=True)
good_ids = set(good_ci["cluster_id"].values)
id_to_rank = {cid: rank for rank, cid in enumerate(good_ci["cluster_id"])}

# ── Filter to first 2 minutes of recording ─────────────────────────────────
RASTER_WINDOW_MIN = 2
t_max_ms = spike_times_ms.min() + RASTER_WINDOW_MIN * 60_000
mask = (spike_clusters >= 0) & np.isin(spike_clusters, list(good_ids)) & (spike_times_ms <= t_max_ms)
st_filt = spike_times_ms[mask] / 1000  # → seconds
sc_filt = spike_clusters[mask]
neuron_y = np.array([id_to_rank[c] for c in sc_filt])

print(f"Spikes in first {RASTER_WINDOW_MIN} min: {len(st_filt):,}  across {len(good_ids)} neurons")

# %%
# ── Build behavioral state color bands ─────────────────────────────────────
STATE_COLORS = {
    "choosing": "rgba(100,160,240,0.15)",
    "correct": "rgba(60,180,80,0.20)",
    "reward":     "rgba(255,200,0,0.30)",
    "choosing":   "rgba(100,160,240,0.30)",
    "hold":       "rgba(160,100,220,0.25)",
    "waiting":    "rgba(100,200,180,0.20)",
    "delay":      "rgba(200,200,100,0.15)",
    "intertrial": "rgba(200,200,200,0.10)",
}
t0_s = spike_times_ms.min() / 1000
t_max_s = t_max_ms / 1000
window_len_s = t_max_s - t0_s   # relative window end (e.g. 120 s)

states_data_raw = metrics.get("metrics", {}).get("states", [])
# Normalize entries: support both [t_ms, state_name] lists and {"t": ..., "state_name": ...} dicts
def _normalize_state(s):
    if isinstance(s, dict) and "t" in s:
        return s
    if isinstance(s, (list, tuple)) and len(s) >= 2:
        return {"t": s[0], "state_name": str(s[1])}
    return None

# Sort by time — raw data can be unsorted (outlier entries cause wrong t_end_s otherwise)
states_data = sorted(
    [n for s in states_data_raw if (n := _normalize_state(s)) is not None],
    key=lambda s: s["t"]
)

shapes = []
for i, state in enumerate(states_data):
    t_start_s = state["t"] / 1000 - t0_s
    t_end_s   = (states_data[i + 1]["t"] / 1000 - t0_s) if i + 1 < len(states_data) else window_len_s
    if t_start_s >= window_len_s:   # state is entirely after the window
        break
    if t_end_s <= 0:               # state is entirely before the window — skip
        continue
    t_start_s = max(t_start_s, 0)  # clip to window start
    t_end_s   = min(t_end_s, window_len_s)
    name = state.get("state_name", state.get("state", "")).lower()
    color = next((v for k, v in STATE_COLORS.items() if k in name), "rgba(200,200,200,0.08)")
    shapes.append(dict(
        type="rect",
        x0=t_start_s, x1=t_end_s,
        y0=0, y1=len(good_ids),
        fillcolor=color,
        line=dict(width=0),
        layer="below",
    ))

print(f"State entries found: {len(states_data)}  |  Shape rectangles built: {len(shapes)}")

# ── Raster figure ──────────────────────────────────────────────────────────
st_plot = st_filt - t0_s  # relative seconds

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=st_plot,
    y=neuron_y,
    mode="markers",
    marker=dict(symbol="line-ns", size=3, line=dict(color="black", width=0.5)),
    hoverinfo="skip",
    name="Spikes",
))

# Add depth tick labels every ~10th neuron
tick_every = max(1, len(good_ci) // 10)
tick_vals = list(range(0, len(good_ci), tick_every))
tick_text = [f"{good_ci.iloc[i]['depth']:.0f} µm" for i in tick_vals]

fig.update_layout(
    shapes=shapes,
    xaxis=dict(title="Time (s)", range=[0, window_len_s]),
    yaxis=dict(title="Neuron (sorted by depth)", tickvals=tick_vals, ticktext=tick_text),
    title=f"Spike Raster — first {RASTER_WINDOW_MIN} min, {len(good_ids)} neurons, sorted by depth",
    height=600,
    width=1100,
    template="plotly_white",
    hovermode="x unified",
)

# Legend patches for state colors (annotations)
# Parse r,g,b from "rgba(R,G,B,A)" by splitting on commas
for i, (name, color) in enumerate(STATE_COLORS.items()):
    parts = color.split("(")[1].rstrip(")").split(",")
    r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
    col_idx = i % 4
    row_idx = i // 4
    fig.add_annotation(
        x=0.01 + col_idx * 0.16, y=1.07 - row_idx * 0.04,
        xref="paper", yref="paper",
        text=f"▮ {name}",
        showarrow=False,
        font=dict(size=11, color=f"rgb({r},{g},{b})"),
    )

write_html_with_caption(fig, OUT_DIR / "08_spike_raster.html")


# %% [markdown]
# ---
# ## 5 · Population Activity
# 
# How does the **population as a whole** fire over time?  
# Gaussian-smoothed population rate + per-neuron heatmap with block overlays.
# 

# %%
# Use 30 Hz dataset for population analysis
ds30 = datasets_by_fs[30]
x30 = ds30.x.toarray()          # (neurons, T)
fs30 = ds30.fs
T = x30.shape[1]
time_s = np.arange(T) / fs30

# Population firing rate = mean across neurons, smoothed
pop_rate = x30.mean(axis=0)      # mean spikes/bin per neuron
sigma_bins = int(2 * fs30)       # 2-second gaussian kernel
pop_rate_smooth = gaussian_filter1d(pop_rate.astype(float), sigma=sigma_bins)

# Block boundaries
block_labels_arr = np.array(ds30.block_labels, dtype=object)
block_changes = []
prev = None
for i, lbl in enumerate(block_labels_arr):
    if lbl is not None and lbl != prev:
        block_changes.append((i, str(lbl)))
        prev = lbl

# Trial choice events
trials = metrics.get("metrics", {}).get("trials", [])
t0_ms = spike_times_ms.min()
trial_t_s = []
for trial in trials:
    t_choice = trial.get("t chosen")
    if t_choice is not None:
        t_rel_s = (t_choice - t0_ms) / 1000
        if 0 < t_rel_s < time_s[-1]:
            trial_t_s.append(t_rel_s)

print(f"Population rate shape: {pop_rate.shape} | Block transitions: {len(block_changes)} | Trials: {len(trial_t_s)}")

# %%
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    row_heights=[1, 3],
    subplot_titles=["Population firing rate (smoothed)", "Per-neuron activation heatmap"],
)

# Row 1: Population rate
fig.add_trace(
    go.Scatter(
        x=time_s, y=pop_rate_smooth,
        mode="lines",
        line=dict(color="steelblue", width=1.5),
        name="Population rate",
        showlegend=False,
    ),
    row=1, col=1,
)

# Row 2: Heatmap
# Sort neurons by their mean firing rate for a cleaner heatmap
neuron_order = np.argsort(x30.mean(axis=1))[::-1]
x30_sorted = x30[neuron_order, :]
# Normalize each neuron to [0, 1] for visual clarity
row_max = x30_sorted.max(axis=1, keepdims=True)
row_max[row_max == 0] = 1
x30_norm = x30_sorted / row_max

fig.add_trace(
    go.Heatmap(
        z=x30_norm,
        x=time_s,
        colorscale="Inferno",
        showscale=True,
        colorbar=dict(title="Norm. activity", thickness=14, len=0.5, y=0.25),
        name="Neurons",
    ),
    row=2, col=1,
)

# Overlay block boundaries on both rows
block_palette = {"left": "green", "right": "dodgerblue"}
for t_idx, lbl in block_changes:
    t = time_s[min(t_idx, len(time_s) - 1)]
    short = "Better L" if "left" in lbl.lower() else ("Better R" if "right" in lbl.lower() else lbl)
    color = "green" if "left" in lbl.lower() else "dodgerblue"
    for row in [1, 2]:
        fig.add_vline(x=t, line_dash="solid", line_color=color, line_width=1.2, row=row, col=1)
    fig.add_annotation(
        x=t, y=1.12, xref="x", yref="paper",
        text=short, showarrow=False, font=dict(color=color, size=10),
    )

fig.update_yaxes(title_text="Mean spikes/bin/neuron", row=1, col=1)
fig.update_yaxes(title_text="Neuron (sorted by FR)", row=2, col=1)
fig.update_xaxes(title_text="Time (s)", row=2, col=1)

fig.update_layout(
    height=640,
    title_text="Population Activity — 30 Hz, Gaussian smoothed (σ=2s)",
    template="plotly_white",
    hovermode="x unified",
)
write_html_with_caption(fig, OUT_DIR / "09_population_activity.html")

# %% [markdown]
# ---
# ## 6 · Trial-Aligned Activity (PSTH)
# 
# Align spikes to the moment of each choice, then average over trials to reveal neurons that  
# ramp up, fire at choice, or respond after the outcome.

# %%
# Parameters
PRE_S  = 0.5   # seconds before choice
POST_S = 1.5   # seconds after choice
BIN_S  = 1 / fs30

n_pre  = int(PRE_S  * fs30)
n_post = int(POST_S * fs30)
n_bins = n_pre + n_post
time_axis_psth = np.linspace(-PRE_S, POST_S, n_bins)

n_neurons = x30.shape[0]
psth_mat = np.zeros((n_neurons, n_bins), dtype=float)
n_valid_trials = 0

for trial in trials:
    t_choice = trial.get("t chosen")
    if t_choice is None:
        continue
    t_rel_s = (t_choice - t0_ms) / 1000
    center_bin = int(t_rel_s * fs30)
    t_start_bin = center_bin - n_pre
    t_end_bin   = center_bin + n_post
    if t_start_bin < 0 or t_end_bin > T:
        continue
    psth_mat += x30[:, t_start_bin:t_end_bin]
    n_valid_trials += 1

psth_mat /= max(n_valid_trials, 1)   # average over trials
# Convert to Hz: divide by bin width
psth_hz = psth_mat / BIN_S

# Smooth PSTH with a small Gaussian
psth_smooth = gaussian_filter1d(psth_hz, sigma=2, axis=1)

print(f"PSTH built from {n_valid_trials} trials  |  Matrix shape: {psth_smooth.shape}")

# %%
# Sort neurons by peak response latency
peak_bins = np.argmax(psth_smooth, axis=1)
sort_idx  = np.argsort(peak_bins)
psth_sorted = psth_smooth[sort_idx, :]

# Normalize rows to [0,1] for heatmap display
row_max_psth = psth_sorted.max(axis=1, keepdims=True)
row_max_psth[row_max_psth == 0] = 1
psth_normed = psth_sorted / row_max_psth

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[3, 1],
    subplot_titles=[
        f"PSTH heatmap — {n_neurons} neurons sorted by peak latency",
        "Mean population PSTH (Hz)",
    ],
)

fig.add_trace(
    go.Heatmap(
        z=psth_normed,
        x=time_axis_psth,
        colorscale="Hot",
        showscale=True,
        colorbar=dict(title="Norm. FR", thickness=14, len=0.55, y=0.65),
        name="PSTH",
    ),
    row=1, col=1,
)

pop_psth_mean = psth_smooth.mean(axis=0)
pop_psth_sem  = psth_smooth.std(axis=0) / np.sqrt(n_neurons)

fig.add_trace(
    go.Scatter(
        x=time_axis_psth,
        y=pop_psth_mean,
        mode="lines",
        line=dict(color="steelblue", width=2),
        name="Mean population",
        showlegend=False,
    ),
    row=2, col=1,
)
fig.add_trace(
    go.Scatter(
        x=np.concatenate([time_axis_psth, time_axis_psth[::-1]]),
        y=np.concatenate([pop_psth_mean + pop_psth_sem, (pop_psth_mean - pop_psth_sem)[::-1]]),
        fill="toself",
        fillcolor="rgba(70,130,180,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
        name="±SEM",
    ),
    row=2, col=1,
)

# Choice time line
for row in [1, 2]:
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1.5, row=row, col=1)
fig.add_annotation(x=0.01, y=1.04, xref="x", yref="paper", text="Choice",
                   showarrow=False, font=dict(color="red", size=11))

fig.update_xaxes(title_text="Time relative to choice (s)", row=2, col=1)
fig.update_yaxes(title_text="Neuron (sorted)", row=1, col=1)
fig.update_yaxes(title_text="FR (Hz)", row=2, col=1)

fig.update_layout(
    height=660,
    title_text=f"Trial-Aligned Activity — averaged over {n_valid_trials} trials",
    template="plotly_white",
    hovermode="x unified",
)
write_html_with_caption(fig, OUT_DIR / "10_psth_heatmap.html")

# %% [markdown]
# ### 6b · Correct vs Incorrect Trials
# 
# Does the population respond differently depending on trial outcome?

# %%
psth_correct   = np.zeros((n_neurons, n_bins), dtype=float)
psth_incorrect = np.zeros((n_neurons, n_bins), dtype=float)
n_correct = 0
n_incorrect = 0

for trial in trials:
    t_choice = trial.get("t chosen")
    outcome  = trial.get("rewarded")
    if t_choice is None or outcome is None:
        continue
    t_rel_s = (t_choice - t0_ms) / 1000
    center_bin = int(t_rel_s * fs30)
    t_start_bin = center_bin - n_pre
    t_end_bin   = center_bin + n_post
    if t_start_bin < 0 or t_end_bin > T:
        continue
    snippet = x30[:, t_start_bin:t_end_bin]
    if outcome:
        psth_correct += snippet
        n_correct += 1
    else:
        psth_incorrect += snippet
        n_incorrect += 1

if n_correct > 0:   psth_correct   /= n_correct
if n_incorrect > 0: psth_incorrect /= n_incorrect

pop_correct   = gaussian_filter1d(psth_correct.mean(axis=0),   sigma=2) / BIN_S
pop_incorrect = gaussian_filter1d(psth_incorrect.mean(axis=0), sigma=2) / BIN_S

print(f"Correct: {n_correct} trials  |  Incorrect: {n_incorrect} trials")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=time_axis_psth, y=pop_correct,
    mode="lines", line=dict(color="seagreen", width=2.5), name=f"Correct (n={n_correct})",
))
fig.add_trace(go.Scatter(
    x=time_axis_psth, y=pop_incorrect,
    mode="lines", line=dict(color="crimson", width=2.5), name=f"Incorrect (n={n_incorrect})",
))
fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1.5)
fig.add_annotation(x=0.02, y=1.06, xref="x", yref="paper", text="Choice",
                   showarrow=False, font=dict(size=11))

fig.update_layout(
    title=f"Mean Population PSTH — Correct vs Incorrect",
    xaxis_title="Time relative to choice (s)",
    yaxis_title="Population firing rate (Hz)",
    height=380,
    template="plotly_white",
    hovermode="x unified",
)
write_html_with_caption(fig, OUT_DIR / "11_psth_correct_vs_incorrect.html")

# %% [markdown]
# ---
# ## 7 · Neuron-Neuron Correlation Structure
# 
# Are there functional groups of co-active neurons?  
# We compute the Pearson correlation matrix and cluster it hierarchically.

# %%
# Subsample time for speed (every 3rd bin)
x30_sub = x30[:, ::3].astype(float)

# Z-score each neuron (avoids correlation artefacts from firing rate differences)
mu  = x30_sub.mean(axis=1, keepdims=True)
std = x30_sub.std(axis=1, keepdims=True)
std[std == 0] = 1
x_z = (x30_sub - mu) / std

corr_mat = np.corrcoef(x_z)   # (n_neurons, n_neurons)
np.fill_diagonal(corr_mat, np.nan)

print(f"Correlation matrix: {corr_mat.shape}")
flat = corr_mat[~np.isnan(corr_mat)]
print(f"Pairwise corr — mean: {flat.mean():.3f}  std: {flat.std():.3f}  max: {np.nanmax(corr_mat):.3f}")

# %%
# Hierarchical clustering to reorder neurons
np.fill_diagonal(corr_mat, 1.0)
dist_mat = 1 - corr_mat
np.fill_diagonal(dist_mat, 0.0)
dist_mat = np.clip(dist_mat, 0, None)   # numerical safety

# Use condensed distance vector
condensed = squareform(dist_mat, checks=False)
Z = linkage(condensed, method="ward")
order = leaves_list(Z)
np.fill_diagonal(corr_mat, np.nan)

corr_reordered = corr_mat[order, :][:, order]

fig = go.Figure(go.Heatmap(
    z=corr_reordered,
    colorscale="RdBu_r",
    zmid=0, zmin=-1, zmax=1,
    colorbar=dict(title="Pearson r", thickness=14),
))
fig.update_layout(
    title=f"Neuron × Neuron Correlation (hierarchically clustered) — {corr_mat.shape[0]} neurons",
    xaxis=dict(title="Neuron (clustered)", showticklabels=False),
    yaxis=dict(title="Neuron (clustered)", showticklabels=False, autorange="reversed"),
    height=580,
    width=620,
    template="plotly_white",
)
write_html_with_caption(fig, OUT_DIR / "12_correlation_heatmap.html")

# %%
np.fill_diagonal(corr_mat, np.nan)
flat_corr = corr_mat[~np.isnan(corr_mat)]

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Distribution of pairwise correlations", "Top-20 most correlated pairs"]
)

# Histogram of all pairwise correlations
fig.add_trace(
    go.Histogram(
        x=flat_corr,
        nbinsx=80,
        marker_color="steelblue",
        name="All pairs",
        showlegend=False,
    ),
    row=1, col=1,
)
fig.update_xaxes(title_text="Pearson r", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=1)

# Top-20 correlated pairs (upper triangle only)
np.fill_diagonal(corr_mat, -2)  # suppress diagonal
iu = np.triu_indices(corr_mat.shape[0], k=1)
pair_corrs = corr_mat[iu]
top_k = 20
top_idx = np.argsort(pair_corrs)[::-1][:top_k]
top_pairs = [(iu[0][i], iu[1][i], pair_corrs[i]) for i in top_idx]
np.fill_diagonal(corr_mat, np.nan)

pair_labels = [f"N{a}–N{b}" for a, b, _ in top_pairs]
pair_vals   = [r for _, _, r in top_pairs]

fig.add_trace(
    go.Bar(
        x=pair_labels,
        y=pair_vals,
        marker=dict(color=pair_vals, colorscale="Reds"),
        showlegend=False,
    ),
    row=1, col=2,
)
fig.update_xaxes(title_text="Neuron pair", tickangle=45, row=1, col=2)
fig.update_yaxes(title_text="Pearson r", row=1, col=2)

fig.update_layout(
    height=420,
    title_text="Pairwise Neuron Correlations",
    template="plotly_white",
)
write_html_with_caption(fig, OUT_DIR / "13_correlation_distribution.html")

# %%
# Plot activity traces of top-5 most correlated pairs side-by-side
N_PAIRS = 5
WIN_PAIRS_S = 60
t_end_pairs = min(int(WIN_PAIRS_S * fs30), T)
t_ax_p = time_s[:t_end_pairs]

fig = make_subplots(
    rows=N_PAIRS, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=[f"Pair N{a}–N{b} (r={r:.3f})" for a, b, r in top_pairs[:N_PAIRS]],
)

for row_idx, (a, b, r) in enumerate(top_pairs[:N_PAIRS], start=1):
    ta = gaussian_filter1d(x30[a, :t_end_pairs].astype(float), sigma=int(fs30 * 0.5))
    tb = gaussian_filter1d(x30[b, :t_end_pairs].astype(float), sigma=int(fs30 * 0.5))
    fig.add_trace(go.Scatter(x=t_ax_p, y=ta, mode="lines", line=dict(color="steelblue", width=1.5), name=f"N{a}", showlegend=False), row=row_idx, col=1)
    fig.add_trace(go.Scatter(x=t_ax_p, y=tb, mode="lines", line=dict(color="crimson", width=1.5), name=f"N{b}", showlegend=False), row=row_idx, col=1)
    fig.update_yaxes(title_text="Activity", row=row_idx, col=1)

fig.update_xaxes(title_text="Time (s)", row=N_PAIRS, col=1)
fig.update_layout(
    height=120 + 180 * N_PAIRS,
    title_text=f"Top {N_PAIRS} Most Correlated Neuron Pairs — first {WIN_PAIRS_S}s",
    template="plotly_white",
    hovermode="x unified",
)
write_html_with_caption(fig, OUT_DIR / "14_top_correlated_pairs.html")

# %% [markdown]
# ---
# ## 8 · Dimensionality & PCA
# 
# How many dimensions does the neural population actually occupy?  
# PCA reveals the dominant axes of co-variation.

# %%
pca = PCA(n_components=min(50, n_neurons))
X_pca = pca.fit_transform(x_z.T)          # (T, n_components)
var_exp   = pca.explained_variance_ratio_
var_cumul = np.cumsum(var_exp)

# Find number of PCs for 80 / 90 / 95 % variance
thresholds = {80: None, 90: None, 95: None}
for t in thresholds:
    idx = int(np.searchsorted(var_cumul, t / 100))
    idx = min(idx, len(var_cumul) - 1)   # clamp in case threshold isn't reached
    thresholds[t] = idx + 1

print("PCs needed for:")
for t, k in thresholds.items():
    print(f"  {t}% variance → {k} PCs  (cumulative: {var_cumul[k-1]*100:.1f}%)")


# %%
n_show = min(30, len(var_exp))
pc_nums = np.arange(1, n_show + 1)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Scree plot (individual variance explained)", "Cumulative variance explained"],
)

fig.add_trace(
    go.Bar(
        x=pc_nums, y=var_exp[:n_show] * 100,
        marker_color="steelblue",
        showlegend=False,
        name="Variance %",
    ),
    row=1, col=1,
)
fig.update_xaxes(title_text="PC", row=1, col=1)
fig.update_yaxes(title_text="Variance explained (%)", row=1, col=1)

fig.add_trace(
    go.Scatter(
        x=pc_nums, y=var_cumul[:n_show] * 100,
        mode="lines+markers",
        line=dict(color="darkorange", width=2),
        marker=dict(size=5),
        showlegend=False,
        name="Cumulative %",
    ),
    row=1, col=2,
)

for t, k in thresholds.items():
    if k <= n_show:
        fig.add_vline(x=k, line_dash="dot", line_color="grey", line_width=1, row=1, col=2)
        fig.add_annotation(
            x=k + 0.3, y=t + 2, xref="x2", yref="y2",
            text=f"{t}% @ PC{k}", showarrow=False, font=dict(size=10, color="grey"),
        )

fig.update_xaxes(title_text="PC", row=1, col=2)
fig.update_yaxes(title_text="Cumulative variance (%)", row=1, col=2)

fig.update_layout(
    height=400,
    title_text=f"PCA Dimensionality — {n_neurons} neurons, {n_show} PCs shown",
    template="plotly_white",
)
write_html_with_caption(fig, OUT_DIR / "15_pca_scree.html")

# %%
# PC1 vs PC2 colored by behavioral state
b_dense = ds30.b.toarray().flatten()
b_labels = ds30.b_labels_dict
b_names_arr = np.array([b_labels.get(int(s), str(s)) for s in b_dense])
# x_z was built from x30[:, ::3], so subsample behaviour labels to match
b_names_arr = b_names_arr[::3]
color_map = ds30.get_color_map_for_plotting()  # state_id → hex
name_to_color = {b_labels.get(int(k), str(k)): v for k, v in color_map.items()}

fig = go.Figure()
for state_name in np.unique(b_names_arr):
    mask = b_names_arr == state_name
    fig.add_trace(go.Scatter(
        x=X_pca[mask, 0],
        y=X_pca[mask, 1],
        mode="markers",
        marker=dict(
            size=2.5,
            color=name_to_color.get(state_name, "grey"),
            opacity=0.5,
        ),
        name=state_name,
    ))

fig.update_layout(
    title=f"PC1 vs PC2 — colored by behavioral state ({var_exp[0]*100:.1f}% + {var_exp[1]*100:.1f}% variance)",
    xaxis_title=f"PC1 ({var_exp[0]*100:.1f}%)",
    yaxis_title=f"PC2 ({var_exp[1]*100:.1f}%)",
    height=520,
    width=680,
    template="plotly_white",
    legend_title="Behavioral state",
)
write_html_with_caption(fig, OUT_DIR / "16_pca_pc1_vs_pc2.html")


# %%
# 3D PCA trajectory (first 3 PCs), colored by behavioral state
# b_names_arr is already subsampled to match X_pca (from previous cell)
fig = go.Figure()
for state_name in np.unique(b_names_arr):
    mask = b_names_arr == state_name
    fig.add_trace(go.Scatter3d(
        x=X_pca[mask, 0],
        y=X_pca[mask, 1],
        z=X_pca[mask, 2],
        mode="markers",
        marker=dict(
            size=1.8,
            color=name_to_color.get(state_name, "grey"),
            opacity=0.4,
        ),
        name=state_name,
    ))

fig.update_layout(
    title=f"3D PCA — PC1 vs PC2 vs PC3 ({sum(var_exp[:3])*100:.1f}% variance)",
    scene=dict(
        xaxis_title=f"PC1 ({var_exp[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({var_exp[1]*100:.1f}%)",
        zaxis_title=f"PC3 ({var_exp[2]*100:.1f}%)",
    ),
    height=580,
    width=700,
    legend_title="Behavioral state",
)
write_html_with_caption(fig, OUT_DIR / "17_pca_3d.html")


# %% [markdown]
# ---
# ## 9 · Laminar Activity Profile
# 
# NeuroPixels probes record across cortical layers. Does activity differ by depth?

# %%
# Map cluster IDs in the 30 Hz dataset to cluster_info depth
# The 30 Hz dataset keeps only 'good' neurons; their order matches cluster_info rows
good_ci_sorted_by_id = good_ci.sort_values("cluster_id").reset_index(drop=True)
depths = good_ci_sorted_by_id["depth"].values    # (n_good_neurons,)

# Make sure lengths match
n_neu_ds = x30.shape[0]
if len(depths) != n_neu_ds:
    print(f"Warning: depth array length {len(depths)} != dataset neurons {n_neu_ds}. Trimming.")
    min_len = min(len(depths), n_neu_ds)
    depths = depths[:min_len]
    x30_depth = x30[:min_len, :]
else:
    x30_depth = x30

# Bin neurons into N_DEPTH_BINS depth windows
N_DEPTH_BINS = 10
depth_bins = np.linspace(depths.min(), depths.max(), N_DEPTH_BINS + 1)
bin_labels = [f"{depth_bins[i]:.0f}–{depth_bins[i+1]:.0f} µm" for i in range(N_DEPTH_BINS)]

mean_fr_per_bin = []
n_per_bin = []
for i in range(N_DEPTH_BINS):
    in_bin = (depths >= depth_bins[i]) & (depths < depth_bins[i + 1])
    if in_bin.sum() == 0:
        mean_fr_per_bin.append(0)
        n_per_bin.append(0)
    else:
        mean_fr_per_bin.append(x30_depth[in_bin, :].mean() * fs30)
        n_per_bin.append(int(in_bin.sum()))

print("Neurons per depth bin:", n_per_bin)

# %%
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Mean firing rate by depth (laminar profile)", "Neurons per depth bin"],
)

# Horizontal bar: firing rate (simulate cortical layer readout)
fig.add_trace(
    go.Bar(
        y=bin_labels,
        x=mean_fr_per_bin,
        orientation="h",
        marker=dict(
            color=mean_fr_per_bin,
            colorscale="Plasma",
            showscale=False,
        ),
        text=[f"{v:.2f} Hz" for v in mean_fr_per_bin],
        textposition="outside",
        showlegend=False,
    ),
    row=1, col=1,
)
fig.update_xaxes(title_text="Mean firing rate (Hz)", row=1, col=1)
fig.update_yaxes(title_text="Depth bin", autorange="reversed", row=1, col=1)

# Neuron count
fig.add_trace(
    go.Bar(
        y=bin_labels,
        x=n_per_bin,
        orientation="h",
        marker_color="steelblue",
        text=n_per_bin,
        textposition="outside",
        showlegend=False,
    ),
    row=1, col=2,
)
fig.update_xaxes(title_text="Neuron count", row=1, col=2)
fig.update_yaxes(autorange="reversed", row=1, col=2)

fig.update_layout(
    height=480,
    title_text="Laminar Activity Profile",
    template="plotly_white",
)
write_html_with_caption(fig, OUT_DIR / "18_laminar_profile.html")

# %%
# Depth × time heatmap: mean activity per depth bin over time
# Downsample time axis for display (every 10th bin)
STRIDE = 10
t_ds = time_s[::STRIDE]
depth_time_mat = np.zeros((N_DEPTH_BINS, len(t_ds)))

for i in range(N_DEPTH_BINS):
    in_bin = (depths >= depth_bins[i]) & (depths < depth_bins[i + 1])
    if in_bin.sum() == 0:
        continue
    bin_activity = x30_depth[in_bin, ::STRIDE].mean(axis=0).astype(float)
    depth_time_mat[i, :] = gaussian_filter1d(bin_activity, sigma=5)

fig = go.Figure(go.Heatmap(
    z=depth_time_mat,
    x=t_ds,
    y=bin_labels,
    colorscale="Plasma",
    colorbar=dict(title="Mean activity", thickness=14),
))

# Block boundaries
for t_idx, lbl in block_changes:
    t = time_s[min(t_idx, len(time_s) - 1)]
    color = "lime" if "left" in lbl.lower() else "cyan"
    fig.add_vline(x=t, line_dash="solid", line_color=color, line_width=1.2)

fig.update_layout(
    title="Depth × Time Heatmap (mean activity per depth bin, smoothed)",
    xaxis_title="Time (s)",
    yaxis=dict(title="Depth bin", autorange="reversed"),
    height=440,
    template="plotly_white",
    hovermode="x unified",
)
write_html_with_caption(fig, OUT_DIR / "19_depth_time_heatmap.html")

# %% [markdown]
# ---
# ## 10 · Autocorrelation & Burstiness
# 
# The temporal structure of individual neuron firing — are neurons bursty or tonic?

# %%
# Pick top-N neurons by firing rate for autocorrelation analysis
N_ACF = 8
top_n_idx = np.argsort(x30.sum(axis=1))[::-1][:N_ACF]
MAX_LAG = int(0.5 * fs30)   # 500 ms

fig = make_subplots(
    rows=2, cols=4,
    subplot_titles=[f"Neuron #{idx} (FR={x30[idx].mean()*fs30:.1f} Hz)" for idx in top_n_idx],
    shared_xaxes=True, shared_yaxes=False,
    vertical_spacing=0.14, horizontal_spacing=0.06,
)

for i, nidx in enumerate(top_n_idx):
    r, c = divmod(i, 4)
    trace = x30[nidx, :].astype(float)
    trace -= trace.mean()
    # Compute autocorrelation via FFT
    n_pad = len(trace) + MAX_LAG
    fft_result = np.fft.rfft(trace, n=2 * n_pad)
    acf_full = np.fft.irfft(fft_result * np.conj(fft_result))[:MAX_LAG + 1]
    acf_norm = acf_full / acf_full[0] if acf_full[0] != 0 else acf_full
    lag_ms = np.arange(MAX_LAG + 1) / fs30 * 1000

    fig.add_trace(
        go.Bar(
            x=lag_ms,
            y=acf_norm,
            marker_color="steelblue",
            showlegend=False,
            width=1000 / fs30,
        ),
        row=r + 1, col=c + 1,
    )
    fig.update_xaxes(title_text="Lag (ms)", row=r + 1, col=c + 1)
    fig.update_yaxes(title_text="ACF", row=r + 1, col=c + 1)

fig.update_layout(
    height=460,
    title_text=f"Autocorrelation — Top {N_ACF} most active neurons (0–500 ms)",
    template="plotly_white",
)
write_html_with_caption(fig, OUT_DIR / "20_autocorrelation.html")

# %%
# Burstiness index: (std - mean) / (std + mean) of ISIs
# Computed from raw spike times per neuron
spike_clusters_raw = np.load(PRIMARY / "spike_clusters.npy").flatten()
spike_times_ms_raw = np.load(PRIMARY / "spike_times_milliseconds_sync_to_behav.npy").flatten()

burstiness_scores = []
cv_isi_scores = []

for cid in good_ci["cluster_id"].values:
    st = np.sort(spike_times_ms_raw[spike_clusters_raw == cid])
    if len(st) < 10:
        burstiness_scores.append(np.nan)
        cv_isi_scores.append(np.nan)
        continue
    isis = np.diff(st)
    mu, sigma = isis.mean(), isis.std()
    B = (sigma - mu) / (sigma + mu) if (sigma + mu) > 0 else 0
    cv = sigma / mu if mu > 0 else 0
    burstiness_scores.append(B)
    cv_isi_scores.append(cv)

good_ci = good_ci.copy()
good_ci["burstiness"]  = burstiness_scores
good_ci["cv_isi"]      = cv_isi_scores

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Burstiness index distribution", "CV of ISI distribution"],
)

fig.add_trace(
    go.Histogram(x=good_ci["burstiness"].dropna(), nbinsx=50, marker_color="#e76f51", showlegend=False),
    row=1, col=1,
)
fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1.5, row=1, col=1)
fig.add_annotation(x=0.03, y=1.06, xref="x", yref="paper",
                   text="B=0: Poisson", showarrow=False, font=dict(size=10))
fig.update_xaxes(title_text="Burstiness B = (σ-μ)/(σ+μ)", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=1)

fig.add_trace(
    go.Histogram(x=good_ci["cv_isi"].dropna(), nbinsx=50, marker_color="#264653", showlegend=False),
    row=1, col=2,
)
fig.add_vline(x=1, line_dash="dash", line_color="black", line_width=1.5, row=1, col=2)
fig.add_annotation(x=1.05, y=1.06, xref="x2", yref="paper",
                   text="CV=1: Poisson", showarrow=False, font=dict(size=10))
fig.update_xaxes(title_text="CV of ISI", row=1, col=2)
fig.update_yaxes(title_text="Count", row=1, col=2)

fig.update_layout(
    height=380,
    title_text="Spike Burstiness & Temporal Regularity",
    template="plotly_white",
)
write_html_with_caption(fig, OUT_DIR / "21_burstiness.html")

B_vals = good_ci["burstiness"].dropna()
print(f"Burstiness — mean: {B_vals.mean():.3f}  median: {B_vals.median():.3f}  ({(B_vals > 0).mean()*100:.1f}% bursty, {(B_vals < 0).mean()*100:.1f}% regular)")

# %% [markdown]
# ---
# ## 11 · ISI Distribution
# 
# Inter-spike interval distributions reveal the fine temporal structure of firing —  
# refractory periods, burst patterns, and overall regularity.

# %%
# Population-level ISI histogram (all good neurons, log scale)
all_isis = []
for cid in good_ci["cluster_id"].values:
    st = np.sort(spike_times_ms_raw[spike_clusters_raw == cid])
    if len(st) > 1:
        all_isis.extend(np.diff(st).tolist())

all_isis = np.array(all_isis)

# Log-spaced bins from 0.1 ms to 10 000 ms
log_bins = np.logspace(-1, 4, 80)
hist_vals, bin_edges = np.histogram(all_isis, bins=log_bins)
bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

fig = go.Figure()
fig.add_trace(go.Bar(
    x=bin_centers,
    y=hist_vals,
    marker=dict(color=bin_centers, colorscale="Viridis", showscale=True,
                colorbar=dict(title="ISI (ms)")),
    showlegend=False,
))

# Mark refractory period
fig.add_vrect(x0=0, x1=2, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Refractory\n<2ms")

fig.update_layout(
    title="Population ISI Distribution (all good neurons, log scale)",
    xaxis=dict(title="ISI (ms)", type="log", dtick=1),
    yaxis_title="Spike count",
    height=400,
    template="plotly_white",
)
write_html_with_caption(fig, OUT_DIR / "22_isi_distribution.html")

print(f"Total ISIs: {len(all_isis):,}  |  median ISI: {np.median(all_isis):.1f} ms  |  <2ms (refractory violations): {(all_isis < 2).mean()*100:.2f}%")

# %% [markdown]
# ---
# ## 12 · UMAP
# 
# PCA is linear — UMAP is not. It preserves local neighbourhood structure and often reveals  
# cluster geometry that PCA collapses into overlapping clouds.  
# Here we run UMAP on the same z-scored population matrix (`x_z`) and color by behavioral state.
# 

# %%
from umap import UMAP

# x_z is (n_neurons, T/3); transpose to (T/3, n_neurons) for UMAP
# Subsample to ≤ 8000 points for speed while preserving temporal coverage
N_UMAP = min(8_000, x_z.shape[1])
rng      = np.random.default_rng(42)
umap_idx = np.sort(rng.choice(x_z.shape[1], N_UMAP, replace=False))

X_umap_in = x_z[:, umap_idx].T   # (N_UMAP, n_neurons)
b_umap    = b_names_arr[umap_idx] # already at x_z stride (::3)

print(f"Fitting UMAP on {X_umap_in.shape} ... ", end="", flush=True)
reducer   = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42, verbose=False)
embedding = reducer.fit_transform(X_umap_in)   # (N_UMAP, 2)
print(f"done  embedding: {embedding.shape}")

fig = go.Figure()
for state_name in np.unique(b_umap):
    mask = b_umap == state_name
    fig.add_trace(go.Scatter(
        x=embedding[mask, 0],
        y=embedding[mask, 1],
        mode="markers",
        marker=dict(size=2.5, color=name_to_color.get(state_name, "grey"), opacity=0.55),
        name=state_name,
    ))

fig.update_layout(
    title=f"UMAP of population activity — {N_UMAP} timepoints, colored by behavioral state",
    xaxis_title="UMAP-1",
    yaxis_title="UMAP-2",
    height=540, width=680,
    template="plotly_white",
    legend_title="Behavioral state",
)
write_html_with_caption(fig, OUT_DIR / "23_umap.html")


# %% [markdown]
# ---
# ## 13 · Neural Trajectory
# 
# The PC scatter (section 8) treats all timepoints as independent. Here we reconnect them in  
# **temporal order**, drawing the path the population traces through its low-dimensional state space.  
# Color encodes elapsed time; diamonds mark block transitions.  
# Loops, spirals, and drift patterns reveal the geometry of the population dynamics.
# 

# %%
# Neural trajectory: PC1 vs PC2 as a time-ordered line
# X_pca is (T/3, n_components); time axis is time_s[::3]
time_xz = time_s[::3]   # (T/3,)

# Stride for rendering — every 4th point, colored by time
TRAJ_STRIDE = 4
traj_idx = np.arange(0, X_pca.shape[0], TRAJ_STRIDE)
traj_t   = time_xz[traj_idx]
traj_x1  = X_pca[traj_idx, 0]
traj_y1  = X_pca[traj_idx, 1]

fig = go.Figure()

# go.Scatter line.color only accepts a scalar; per-point color requires marker.color.
# Render as densely packed markers — visually identical to a colored line.
fig.add_trace(go.Scatter(
    x=traj_x1, y=traj_y1,
    mode="markers",
    marker=dict(
        size=3,
        color=traj_t,
        colorscale="Plasma",
        showscale=True,
        colorbar=dict(title="Time (s)", thickness=12, len=0.55, y=0.5),
        opacity=0.85,
    ),
    hovertemplate="t=%{text:.1f}s<extra></extra>",
    text=traj_t,
    showlegend=False,
    name="Trajectory",
))

# Mark block transitions as diamond markers
for t_idx, lbl in block_changes:
    xz_idx = min(t_idx // 3, X_pca.shape[0] - 1)
    col   = "lime" if "left" in lbl.lower() else "cyan"
    short = "→L" if "left" in lbl.lower() else "→R"
    fig.add_trace(go.Scatter(
        x=[X_pca[xz_idx, 0]], y=[X_pca[xz_idx, 1]],
        mode="markers+text",
        marker=dict(size=11, color=col, symbol="diamond", line=dict(width=1, color="black")),
        text=[short], textposition="top center",
        textfont=dict(color=col, size=10),
        showlegend=False,
    ))

# Mark start and end
for idx, label, sym in [(0, "Start", "triangle-right"), (len(traj_idx)-1, "End", "square")]:
    fig.add_trace(go.Scatter(
        x=[traj_x1[idx]], y=[traj_y1[idx]],
        mode="markers+text",
        marker=dict(size=10, color="white", symbol=sym, line=dict(width=1.5, color="grey")),
        text=[label], textposition="top right",
        textfont=dict(color="grey", size=10),
        showlegend=False,
    ))

fig.update_layout(
    title="Neural Trajectory — PC1 × PC2, colored by time (diamonds = block transitions)",
    xaxis_title=f"PC1 ({var_exp[0]*100:.1f}%)",
    yaxis_title=f"PC2 ({var_exp[1]*100:.1f}%)",
    height=560, width=700,
    template="plotly_dark",
    plot_bgcolor="#0d1117",
    paper_bgcolor="#0d1117",
    font=dict(color="#e2e8f0"),
)
write_html_with_caption(fig, OUT_DIR / "24_neural_trajectory.html")
print(f"Trajectory: {len(traj_idx)} points, {len(block_changes)} transitions")


# %% [markdown]
# ---
# ## 14 · Block-Transition-Aligned PSTH
# 
# Align population activity to the moment the reward probabilities switch (block transitions),  
# not to individual choices. This reveals whether the population adapts its firing pattern  
# after the environment changes — a signature of **surprise / context updating**.
# 

# %%
# Block-transition-aligned PSTH
# Align population firing rate to reward-probability switches
BLK_PRE_S  = 30.0
BLK_POST_S = 60.0
n_pre_blk  = int(BLK_PRE_S  * fs30)
n_post_blk = int(BLK_POST_S * fs30)
n_bins_blk = n_pre_blk + n_post_blk
time_blk   = np.linspace(-BLK_PRE_S, BLK_POST_S, n_bins_blk)

sigma_blk = int(2 * fs30)

# Collect individual snippets per direction
snippets_left  = []
snippets_right = []

for t_idx, lbl in block_changes:
    center = int(t_idx)
    ts = center - n_pre_blk
    te = center + n_post_blk
    if ts < 0 or te > T:
        continue
    seg = x30[:, ts:te].mean(axis=0)
    seg_hz = gaussian_filter1d(seg / BIN_S, sigma=sigma_blk)
    if "left" in lbl.lower():
        snippets_left.append(seg_hz)
    else:
        snippets_right.append(seg_hz)

n_left_blk  = len(snippets_left)
n_right_blk = len(snippets_right)
print(f"Block-transition PSTH: {n_left_blk} → Better-Left  |  {n_right_blk} → Better-Right")

# Means (NaN-safe)
mean_left  = np.mean(snippets_left,  axis=0) if n_left_blk  > 0 else np.zeros(n_bins_blk)
mean_right = np.mean(snippets_right, axis=0) if n_right_blk > 0 else np.zeros(n_bins_blk)

fig = go.Figure()

# Individual traces — thin, semi-transparent
for k, tr in enumerate(snippets_left):
    fig.add_trace(go.Scatter(
        x=time_blk, y=tr,
        mode="lines", line=dict(color="seagreen", width=1),
        opacity=0.35, showlegend=(k == 0),
        name=f"→ Better-Left (trial)",
        legendgroup="left",
    ))
for k, tr in enumerate(snippets_right):
    fig.add_trace(go.Scatter(
        x=time_blk, y=tr,
        mode="lines", line=dict(color="dodgerblue", width=1),
        opacity=0.35, showlegend=(k == 0),
        name=f"→ Better-Right (trial)",
        legendgroup="right",
    ))

# Mean traces — thick
fig.add_trace(go.Scatter(
    x=time_blk, y=mean_left,
    mode="lines", line=dict(color="seagreen", width=3),
    name=f"→ Better-Left mean (n={n_left_blk})",
    legendgroup="left",
))
fig.add_trace(go.Scatter(
    x=time_blk, y=mean_right,
    mode="lines", line=dict(color="dodgerblue", width=3),
    name=f"→ Better-Right mean (n={n_right_blk})",
    legendgroup="right",
))

fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1.5)
fig.add_annotation(x=1.5, y=1.06, xref="x", yref="paper",
                   text="Block switch", showarrow=False, font=dict(size=11))
fig.update_layout(
    title=f"Block-Transition-Aligned Population PSTH<br>"
          f"<sup>Individual trials + mean  (n={n_left_blk} left, n={n_right_blk} right)</sup>",
    xaxis_title="Time relative to block switch (s)",
    yaxis_title="Population firing rate (Hz)",
    height=400,
    template="plotly_white",
    hovermode="x unified",
)
write_html_with_caption(fig, OUT_DIR / "25_block_psth.html")
print("Saved 25_block_psth.html")


# %% [markdown]
# ---
# ## 15 · Fano Factor Over Time
# 
# The Fano factor (variance/mean) measures spiking variability relative to the mean rate.  
# - **F = 1** → Poisson-like (maximum entropy for the rate)  
# - **F < 1** → sub-Poisson / quenched variability (often seen around movement onset)  
# - **F > 1** → super-Poisson / bursty  
# 
# Block transitions are marked with vertical lines.
# 

# %%
# Fano factor = variance / mean spike counts in sliding windows
FANO_WIN_S  = 5.0   # seconds
FANO_STEP_S = 1.0
win  = int(FANO_WIN_S  * fs30)
step = int(FANO_STEP_S * fs30)

fano_times  = []
fano_mean   = []
fano_median = []

t_bin = 0
while t_bin + win <= T:
    window = x30[:, t_bin : t_bin + win].astype(float)   # (n_neurons, win)
    mu_w   = window.mean(axis=1)
    var_w  = window.var(axis=1)
    valid  = mu_w > 0
    fano_n = np.where(valid, var_w / np.where(mu_w > 0, mu_w, 1.0), np.nan)
    fano_times.append(t_bin / fs30 + FANO_WIN_S / 2)
    fano_mean.append(np.nanmean(fano_n))
    fano_median.append(np.nanmedian(fano_n))
    t_bin += step

fano_times  = np.array(fano_times)
fano_mean   = np.array(fano_mean)
fano_median = np.array(fano_median)

print(f"Fano factor: {len(fano_times)} windows | population mean range: {fano_mean.min():.3f} – {fano_mean.max():.3f}")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=fano_times, y=fano_mean,
    mode="lines", line=dict(color="steelblue", width=1.5), name="Population mean",
))
fig.add_trace(go.Scatter(
    x=fano_times, y=fano_median,
    mode="lines", line=dict(color="darkorange", width=1.5, dash="dot"), name="Population median",
))
fig.add_hline(y=1, line_dash="dash", line_color="grey", line_width=1.2,
              annotation_text="Poisson (F=1)", annotation_position="right")

for t_idx, lbl in block_changes:
    t_sec = time_s[min(t_idx, len(time_s) - 1)]
    col = "green" if "left" in lbl.lower() else "dodgerblue"
    fig.add_vline(x=t_sec, line_dash="solid", line_color=col, line_width=1, opacity=0.5)

fig.update_layout(
    title=f"Population Fano Factor Over Time (window={FANO_WIN_S:.0f}s, step={FANO_STEP_S:.0f}s)",
    xaxis_title="Time (s)",
    yaxis_title="Fano factor (variance / mean)",
    height=380,
    template="plotly_white",
    hovermode="x unified",
)
write_html_with_caption(fig, OUT_DIR / "26_fano_factor.html")


# %% [markdown]
# ---
# ## 16 · Single-Trial Population Activity
# 
# Instead of averaging over trials, every trial is shown as its own row — sorted by reaction time  
# (fast choices at top, slow at bottom). Color on the right indicates correct (green) vs incorrect (red).  
# This reveals whether fast choices differ in their pre-choice population state.
# 

# %%
# Single-trial PSTH: each row = one trial's population-mean activity, sorted by reaction time
single_trial_rows     = []
single_trial_rts      = []
single_trial_outcomes = []

for trial in trials:
    t_choice   = trial.get("t chosen")
    t_choosing = trial.get("t choosing")
    outcome    = trial.get("rewarded")
    if t_choice is None or t_choosing is None:
        continue
    rt_ms = t_choice - t_choosing
    if rt_ms <= 0:
        continue
    t_rel_s    = (t_choice - t0_ms) / 1000
    center_bin = int(t_rel_s * fs30)
    ts = center_bin - n_pre
    te = center_bin + n_post
    if ts < 0 or te > T:
        continue
    pop_snippet = x30[:, ts:te].mean(axis=0)   # (n_bins,)
    single_trial_rows.append(pop_snippet)
    single_trial_rts.append(rt_ms)
    single_trial_outcomes.append(bool(outcome) if outcome is not None else None)

single_trial_rows = np.array(single_trial_rows)
single_trial_rts  = np.array(single_trial_rts)

# Sort by RT (fast → slow)
rt_sort = np.argsort(single_trial_rts)
rows_sorted     = single_trial_rows[rt_sort]
rts_sorted_st   = single_trial_rts[rt_sort]
outcomes_sorted = [single_trial_outcomes[i] for i in rt_sort]

# Smooth + normalise
rows_smooth = gaussian_filter1d(rows_sorted.astype(float), sigma=2, axis=1) / BIN_S
row_mx = rows_smooth.max(axis=1, keepdims=True)
row_mx[row_mx == 0] = 1
rows_norm = rows_smooth / row_mx

print(f"Single-trial matrix: {rows_norm.shape}  |  RT range: {rts_sorted_st[0]:.0f} – {rts_sorted_st[-1]:.0f} ms")

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Population activity per trial (sorted by RT, fast→slow)",
                    "Reaction time"],
    column_widths=[0.78, 0.22],
)

fig.add_trace(
    go.Heatmap(z=rows_norm, x=time_axis_psth, colorscale="Hot",
               showscale=True,
               colorbar=dict(title="Norm. FR", thickness=12, len=0.6, x=0.74),
               name="Trials"),
    row=1, col=1,
)

colors_oc = ["seagreen" if o else ("crimson" if o is False else "grey") for o in outcomes_sorted]
fig.add_trace(
    go.Scatter(x=rts_sorted_st, y=np.arange(len(rts_sorted_st)),
               mode="markers",
               marker=dict(size=3, color=colors_oc),
               showlegend=False, name="RT"),
    row=1, col=2,
)

fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1.5, row=1, col=1)
fig.update_xaxes(title_text="Time relative to choice (s)", row=1, col=1)
fig.update_yaxes(title_text="Trial (fast → slow)", row=1, col=1)
fig.update_xaxes(title_text="RT (ms)", row=1, col=2)
fig.update_yaxes(showticklabels=False, row=1, col=2)

fig.update_layout(
    height=620,
    title_text=f"Single-Trial Population Activity — {rows_norm.shape[0]} trials sorted by RT",
    template="plotly_white",
)
write_html_with_caption(fig, OUT_DIR / "27_singletrial_psth.html")


# %% [markdown]
# ---
# ## 17 · Normalized Inter-Trial Decoder
# 
# **Question:** Across the full interval between the previous and next choice, when does the
# population start encoding which way the animal will go *this* trial?
# 
# **X-axis (τ):** 0 = previous choice · 0.5 = this trial's choice (always) · 1 = next choice  
# The left half [0, 0.5] is independently normalized to the prev→curr ITI;  
# the right half [0.5, 1] to the curr→next ITI.
# 
# At each of 100 τ grid points, logistic regression decodes left vs right from the full
# population vector using 5-fold stratified CV.
# 

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

CHOICE_MAP = {0: 0, 1: 1, 0.0: 0, 1.0: 1, "l": 0, "r": 1, "left": 0, "right": 1}
GRID_N   = 100   # 50 points per half
tau_grid = np.linspace(0.0, 1.0, GRID_N)

# ── Pass 1: extract valid (center_bin, choice_int) for every trial ────────────
valid_trials_dec = []   # list of (center_bin, choice_int)
for trial in trials:
    t_choice = trial.get("t chosen")
    choice   = trial.get("choice")
    if t_choice is None or choice is None:
        continue
    choice_raw = choice.lower() if isinstance(choice, str) else choice
    choice_int = CHOICE_MAP.get(choice_raw)
    if choice_int is None:
        continue
    center_bin = int((t_choice - t0_ms) / 1000 * fs30)
    if center_bin < 0 or center_bin >= T:
        continue
    valid_trials_dec.append((center_bin, choice_int))

# ── Pass 2: build triplets (prev, curr, next) — interior trials only ─────────
triplets = []   # (b_prev, b_curr, b_next, choice_int)
for i in range(1, len(valid_trials_dec) - 1):
    b_prev, _          = valid_trials_dec[i - 1]
    b_curr, choice_int = valid_trials_dec[i]
    b_next, _          = valid_trials_dec[i + 1]
    if b_curr <= b_prev or b_next <= b_curr:
        continue   # guard against zero-length intervals
    triplets.append((b_prev, b_curr, b_next, choice_int))

n_trip       = len(triplets)
choices_norm = np.array([ch for *_, ch in triplets], dtype=np.int32)
counts_norm  = dict(zip(*np.unique(choices_norm, return_counts=True))) if n_trip > 0 else {}
left_cnt     = int(counts_norm.get(0, 0))
right_cnt    = int(counts_norm.get(1, 0))
print(f"Inter-trial decoder: {n_trip} triplets | left={left_cnt}, right={right_cnt}")

# ── Pass 3: sample x30 at each τ grid point for every triplet ────────────────
# X_norm shape: (n_trip, n_neurons, GRID_N)
X_norm = np.empty((n_trip, n_neurons, GRID_N), dtype=np.float32)
for j, (b_prev, b_curr, b_next, _) in enumerate(triplets):
    for k, tau in enumerate(tau_grid):
        if tau <= 0.5:
            b = b_prev + (tau / 0.5) * (b_curr - b_prev)
        else:
            b = b_curr + ((tau - 0.5) / 0.5) * (b_next - b_curr)
        X_norm[j, :, k] = x30[:, int(np.clip(round(b), 0, T - 1))]

# ── Decode at each τ using 5-fold stratified CV ───────────────────────────────
decoder_acc_norm = np.full(GRID_N, np.nan)

if len(np.unique(choices_norm)) >= 2:
    cv_norm     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits_norm = list(cv_norm.split(X_norm[:, :, 0], choices_norm))

    for k in range(GRID_N):
        X_bin = X_norm[:, :, k].astype(float)
        fold_accs = []
        for train_idx, test_idx in splits_norm:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_bin[train_idx])
            X_te = scaler.transform(X_bin[test_idx])
            clf  = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
            clf.fit(X_tr, choices_norm[train_idx])
            fold_accs.append(clf.score(X_te, choices_norm[test_idx]))
        decoder_acc_norm[k] = np.mean(fold_accs)

    print(f"Accuracy: min={np.nanmin(decoder_acc_norm):.3f}  max={np.nanmax(decoder_acc_norm):.3f}  chance=0.5")
else:
    print("Only one choice class found — decoder skipped")
    decoder_acc_norm[:] = 0.5

# ── Median ITI for subtitle ───────────────────────────────────────────────────
iti_prev_s = float(np.median([(b_curr - b_prev) / fs30 for b_prev, b_curr, b_next, _ in triplets]))
iti_next_s = float(np.median([(b_next - b_curr) / fs30 for b_prev, b_curr, b_next, _ in triplets]))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=tau_grid, y=decoder_acc_norm,
    mode="lines",
    line=dict(color="darkorchid", width=2.5),
    name="5-fold CV accuracy",
    hovertemplate="τ=%{x:.2f}  acc=%{y:.3f}<extra></extra>",
))
fig.add_hline(y=0.5, line_dash="dash", line_color="grey", line_width=1.5,
              annotation_text="Chance (0.50)", annotation_position="right")

for tau_v, label, col in [
    (0.0, "Prev choice", "steelblue"),
    (0.5, "This choice", "crimson"),
    (1.0, "Next choice", "steelblue"),
]:
    fig.add_vline(x=tau_v, line_dash="dash", line_color=col, line_width=1.5)
    fig.add_annotation(
        x=tau_v, y=1.06, xref="x", yref="paper",
        text=label, showarrow=False,
        font=dict(color=col, size=10),
    )

fig.update_layout(
    title=(
        f"Normalized Inter-Trial Decoder — {n_trip} trials × {n_neurons} neurons<br>"
        f"<sup>Median ITI: {iti_prev_s:.1f}s (prev→curr)  ·  {iti_next_s:.1f}s (curr→next)</sup>"
    ),
    xaxis=dict(
        title="Normalized inter-trial position (τ)",
        tickvals=[0, 0.25, 0.5, 0.75, 1.0],
        ticktext=["0 · prev", "0.25", "0.5 · choice", "0.75", "1 · next"],
    ),
    yaxis_title="Decoding accuracy (5-fold CV)",
    yaxis=dict(range=[0.3, 1.0]),
    height=420,
    template="plotly_white",
    hovermode="x unified",
)

orig_caption = _FIGURE_CAPTIONS.get("28_decoder.html", "")
extra = (
    f" {n_trip} trials used (left={left_cnt}, right={right_cnt})."
    f" Median ITI: {iti_prev_s:.1f}s (prev→curr), {iti_next_s:.1f}s (curr→next)."
    " 5-fold stratified CV; LogisticRegression (lbfgs, C=1.0); activity z-scored per fold."
)
_FIGURE_CAPTIONS["28_decoder.html"] = (orig_caption + " " + extra).strip()
write_html_with_caption(fig, OUT_DIR / "28_decoder.html")
print("Saved 28_decoder.html")
# %% [markdown]
# ---
# ## 18 · Normalized Inter-Trial Reward Decoder
# 
# Can the population vector predict whether a trial will be rewarded (vs not rewarded)?
# Time is normalized between the previous and next choice (τ=0..1) with the current
# trial's choice pinned at τ=0.5. Each half is independently rescaled to the
# prev→curr and curr→next ITIs. A logistic regression decodes rewarded (True)
# vs not rewarded (False) at 100 τ grid points using 5-fold stratified CV.
# 

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

GRID_N = 100
tau_grid = np.linspace(0.0, 1.0, GRID_N)

# Gather valid trials with a choice time and a defined 'rewarded' flag
valid_trials_reward = []  # list of (center_bin, rewarded_int)
for trial in trials:
    t_choice = trial.get("t chosen")
    outcome  = trial.get("rewarded")
    if t_choice is None or outcome is None:
        continue
    center_bin = int((t_choice - t0_ms) / 1000 * fs30)
    if center_bin < 0 or center_bin >= T:
        continue
    valid_trials_reward.append((center_bin, int(bool(outcome))))

# Build triplets (prev, curr, next) — interior trials only
triplets_reward = []  # (b_prev, b_curr, b_next, rewarded_int)
for i in range(1, len(valid_trials_reward) - 1):
    b_prev, _          = valid_trials_reward[i - 1]
    b_curr, rewarded   = valid_trials_reward[i]
    b_next, _          = valid_trials_reward[i + 1]
    if b_curr <= b_prev or b_next <= b_curr:
        continue
    triplets_reward.append((b_prev, b_curr, b_next, rewarded))

n_trip_r = len(triplets_reward)
labels_r = np.array([lbl for *_, lbl in triplets_reward], dtype=np.int32) if n_trip_r > 0 else np.array([], dtype=np.int32)
counts_r = dict(zip(*np.unique(labels_r, return_counts=True))) if n_trip_r > 0 else {}
no_reward_cnt = int(counts_r.get(0, 0))
reward_cnt    = int(counts_r.get(1, 0))

print(f"Inter-trial reward decoder: {n_trip_r} triplets | reward={reward_cnt}, no_reward={no_reward_cnt}")

# Sample population vector at each τ grid point
X_reward = np.empty((n_trip_r, n_neurons, GRID_N), dtype=np.float32)
for j, (b_prev, b_curr, b_next, _) in enumerate(triplets_reward):
    for k, tau in enumerate(tau_grid):
        if tau <= 0.5:
            b = b_prev + (tau / 0.5) * (b_curr - b_prev)
        else:
            b = b_curr + ((tau - 0.5) / 0.5) * (b_next - b_curr)
        b_idx = int(np.clip(round(b), 0, T - 1))
        X_reward[j, :, k] = x30[:, b_idx]

decoder_acc_reward = np.full(GRID_N, np.nan)
if len(np.unique(labels_r)) >= 2:
    cv_r = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits_r = list(cv_r.split(X_reward[:, :, 0], labels_r))
    for k in range(GRID_N):
        X_bin = X_reward[:, :, k].astype(float)
        fold_accs = []
        for train_idx, test_idx in splits_r:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_bin[train_idx])
            X_te = scaler.transform(X_bin[test_idx])
            clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
            clf.fit(X_tr, labels_r[train_idx])
            fold_accs.append(clf.score(X_te, labels_r[test_idx]))
        decoder_acc_reward[k] = np.mean(fold_accs)
    print(f"Reward accuracy: min={np.nanmin(decoder_acc_reward):.3f}  max={np.nanmax(decoder_acc_reward):.3f}  chance=0.5")
else:
    print("Only one reward class found — reward decoder skipped")
    decoder_acc_reward[:] = 0.5

chance = 0.5

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=tau_grid, y=decoder_acc_reward,
    mode="lines", line=dict(color="seagreen", width=2.5),
    name="5-fold CV accuracy",
    hovertemplate="τ=%{x:.2f}  acc=%{y:.3f}<extra></extra>",
))
fig.add_hline(y=chance, line_dash="dash", line_color="grey", line_width=1.5,
              annotation_text="Chance (0.50)", annotation_position="right")
for tau_v, label, col in [
    (0.0, "Prev choice", "steelblue"),
    (0.5, "This choice", "crimson"),
    (1.0, "Next choice", "steelblue"),
]:
    fig.add_vline(x=tau_v, line_dash="dash", line_color=col, line_width=1.5)
    fig.add_annotation(x=tau_v, y=1.06, xref="x", yref="paper", text=label, showarrow=False,
                       font=dict(color=col, size=10))

iti_prev_s_r = float(np.median([(b_curr - b_prev) / fs30 for b_prev, b_curr, b_next, _ in triplets_reward])) if n_trip_r > 0 else 0.0
iti_next_s_r = float(np.median([(b_next - b_curr) / fs30 for b_prev, b_curr, b_next, _ in triplets_reward])) if n_trip_r > 0 else 0.0

fig.update_layout(
    title=(f"Normalized Inter-Trial Reward Decoder — {n_trip_r} trials × {n_neurons} neurons<br>"
           f"<sup>Median ITI: {iti_prev_s_r:.1f}s (prev→curr) · {iti_next_s_r:.1f}s (curr→next)</sup>"),
    xaxis=dict(title="Normalized inter-trial position (τ)", tickvals=[0, 0.25, 0.5, 0.75, 1.0],
               ticktext=["0 · prev", "0.25", "0.5 · choice", "0.75", "1 · next"],),
    yaxis_title="Decoding accuracy (5-fold CV)",
    yaxis=dict(range=[0.3, 1.0]),
    height=420, template="plotly_white", hovermode="x unified",
)

orig_caption = _FIGURE_CAPTIONS.get("29_reward_decoder.html", "")
extra = (f" {n_trip_r} triplets used (reward={reward_cnt}, no_reward={no_reward_cnt})."
         f" Median ITI: {iti_prev_s_r:.1f}s (prev→curr), {iti_next_s_r:.1f}s (curr→next)."
         " 5-fold stratified CV; LogisticRegression (lbfgs, C=1.0); activity z-scored per fold.")
_FIGURE_CAPTIONS["29_reward_decoder.html"] = (orig_caption + " " + extra).strip()
write_html_with_caption(fig, OUT_DIR / "29_reward_decoder.html")
print("Saved 29_reward_decoder.html")

# %% [markdown]
# ---
# ## 19 · Absolute-Time vs Normalized TI-decoder Checks
#
# Quick checks to compare the normalized-τ decoder above with an absolute-time
# decoder (fixed seconds relative to the current choice) and with ITI-stratified
# normalized decoders. Also compute a permutation null (default `N_PERM=200`) to
# assess significance per τ (Benjamini-Hochberg FDR across τ).

# %%
from sklearn.metrics import balanced_accuracy_score

# Parameters for absolute-time decoder
PRE_S_ABS = 5.0   # seconds before choice
POST_S_ABS = 1.0  # seconds after choice
GRID_N_ABS = 100
abs_time_grid = np.linspace(-PRE_S_ABS, POST_S_ABS, GRID_N_ABS)

# Use the same triplets and labels as the normalized reward decoder (ensures comparability)
if n_trip_r > 0:
    # Filter triplets so the absolute window fits inside recording
    abs_valid_idx = []
    for j, (b_prev, b_curr, b_next, lbl) in enumerate(triplets_reward):
        ts = int(round(b_curr + abs_time_grid[0] * fs30))
        te = int(round(b_curr + abs_time_grid[-1] * fs30))
        if ts >= 0 and te < T:
            abs_valid_idx.append(j)

    if len(abs_valid_idx) == 0:
        print("Absolute-time decoder: no triplets fit the requested window; skipping")
        decoder_acc_abs = np.full(GRID_N_ABS, 0.5)
    else:
        X_abs = np.empty((len(abs_valid_idx), n_neurons, GRID_N_ABS), dtype=np.float32)
        y_abs = np.empty((len(abs_valid_idx),), dtype=np.int32)
        for ii, j in enumerate(abs_valid_idx):
            b_prev, b_curr, b_next, lbl = triplets_reward[j]
            y_abs[ii] = lbl
            for k, t_rel in enumerate(abs_time_grid):
                b_idx = int(np.clip(round(b_curr + t_rel * fs30), 0, T - 1))
                X_abs[ii, :, k] = x30[:, b_idx]

        # Helper to compute per-τ decoding accuracy with stratified CV (adapt n_splits if needed)
        def decode_time_series(X_all, y_all, grid_n=GRID_N_ABS, n_splits=5):
            n_samples = X_all.shape[0]
            decoder_acc = np.full(grid_n, np.nan)
            if len(np.unique(y_all)) < 2:
                print("Only one class present — decoder skipped")
                return np.full(grid_n, 0.5)
            class_counts = np.bincount(y_all)
            max_splits = int(np.min(class_counts))
            n_splits_eff = min(n_splits, max_splits) if max_splits >= 2 else 0
            if n_splits_eff < 2:
                print("Too few samples per class for StratifiedKFold — skipping")
                return np.full(grid_n, 0.5)
            cv = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=42)
            splits = list(cv.split(X_all[:, :, 0], y_all))
            for k in range(grid_n):
                X_bin = X_all[:, :, k]
                fold_accs = []
                for train_idx, test_idx in splits:
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_bin[train_idx])
                    X_te = scaler.transform(X_bin[test_idx])
                    clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
                    clf.fit(X_tr, y_all[train_idx])
                    fold_accs.append(clf.score(X_te, y_all[test_idx]))
                decoder_acc[k] = np.mean(fold_accs)
            return decoder_acc

        decoder_acc_abs = decode_time_series(X_abs, y_abs, grid_n=GRID_N_ABS)
        print(f"Absolute-time decoder computed on {len(abs_valid_idx)} triplets")

    # Compute majority and balanced baselines for reporting
    try:
        majority_baseline = max(np.bincount(y_abs)) / float(len(y_abs))
    except Exception:
        majority_baseline = 0.5
else:
    decoder_acc_abs = np.full(GRID_N_ABS, 0.5)
    majority_baseline = 0.5

# ── ITI-stratified normalized decoder (split by prev→curr ITI tertiles) ─────
iti_prev_s = np.array([(b_curr - b_prev) / fs30 for b_prev, b_curr, b_next, _ in triplets_reward]) if n_trip_r > 0 else np.array([])
if n_trip_r > 0 and len(iti_prev_s) > 0:
    # compute tertile edges
    q1, q2 = np.percentile(iti_prev_s, [33.333, 66.666])
    groups = {
        'short': np.where(iti_prev_s <= q1)[0].tolist(),
        'medium': np.where((iti_prev_s > q1) & (iti_prev_s <= q2))[0].tolist(),
        'long': np.where(iti_prev_s > q2)[0].tolist(),
    }
else:
    groups = {'short': [], 'medium': [], 'long': []}

decoder_acc_iti = {}
for name, idxs in groups.items():
    if len(idxs) == 0:
        decoder_acc_iti[name] = np.full(GRID_N, 0.5)
        continue
    X_grp = X_reward[np.array(idxs), :, :]
    y_grp = labels_r[np.array(idxs)]
    decoder_acc_iti[name] = decode_time_series(X_grp, y_grp, grid_n=GRID_N)
    print(f"ITI group '{name}': {len(idxs)} triplets — decoding done")

# ── Permutation test (on normalized reward decoder across all triplets) ─────
N_PERM = 200
perm_acc = None
pvals = None
significant_mask = None
if n_trip_r > 0 and len(np.unique(labels_r)) >= 2:
    # use same splitting logic as above
    class_counts = np.bincount(labels_r)
    max_splits = int(np.min(class_counts))
    n_splits_eff = min(5, max_splits) if max_splits >= 2 else 0
    if n_splits_eff >= 2:
        cv_perm = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=42)
        splits_perm = list(cv_perm.split(X_reward[:, :, 0], labels_r))
        perm_acc = np.empty((N_PERM, GRID_N), dtype=np.float32)
        rng = np.random.default_rng(42)
        for p in range(N_PERM):
            y_perm = rng.permutation(labels_r)
            fold_means = np.full(GRID_N, np.nan)
            # compute per-τ accuracy for this perm using the same splits
            for k in range(GRID_N):
                X_bin = X_reward[:, :, k]
                fold_accs = []
                for train_idx, test_idx in splits_perm:
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_bin[train_idx])
                    X_te = scaler.transform(X_bin[test_idx])
                    clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
                    clf.fit(X_tr, y_perm[train_idx])
                    fold_accs.append(clf.score(X_te, y_perm[test_idx]))
                perm_acc[p, k] = np.mean(fold_accs)
        # p-values (one-sided: how often perm >= real)
        pvals = (np.sum(perm_acc >= decoder_acc_norm[None, :], axis=0) + 1) / (N_PERM + 1)
        # Benjamini-Hochberg FDR
        alpha = 0.05
        m = len(pvals)
        order = np.argsort(pvals)
        sorted_p = pvals[order]
        thresh = np.arange(1, m + 1) * alpha / m
        below = np.where(sorted_p <= thresh)[0]
        if below.size > 0:
            k_max = below[-1]
            p_crit = sorted_p[k_max]
            significant_mask = pvals <= p_crit
        else:
            significant_mask = np.zeros_like(pvals, dtype=bool)
        print(f"Permutation test done (N={N_PERM}); significant τ count: {np.sum(significant_mask)}")
    else:
        print("Too few samples per class to run permuted StratifiedKFold; skipping permutation test")
else:
    print("No triplets or single-class labels; skipping permutation test")

# ── Save comparison plots
# Absolute-time figure
fig_abs = go.Figure()
fig_abs.add_trace(go.Scatter(x=abs_time_grid, y=decoder_acc_abs, mode='lines', line=dict(color='darkorange', width=2.5), name='5-fold CV acc'))
fig_abs.add_hline(y=majority_baseline, line_dash='dash', line_color='grey', line_width=1.5, annotation_text=f'Majority ({majority_baseline:.2f})', annotation_position='right')
fig_abs.update_layout(title=f'Absolute-Time Reward Decoder — {len(abs_valid_idx) if n_trip_r>0 else 0} triplets', xaxis_title='Time relative to choice (s)', yaxis_title='Decoding accuracy')
write_html_with_caption(fig_abs, OUT_DIR / '30_abs_time_decoder.html')
print('Saved 30_abs_time_decoder.html')

# ITI-stratified normalized decoder figure
fig_iti = go.Figure()
for name, acc in decoder_acc_iti.items():
    fig_iti.add_trace(go.Scatter(x=tau_grid, y=acc, mode='lines+markers', name=f'{name} ITI'))
fig_iti.add_trace(go.Scatter(x=tau_grid, y=decoder_acc_norm, mode='lines', line=dict(color='black', width=2), name='all triplets'))
if significant_mask is not None:
    sig_x = tau_grid[significant_mask]
    sig_y = np.clip(decoder_acc_norm[significant_mask] + 0.02, 0.5, 0.99)
    fig_iti.add_trace(go.Scatter(x=sig_x, y=sig_y, mode='markers', marker=dict(color='black', size=6), name='FDR p<0.05'))
fig_iti.update_layout(title='ITI-stratified Normalized Reward Decoder', xaxis_title='τ', yaxis_title='Decoding accuracy')
write_html_with_caption(fig_iti, OUT_DIR / '31_iti_stratified_decoder.html')
print('Saved 31_iti_stratified_decoder.html')


# %% [markdown]
# ---
# ## 18b · Absolute-Time Choice Decoder
#
# Use the mean distance between consecutive choice times (ms) + 1s as an absolute
# window. We sample `GRID_N` points from `-window` .. `+window` around each trial's
# choice time (in seconds) and train a separate linear decoder per timepoint.
# This complements the normalized inter-trial decoder by testing absolute-time-locked
# signals that do not scale with the ITI.


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

CHOICE_MAP = {0: 0, 1: 1, 0.0: 0, 1.0: 1, "l": 0, "r": 1, "left": 0, "right": 1}

# Collect valid choice times (ms) and compute mean inter-choice distance
choice_times = []
choice_pairs = []  # (t_choice_ms, choice_int)
for trial in trials:
    t_choice = trial.get("t chosen")
    choice = trial.get("choice")
    if t_choice is None or choice is None:
        continue
    choice_raw = choice.lower() if isinstance(choice, str) else choice
    choice_int = CHOICE_MAP.get(choice_raw)
    if choice_int is None:
        continue
    choice_times.append(t_choice)
    choice_pairs.append((t_choice, choice_int))

if len(choice_times) < 2:
    mean_iti_s = 2.0
else:
    choice_times = np.sort(np.array(choice_times))
    diffs = np.diff(choice_times)
    mean_iti_s = float(np.mean(diffs)) / 1000.0

window_s = mean_iti_s + 1.0
PRE_S = POST_S = window_s
GRID_N = 100
time_grid = np.linspace(-PRE_S, POST_S, GRID_N)

print(f"Mean inter-choice distance: {mean_iti_s:.2f}s → window: {window_s:.2f}s (±{window_s:.2f}s)")

# Build trial-level samples: one population vector per trial at each absolute offset
trial_data = []
trial_labels = []
pre_bins = int(np.ceil(PRE_S * fs30))
post_bins = int(np.ceil(POST_S * fs30))
n_skipped = 0

for t_choice_ms, choice_int in sorted(choice_pairs, key=lambda x: x[0]):
    center_bin = int((t_choice_ms - t0_ms) / 1000 * fs30)
    if center_bin - pre_bins < 0 or center_bin + post_bins >= T:
        n_skipped += 1
        continue
    feat = np.zeros((n_neurons, GRID_N), dtype=np.float32)
    for k, t_rel in enumerate(time_grid):
        b_idx = int(np.clip(round(center_bin + t_rel * fs30), 0, T - 1))
        feat[:, k] = x30[:, b_idx]
    trial_data.append(feat)
    trial_labels.append(choice_int)

trial_data = np.array(trial_data, dtype=np.float32)  # (n_trials, n_neurons, GRID_N)
trial_labels = np.array(trial_labels, dtype=np.int32)
n_trials_abs = trial_data.shape[0]
counts_abs = dict(zip(*np.unique(trial_labels, return_counts=True))) if n_trials_abs > 0 else {}
left_cnt_abs = int(counts_abs.get(0, 0))
right_cnt_abs = int(counts_abs.get(1, 0))

print(f"Absolute-time decoder: {n_trials_abs} trials used (skipped {n_skipped}) | left={left_cnt_abs} right={right_cnt_abs}")

decoder_acc_abs = np.full(GRID_N, np.nan)

if len(np.unique(trial_labels)) >= 2:
    cv_abs = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits_abs = list(cv_abs.split(trial_data[:, :, 0], trial_labels))
    for k in range(GRID_N):
        X_bin = trial_data[:, :, k]
        fold_accs = []
        for train_idx, test_idx in splits_abs:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_bin[train_idx])
            X_te = scaler.transform(X_bin[test_idx])
            clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
            clf.fit(X_tr, trial_labels[train_idx])
            fold_accs.append(clf.score(X_te, trial_labels[test_idx]))
        decoder_acc_abs[k] = np.mean(fold_accs)
    print(f"Absolute-time accuracy: min={np.nanmin(decoder_acc_abs):.3f}  max={np.nanmax(decoder_acc_abs):.3f}")
else:
    print("Only one choice class found — absolute-time decoder skipped")
    decoder_acc_abs[:] = 0.5

# Plot absolute-time decoder
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=time_grid, y=decoder_acc_abs,
    mode="lines", line=dict(color="darkorange", width=2.5),
    name="5-fold CV accuracy",
    hovertemplate="t=%{x:.2f}s  acc=%{y:.3f}<extra></extra>",
))
chance_line = 0.5
majority_rate = max(counts_abs.values()) / max(sum(counts_abs.values()), 1) if counts_abs else 0.5
fig.add_hline(y=chance_line, line_dash="dash", line_color="grey", line_width=1.2,
              annotation_text=f"Chance ({chance_line:.2f})", annotation_position="right")
fig.add_hline(y=majority_rate, line_dash="dot", line_color="silver", line_width=1.2,
              annotation_text=f"Majority ({majority_rate:.2f})", annotation_position="left")
fig.add_vline(x=0.0, line_dash="dash", line_color="crimson", line_width=1.5)

fig.update_layout(
    title=(f"Absolute-Time Decoder — Left/Right choice — window={window_s:.1f}s (±{window_s:.1f}s) \n"
           f"{n_trials_abs} trials × {n_neurons} neurons (skipped {n_skipped})"),
    xaxis=dict(title="Time relative to choice (s)", tickvals=[-PRE_S, -PRE_S/2, 0, POST_S/2, POST_S],
               ticktext=[f"-{PRE_S:.1f}s", f"-{PRE_S/2:.1f}s", "0s", f"{POST_S/2:.1f}s", f"{POST_S:.1f}s"]),
    yaxis_title="Decoding accuracy (5-fold CV)",
    yaxis=dict(range=[0.3, 1.0]),
    height=420, template="plotly_white", hovermode="x unified",
)

orig_caption = _FIGURE_CAPTIONS.get("30_absolute_time_decoder.html", "")
extra = (f" {n_trials_abs} trials used (left={left_cnt_abs}, right={right_cnt_abs}); skipped {n_skipped}."
         f" Window: ±{window_s:.1f}s around choice (mean inter-choice {mean_iti_s:.2f}s + 1s)."
         " 5-fold stratified CV; LogisticRegression (lbfgs, C=1.0); activity z-scored per fold.")
_FIGURE_CAPTIONS["30_absolute_time_decoder.html"] = (orig_caption + " " + extra).strip()
write_html_with_caption(fig, OUT_DIR / "30_absolute_time_decoder.html")
print("Saved 30_absolute_time_decoder.html")

# %%

# ── Generate index.html ─────────────────────────────────────────────────────
SECTIONS = [
    ("0 · Session Overview", [
        ("01_session_summary.html", "Session Summary — neuron counts, firing rates, recording duration"),
    ]),
    ("1 · Probe Anatomy & Neuron Quality", [
        ("02_probe_anatomy.html",         "Probe Anatomy — neuron positions colored by firing rate"),
        ("03_neuron_quality_metrics.html","Neuron Quality Metrics — SNR, ISI violations, amplitude…"),
        ("04_depth_vs_firing_rate.html",  "Depth vs Firing Rate — scatter by probe depth"),
    ]),
    ("2 · Firing Rate Distribution", [
        ("05_firing_rate_distribution.html", "FR Distribution — histogram, rank plot, CDF"),
    ]),
    ("3 · Multi-Resolution Sampling", [
        ("06_multiresolution_sampling.html", "Multi-Resolution — same neuron at 5 / 30 / 100 / 300 Hz"),
        ("07_binning_methods.html",          "Binning Methods — binary vs count vs rate vs Gaussian"),
    ]),
    ("4 · Spike Raster", [
        ("08_spike_raster.html", "Spike Raster — first 2 min, neurons sorted by depth"),
    ]),
    ("5 · Population Activity", [
        ("09_population_activity.html", "Population Activity — smoothed rate + per-neuron heatmap"),
    ]),
    ("6 · Trial-Aligned Activity (PSTH)", [
        ("10_psth_heatmap.html",              "PSTH Heatmap — all neurons sorted by peak latency"),
        ("11_psth_correct_vs_incorrect.html", "PSTH Correct vs Incorrect — population mean by outcome"),
    ]),
    ("7 · Neuron-Neuron Correlation Structure", [
        ("12_correlation_heatmap.html",      "Correlation Heatmap — hierarchically clustered"),
        ("13_correlation_distribution.html", "Correlation Distribution — pairwise histogram + top pairs"),
        ("14_top_correlated_pairs.html",     "Top Correlated Pairs — activity traces of top-5 pairs"),
    ]),
    ("8 · Dimensionality & PCA", [
        ("15_pca_scree.html",      "Scree Plot — variance explained per PC"),
        ("16_pca_pc1_vs_pc2.html", "PC1 vs PC2 — colored by behavioral state"),
        ("17_pca_3d.html",         "3D PCA — PC1 × PC2 × PC3 colored by state"),
    ]),
    ("9 · Laminar Activity Profile", [
        ("18_laminar_profile.html",    "Laminar Profile — mean FR per depth bin"),
        ("19_depth_time_heatmap.html", "Depth × Time Heatmap — activity by depth over session"),
    ]),
    ("10 · Autocorrelation & Burstiness", [
        ("20_autocorrelation.html", "Autocorrelation — top-8 neurons, 0–500 ms"),
        ("21_burstiness.html",      "Burstiness & CV-ISI — population distributions"),
    ]),
    ("11 · ISI Distribution", [
        ("22_isi_distribution.html", "ISI Distribution — log-scale histogram, refractory period"),
    ]),
    ("12 · UMAP", [
        ("23_umap.html", "UMAP — non-linear manifold, colored by behavioral state"),
    ]),
    ("13 · Neural Trajectory", [
        ("24_neural_trajectory.html", "Neural Trajectory — PC1 × PC2 time-ordered path, colored by time"),
    ]),
    ("14 · Block-Transition PSTH", [
        ("25_block_psth.html", "Block-Transition PSTH — population response to reward-prob switches"),
    ]),
    ("15 · Fano Factor", [
        ("26_fano_factor.html", "Fano Factor Over Time — population variability vs Poisson baseline"),
    ]),
    ("16 · Single-Trial Population Activity", [
        ("27_singletrial_psth.html", "Single-Trial Heatmap — every trial as a row, sorted by RT"),
    ]),
    ("17 · Linear Decoder", [
        ("28_decoder.html", "Linear Decoder — left/right choice decoded from population at each time bin"),
        ("29_reward_decoder.html", "Normalized Inter-Trial Reward Decoder — rewarded vs not rewarded decoded from population"),
        ("30_absolute_time_decoder.html", "Absolute-Time Decoder — left/right choice decoded in seconds (mean ITI +1s window)"),
        ("30_abs_time_decoder.html", "Absolute-Time Decoder (alternate) — compact absolute-time summary"),
        ("31_iti_stratified_decoder.html", "ITI-stratified Normalized Decoder — short/medium/long ITI comparisons"),
    ]),
]
 
html_parts = []
# Compute number of figures from SECTIONS so header stays in sync
n_figs = sum(len(figs) for _, figs in SECTIONS)
html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neuronal Activity Exploration — JPAS_0023_20230922</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    min-height: 100vh;
    padding: 2.5rem 1.5rem;
  }
  header {
    max-width: 860px;
    margin: 0 auto 2.5rem;
    border-bottom: 1px solid #2d3748;
    padding-bottom: 1.5rem;
  }
  header h1 { font-size: 1.7rem; font-weight: 700; color: #90cdf4; }
  header p  { margin-top: 0.4rem; color: #a0aec0; font-size: 0.92rem; }
  .toc {
    max-width: 860px;
    margin: 0 auto 2.5rem;
    background: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
  }
  .toc h2 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em;
             color: #718096; margin-bottom: 0.8rem; }
  .toc ol { padding-left: 1.2rem; column-count: 2; column-gap: 2rem; }
  .toc li { margin: 0.25rem 0; break-inside: avoid; }
  .toc a  { color: #76e4f7; text-decoration: none; font-size: 0.88rem; }
  .toc a:hover { text-decoration: underline; }
  .section {
    max-width: 860px;
    margin: 0 auto 2rem;
  }
  .section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #fbd38d;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #2d3748;
  }
  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 0.75rem;
  }
  .card {
    background: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    transition: border-color 0.15s, background 0.15s;
    text-decoration: none;
    display: block;
  }
  .card:hover { border-color: #63b3ed; background: #2d3748; }
  .card-num   { font-size: 0.72rem; color: #718096; font-weight: 600;
                letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.35rem; }
  .card-title { font-size: 0.9rem; color: #e2e8f0; font-weight: 500; }
  footer {
    max-width: 860px;
    margin: 3rem auto 0;
    border-top: 1px solid #2d3748;
    padding-top: 1rem;
    font-size: 0.78rem;
    color: #4a5568;
    text-align: center;
  }
</style>
</head>
<body>
<header>
  <h1>Neuronal Activity Exploration</h1>
    <p>Primary session: <strong>JPAS_0023_20230922</strong> &nbsp;·&nbsp; {n_figs} interactive figures</p>
</header>
""")

# Table of contents
html_parts.append('<nav class="toc"><h2>Contents</h2><ol>\n')
for sec_title, _ in SECTIONS:
    anchor = "-".join(sec_title.replace("·", "").replace("(", "").replace(")", "").split()).lower()
    html_parts.append(f'  <li><a href="#{anchor}">{sec_title}</a></li>\n')
html_parts.append("</ol></nav>\n")

# Sections + cards
for sec_title, figures in SECTIONS:
    anchor = "-".join(sec_title.replace("·", "").replace("(", "").replace(")", "").split()).lower()
    html_parts.append(f'<section class="section" id="{anchor}">\n')
    html_parts.append(f'  <div class="section-title">{sec_title}</div>\n')
    html_parts.append('  <div class="cards">\n')
    for fname, desc in figures:
        num = fname.split("_")[0]
        html_parts.append(
            f'    <a class="card" href="{fname}" target="_blank">\n'
            f'      <div class="card-num">Figure {int(num):02d}</div>\n'
            f'      <div class="card-title">{desc}</div>\n'
            f'    </a>\n'
        )
    html_parts.append('  </div>\n</section>\n')

html_parts.append("""<footer>Generated by neuronal_activity_exploration.ipynb</footer>
</body>
</html>
""")

index_path = OUT_DIR / "index.html"
index_path.write_text("".join(html_parts), encoding="utf-8")
print(f"Written → {index_path}")
print(f"\nServe with:  python -m http.server 8080 --directory \"{OUT_DIR}\"")
print(f"Then open:   http://localhost:8080/")



