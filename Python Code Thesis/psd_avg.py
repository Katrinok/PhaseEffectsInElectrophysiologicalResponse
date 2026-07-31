#%%
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from helper_funcs import preprocess_subject, EVENT_ID
from ssr_helpers import compute_power
from pathlib import Path

# -------------------------
# Output
# -------------------------
OUT_DIR = Path("/Users/katrinosk/Desktop/Thesis/figures_final/spectrum")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Subject definitions  (single session per participant; 10 unique subjects)
# (bdf_file, mat_files, subject_id, bad_channels, eog_chs, drop_chs, has_rest_blocks)
# -------------------------
SUBJECTS = [
    (
        "/Users/katrinosk/Desktop/Thesis/data/Testdata-S01-session2-29-04-26.bdf",
        ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S01-session2.mat"],
        "S01", [], ['EXG3', 'EXG4'], ['EXG5', 'EXG6', 'EXG7', 'EXG8'], False
    ),
    (
        "/Users/katrinosk/Desktop/Thesis/data/Testdata-S02-session2-30-04-2026.bdf",
        ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S02-session2.mat"],
        "S02", ['Fp2'], ['EXG3', 'EXG4'], ['EXG5', 'EXG6', 'EXG7', 'EXG8'], False
    ),
    (
        "/Users/katrinosk/Desktop/Thesis/data/Testdata-S03-18-03-26.bdf",
        ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S03.mat"],
        "S03", ['F6'], ['EXG3', 'EXG4'], ['EXG5', 'EXG6', 'EXG7', 'EXG8'], False
    ),
    (
        "/Users/katrinosk/Desktop/Thesis/data/Testdata-S04-08-05-26.bdf",
        ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S04-1.mat",
         "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S04-2.mat"],
        "S04", ['F7'], ['EXG3', 'EXG5'], ['EXG4', 'EXG6', 'EXG7', 'EXG8'], True
    ),
    (
        "/Users/katrinosk/Desktop/Thesis/data/Testdata-S05-11-05-26.bdf",
        ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S05-1.mat",
         "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S05-2.mat"],
        "S05", ['F3', 'F5', 'FC3', 'FC5'], ['EXG3', 'EXG4'], ['EXG5', 'EXG6', 'EXG7', 'EXG8'], True
    ),
    (
        "/Users/katrinosk/Desktop/Thesis/data/Testdata-S06-12-05-26.bdf",
        ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S06-1.mat",
         "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S06-2.mat"],
        "S06", [], ['EXG3', 'EXG4'], ['EXG5', 'EXG6', 'EXG7', 'EXG8'], True
    ),
    (
        "/Users/katrinosk/Desktop/Thesis/data/Testdata-S07-14-05-26.bdf",
        ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S07.mat",
         "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S07-2.mat"],
        "S07", [], ['EXG3', 'EXG4'], ['EXG5', 'EXG6', 'EXG7', 'EXG8'], True
    ),
    (
        "/Users/katrinosk/Desktop/Thesis/data/Testdata-S08-14-05-26.bdf",
        ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S08.mat",
         "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S08-2.mat"],
        "S08", ['T7', 'T8'], ['EXG3', 'EXG4'], ['EXG5', 'EXG6', 'EXG7', 'EXG8'], True
    ),
    (
        "/Users/katrinosk/Desktop/Thesis/data/Testdata-S09-21-05-26.bdf",
        ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S09.mat",
         "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S09-2.mat"],
        "S09", [], ['EXG3', 'EXG4'], ['EXG5', 'EXG6', 'EXG7', 'EXG8'], True
    ),
    # (
    #     "/Users/katrinosk/Desktop/Thesis/data/Testdata-S10-21-05-26.bdf",
    #     ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S10.mat",
    #      "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S10-2.mat"],
    #     "S10", [], ['EXG3', 'EXG4'], ['EXG5', 'EXG6', 'EXG7', 'EXG8'], True
    # ),
]

# -------------------------
# Settings
# -------------------------
PHI_LABELS = {"phi1": "φ = 0", "phi2": "φ = π/4", "phi3": "φ = π/2",
              "phi4": "φ = 3π/4", "phi5": "φ = π"}

# Single-channel "ROIs" for the PSD, matching the TFR figures
PSD_CHANNELS = {'Oz': 'Visual', 'Cz': 'Auditory'}

# Frequency range to display in the PSD
PLOT_FMIN = 2.0
PLOT_FMAX = 90.0

crop_tmin = 0.3125
crop_tmax = 1.4375

# -------------------------
# Storage: {channel: {cond: [per-subject 40Hz-spectrum-in-dB arrays]}}
# -------------------------
psd_data = {ch: {cond: [] for cond in EVENT_ID} for ch in PSD_CHANNELS}
freqs_ref = None

# -------------------------
# Process each subject
# -------------------------
n_total = len(SUBJECTS)
for subj_idx, (bdf_file, mat_files, subject_id, bad_channels,
               eog_chs, drop_chs, has_rest_blocks) in enumerate(SUBJECTS):

    print(f"\n[{subj_idx + 1}/{n_total}] Processing {subject_id} ...")

    epochs, pct_rejected = preprocess_subject(
        bdf_file, mat_files, bad_channels, eog_chs, drop_chs, has_rest_blocks,
        reject_uv=100e-6,
        epoch_tmin=0.3, epoch_tmax=1.5,
        reject_tmin=0.3, reject_tmax=1.5,
        crop_tmin=crop_tmin, crop_tmax=crop_tmax,
        baseline=None,
        plot_drops=False,        # don't pop up 10 windows in a loop
    )

    for ch in PSD_CHANNELS:
        if ch not in epochs.ch_names:
            print(f"  {ch} not found, skipping.")
            continue
        epochs_ch = epochs.copy().pick([ch])
        for cond_name in EVENT_ID:
            ep = epochs_ch[cond_name]
            if len(ep) == 0:
                continue
            pow_sub, _, freqs, _ = compute_power(
                ep, picks=None, n_fft=None, apply_hann=True, demean=True
            )
            spec_db = 10 * np.log10(pow_sub[0] * 1e12 + 1e-30)
            psd_data[ch][cond_name].append(spec_db)
            if freqs_ref is None:
                freqs_ref = freqs

# -------------------------
# Plot: 1 row x 2 cols (Oz, Cz), five phi conditions overlaid
# -------------------------
n_included = len(SUBJECTS)
fmask = (freqs_ref >= PLOT_FMIN) & (freqs_ref <= PLOT_FMAX)

# distinct colors for the five phis
phi_colors = plt.cm.RdYlBu(np.linspace(0, 0.9, len(EVENT_ID)))

fig, axes = plt.subplots(1, len(PSD_CHANNELS), figsize=(14, 5.5), sharey=True)

for ax, (ch, ch_label) in zip(axes, PSD_CHANNELS.items()):
    for ci, cond_name in enumerate(EVENT_ID):
        stack = psd_data[ch][cond_name]
        if len(stack) == 0:
            continue
        grand = np.mean(stack, axis=0)  # grand-average spectrum across subjects
        ax.plot(freqs_ref[fmask], grand[fmask],
                color=phi_colors[ci], lw=1.0,
                label=PHI_LABELS[cond_name])
    ax.axvline(40, color='grey', ls=':', lw=1, alpha=0.7)
    ax.axvline(80, color='grey', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_title(f"{ch} — {ch_label}", fontsize=14)
    ax.grid(True, alpha=0.3)
    if ax is axes[0]:
        ax.set_ylabel(r"Power (dB)", fontsize=14)
    ax.legend(fontsize=12, title="Phase Condition")

fig.suptitle(f"Overall Average 40 Hz Power Spectrum by Phase Condition (n=9)",
             fontsize=16, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_DIR / "psd_byphi_Oz_Cz.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved:", OUT_DIR / "psd_overall_avg_OzCz_dB.png")
