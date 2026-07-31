import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from helper_funcs import preprocess_subject, EVENT_ID, SCALP_CHS
from ssr_helpers import compute_power, compute_snr

# -------------------------
# Subject definitions
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
# Analysis settings (identical to your main pipeline)
# -------------------------
PHI_VALUES = {"phi1": 0, "phi2": np.pi / 4, "phi3": np.pi / 2,
              "phi4": 3 * np.pi / 4, "phi5": np.pi}
PHI_LABELS = ['0', 'π/4', 'π/2', '3π/4', 'π']
PHI_TICKS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]


BANDS = {
    'Frontal':   ['Fp1','Fpz','Fp2','AF7','AF3','AFz','AF4','AF8','F7','F5','F3','F1','Fz','F2','F4','F6','F8'],
    'Fronto-central': ['FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8'],
    'Central':   ['T7','C5','C3','C1','Cz','C2','C4','C6','T8'],
    'Centro-parietal': ['TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8'],
    'Parietal':  ['P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10'],
    'Occipital': ['PO7','PO3','POz','PO4','PO8','O1','Oz','O2','Iz'],
}

CROP_TMIN = 0.3
CROP_TMAX = 1.5
crop_tmin = 0.3125
crop_tmax = 1.4375

# -------------------------
# Storage: per-channel SNR curves per subject
# {subject_id: {ch_name: snr_array (5,)}}
# -------------------------
per_subject_snr = {}
info_ref = None  # store an info from the first subject for channel positions

n_total = len(SUBJECTS)

for subj_idx, (bdf_file, mat_files, subject_id, bad_channels,
               eog_chs, drop_chs, has_rest_blocks) in enumerate(SUBJECTS):

    print(f"\nProcessing {subject_id} ({subj_idx + 1}/{n_total})...")

    # --- Load and preprocess (identical to main pipeline) ---
    epochs, pct_rejected = preprocess_subject(
        bdf_file, mat_files, bad_channels, eog_chs, drop_chs, has_rest_blocks,
        reject_uv=100e-6,
        epoch_tmin=CROP_TMIN, epoch_tmax=CROP_TMAX,
        reject_tmin=CROP_TMIN, reject_tmax=CROP_TMAX,   # reject on -0.5–1.5
        crop_tmin=crop_tmin,  crop_tmax=crop_tmax,        # steady-state 0.3–1.5
        baseline=None,
    )

    scalp_present = [ch for ch in SCALP_CHS if ch in epochs.ch_names]
    epochs_scalp = epochs.copy().pick(scalp_present)

    if info_ref is None:
        info_ref = epochs_scalp.info.copy()

    # --- Per-channel SNR per condition ---
    ch_snr = {ch: np.full(len(EVENT_ID), np.nan) for ch in scalp_present}

    for ci, cond_name in enumerate(EVENT_ID.keys()):
        ep = epochs_scalp[cond_name]
        if len(ep) == 0:
            continue

        pow_sub, itpc_sub, freqs, ch_names = compute_power(
            ep, picks=None, n_fft=None, apply_hann=False, demean=True
        )
        # per-channel SNR (no averaging across channels here)
        snr_db, snr, signal_power_db, _, _ = compute_snr(
            pow_sub, freqs, target_freq=40, noise_band=(35, 45), center=False
        )
        for ch_name, val in zip(ch_names, snr_db):
            if ch_name in ch_snr:
                ch_snr[ch_name][ci] = val

    per_subject_snr[subject_id] = ch_snr
    print(f"  done ({len(scalp_present)} channels)")

# -------------------------
# Average each channel's SNR curve across participants
# (only channels present in every subject contribute their available values;
#  we use nanmean so an interpolated/missing channel in one subject is handled)
# -------------------------
all_channels = info_ref.ch_names  # ordered as in the montage
phi_vals = np.array([PHI_VALUES[c] for c in EVENT_ID.keys()])

grand_ch_snr = {}

for ch in all_channels:
    stack = []

    for sid in per_subject_snr:
        if ch in per_subject_snr[sid]:
            stack.append(per_subject_snr[sid][ch])  # 5 phase values

    if stack:
        grand_ch_snr[ch] = np.nanmean(np.vstack(stack), axis=0)


channel_stats = {}
for ch in all_channels:
    vals = []

    for sid in per_subject_snr:
        if ch in per_subject_snr[sid]:
            subj_curve = per_subject_snr[sid][ch]  # 5 phase values
            vals.append(np.nanmean(subj_curve))    # average over phase

    vals = np.array(vals)

    if len(vals) > 1:
        channel_stats[ch] = {
            "mean": np.nanmean(vals),
            "sd": np.nanstd(vals, ddof=1),
            "n": len(vals)
        }
# -------------------------
# Color-code channels front -> back using their y-coordinate (anterior-posterior)
# In MNE head coords, +y is anterior (front), -y is posterior (back)
# -------------------------
pos = info_ref.get_montage().get_positions()['ch_pos']
ap_coord = {ch: pos[ch][1] for ch in grand_ch_snr if ch in pos}  # y = anterior-posterior

# normalize: front (high y) -> 0, back (low y) -> 1 so colormap runs front->back
y_vals = np.array(list(ap_coord.values()))
norm = Normalize(vmin=y_vals.min(), vmax=y_vals.max())
cmap = cm.get_cmap('RdYlBu')   # was 'coolwarm'

def channel_color(ch):
    # higher y (front) -> 0.0 end, lower y (back) -> 1.0 end
    return cmap(1.0 - norm(ap_coord[ch]))


# -------------------------
# Rank channels by overall 40 Hz SNR (mean across the 5 phase conditions)
# -------------------------
# mean SNR per channel across phases
mean_snr = {
    ch: stats["mean"]
    for ch, stats in channel_stats.items()
}

# boundary between the CPz row and the Pz row
y_cpz = ap_coord['CPz']
y_pz  = ap_coord['Pz']
boundary_y = (y_cpz + y_pz) / 2   # halfway between the two rows

# back = everything at or behind the Pz line (P*, PO*, O*, Iz); front = the rest
back_chs  = [ch for ch in mean_snr if ap_coord.get(ch, 1) <= boundary_y]
front_chs = [ch for ch in mean_snr if ap_coord.get(ch, 1) >  boundary_y]
print("BACK group:", sorted(back_chs))
print("FRONT group:", sorted(front_chs))

def top_n(chs, n=10):
    return sorted(chs, key=lambda c: mean_snr[c], reverse=True)[:n]

print("\nTop 10 FRONT channels by mean 40 Hz SNR:")
for ch in top_n(front_chs):
    stats = channel_stats[ch]
    print(f"  {ch:<6} {stats['mean']:6.2f} ± {stats['sd']:.2f} dB "
          f"(N={stats['n']})")

print("\nTop 10 POSTERIOR channels by mean 40 Hz SNR:")
for ch in top_n(back_chs):
    stats = channel_stats[ch]
    print(f"  {ch:<6} {stats['mean']:6.2f} ± {stats['sd']:.2f} dB "
          f"(N={stats['n']})")

# -------------------------
# Plot: ROI channel's SNR-vs-phase curve
# -------------------------
band_colors = plt.cm.RdYlBu_r(np.linspace(1, 0, len(BANDS)))   # was plt.cm.coolwarm(...)

fig, ax = plt.subplots(figsize=(9, 6))
for (name, chs), col in zip(BANDS.items(), band_colors):
    present = [c for c in chs if c in grand_ch_snr]
    if not present:
        continue
    band_mean = np.nanmean(np.vstack([grand_ch_snr[c] for c in present]), axis=0)
    ax.plot(phi_vals, band_mean, 'o-', color=col, lw=2, label=name)

ax.set_xticks(PHI_TICKS); ax.set_xticklabels(PHI_LABELS)
ax.set_xlabel('Phase Condition φ (rad)', fontsize=14); ax.set_ylabel('40 Hz SNR (dB)', fontsize=14)
ax.set_title('40 Hz SNR vs Phase — Anterior-to-Posterior Bands', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3); ax.legend(title='Scalp region', fontsize=12, title_fontsize=13)

plt.tight_layout()
plt.savefig('/Users/katrinosk/Desktop/Thesis/figures_final/snr_ROI_channels_vs_phase.png', dpi=150)
plt.show()

# -------------------------
# Plot: every channel's SNR-vs-phase curve
# -------------------------
fig2, ax2 = plt.subplots(figsize=(11, 7))

ordered = sorted(ap_coord.keys(), key=lambda c: ap_coord[c])
for ch in ordered:
    curve = grand_ch_snr[ch]
    if np.all(np.isnan(curve)):
        continue
    ax2.plot(phi_vals, curve, '-', color=channel_color(ch), lw=1.2, alpha=0.8)

ax2.set_xticks(PHI_TICKS); ax2.set_xticklabels(PHI_LABELS)
ax2.set_xlabel('Phase Condition φ (rad)', fontsize=14); ax2.set_ylabel('40 Hz SNR (dB)', fontsize=14)
ax2.set_title('40 Hz SNR vs Phase Condition — all channels (overall average across participants)\n', fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.3)

sm = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap='RdYlBu')
sm.set_array([])
cbar = fig2.colorbar(sm, ax=ax2, ticks=[0, 1])
cbar.ax.set_yticklabels(['Front', 'Back'])
cbar.ax.invert_yaxis()
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig('/Users/katrinosk/Desktop/Thesis/figures_final/snr_all_channels_vs_phase.png', dpi=150)
plt.show()
