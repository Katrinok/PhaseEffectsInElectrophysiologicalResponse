import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
import mne
from mne.stats import permutation_cluster_1samp_test
from mne.channels import find_ch_adjacency
from helper_funcs import preprocess_subject, EVENT_ID, SCALP_CHS
from ssr_helpers import compute_power, compute_snr
from mne.preprocessing import compute_current_source_density

# -------------------------
# Subject definitions
# Each entry: (bdf_file, mat_files, subject_id, bad_channels, eog_chs, drop_chs, has_rest_blocks)
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
# Analysis settings
# -------------------------
PHI_ROW_LABELS = ['φ=0', 'φ=π/4', 'φ=π/2', 'φ=3π/4', 'φ=π']

# Fixed colormap limit for all three measures
VLIM = 10.0  # ±10 dB for power and SNR, 0-0.5 for ITPC separately
n_subjects = len(SUBJECTS)
info_ref        = None  # store channel info from first subject for plotting
crop_tmin = 0.3125
crop_tmax = 1.4375

# -------------------------
# Plot per phi condition grand average topoplots
# Layout: 3 rows (measures) x 5 columns (phi conditions)
# -------------------------

# Re-run per condition for the per-phi topoplot
# Storage: {measure: {cond: [per-subject arrays]}}
per_phi = {
    'abs_power': {c: [] for c in EVENT_ID.keys()},
    'snr':       {c: [] for c in EVENT_ID.keys()},
    'itpc':      {c: [] for c in EVENT_ID.keys()},
}

print("\nRe-computing per phi condition for topoplot grid...")

for subj_idx, (bdf_file, mat_files, subject_id, bad_channels,
               eog_chs, drop_chs, has_rest_blocks) in enumerate(SUBJECTS):

    pct = (subj_idx / n_subjects) * 100
    bar = chr(9608) * int(pct / 5) + chr(9617) * (20 - int(pct / 5))
    print(f"\n[{bar}] Re-processing {subject_id} ({subj_idx + 1}/{n_subjects})...")

    epochs, pct_rejected = preprocess_subject(
        bdf_file, mat_files, bad_channels, eog_chs, drop_chs, has_rest_blocks,
        reject_uv=100e-6,
        epoch_tmin=-0.5, epoch_tmax=1.5,
        reject_tmin=0.3, reject_tmax=1.5,
        crop_tmin=crop_tmin, crop_tmax=crop_tmax,
        baseline=None,
        plot_drops=False,
    )

    epochs_csd = compute_current_source_density(epochs) # compute CSD to get better scalp topography for 40 Hz
    epochs_scalp = epochs.copy().pick(
        [ch for ch in SCALP_CHS if ch in epochs.ch_names]
    )

    
        # Save channel info from first subject for plotting
    if info_ref is None:
        info_ref = epochs_scalp.info.copy()

    for cond_name in EVENT_ID.keys():
        ep = epochs_scalp[cond_name]
        if len(ep) == 0:
            continue

        # Hann window: used for 40 Hz power and ITPC
        pow_h, itpc_h, freqs, _ = compute_power(
            ep, picks=None, n_fft=None, apply_hann=True, demean=True
        )
        idx_40 = np.argmin(np.abs(freqs - 40))

        abs_pow_db = 10 * np.log10(pow_h[:, idx_40] * 1e12 + 1e-30)
        per_phi['abs_power'][cond_name].append(abs_pow_db)

        itpc_40 = itpc_h[:, idx_40]
        per_phi['itpc'][cond_name].append(itpc_40)

        # No Hann window: used for SNR
        pow_nh, _, freqs_nh, _ = compute_power(
            ep, picks=None, n_fft=None, apply_hann=False, demean=True
        )
        snr_db, _, _, _, _ = compute_snr(
            pow_nh, freqs_nh,
            target_freq=40, noise_band=(35, 45), center=False
        )
        per_phi['snr'][cond_name].append(snr_db)

# --- common color scale for absolute power, computed across all conditions ---
all_power = np.concatenate([
    np.mean(per_phi['abs_power'][c], axis=0)
    for c in EVENT_ID.keys() if len(per_phi['abs_power'][c]) > 0
])
POWER_VLIM = (np.percentile(all_power, 2), np.percentile(all_power, 98))
print(f"\nPower color scale limits (dB): {POWER_VLIM[0]:.1f} to {POWER_VLIM[1]:.1f}")

# Get the SNR limits across all conditions for a common color scale
all_snr = np.concatenate([
    np.mean(per_phi['snr'][c], axis=0)
    for c in EVENT_ID.keys() if len(per_phi['snr'][c]) > 0
])
SNR_VLIM = (np.percentile(all_snr, 2), np.percentile(all_snr, 98))
print(f"SNR color scale limits (dB): {SNR_VLIM[0]:.1f} to {SNR_VLIM[1]:.1f}")

# -------------------------
# Cluster-based permutation test: linear trend across phase, per channel
# -------------------------
adjacency, ch_names = find_ch_adjacency(info_ref, ch_type='eeg')
conds = list(EVENT_ID.keys())
N_LEVELS = len(EVENT_ID)
weights = np.arange(N_LEVELS) - (N_LEVELS - 1) / 2   # [-2,-1,0,1,2]

for MEASURE in ['abs_power', 'snr', 'itpc']:
    n_subj = len(per_phi[MEASURE][conds[0]])
    n_ch = len(info_ref.ch_names)

    trend_maps = np.zeros((n_subj, n_ch))
    for s in range(n_subj):
        subj_stack = np.vstack([per_phi[MEASURE][c][s] for c in conds])
        trend_maps[s] = weights @ subj_stack

    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        trend_maps, adjacency=adjacency, n_permutations=5000,
        tail=0, seed=42, out_type='mask', verbose=False,
    )

    print(f"\nCluster permutation ({MEASURE}, linear φ trend):")
    sig = [i for i, p in enumerate(cluster_pv) if p < 0.05]
    if not sig:
        print("  No significant clusters (p < 0.05).")
    for i in sig:
        ch_in = [info_ref.ch_names[j] for j in np.where(clusters[i])[0]]
        print(f"  Cluster {i}: p = {cluster_pv[i]:.4f}, "
              f"{len(ch_in)} channels: {ch_in}")
        
# -------------------------
# Plot: 3 rows x 5 columns per-phi grand average topoplots
# -------------------------
n_conditions = len(EVENT_ID)
#measure_labels = ['CSD power (dB) - Centered', 'CSD SNR (dB) - Centered', 'ITPC']
measure_labels = ['Power (dB)', 'SNR (dB)', 'ITPC']
measure_keys   = ['abs_power', 'snr', 'itpc']

fig2 = plt.figure(figsize=(4 * n_conditions, 11))
gs2 = fig2.add_gridspec(
    3, n_conditions + 1,
    width_ratios=[1] * n_conditions + [0.08],
    hspace=0.08, wspace=0.05
)

axes2 = np.array([[fig2.add_subplot(gs2[row, col])
                   for col in range(n_conditions)]
                  for row in range(3)])
cbar_axes2 = [fig2.add_subplot(gs2[row, -1]) for row in range(3)]

fig2.suptitle(f"Overall Average 40 Hz Topoplots by Phase Condition",
              fontsize=22, fontweight='bold', y=0.99)

im_last = [None, None, None]

for col, cond_name in enumerate(EVENT_ID.keys()):
    for row, mkey in enumerate(measure_keys):
        data_list = per_phi[mkey][cond_name]
        if len(data_list) == 0:
            continue

        grand = np.mean(data_list, axis=0)   # mean across subjects, per channel

        if mkey == 'itpc':
            grand_plot = grand
            vlim_use   = (0.0, np.percentile(grand, 100))
            cmap_use   = 'RdYlBu_r'

        elif mkey == 'abs_power':
            grand_plot = grand  # raw absolute dB
            vlim_use   = POWER_VLIM
            cmap_use   = 'RdYlBu_r'

        else: # snr
            grand_plot = grand # not centered, just raw SNR in dB
            cmap_use   = 'RdYlBu_r'
            vlim_use   = SNR_VLIM

        im, _ = mne.viz.plot_topomap(
            grand_plot, info_ref,
            axes=axes2[row, col],
            vlim=vlim_use,
            cmap=cmap_use,
            contours=6,
            show=False
        )
        im_last[row] = im
        axes2[0, col].set_title(PHI_ROW_LABELS[col], fontsize=19, fontweight='bold')

cbar_labels = ['dB', 'dB', 'ITPC']
for row, (label, cbar_label) in enumerate(zip(measure_labels, cbar_labels)):
    axes2[row, 0].set_ylabel(label, fontsize=19, labelpad=10)
    if im_last[row] is not None:
        fig2.colorbar(im_last[row], cax=cbar_axes2[row], label=cbar_label)

plt.savefig('/Users/katrinosk/Desktop/Thesis/figures_final/topo/avg_topo.png', dpi=150)
plt.show()
