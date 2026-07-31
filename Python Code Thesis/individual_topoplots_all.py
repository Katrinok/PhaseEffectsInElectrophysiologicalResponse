import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from helper_funcs import preprocess_subject, EVENT_ID, SCALP_CHS
from ssr_helpers import compute_power, compute_snr

# =========================================================
# 12-subject list (S01 + S02 each as TWO sessions, adjacent)
# (bdf_file, mat_files, col_label, bad_channels, eog_chs, drop_chs, has_rest_blocks)
# =========================================================
SUBJECTS = [
    # --- S01-S03 (no rest blocks, single MAT file) ---
    (
        "/Users/katrinosk/Desktop/Thesis/data/Testdata-S01-session1-18-03-26.bdf",
        ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S01-session1.mat"],
        "S01 session 1", [], ['EXG3','EXG4'], ['EXG5','EXG6','EXG7','EXG8'], False
    ),
    (
        "/Users/katrinosk/Desktop/Thesis/data/Testdata-S01-session2-29-04-26.bdf",
        ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S01-session2.mat"],
        "S01 session 2", [], ['EXG3', 'EXG4'], ['EXG5', 'EXG6', 'EXG7', 'EXG8'], False
    ),
    (
        "/Users/katrinosk/Desktop/Thesis/data/Testdata-S02-session1-20-03-26.bdf",
        ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S02-session1.mat"],
        "S02 session 1", [], ['EXG3','EXG4'], ['EXG5','EXG6','EXG7','EXG8'], False
    ),
    (
        "/Users/katrinosk/Desktop/Thesis/data/Testdata-S02-session2-30-04-2026.bdf",
        ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S02-session2.mat"],
        "S02 session 2", ['Fp2'], ['EXG3', 'EXG4'], ['EXG5', 'EXG6', 'EXG7', 'EXG8'], False
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

EPOCH_TMIN, EPOCH_TMAX = 0.3, 1.5

COLS_PER_IMAGE = 6   # 12 subjects -> two images of 6 columns each

# Storage: data[measure][col_label][phi_idx] = array over info_ref channels
measures = ['power', 'snr', 'itpc']
data = {m: {} for m in measures}
col_labels = []
info_ref = None
crop_tmin = 0.3125
crop_tmax = 1.4375

# ------------------------------
# Process each subject/session
# ------------------------------
for subj_idx, (bdf_file, mat_files, col_label, bad_channels,
               eog_chs, drop_chs, has_rest_blocks) in enumerate(SUBJECTS):

    print(f"\nProcessing {col_label} ({subj_idx + 1}/{len(SUBJECTS)})...")
    col_labels.append(col_label)

    epochs, pct_rejected = preprocess_subject(
        bdf_file, mat_files, bad_channels, eog_chs, drop_chs, has_rest_blocks,
        reject_uv=100e-6,
        epoch_tmin=EPOCH_TMIN, epoch_tmax=EPOCH_TMAX,
        reject_tmin=EPOCH_TMIN, reject_tmax=EPOCH_TMAX,
        crop_tmin=crop_tmin, crop_tmax=crop_tmax,
        baseline=None
    )

    scalp_present = [ch for ch in SCALP_CHS if ch in epochs.ch_names]
    epochs_scalp = epochs.copy().pick(scalp_present)

    if info_ref is None:
        info_ref = epochs_scalp.info.copy()
    ref_names = info_ref.ch_names

    # init storage for this column: arrays of (n_phi, n_ref_channels), nan-filled
    n_phi = len(EVENT_ID)
    for m in measures:
        data[m][col_label] = np.full((n_phi, len(ref_names)), np.nan)

    for ci, cond_name in enumerate(EVENT_ID.keys()):
        ep = epochs_scalp[cond_name]
        if len(ep) == 0:
            continue

        # Hann window for 40 Hz power and ITPC
        pow_h, itpc_sub, freqs, ch_names = compute_power(ep, picks=None, n_fft=None, apply_hann=True, demean=True)
        idx_40 = np.argmin(np.abs(freqs - 40))

        pow_db = 10 * np.log10(pow_h[:, idx_40] * 1e12 + 1e-30)
        itpc_40 = itpc_sub[:, idx_40]

        # No Hann window for SNR
        pow_nh, _, freqs_nh, _ = compute_power(ep, picks=None, n_fft=None, apply_hann=False, demean=True)
        snr_db, _, _, _, _ = compute_snr(pow_nh, freqs_nh, target_freq=40, noise_band=(35, 45), center=False)

        # map each channel back to info_ref order
        name_to_idx = {n: k for k, n in enumerate(ref_names)}
        for ch_name, p_v, s_v, i_v in zip(ch_names, pow_db, snr_db, itpc_40):
            if ch_name in name_to_idx:
                j = name_to_idx[ch_name]
                data['power'][col_label][ci, j] = p_v
                data['snr'][col_label][ci, j] = s_v
                data['itpc'][col_label][ci, j] = i_v

    print(f"  done ({len(scalp_present)} channels)")

# =========================================================
# Common color scales per measure (across ALL subjects + conditions)
# =========================================================
def common_vlim(measure, pct=(2, 98)):
    vals = np.concatenate([data[measure][c].ravel() for c in col_labels])
    vals = vals[np.isfinite(vals)]
    return np.percentile(vals, pct[0]), np.percentile(vals, pct[1])

VLIM = {
    'power': common_vlim('power'),
    'snr':   common_vlim('snr'),
    'itpc':  (0.0, common_vlim('itpc')[1]),
}
CMAP = {'power': 'RdYlBu_r', 'snr': 'RdYlBu_r', 'itpc': 'RdYlBu_r'}
CBAR_LABEL = {'power': 'Power (dB)', 'snr': 'SNR (dB)', 'itpc': 'ITPC'}
TITLE = {'power': '40 Hz Absolute Power', 'snr': '40 Hz SNR', 'itpc': '40 Hz ITPC'}

print("\nColor scales:")
for m in measures:
    print(f"  {m}: {VLIM[m][0]:.2f} to {VLIM[m][1]:.2f}")

    

# =========================================================
# Build the figures: for each measure, two images (6 columns each)
# rows = phi conditions, columns = subjects
# =========================================================
n_rows = len(EVENT_ID)

def make_figure(measure, cols_subset, part_idx):
    n_cols = len(cols_subset)
    fig = plt.figure(figsize=(2.2 * n_cols + 1.0, 2.2 * n_rows + 0.6))
    gs = fig.add_gridspec(n_rows, n_cols + 1,
                          width_ratios=[1] * n_cols + [0.07],
                          hspace=0.06, wspace=0.05)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)]
                     for r in range(n_rows)])
    cbar_ax = fig.add_subplot(gs[:, -1])

    fig.suptitle(f"{TITLE[measure]} by phase condition — individual subjects "
                 f"(part {part_idx})", fontsize=20, fontweight='bold', y=0.995)

    im_last = None
    for c, col_label in enumerate(cols_subset):
        arr = data[measure][col_label]   # (n_phi, n_channels)
        for r in range(n_rows):
            vec = arr[r]
            ax = axes[r, c]
            if np.all(np.isnan(vec)):
                ax.set_axis_off()
                continue
            im, _ = mne.viz.plot_topomap(
                vec, info_ref, axes=ax,
                vlim=VLIM[measure], cmap=CMAP[measure],
                contours=4, show=False
            )
            im_last = im
            if r == 0:
                ax.set_title(col_label, fontsize=14)
            if c == 0:
                ax.text(-0.25, 0.5, PHI_ROW_LABELS[r], transform=ax.transAxes,
                        rotation=90, va='center', ha='center', fontsize=14)

    if im_last is not None:
        fig.colorbar(im_last, cax=cbar_ax, label=CBAR_LABEL[measure])

    out = f"/Users/katrinosk/Desktop/Thesis/figures_final/topo/topo_{measure}_part{part_idx}_RdYlBu.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    return out

# split the 12 columns into chunks of 6
chunks = [col_labels[i:i + COLS_PER_IMAGE]
          for i in range(0, len(col_labels), COLS_PER_IMAGE)]

saved = []
for measure in measures:
    for p, chunk in enumerate(chunks, start=1):
        saved.append(make_figure(measure, chunk, p))

print("\nSaved figures:")
for s in saved:
    print(" ", s)
