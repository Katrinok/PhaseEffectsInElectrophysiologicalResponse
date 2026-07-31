import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from helper_funcs import plot_measure_vs_phase, rm_anova_table, linear_trend_test, preprocess_subject, plot_relative_to_phi1, EVENT_ID
from ssr_helpers import compute_power
import pingouin as pg
import pandas as pd

# -------------------------
# Subject definitions
# Each entry: (bdf_file, mat_files, subject_id, bad_channels, eog_chs, drop_chs, has_rest_blocks)
# Comment out any subjects you want to exclude
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
PHI_VALUES = {
    "phi1": 0,
    "phi2": np.pi / 4,
    "phi3": np.pi / 2,
    "phi4": 3 * np.pi / 4,
    "phi5": np.pi,
}
PHI_LABELS = ['0', 'π/4', 'π/2', '3π/4', 'π']
PHI_TICKS  = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]

#OCCIPITAL_CHANNELS = ['Oz']
#AUDITORY_CHANNELS  = ['Cz']
OCCIPITAL_CHANNELS = ['POz', 'PO3', 'PO4', 'PO8', 'PO7', 'Oz', 'O1', 'O2']
AUDITORY_CHANNELS  = ['F1', 'Fz', 'F2', 'FCz', 'Cz', 'FC1', 'FC2', 'C1', 'C2']

EPOCH_TMIN = -0.5
EPOCH_TMAX = 1.5
CROP_TMIN  = 0.3
CROP_TMAX  = 1.5

crop_tmin = 0.3125
crop_tmax = 1.4375

# -------------------------
# Storage
# -------------------------
all_itpc = {
    'Occipital': {},
    'Auditory':  {},
}

# -------------------------
# Process each subject
# -------------------------
n_total = len(SUBJECTS)

for subj_idx, (bdf_file, mat_files, subject_id, bad_channels,
               eog_chs, drop_chs, has_rest_blocks) in enumerate(SUBJECTS):

    pct = (subj_idx / n_total) * 100
    bar = chr(9608) * int(pct / 5) + chr(9617) * (20 - int(pct / 5))
    print(f"\n[{bar}] {subj_idx}/{n_total} ({pct:.0f}%)")
    print(f"Processing {subject_id} ({subj_idx + 1} of {n_total})...")
    print("=" * 55)

    epochs, pct_rejected = preprocess_subject(
        bdf_file, mat_files, bad_channels, eog_chs, drop_chs, has_rest_blocks,
        reject_uv=100e-6,
        epoch_tmin=0.3, epoch_tmax=1.5,
        reject_tmin=0.3, reject_tmax=1.5,
        crop_tmin=crop_tmin, crop_tmax=crop_tmax,
        baseline=None,
        plot_drops=False,
    )

    print(f"  Epochs: {dict((n, len(epochs[n])) for n in EVENT_ID.keys())}")

    for roi_name, roi_channels in [('Occipital', OCCIPITAL_CHANNELS),
                                    ('Auditory',  AUDITORY_CHANNELS)]:

        roi_chs = [ch for ch in roi_channels if ch in epochs.ch_names]
        if len(roi_chs) == 0:
            print(f"  No channels for {roi_name} ROI, skipping.")
            continue

        epochs_roi = epochs.copy().pick(roi_chs)
        itpc_per_phi = []

        for cond_name in EVENT_ID.keys():
            ep = epochs_roi[cond_name]
            if len(ep) == 0:
                itpc_per_phi.append(np.nan)
                continue

            # compute_power from ssr_helpers returns itpc_sub (n_channels, n_freqs)
            pow_sub, itpc_sub, freqs, ch_names = compute_power(
                ep, picks=None, n_fft=None, apply_hann=True, demean=True
            )

            idx_40 = np.argmin(np.abs(freqs - 40))
            itpc_40hz = float(np.mean(itpc_sub[:, idx_40]))
            itpc_per_phi.append(itpc_40hz)

        all_itpc[roi_name][subject_id] = np.array(itpc_per_phi)
        print(f"  {roi_name} ITPC@40Hz: "
              f"{dict(zip(EVENT_ID.keys(), np.round(itpc_per_phi, 3)))}")

    pct_done = ((subj_idx + 1) / n_total) * 100
    bar_done = chr(9608) * int(pct_done / 5) + chr(9617) * (20 - int(pct_done / 5))
    print(f"  [{bar_done}] {subj_idx + 1}/{n_total} complete ({pct_done:.0f}%)")

# -------------------------
# Plot
# -------------------------
phi_vals = np.array([PHI_VALUES[c] for c in EVENT_ID.keys()])


roi_colors = {'Occipital': 'steelblue', 'Auditory': 'crimson'}
roi_titles = {'Occipital': 'Visual ROI',
              'Auditory':  'Auditory ROI'}

plot_measure_vs_phase(all_itpc, phi_vals, "40 Hz ITPC",
                      "40 Hz ITPC vs Phase Condition — All Subjects",
                      roi_titles, roi_colors, PHI_TICKS, PHI_LABELS, y_min=0, y_max=1, out_path="itpc_vs_phase.png")

plot_relative_to_phi1(all_itpc, phi_vals, "Change in 40 Hz ITPC relative to φ = 0",
                      "Relative 40 Hz ITPC vs Phase Condition",
                      roi_titles, roi_colors, PHI_TICKS, PHI_LABELS, out_path="itpc_relative.png")

# -------------------------
# Summary table
# -------------------------
print(f"\n{'='*70}")
print("40 Hz ITPC per phi condition - all subjects")
print(f"{'='*70}")

for roi_name in ['Occipital', 'Auditory']:
    print(f"\n{roi_name} ROI:")
    subject_data = all_itpc[roi_name]
    header = f"{'Subject':<20}" + "".join(f"{c:>10}" for c in EVENT_ID.keys())
    print(header)
    print("-" * 70)
    for subject_id, itpc_vals in subject_data.items():
        row = f"{subject_id:<20}" + "".join(f"{v:>10.3f}" for v in itpc_vals)
        print(row)
    if subject_data:
        all_vals = np.array(list(subject_data.values()))
        mean_row = f"{'Mean':<20}" + "".join(
            f"{np.nanmean(all_vals[:, i]):>10.3f}" for i in range(5))
        print("-" * 70)
        print(mean_row)

print(f"{'='*70}")

# -------------------------
# Statistical analysis
# -------------------------
rm_anova_table(all_itpc, EVENT_ID, "ITPC")
linear_trend_test(all_itpc, EVENT_ID, "ITPC")
