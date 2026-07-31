import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from helper_funcs import plot_measure_vs_phase, rm_anova_table, linear_trend_test, preprocess_subject, plot_relative_to_phi1, EVENT_ID
from ssr_helpers import compute_power, compute_snr
import pingouin as pg
import pandas as pd

# -------------------------
# Subject definitions
# Each entry: (bdf_file, mat_files, subject_id, bad_channels, eog_chs, drop_chs, has_rest_blocks)
# Comment out any subjects you want to exclude
# -------------------------
SUBJECTS = [
    # --- S01-S03 (no rest blocks, single MAT file) ---
    # (
    #     "/Users/katrinosk/Desktop/Thesis/data/Testdata-S01-session1-18-03-26.bdf",
    #     ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S01-session1.mat"],
    #     "S01 session 1", [], ['EXG3','EXG4'], ['EXG5','EXG6','EXG7','EXG8'], False
    # ),
    (
        "/Users/katrinosk/Desktop/Thesis/data/Testdata-S01-session2-29-04-26.bdf",
        ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S01-session2.mat"],
        "S01", [], ['EXG3', 'EXG4'], ['EXG5', 'EXG6', 'EXG7', 'EXG8'], False
    ),
    # (
    #     "/Users/katrinosk/Desktop/Thesis/data/Testdata-S02-session1-20-03-26.bdf",
    #     ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S02-session1.mat"],
    #     "S02 session 1", [], ['EXG3','EXG4'], ['EXG5','EXG6','EXG7','EXG8'], False
    # ),
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
PHI_TICKS = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]

# ROI definitions
#OCCIPITAL_CHANNELS = ['Oz']
#AUDITORY_CHANNELS = ['Cz']
VISUAL_CHANNELS = ['POz', 'PO3', 'PO4', 'PO8', 'PO7', 'Oz', 'O1', 'O2']
AUDITORY_CHANNELS  = ['F1', 'Fz', 'F2', 'FCz', 'Cz', 'FC1', 'FC2', 'C1', 'C2']

crop_tmin = 0.3125
crop_tmax = 1.4375

# -------------------------
# Storage: SNR per subject per ROI per condition
# -------------------------
# Structure: {roi_name: {subject_id: snr_array (5,)}}
all_snr = {'Visual': {}, 'Auditory': {}}   # SNR in dB
all_pow = {'Visual': {}, 'Auditory': {}}   # absolute power, dB

def extract_40hz_power_db(pow_arr, freqs, target_freq=40):
    """
    Extract mean 40 Hz power across channels and convert from V² to dB µV².
    """
    idx = np.argmin(np.abs(freqs - target_freq))

    if not np.isclose(freqs[idx], target_freq):
        print(f"Warning: closest frequency bin is {freqs[idx]:.3f} Hz, not exactly {target_freq} Hz")

    power_uv2 = pow_arr[:, idx] * 1e12  # V² -> µV²
    return 10 * np.log10(np.mean(power_uv2) + 1e-30)
# -------------------------
# Process each subject
# -------------------------
n_total = len(SUBJECTS)

for subj_idx, (bdf_file, mat_files, subject_id, bad_channels,
               eog_chs, drop_chs, has_rest_blocks) in enumerate(SUBJECTS):

    pct = (subj_idx / n_total) * 100
    bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
    print(f"\n[{bar}] {subj_idx}/{n_total} ({pct:.0f}%)")
    print(f"Processing {subject_id} ({subj_idx + 1} of {n_total})...")
    print(f"{'='*55}")

    # --- Load and preprocess ---
    epochs, pct_rejected = preprocess_subject(
        bdf_file, mat_files, bad_channels, eog_chs, drop_chs, has_rest_blocks,
        reject_uv=100e-6,
        epoch_tmin=-0.3, epoch_tmax=1.5,
        reject_tmin=0.3, reject_tmax=1.5,
        crop_tmin=crop_tmin, crop_tmax=crop_tmax,
        baseline=None,
        plot_drops=False,
    )


    print(f"  Epoch counts: "
          f"{dict((n, len(epochs[n])) for n in EVENT_ID.keys())}")

    # --- Compute SNR per ROI using ssr_helpers ---
    for roi_name, roi_channels in [('Visual', VISUAL_CHANNELS), ('Auditory',  AUDITORY_CHANNELS)]:

        roi_chs = [ch for ch in roi_channels if ch in epochs.ch_names]
        if len(roi_chs) == 0:
            print(f"  No channels for {roi_name} ROI, skipping.")
            continue

        epochs_roi = epochs.copy().pick(roi_chs)
        snr_per_phi = []
        pow_per_phi = []

        for cond_name in EVENT_ID.keys():
            ep = epochs_roi[cond_name]
            if len(ep) == 0:
                snr_per_phi.append(np.nan)
                pow_per_phi.append(np.nan)
                continue
            
            # Use Hanning for power
            pow_hann, _, freqs_hann, _ = compute_power(
                ep,
                picks=None,
                n_fft=None,
                apply_hann=True,
                demean=True
            )

            power_40_db = extract_40hz_power_db(
                pow_hann,
                freqs_hann,
                target_freq=40
            )
            pow_per_phi.append(power_40_db)

            # compute_power returns per-channel power — average across ROI channels
            pow_snr, _, freqs, _ = compute_power(
                ep,
                picks=None,
                n_fft=None,
                apply_hann=False,
                demean=True
            )

            # compute_snr returns per-channel SNR — average across ROI channels
            snr_db, snr, _, signal_power, _ = compute_snr(
                pow_snr,
                freqs,
                target_freq=40,
                noise_band=(35, 45),
                center=False
            )

            snr_per_phi.append(np.mean(snr_db))

        all_snr[roi_name][subject_id] = np.array(snr_per_phi)
        all_pow[roi_name][subject_id] = np.array(pow_per_phi)
        print(f"  {roi_name} SNR: "
              f"{dict(zip(EVENT_ID.keys(), np.round(snr_per_phi, 2)))}")
        print(f"  {roi_name} power: "
              f"{dict(zip(EVENT_ID.keys(), np.round(pow_per_phi, 2)))}")

    pct_done = ((subj_idx + 1) / n_total) * 100
    bar_done = '█' * int(pct_done / 5) + '░' * (20 - int(pct_done / 5))
    print(f"  [{bar_done}] {subj_idx + 1}/{n_total} complete ({pct_done:.0f}%)")

# -------------------------
# Plot: power vs phase for each ROI
# -------------------------
phi_vals = np.array([PHI_VALUES[c] for c in EVENT_ID.keys()])

roi_colors = {'Visual': 'steelblue', 'Auditory': 'crimson'}
roi_titles = {'Visual': 'Visual ROI', 'Auditory':  'Auditory ROI'}

# --- SNR ---
plot_measure_vs_phase(all_snr, phi_vals, "40 Hz SNR (dB)",
                      "40 Hz SNR vs Phase Condition - Overall Average",
                      roi_titles, roi_colors, PHI_TICKS, PHI_LABELS, y_min=0, y_max=29, out_path="snr_vs_phase.png")

plot_relative_to_phi1(all_snr, phi_vals, "Change in 40 Hz SNR (dB) relative to φ = 0 (dB)",
                      "Relative 40 Hz SNR vs Phase Condition",
                      roi_titles, roi_colors, PHI_TICKS, PHI_LABELS, out_path="snr_relative.png")

# --- Absolute power ---
plot_measure_vs_phase(all_pow, phi_vals, "40 Hz power (dB)",
                      "40 Hz Power vs Phase Condition - Overall Average",
                      roi_titles, roi_colors, PHI_TICKS, PHI_LABELS, out_path="power_vs_phase.png")

plot_relative_to_phi1(all_pow, phi_vals, "Change in 40 Hz power relative to φ = 0 (dB)",
                      "Relative 40 Hz Power vs Phase Condition", roi_titles, roi_colors,
                      PHI_TICKS, PHI_LABELS, out_path="power_relative_to_phi1.png")

# -------------------------
# Statistical analysis: Repeated measures ANOVA to test for phase effect on SNR
# -------------------------

rm_anova_table(all_snr, EVENT_ID, "SNR")
linear_trend_test(all_snr, EVENT_ID, "SNR")


rm_anova_table(all_pow, EVENT_ID, "absolute power")
linear_trend_test(all_pow, EVENT_ID, "absolute power")