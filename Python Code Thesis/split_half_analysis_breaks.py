#%%
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from helper_funcs import load_conditions_from_mat, preprocess_subject, EVENT_ID
from ssr_helpers import compute_power, compute_snr

# -------------------------
# Settings — adjust per subject
# -------------------------

# --- S01 session 1 (single MAT file, no rest blocks) --------------
# BDF_FILE = "/Users/katrinosk/Desktop/Thesis/data/Testdata-S01-session1-18-03-26.bdf"
# MAT_FILES = ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S01-session1.mat"]
# SUBJECT_ID = "S01 session 1"
# EOG_CHS = ['EXG3', 'EXG4']
# DROP_CHS = ['EXG5', 'EXG6', 'EXG7', 'EXG8']
# BAD_CHANNELS = []
# HAS_REST_BLOCKS = False

# --- S01 session 2 ---
# BDF_FILE = "/Users/katrinosk/Desktop/Thesis/data/Testdata-S01-session2-29-04-26.bdf"
# MAT_FILES = ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S01-session2.mat"]
# SUBJECT_ID = "S01 session 2"
# EOG_CHS = ['EXG3', 'EXG4']
# DROP_CHS = ['EXG5', 'EXG6', 'EXG7', 'EXG8']
# BAD_CHANNELS = []
# HAS_REST_BLOCKS = False

# --- S02 session 1 ---
# BDF_FILE = "/Users/katrinosk/Desktop/Thesis/data/Testdata-S02-session1-20-03-26.bdf"
# MAT_FILES = ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S02-session1.mat"]
# SUBJECT_ID = "S02 session 1"
# EOG_CHS = ['EXG3', 'EXG4']
# DROP_CHS = ['EXG5', 'EXG6', 'EXG7', 'EXG8']
# BAD_CHANNELS = []
# HAS_REST_BLOCKS = False

# --- S02 session 2 ---
# BDF_FILE = "/Users/katrinosk/Desktop/Thesis/data/Testdata-S02-session2-30-04-2026.bdf"
# MAT_FILES = ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S02-session2.mat"]
# SUBJECT_ID = "S02 session 2"
# EOG_CHS = ['EXG3', 'EXG4']
# DROP_CHS = ['EXG5', 'EXG6', 'EXG7', 'EXG8']
# BAD_CHANNELS = ['Fp2']
# HAS_REST_BLOCKS = False

#--- S03 ---
# BDF_FILE = "/Users/katrinosk/Desktop/Thesis/data/Testdata-S03-18-03-26.bdf"
# MAT_FILES = ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S03.mat"]
# SUBJECT_ID = "S03"
# EOG_CHS = ['EXG3', 'EXG4']
# DROP_CHS = ['EXG5', 'EXG6', 'EXG7', 'EXG8']
# BAD_CHANNELS = ['F6']
# HAS_REST_BLOCKS = False


# --- S04 (two MAT files, rest blocks, EXG5 for EOG) ---
# BDF_FILE = "/Users/katrinosk/Desktop/Thesis/data/Testdata-S04-08-05-26.bdf"
# MAT_FILES = [
#     "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S04-1.mat",
#     "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S04-2.mat",
# ]
# SUBJECT_ID = "S04"
# EOG_CHS = ['EXG3', 'EXG5'] # EXG5 used due to EXG4 noise
# DROP_CHS = ['EXG4', 'EXG6', 'EXG7', 'EXG8']
# BAD_CHANNELS = ['F7']
# HAS_REST_BLOCKS = True

# --- S05 ---
BDF_FILE = "/Users/katrinosk/Desktop/Thesis/data/Testdata-S05-11-05-26.bdf"
MAT_FILES = [
    "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S05-1.mat",
    "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S05-2.mat",
]
SUBJECT_ID = "S05"
EOG_CHS = ['EXG3', 'EXG4']
DROP_CHS = ['EXG5', 'EXG6', 'EXG7', 'EXG8']
BAD_CHANNELS = ['F3', 'F5', 'FC3', 'FC5']
HAS_REST_BLOCKS = True

# --- S06 ---
# BDF_FILE = "/Users/katrinosk/Desktop/Thesis/data/Testdata-S06-12-05-26.bdf"
# MAT_FILES = [
#     "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S06-1.mat",
#     "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S06-2.mat",
# ]
# SUBJECT_ID = "S06"
# EOG_CHS = ['EXG3', 'EXG4']
# DROP_CHS = ['EXG5', 'EXG6', 'EXG7', 'EXG8']
# BAD_CHANNELS = []
# HAS_REST_BLOCKS = True

# --- S07 ---
# BDF_FILE = "/Users/katrinosk/Desktop/Thesis/data/Testdata-S07-14-05-26.bdf"
# MAT_FILES = [
#     "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S07.mat",
#     "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S07-2.mat",
# ]
# SUBJECT_ID = "S07"
# EOG_CHS = ['EXG3', 'EXG4']
# DROP_CHS = ['EXG5', 'EXG6', 'EXG7', 'EXG8']
# BAD_CHANNELS = []
# HAS_REST_BLOCKS = True

# --- S08 ---
# BDF_FILE = "/Users/katrinosk/Desktop/Thesis/data/Testdata-S08-14-05-26.bdf"
# MAT_FILES = [
#     "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S08.mat",
#     "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S08-2.mat",
# ]
# SUBJECT_ID = "S08"
# EOG_CHS = ['EXG3', 'EXG4']
# DROP_CHS = ['EXG5', 'EXG6', 'EXG7', 'EXG8']
# BAD_CHANNELS = ['T8', 'T7']
# HAS_REST_BLOCKS = True

# --- S09 ---
# BDF_FILE = "/Users/katrinosk/Desktop/Thesis/data/Testdata-S09-21-05-26.bdf"
# MAT_FILES = [
#     "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S09.mat",
#     "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S09-2.mat",
# ]
# SUBJECT_ID = "S09"
# EOG_CHS = ['EXG3', 'EXG4']
# DROP_CHS = ['EXG5', 'EXG6', 'EXG7', 'EXG8']
# BAD_CHANNELS = []
# HAS_REST_BLOCKS = True

# --- S10 ---
# BDF_FILE = "/Users/katrinosk/Desktop/Thesis/data/Testdata-S10-21-05-26.bdf"
# MAT_FILES = [
#     "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S10.mat",
#     "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S10-2.mat",
# ]
# SUBJECT_ID = "S10"
# EOG_CHS = ['EXG3', 'EXG4']
# DROP_CHS = ['EXG5', 'EXG6', 'EXG7', 'EXG8']
# BAD_CHANNELS = []
# HAS_REST_BLOCKS = True


SEED = 42
PHI_VALUES = {
    "phi1": 0,
    "phi2": np.pi / 4,
    "phi3": np.pi / 2,
    "phi4": 3 * np.pi / 4,
    "phi5": np.pi,
}

#OCCIPITAL_CHANNELS = ['Oz']
#AUDITORY_CHANNELS = ['Cz']
OCCIPITAL_CHANNELS = ['POz','PO3','PO4','PO8','PO7','Oz','O1','O2']
AUDITORY_CHANNELS  = ['F1', 'Fz', 'F2', 'FCz', 'Cz', 'FC1', 'FC2', 'C1', 'C2']
crop_tmin = 0.3125
crop_tmax = 1.4375

# -------------------------
# Helper functions
# -------------------------
def compute_40hz_curve_snr(epochs_obj):
    """Compute 40 Hz SNR per phi condition, averaged across ROI channels."""
    phi_list = []
    snr_list = []

    for cond_name in EVENT_ID.keys():
        ep = epochs_obj[cond_name]
        if len(ep) == 0:
            print(f"Warning: no epochs for {cond_name}, skipping.")
            continue

        pow_sub, itpc_sub, freqs, ch_names = compute_power(
            ep,
            picks=None,
            n_fft=None,
            apply_hann=False,
            demean=True
        )

        snr_db, snr, signal_power_db, signal_power, rel_signal_power = compute_snr(
            pow_sub,
            freqs,
            target_freq=40,
            noise_band=(35, 45),
            center=False
        )

        # Average SNR across ROI channels
        roi_40hz_snr = np.mean(snr_db)
        phi_list.append(PHI_VALUES[cond_name])
        snr_list.append(roi_40hz_snr)

    phi_arr = np.array(phi_list)
    snr_arr = np.array(snr_list)
    sort_idx = np.argsort(phi_arr)
    return phi_arr[sort_idx], snr_arr[sort_idx]


def make_split_indices(epochs_obj, seed=SEED):
    """Split epochs into two balanced halves per condition."""
    rng = np.random.default_rng(seed)
    split1_idx = []
    split2_idx = []

    for cond_name, event_code in EVENT_ID.items():
        cond_idx = np.where(epochs_obj.events[:, 2] == event_code)[0]
        n = len(cond_idx)
        if n < 2:
            print(f"Warning: {cond_name} has fewer than 2 trials.")
            continue

        perm = rng.permutation(n)
        half = n // 2
        split1_idx.extend(cond_idx[perm[:half]])
        split2_idx.extend(cond_idx[perm[half:2 * half]])

    return np.sort(np.array(split1_idx)), np.sort(np.array(split2_idx))


def compute_split_half_for_roi(epochs_obj, channel_list, roi_name,
                                split1_idx, split2_idx):
    """Compute split-half 40 Hz SNR curves for one ROI."""
    roi_chs = [ch for ch in channel_list if ch in epochs_obj.ch_names]
    if len(roi_chs) == 0:
        raise ValueError(f"No channels found for {roi_name} ROI.")
    print(f"\n{roi_name} ROI channels: {roi_chs}")

    roi_epochs = epochs_obj.copy().pick(roi_chs)
    phi_1, snr_1 = compute_40hz_curve_snr(roi_epochs[split1_idx])
    phi_2, snr_2 = compute_40hz_curve_snr(roi_epochs[split2_idx])

    r = np.corrcoef(snr_1, snr_2)[0, 1]
    return phi_1, snr_1, phi_2, snr_2, r


# -------------------------
# Load and preprocess
# -------------------------
epochs, pct_rejected = preprocess_subject(
    BDF_FILE, MAT_FILES, BAD_CHANNELS, EOG_CHS, DROP_CHS, HAS_REST_BLOCKS,
    reject_uv=100e-6,
    epoch_tmin=0.3, epoch_tmax=1.5,
    reject_tmin=0.3, reject_tmax=1.5,
    crop_tmin=crop_tmin, crop_tmax=crop_tmax, # crop to get only the steady-state part of the response
    baseline=None, plot_drops=True,
)

print("\nEpoch counts per condition:")
for cond_name in EVENT_ID.keys():
    print(f"  {cond_name}: {len(epochs[cond_name])}")
#%%
# -------------------------
# Split-half indices
# -------------------------
split1_idx, split2_idx = make_split_indices(epochs, seed=SEED)
print(f"\nSplit 1: {len(split1_idx)} epochs")
print(f"Split 2: {len(split2_idx)} epochs")

# -------------------------
# Compute split-half reliability for both ROIs
# -------------------------
phi_occ_1, occ_snr_1, phi_occ_2, occ_snr_2, r_occ = \
    compute_split_half_for_roi(epochs, OCCIPITAL_CHANNELS,
                                "Occipital", split1_idx, split2_idx)

phi_aud_1, aud_snr_1, phi_aud_2, aud_snr_2, r_aud = \
    compute_split_half_for_roi(epochs, AUDITORY_CHANNELS,
                                "Auditory/Central", split1_idx, split2_idx)

# -------------------------
# Plot
# -------------------------
phi_labels = ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']

fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
fig.suptitle(
    f'Split-half reliability — 40 Hz SNR vs phase offset — {SUBJECT_ID}',
    fontsize=14, fontweight='bold'
)

# Occipital ROI
axes[0].plot(phi_occ_1, occ_snr_1, 'o-', color='steelblue', label='Split 1')
axes[0].plot(phi_occ_2, occ_snr_2, 's-', color='crimson', label='Split 2')
axes[0].set_ylabel('40 Hz SNR (dB)')
axes[0].set_title(f'Occipital ROI — split-half r = {r_occ:.3f}')
axes[0].legend()
axes[0].grid(True, alpha=0.4)
axes[0].set_ylim(-2, 28)

# Auditory/Central ROI
axes[1].plot(phi_aud_1, aud_snr_1, 'o-', color='steelblue', label='Split 1')
axes[1].plot(phi_aud_2, aud_snr_2, 's-', color='crimson', label='Split 2')
axes[1].set_xticks(phi_occ_1)
axes[1].set_xticklabels(phi_labels)
axes[1].set_xlabel(r'Phase offset $\phi$')
axes[1].set_ylabel('40 Hz SNR (dB)')
axes[1].set_title(f'Auditory/Central ROI — split-half r = {r_aud:.3f}')
axes[1].legend()
axes[1].grid(True, alpha=0.4)
axes[1].set_ylim(-2, 28)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# -------------------------
# Summary
# -------------------------
print(f"\n{'='*50}")
print(f"Split-half reliability — {SUBJECT_ID}")
print(f"{'='*50}")
print(f"Occipital ROI:       r = {r_occ:.3f}")
print(f"Auditory/Central ROI: r = {r_aud:.3f}")
print(f"\nOccipital SNR per phi (split 1 / split 2):")
for phi, s1, s2 in zip(phi_occ_1, occ_snr_1, occ_snr_2):
    print(f"  phi={phi:.3f}: {s1:.2f} dB / {s2:.2f} dB")
print(f"\nAuditory SNR per phi (split 1 / split 2):")
for phi, s1, s2 in zip(phi_aud_1, aud_snr_1, aud_snr_2):
    print(f"  phi={phi:.3f}: {s1:.2f} dB / {s2:.2f} dB")
print(f"{'='*50}")
#%%
# -------------------------
# Repeated split-half reliability
# -------------------------
N_SPLITS = 200          # number of random half-splits
RNG = np.random.default_rng(SEED)

def fisher_mean(rs):
    """Average correlations via Fisher z-transform (handles r outside ±1 safely)."""
    rs = np.asarray(rs, dtype=float)
    rs = rs[np.isfinite(rs)]                 # drop any nan (e.g. constant curve)
    rs = np.clip(rs, -0.999999, 0.999999)    # avoid inf at exactly ±1
    if len(rs) == 0:
        return np.nan
    z = np.arctanh(rs)
    return np.tanh(np.mean(z))

def spearman_brown(r):
    """Correct a half-length reliability up to full-length."""
    if not np.isfinite(r):
        return np.nan
    return (2 * r) / (1 + r)

def repeated_split_half(epochs_obj, channel_list, roi_name, n_splits=N_SPLITS):
    """Run n random half-splits, correlate the 5-point SNR curves each time."""
    roi_chs = [ch for ch in channel_list if ch in epochs_obj.ch_names]
    if len(roi_chs) == 0:
        raise ValueError(f"No channels found for {roi_name} ROI.")
    roi_epochs = epochs_obj.copy().pick(roi_chs)

    r_vals = []
    for _ in range(n_splits):
        seed_i = int(RNG.integers(0, 2**31 - 1))
        s1, s2 = make_split_indices(roi_epochs, seed=seed_i)
        _, snr_1 = compute_40hz_curve_snr(roi_epochs[s1])
        _, snr_2 = compute_40hz_curve_snr(roi_epochs[s2])
        # skip degenerate splits (a flat curve gives undefined correlation)
        if np.std(snr_1) == 0 or np.std(snr_2) == 0:
            continue
        if np.any(~np.isfinite(snr_1)) or np.any(~np.isfinite(snr_2)):
            continue
        r_vals.append(np.corrcoef(snr_1, snr_2)[0, 1])

    r_vals = np.array(r_vals)
    r_mean = fisher_mean(r_vals)            # half-length reliability
    r_sb   = spearman_brown(r_mean)         # full-length, Spearman-Brown corrected
    lo, hi = (np.nanpercentile(r_vals, [5, 95]) if len(r_vals) else (np.nan, np.nan))
    return r_vals, r_mean, r_sb, lo, hi

# Run for both ROIs
occ_rvals, occ_rmean, occ_rsb, occ_lo, occ_hi = \
    repeated_split_half(epochs, OCCIPITAL_CHANNELS, "Occipital")
aud_rvals, aud_rmean, aud_rsb, aud_lo, aud_hi = \
    repeated_split_half(epochs, AUDITORY_CHANNELS, "Auditory/Central")

# -------------------------
# Plot: distribution of split-half r across resamples
# -------------------------
fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
fig.suptitle(
    f'Repeated split-half reliability ({N_SPLITS} splits) — '
    f'40 Hz SNR vs phase — {SUBJECT_ID}',
    fontsize=13, fontweight='bold'
)

for ax, rvals, rmean, rsb, lo, hi, name in [
    (axes[0], occ_rvals, occ_rmean, occ_rsb, occ_lo, occ_hi, 'Occipital'),
    (axes[1], aud_rvals, aud_rmean, aud_rsb, aud_lo, aud_hi, 'Auditory/Central'),
]:
    ax.hist(rvals, bins=30, range=(-1, 1), color='steelblue', alpha=0.7)
    ax.axvline(rmean, color='crimson', lw=2,
               label=f'mean r = {rmean:.3f} (Fisher z)')
    ax.axvline(lo, color='gray', ls='--', lw=1)
    ax.axvline(hi, color='gray', ls='--', lw=1, label='5–95th pct')
    ax.set_title(f'{name} — mean r = {rmean:.3f}, '
                 f'Spearman-Brown r = {rsb:.3f}')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes[1].set_xlabel('Split-half correlation r')
axes[1].set_xlim(-1, 1)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# -------------------------
# Summary
# -------------------------
print(f"\n{'='*55}")
print(f"Repeated split-half reliability — {SUBJECT_ID} "
      f"({N_SPLITS} splits)")
print(f"{'='*55}")
print(f"Occipital ROI:")
print(f"  mean r (Fisher z)      = {occ_rmean:.3f}")
print(f"  Spearman-Brown r       = {occ_rsb:.3f}")
print(f"  5-95th percentile      = [{occ_lo:.3f}, {occ_hi:.3f}]")
print(f"  valid splits           = {len(occ_rvals)}/{N_SPLITS}")
print(f"\nAuditory/Central ROI:")
print(f"  mean r (Fisher z)      = {aud_rmean:.3f}")
print(f"  Spearman-Brown r       = {aud_rsb:.3f}")
print(f"  5-95th percentile      = [{aud_lo:.3f}, {aud_hi:.3f}]")
print(f"  valid splits           = {len(aud_rvals)}/{N_SPLITS}")
print(f"{'='*55}")