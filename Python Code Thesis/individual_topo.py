import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from helper_funcs import load_conditions_from_mat
from ssr_helpers import compute_power, compute_snr

# =========================================================
# >>> EDIT THIS BLOCK FOR THE SUBJECT YOU WANT <<<
# (bdf_file, mat_files, col_label, bad_channels, eog_chs, drop_chs, has_rest_blocks)
# =========================================================
SUBJECT = (
        "/Users/katrinosk/Desktop/Thesis/data/Testdata-S01-session2-29-04-26.bdf",
        ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S01-session2.mat"],
        "S01", [], ['EXG3', 'EXG4'], ['EXG5', 'EXG6', 'EXG7', 'EXG8'], False
)

# -------------------------
# Analysis settings (identical to main pipeline)
# -------------------------
TRIGGER_CODES = [10, 20, 30, 40, 50]
EVENT_ID = {"phi1": 10, "phi2": 20, "phi3": 30, "phi4": 40, "phi5": 50}
PHI_ROW_LABELS = ['φ=0', 'φ=π/4', 'φ=π/2', 'φ=3π/4', 'φ=π']

SCALP_CHS = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
    'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7',
    'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2',
    'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4',
    'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
    'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'
]

ICA_THRESHOLD = 2.5
EPOCHS_REJECT_CRITERIA = dict(eeg=100e-6)
REST_TRIGGER = 21
EPOCH_TMIN, EPOCH_TMAX = -0.5, 1.5
CROP_TMIN, CROP_TMAX = 0.3, 1.5

# -------------------------
# Storage: data[measure][phi_idx] = array over info_ref channels
# -------------------------
measures = ['power', 'snr', 'itpc']
data = {m: None for m in measures}
info_ref = None

# -----------------------------------------------------------
# Process the single subject/session
# -----------------------------------------------------------
(bdf_file, mat_files, col_label, bad_channels,
 eog_chs, drop_chs, has_rest_blocks) = SUBJECT

print(f"\nProcessing {col_label}...")

raw = mne.io.read_raw_bdf(bdf_file, preload=True, verbose=False)
raw.set_montage('biosemi64', on_missing='ignore')
raw.drop_channels([ch for ch in drop_chs if ch in raw.ch_names])

ch_types = {'EXG1': 'misc', 'EXG2': 'misc'}
for eog_ch in eog_chs:
    if eog_ch in raw.ch_names:
        ch_types[eog_ch] = 'eog'
raw.set_channel_types(ch_types)

if bad_channels:
    raw.info['bads'] = bad_channels
    raw.interpolate_bads(reset_bads=True, verbose=False)

raw.filter(1, 100, verbose=False)
raw.notch_filter(50, verbose=False)
raw.set_eeg_reference(ref_channels=['EXG1', 'EXG2'], verbose=False)

for ch in raw.info["chs"]:
    print(ch["ch_name"], ch["unit"], ch["unit_mul"])

scalp_picks = mne.pick_channels(raw.ch_names, include=SCALP_CHS)
ica = ICA(n_components=20, random_state=97, max_iter='auto', verbose=False)
ica.fit(raw.copy().filter(1, None, verbose=False), picks=scalp_picks)
eog_inds, _ = ica.find_bads_eog(raw, ch_name=eog_chs,
                                threshold=ICA_THRESHOLD, verbose=False)
ica.exclude = eog_inds
raw = ica.apply(raw.copy(), verbose=False)

events = mne.find_events(raw, stim_channel='Status',
                         shortest_event=1, verbose=False)

if has_rest_blocks:
    rest_ev = events[events[:, 2] == REST_TRIGGER].copy()
    gaps = np.diff(rest_ev[:, 0]) / raw.info['sfreq']
    breaks = np.where(gaps > 10)[0]
    rest_blocks = np.split(rest_ev, breaks + 1)
    mid_start = rest_blocks[1][0, 0]
    mid_end = rest_blocks[1][-1, 0]
    all_stim = events[np.isin(events[:, 2], TRIGGER_CODES)].copy()
    sb1 = all_stim[all_stim[:, 0] < mid_start]
    sb2 = all_stim[all_stim[:, 0] > mid_end]
    c1 = np.asarray(load_conditions_from_mat(mat_files[0])).astype(int).ravel()
    c2 = np.asarray(load_conditions_from_mat(mat_files[1])).astype(int).ravel()
    n1 = min(len(sb1), len(c1)); n2 = min(len(sb2), len(c2))
    sb1 = sb1[:n1].copy(); sb1[:, 2] = c1[:n1] * 10
    sb2 = sb2[:n2].copy(); sb2[:, 2] = c2[:n2] * 10
    stim_events = np.vstack([sb1, sb2])
else:
    stim_ev = events[np.isin(events[:, 2], TRIGGER_CODES)].copy()
    cond = np.asarray(load_conditions_from_mat(mat_files[0])).astype(int).ravel()
    n_use = min(len(stim_ev), len(cond))
    stim_events = stim_ev[:n_use].copy()
    stim_events[:, 2] = cond[:n_use] * 10

epochs = mne.Epochs(
    raw, stim_events, event_id=EVENT_ID,
    tmin=EPOCH_TMIN, tmax=EPOCH_TMAX - 1 / raw.info['sfreq'],
    baseline=(EPOCH_TMIN, 0), preload=True, reject=None, verbose=False
)
epochs_clean = epochs.copy().crop(tmin=EPOCH_TMIN, tmax=CROP_TMAX - 1 / raw.info['sfreq'])
epochs_clean.drop_bad(reject=EPOCHS_REJECT_CRITERIA, verbose=False)
kept = np.where(np.isin(epochs.selection, epochs_clean.selection))[0]
epochs = epochs[kept].copy()
epochs.crop(tmin=CROP_TMIN, tmax=CROP_TMAX - 1 / raw.info['sfreq'])

scalp_present = [ch for ch in SCALP_CHS if ch in epochs.ch_names]
epochs_scalp = epochs.copy().pick(scalp_present)

info_ref = epochs_scalp.info.copy()
ref_names = info_ref.ch_names

# init storage: arrays of (n_phi, n_ref_channels), nan-filled
n_phi = len(EVENT_ID)
for m in measures:
    data[m] = np.full((n_phi, len(ref_names)), np.nan)

for ci, cond_name in enumerate(EVENT_ID.keys()):
    ep = epochs_scalp[cond_name]
    if len(ep) == 0:
        continue

    pow_sub, itpc_sub, freqs, ch_names = compute_power(
        ep, picks=None, n_fft=None, apply_hann=False, demean=True
    )
    idx_40 = np.argmin(np.abs(freqs - 40))

    pow_db = 10 * np.log10(pow_sub[:, idx_40] * 1e12 + 1e-30)
    snr_db, _, _, _, _ = compute_snr(
        pow_sub, freqs, target_freq=40, noise_band=(35, 45), center=False
    )
    itpc_40 = itpc_sub[:, idx_40]

    name_to_idx = {n: k for k, n in enumerate(ref_names)}
    for ch_name, p_v, s_v, i_v in zip(ch_names, pow_db, snr_db, itpc_40):
        if ch_name in name_to_idx:
            j = name_to_idx[ch_name]
            data['power'][ci, j] = p_v
            data['snr'][ci, j] = s_v
            data['itpc'][ci, j] = i_v

print(f"  done ({len(scalp_present)} channels)")

# -----------------------------------------------------------
# Color scales per measure (across this subject's conditions)
# -----------------------------------------------------------
def common_vlim(measure, pct=(2, 98)):
    vals = data[measure].ravel()
    vals = vals[np.isfinite(vals)]
    return np.percentile(vals, pct[0]), np.percentile(vals, pct[1])

VLIM = {
    'power': common_vlim('power'),
    'snr':   common_vlim('snr'),
    'itpc':  (0.0, common_vlim('itpc')[1]),
}
CMAP = {'power': 'RdYlBu_r', 'snr': 'RdYlBu_r', 'itpc': 'RdYlBu_r'}
CBAR_LABEL = {'power': 'Power (dB re µV²)', 'snr': 'SNR (dB)', 'itpc': 'ITPC'}
TITLE = {'power': '40 Hz Absolute Power', 'snr': '40 Hz SNR', 'itpc': '40 Hz ITPC'}

print("\nColor scales:")
for m in measures:
    print(f"  {m}: {VLIM[m][0]:.2f} to {VLIM[m][1]:.2f}")

# -----------------------------------------------------------
# Build ONE figure: columns = measures (power/snr/itpc), rows = phi conditions
# each measure column has its own colorbar
# -----------------------------------------------------------
n_rows = len(EVENT_ID)
n_meas = len(measures)

fig = plt.figure(figsize=(2.6 * n_meas + 0.8, 2.2 * n_rows + 0.8))
# per measure: 1 topo column + 1 thin colorbar column
width_ratios = []
for _ in range(n_meas):
    width_ratios += [1, 0.07]
gs = fig.add_gridspec(n_rows, 2 * n_meas,
                      width_ratios=width_ratios,
                      hspace=0.06, wspace=0.10)

fig.suptitle(f"40 Hz responses by phase condition — {col_label}",
             fontsize=16, fontweight='bold', y=0.995)

for mi, measure in enumerate(measures):
    topo_col = 2 * mi
    cbar_col = 2 * mi + 1
    arr = data[measure]   # (n_phi, n_channels)
    cbar_ax = fig.add_subplot(gs[:, cbar_col])
    im_last = None
    for r in range(n_rows):
        ax = fig.add_subplot(gs[r, topo_col])
        vec = arr[r]
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
            ax.set_title(TITLE[measure], fontsize=13)
        if mi == 0:
            ax.text(-0.28, 0.5, PHI_ROW_LABELS[r], transform=ax.transAxes,
                    rotation=90, va='center', ha='center', fontsize=14)
    if im_last is not None:
        fig.colorbar(im_last, cax=cbar_ax, label=CBAR_LABEL[measure])

out = f"/Users/katrinosk/Desktop/Thesis/figures_final/topo_all_{col_label}_RdYlBu.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved figure:\n  {out}")