import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_morlet
from mne.preprocessing import ICA
from helper_funcs import load_conditions_from_mat

# -------------------------
# Subject definitions
# Each entry: (bdf_file, mat_files, subject_id, bad_channels, eog_chs, drop_chs, has_rest_blocks)
# Comment out any subjects you want to exclude from the grand average
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
        "S01 session 2", [], ['EXG3', 'EXG4'], ['EXG5', 'EXG6', 'EXG7', 'EXG8'], False
    ),
    # (
    #     "/Users/katrinosk/Desktop/Thesis/data/Testdata-S02-session1-20-03-26.bdf",
    #     ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S02-session1.mat"],
    #     "S02 session 1", [], ['EXG3','EXG4'], ['EXG5','EXG6','EXG7','EXG8'], False
    # ),
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
    (
        "/Users/katrinosk/Desktop/Thesis/data/Testdata-S10-21-05-26.bdf",
        ["/Users/katrinosk/Desktop/Thesis/data/gammaphase_S10.mat",
         "/Users/katrinosk/Desktop/Thesis/data/gammaphase_S10-2.mat"],
        "S10", [], ['EXG3', 'EXG4'], ['EXG5', 'EXG6', 'EXG7', 'EXG8'], True
    ),
]

# -------------------------
# Analysis settings
# -------------------------
TRIGGER_CODES = [10, 20, 30, 40, 50]
EVENT_ID = {
    "phi1": 10,
    "phi2": 20,
    "phi3": 30,
    "phi4": 40,
    "phi5": 50,
}
SCALP_CHS = [
    'Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1',
    'C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7',
    'P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz','Fp2',
    'AF8','AF4','AFz','Fz','F2','F4','F6','F8','FT8','FC6','FC4',
    'FC2','FCz','Cz','C2','C4','C6','T8','TP8','CP6','CP4','CP2',
    'P2','P4','P6','P8','P10','PO8','PO4','O2'
]

ICA_THRESHOLD = 2.5
EPOCHS_REJECT_CRITERIA = dict(eeg=150e-6)
REST_TRIGGER = 21

FREQS = np.linspace(1, 60, 120)
N_CYCLES = np.maximum(FREQS / 2, 3)

DISPLAY_TMIN = -7.5
DISPLAY_TMAX = 10.0
BASELINE_TMIN = -0.5
BASELINE_TMAX = 0

VMAX = 7.0
VMIN = -7.0

COMPARE_CHANNELS = ['Oz', 'Cz']

# -------------------------
# Storage for grand average accumulation
# Structure: {channel: {condition: [subject_power_arrays]}}
# -------------------------
grand_avg_data = {
    ch: {cond: [] for cond in EVENT_ID.keys()}
    for ch in COMPARE_CHANNELS
}
times_ref = None  # will be set from first subject

# -------------------------
# Process each subject
# -------------------------
for subj_idx, (bdf_file, mat_files, subject_id, bad_channels,
               eog_chs, drop_chs, has_rest_blocks) in enumerate(SUBJECTS):

    n_total = len(SUBJECTS)
    pct = (subj_idx / n_total) * 100
    bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
    print(f"\n[{bar}] {subj_idx}/{n_total} ({pct:.0f}%)")
    print(f"Processing {subject_id} ({subj_idx + 1} of {n_total})...")
    print(f"{'='*60}")

    # Load and preprocess
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
        print(f"  Interpolated: {bad_channels}")

    raw.filter(1, 100, verbose=False)
    raw.notch_filter(50, verbose=False)
    raw.set_eeg_reference(ref_channels=['EXG1', 'EXG2'], verbose=False)

    scalp_picks = mne.pick_channels(raw.ch_names, include=SCALP_CHS)
    ica = ICA(n_components=20, random_state=97, max_iter='auto', verbose=False)
    ica.fit(raw.copy().filter(1, None, verbose=False), picks=scalp_picks)
    eog_inds, _ = ica.find_bads_eog(raw, ch_name=eog_chs,
                                     threshold=ICA_THRESHOLD, verbose=False)
    ica.exclude = eog_inds
    print(f"  ICA excluded: {eog_inds}")
    raw = ica.apply(raw.copy(), verbose=False)

    # Build stimulus event list
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
        n1 = min(len(sb1), len(c1))
        n2 = min(len(sb2), len(c2))
        sb1 = sb1[:n1].copy(); sb1[:, 2] = c1[:n1] * 10
        sb2 = sb2[:n2].copy(); sb2[:, 2] = c2[:n2] * 10
        stim_events = np.vstack([sb1, sb2])
    else:
        stim_events = events[np.isin(events[:, 2], TRIGGER_CODES)].copy()
        cond = np.asarray(
            load_conditions_from_mat(mat_files[0])).astype(int).ravel()
        n_use = min(len(stim_events), len(cond))
        stim_events = stim_events[:n_use].copy()
        stim_events[:, 2] = cond[:n_use] * 10  # use MAT-derived condition codes

    # Create epochs
    epochs = mne.Epochs(
        raw, stim_events, 
        event_id=EVENT_ID,
        tmin=DISPLAY_TMIN,
        tmax=DISPLAY_TMAX - 1/raw.info['sfreq'],
        baseline=(DISPLAY_TMIN, 0),
        preload=True,
        reject=None,
        verbose=False
    )

    # Artifact rejection
    epochs_clean = epochs.copy().crop(tmin=0.3, tmax=1.5 - 1/raw.info['sfreq']) # to keep artifact rejection focused on stimulus period
    epochs_clean.drop_bad(reject=EPOCHS_REJECT_CRITERIA, verbose=False)
    kept = np.where(np.isin(epochs.selection, epochs_clean.selection))[0]

    print(f"  Epochs kept: {len(kept)} / {len(epochs)}")

    # Compute TFR for each compare channel
    for ch in COMPARE_CHANNELS:
        if ch not in epochs.ch_names:
            print(f"  {ch} not found, skipping.")
            continue

        epochs_ch = epochs[kept].copy().pick([ch])

        for cond_name in EVENT_ID.keys():
            ep_cond = epochs_ch[cond_name]
            if len(ep_cond) == 0:
                print(f"  No epochs for {cond_name}, skipping.")
                continue

            tfr = tfr_morlet(
                ep_cond, freqs=FREQS, 
                n_cycles=N_CYCLES,
                use_fft=True, 
                return_itc=False, 
                average=True,
                picks='eeg', 
                verbose=False
            )
            tfr.apply_baseline(
                baseline=(BASELINE_TMIN, BASELINE_TMAX), mode='logratio'
            )
            tfr.crop(tmin=DISPLAY_TMIN, tmax=DISPLAY_TMAX - 1/raw.info['sfreq'])

            # tfr.data shape: (1, n_freqs, n_times) for single channel
            power_db = 10 * tfr.data[0]  # (n_freqs, n_times)
            grand_avg_data[ch][cond_name].append(power_db)

            if times_ref is None:
                times_ref = tfr.times

    pct_done = ((subj_idx + 1) / n_total) * 100
    bar_done = '█' * int(pct_done / 5) + '░' * (20 - int(pct_done / 5))
    print(f"  [{bar_done}] {subj_idx + 1}/{n_total} complete ({pct_done:.0f}%)")

# -------------------------
# Compute grand averages
# -------------------------
n_subjects = len(SUBJECTS)
print(f"\nComputed grand average from {n_subjects} subjects")

grand_avg = {
    ch: {
        cond: np.mean(grand_avg_data[ch][cond], axis=0)
        for cond in EVENT_ID.keys()
        if len(grand_avg_data[ch][cond]) > 0
    }
    for ch in COMPARE_CHANNELS
}

# -------------------------
# Helper: plot one grand average TFR figure
# -------------------------
def plot_grand_avg_figure(tfr_dict, title, freqs, times, vmin, vmax):
    n_cond = len(EVENT_ID)
    fig, axes = plt.subplots(1, n_cond,
                              figsize=(5 * n_cond, 5), sharey=True)
    im = None
    for ax, cond_name in zip(axes, EVENT_ID.keys()):
        power_db = tfr_dict[cond_name]
        im = ax.imshow(
            power_db,
            aspect='auto', origin='lower',
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            vmin=vmin, vmax=vmax,
            cmap='RdBu_r', interpolation='bilinear',
        )
        ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
        ax.axvline(1.5, color='black', linewidth=1.5, linestyle=':')
        ax.axhline(40, color='white', linewidth=1, linestyle=':', alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_title(cond_name)
        if ax == axes[0]:
            ax.set_ylabel("Frequency (Hz)")

    plt.colorbar(im, ax=axes[-1], label='Power (dB re. pre-stimulus baseline)')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    plt.show()

# -------------------------
# Plot overall average per channel
# -------------------------
n_included = len(SUBJECTS)

for ch in COMPARE_CHANNELS:
    if ch not in grand_avg or len(grand_avg[ch]) == 0:
        print(f"No data for {ch}, skipping.")
        continue

    ch_label = "visual (Oz)" if ch == 'Oz' else "auditory (Cz)"
    plot_grand_avg_figure(
        grand_avg[ch],
        f"Grand Average TFR — {ch} ({ch_label}) — N={n_included}",
        FREQS, times_ref, VMIN, VMAX
    )

