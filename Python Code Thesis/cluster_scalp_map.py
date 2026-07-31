import numpy as np
import matplotlib.pyplot as plt
import mne

# -------------------------
# Cluster electrodes (hardcoded from the permutation results)
# -------------------------
CLUSTERS = {
    '40 Hz Power': {
        'p': 0.0156,
        'chs': ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FC5', 'FC3', 'FC1', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'T8'],
    },
    '40 Hz ITPC': {
        'p': 0.0117,
        'chs': ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FC5', 'FC3', 'FC1', 'C1', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'CP4', 'CP2'],
    },
}

# -------------------------
# Build a biosemi64 info just for plotting positions
# -------------------------
montage = mne.channels.make_standard_montage('biosemi64')
ch_names = montage.ch_names
info = mne.create_info(ch_names, sfreq=512., ch_types='eeg')
info.set_montage(montage)

# -------------------------
# Plot: 3 heads, cluster electrodes filled red, others light grey
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

for ax, (title, d) in zip(axes, CLUSTERS.items()):
    # value = 1 for cluster electrodes, 0 otherwise -> two-tone map
    vals = np.array([1.0 if ch in d['chs'] else 0.0 for ch in ch_names])
    mask = np.array([ch in d['chs'] for ch in ch_names])

    mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k',
                       linewidth=0, markersize=7)

    im, _ = mne.viz.plot_topomap(
        vals, info, axes=ax, show=False,
        cmap='Reds', vlim=(0, 1),
        mask=mask, mask_params=mask_params,
        contours=0,
    )
    ax.set_title(f"{title}\n(cluster p = {d['p']:.4f})", fontsize=13)

fig.suptitle("Significant cluster electrodes by measure (linear phase trend)",
             fontsize=15, fontweight='bold')
fig.savefig('/Users/katrinosk/Desktop/Thesis/figures_final/cluster_electrodes.png',
            dpi=150, bbox_inches='tight')
plt.show()