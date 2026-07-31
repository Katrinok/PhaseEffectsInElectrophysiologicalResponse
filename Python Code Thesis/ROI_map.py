import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import mne

# -------------------------
# ROI definitions
# -------------------------
OCCIPITAL_CHANNELS = ['POz', 'PO3', 'PO4', 'PO8', 'PO7', 'Oz', 'O1', 'O2']
AUDITORY_CHANNELS  = ['F1', 'Fz', 'F2', 'FCz', 'Cz', 'FC1', 'FC2', 'C1', 'C2']

ROI_COLOUR = {
    'Occipital (visual)':  '#2c7fb8',   # blue
    'Central (auditory)':  '#d7301f',   # red
}
NONROI_COLOUR = '#d9d9d9'               # light grey for context channels

# channel -> ROI lookup
ch_roi = {}
for ch in OCCIPITAL_CHANNELS:
    ch_roi[ch] = 'Occipital (visual)'
for ch in AUDITORY_CHANNELS:
    ch_roi[ch] = 'Central (auditory)'

# -------------------------
# Full biosemi64 layout for context
# -------------------------
montage = mne.channels.make_standard_montage('biosemi64')
ALL_CHS = montage.ch_names

info = mne.create_info(ALL_CHS, sfreq=1.0, ch_types='eeg')
info.set_montage(montage, on_missing='ignore')
pos3d = info.get_montage().get_positions()['ch_pos']

names = [ch for ch in ALL_CHS
         if ch in pos3d and not np.any(np.isnan(pos3d[ch]))]
P = np.array([pos3d[ch] for ch in names])

# -------------------------
# Azimuthal-equidistant projection from the vertex (nose up, left ear left)
#   +x = right, +y = front, +z = up   (MNE head coords)
# -------------------------
x, y, z = P[:, 0], P[:, 1], P[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)
theta = np.arccos(np.clip(z / r, -1, 1))   # polar angle from top (Cz ~ 0)
phi = np.arctan2(y, x)                      # azimuth (+y front)
X = theta * np.cos(phi)                     # left/right
Y = theta * np.sin(phi)                     # front/back (nose up)

# normalise so the outermost electrode sits at radius 0.45 (head circle = 0.5)
rad = np.sqrt(X**2 + Y**2)
scale = 0.5 / rad.max()
X *= scale
Y *= scale

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(9, 9))

# head outline, nose, ears
HEAD_R = 0.5
ax.add_patch(Circle((0, 0), HEAD_R, fill=False, lw=2, color='black', zorder=1))
ax.plot([-0.05, 0, 0.05], [HEAD_R - 0.005, HEAD_R + 0.08, HEAD_R - 0.005],
        color='black', lw=2, zorder=1)                      # nose
for sgn in (-1, 1):                                          # ears
    ax.add_patch(Circle((sgn * HEAD_R, 0), 0.05, fill=False,
                        lw=2, color='black', zorder=1))

# electrodes: ROI channels highlighted, the rest greyed out
for ch, xx, yy in zip(names, X, Y):
    if ch in ch_roi:
        col = ROI_COLOUR[ch_roi[ch]]
        ax.scatter(xx, yy, s=360, color=col, edgecolors='black',
                   linewidths=0.9, zorder=3)
        ax.text(xx, yy, ch, ha='center', va='center',
                fontsize=6.5, fontweight='bold', color='white', zorder=4)
    else:
        ax.scatter(xx, yy, s=180, color=NONROI_COLOUR, edgecolors='grey',
                   linewidths=0.5, zorder=2)
        ax.text(xx, yy, ch, ha='center', va='center',
                fontsize=5.5, color='dimgrey', zorder=2)

# legend = ROI -> colour
handles = [Line2D([0], [0], marker='o', linestyle='',
                  markerfacecolor=ROI_COLOUR[roi], markeredgecolor='black',
                  markersize=11, label=roi) for roi in ROI_COLOUR]
ax.legend(handles=handles, title='Region of interest', loc='center left',
          bbox_to_anchor=(1.0, 0.5), frameon=True, fontsize=12)

ax.set_xlim(-0.72, 0.72)
ax.set_ylim(-0.62, 0.66)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('ROI electrode selection\n(from above, nose up)',
             fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/katrinosk/Desktop/Thesis/figures_final/roi_electrode_map.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved roi_electrode_map.png")