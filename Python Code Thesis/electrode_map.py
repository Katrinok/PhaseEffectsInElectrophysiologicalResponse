import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import mne

# -------------------------
# Same band definitions as the SNR band plot
# -------------------------
BANDS = {
    'Frontal':         ['Fp1','Fpz','Fp2','AF7','AF3','AFz','AF4','AF8','F7','F5','F3','F1','Fz','F2','F4','F6','F8'],
    'Fronto-central':  ['FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8'],
    'Central':         ['T7','C5','C3','C1','Cz','C2','C4','C6','T8'],
    'Centro-parietal': ['TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8'],
    'Parietal':        ['P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10'],
    'Occipital':       ['PO7','PO3','POz','PO4','PO8','O1','Oz','O2','Iz'],
}

# Colors must match the band SNR plot: Frontal = red ... Occipital = blue (RdYlBu)
band_order = list(BANDS.keys())
band_cols = plt.cm.RdYlBu(np.linspace(0, 1, len(band_order)))   # red -> blue
band_color = {b: band_cols[i] for i, b in enumerate(band_order)}

# channel -> band lookup
ch_band = {ch: b for b, chs in BANDS.items() for ch in chs}
ALL_CHS = [ch for chs in BANDS.values() for ch in chs]

# -------------------------
# Get 3D electrode positions from the biosemi64 montage
# -------------------------
info = mne.create_info(ALL_CHS, sfreq=1.0, ch_types='eeg')
info.set_montage('biosemi64', on_missing='ignore')
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
scale = 0.5 / rad.max() # the scale of the radius of the outermost electrode to the desired radius
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

# electrodes, colored by band
for ch, xx, yy in zip(names, X, Y):
    col = band_color[ch_band[ch]]
    ax.scatter(xx, yy, s=320, color=col, edgecolors='black',
               linewidths=0.6, zorder=2)
    ax.text(xx, yy, ch, ha='center', va='center',
            fontsize=6.5, zorder=3,
            color='black')

# legend = band -> color
handles = [Line2D([0], [0], marker='o', linestyle='',
                  markerfacecolor=band_color[b], markeredgecolor='black',
                  markersize=11, label=b) for b in band_order]
ax.legend(handles=handles, title='Scalp region', loc='center left',
          bbox_to_anchor=(1.0, 0.5), frameon=True)

ax.set_xlim(-0.72, 0.72)
ax.set_ylim(-0.62, 0.66)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Electrode layout by anterior-posterior band\n(from above, nose up)',
             fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/katrinosk/Desktop/Thesis/figures_final/band_electrode_map.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved band_electrode_map.png")