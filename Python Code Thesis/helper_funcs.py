import h5py
import numpy as np
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

TRIGGER_CODES = [10, 20, 30, 40, 50]
EVENT_ID = {"phi1": 10, "phi2": 20, "phi3": 30, "phi4": 40, "phi5": 50}
REST_TRIGGER = 21
ICA_THRESHOLD = 2.5
SCALP_CHS = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
    'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7',
    'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2',
    'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4',
    'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
    'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'
]

def load_conditions_from_mat(mat_file):
    """
    Loads dat.conditions from the gammaphase_<SID>.mat file.
    Assumes MATLAB struct saved as:
        save(..., 'dat', '-v7.3')  OR regular MAT struct
    """
    with h5py.File(mat_file, 'r') as f:
        # Inspect keys if unsure
        print("Keys in MAT file:", list(f.keys()))

        # Access 'dat' struct
        dat = f['dat']

        print("Fields in 'dat':", list(dat.keys()))

        # Load conditions
        conditions = np.array(dat['conditions']).squeeze()

        # MATLAB stores as column vector → flatten
        conditions = conditions.astype(int)

    return conditions


def build_events(raw, mat_files, has_rest_blocks):
    """Build the stimulus event array with phase-condition codes."""
    events = mne.find_events(raw, stim_channel='Status',
                             shortest_event=1, verbose=False)
    sfreq = raw.info['sfreq']

    if has_rest_blocks:
        rest_ev = events[events[:, 2] == REST_TRIGGER].copy()
        gaps = np.diff(rest_ev[:, 0]) / sfreq
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
        return np.vstack([sb1, sb2])
    else:
        stim_ev = events[np.isin(events[:, 2], TRIGGER_CODES)].copy()
        cond = np.asarray(load_conditions_from_mat(mat_files[0])).astype(int).ravel()
        n_use = min(len(stim_ev), len(cond))
        stim_ev = stim_ev[:n_use].copy()
        stim_ev[:, 2] = cond[:n_use] * 10
        return stim_ev


def preprocess_subject(bdf_file, mat_files, bad_channels, eog_chs, drop_chs,
                       has_rest_blocks,
                       reject_uv=100e-6,
                       epoch_tmin=-0.5, epoch_tmax=1.5,
                       reject_tmin=-0.5, reject_tmax=1.5,
                       crop_tmin=0.3, crop_tmax=1.5,
                       baseline=(-0.5, 0), plot_drops=False):
    raw = mne.io.read_raw_bdf(bdf_file, preload=True, verbose=False)
    raw.set_montage('biosemi64', on_missing='ignore')
    raw.drop_channels([ch for ch in drop_chs if ch in raw.ch_names])

    # type EOG now; reference to mastoids BEFORE making them misc
    ch_types = {ch: 'eog' for ch in eog_chs if ch in raw.ch_names}
    raw.set_channel_types(ch_types)

    raw.filter(1, 100, verbose=False)
    raw.notch_filter(50, verbose=False)
    raw.set_eeg_reference(ref_channels=['EXG1', 'EXG2'], verbose=False)

    # now mastoids are misc -> excluded from interpolation, ICA, rejection
    raw.set_channel_types({'EXG1': 'misc', 'EXG2': 'misc'})

    if bad_channels:
        raw.info['bads'] = bad_channels
        raw.interpolate_bads(reset_bads=True, verbose=False)

    scalp_picks = mne.pick_channels(raw.ch_names, include=SCALP_CHS)
    ica = ICA(n_components=20, random_state=97, max_iter='auto', verbose=False)
    ica.fit(raw.copy().filter(1, None, verbose=False), picks=scalp_picks)
    eog_inds, _ = ica.find_bads_eog(raw, ch_name=eog_chs,
                                    threshold=ICA_THRESHOLD, verbose=False)
    ica.exclude = eog_inds
    raw = ica.apply(raw.copy(), verbose=False)

    stim_events = build_events(raw, mat_files, has_rest_blocks)

    sfreq = raw.info['sfreq']
    epochs = mne.Epochs(
        raw, stim_events, 
        event_id=EVENT_ID,
        tmin=epoch_tmin, 
        tmax=epoch_tmax - 1/sfreq,
        baseline=baseline, 
        preload=True,
        reject=None, 
        verbose=False
        )

   # reject epochs based on EEG amplitude in the specified time window
    ec = epochs.copy().crop(tmin=reject_tmin, tmax=reject_tmax - 1/sfreq, verbose=False)
    ec.drop_bad(reject=dict(eeg=reject_uv), verbose=False)

    if plot_drops:
        ec.plot_drop_log() 
    kept = np.where(np.isin(epochs.selection, ec.selection))[0]

    if len(kept) == 0:
        raise RuntimeError(
            f"All epochs dropped. Check your channels."
        )

    epochs = epochs[kept].copy()
    n_total = len(epochs.drop_log)
    n_kept = len(epochs)
    pct_rejected = 100 * (1 - n_kept / n_total) if n_total else 0
    epochs.crop(tmin=crop_tmin, tmax=crop_tmax - 1/sfreq)
    return epochs, pct_rejected

# def fit_cosine(phi, y):
#     """Fit y(phi) = b0 + b1 cos(phi) + b2 sin(phi).

#     Returns
#     -------
#     phi_opt : float   phase of maximal response, atan2(b2, b1), in (-pi, pi]
#     depth   : float   modulation amplitude, hypot(b1, b2)
#     r2      : float   coefficient of determination of the fit
#     """
#     phi = np.asarray(phi, float)
#     y = np.asarray(y, float)
#     X = np.column_stack([np.ones_like(phi), np.cos(phi), np.sin(phi)])
#     b, *_ = np.linalg.lstsq(X, y, rcond=None)
#     phi_opt = np.arctan2(b[2], b[1])
#     depth = np.hypot(b[1], b[2])
#     yhat = X @ b
#     ss_res = np.sum((y - yhat) ** 2)
#     ss_tot = np.sum((y - y.mean()) ** 2)
#     r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
#     return phi_opt, depth, r2


def plot_measure_vs_phase(all_data, phi_vals, measure_label, fig_title,
                          roi_titles=None, roi_colors=None,
                          phi_ticks=None, phi_labels=None,
                          out_path=None, y_min=None, y_max=None):
    """Plot a per-ROI measure as a function of phase offset.

    Reproduces the SNR figure layout for any measure (SNR, absolute power, ...).
    Grey per-subject curves, group mean +/- SEM, one panel per ROI.

    Parameters
    ----------
    all_data : dict
        {roi_name: {subject_id: array over the phi conditions}}.
        e.g. all_snr  or  all_pow
    phi_vals : array
        Phase-offset values (radians), one per condition, in plotting order.
    measure_label : str
        Y-axis label, e.g. '40 Hz SNR (dB)' or '40 Hz power (dB)'.
    fig_title : str
        Figure suptitle.
    roi_titles : dict | None
        {roi_name: panel_title}. Defaults to the ROI name.
    roi_colors : dict | None
        {roi_name: colour}. Falls back to a default palette.
    phi_ticks, phi_labels : list | None
        X tick positions and labels. If None, uses phi_vals as ticks.
    out_path : str | Path | None
        If given, the figure is saved there.
    """
    roi_names = list(all_data.keys())
    if roi_titles is None:
        roi_titles = {r: r for r in roi_names}
    if roi_colors is None:
        default = ['steelblue', 'crimson', 'darkgreen', 'darkorange']
        roi_colors = {r: default[i % len(default)]
                      for i, r in enumerate(roi_names)}

    fig, axes = plt.subplots(1, len(roi_names), figsize=(7 * len(roi_names), 6), sharey=True, squeeze=False)
    axes = axes[0]
    fig.suptitle(fig_title, fontsize=14, fontweight='bold')

    for ax, roi_name in zip(axes, roi_names):
        subject_data = all_data[roi_name]
        if len(subject_data) == 0:
            ax.set_title(f"{roi_titles.get(roi_name, roi_name)}\n(no data)")
            continue

        # individual subjects (grey)
        for subject_id, vals in subject_data.items():
            ax.plot(phi_vals, vals, color='lightgrey', lw=1.2, alpha=0.7, marker='o', ms=4, zorder=1)
            ax.text(phi_vals[-1] + 0.05, vals[-1], subject_id, fontsize=7, color='grey', va='center', zorder=2)

        # group mean +/- SEM
        all_vals = np.array(list(subject_data.values()))   # (n_subj, n_phi)
        mean_v = np.nanmean(all_vals, axis=0)
        n_per = np.sum(~np.isnan(all_vals), axis=0)
        sem_v = np.nanstd(all_vals, axis=0) / np.sqrt(np.maximum(n_per, 1))

        color = roi_colors.get(roi_name, 'steelblue')
        ax.plot(phi_vals, mean_v, color=color, lw=2.5, marker='o', ms=7, zorder=3, label=f'Mean (n={all_vals.shape[0]})')
        ax.fill_between(phi_vals, mean_v - sem_v, mean_v + sem_v, color=color, alpha=0.2, zorder=2)

        if phi_ticks is not None:
            ax.set_xticks(phi_ticks)
            ax.set_xticklabels(phi_labels if phi_labels is not None else phi_ticks, fontsize=13)
        ax.set_xlabel("Phase Condition φ (rad)", fontsize=16)
        ax.set_ylabel(measure_label, fontsize=14)
        ax.set_title(roi_titles.get(roi_name, roi_name), fontsize=14)
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig

def plot_relative_to_phi1(all_data, phi_vals, measure_label, fig_title,
                          roi_titles=None, roi_colors=None,
                          phi_ticks=None, phi_labels=None,
                          out_path=None, y_min=None, y_max=None):
    """
    Plot change in a measure relative to the phi1 / in-phase condition.
    For dB measures, this gives dB difference relative to phi=0.
    """

    roi_names = list(all_data.keys())

    if roi_titles is None:
        roi_titles = {r: r for r in roi_names}

    if roi_colors is None:
        default = ['steelblue', 'crimson']
        roi_colors = {r: default[i % len(default)] for i, r in enumerate(roi_names)}

    fig, axes = plt.subplots(
        1, len(roi_names),
        figsize=(7 * len(roi_names), 6),
        sharey=True,
        squeeze=False
    )
    axes = axes[0]

    fig.suptitle(fig_title, fontsize=14, fontweight='bold')

    for ax, roi_name in zip(axes, roi_names):
        subject_data = all_data[roi_name]

        if len(subject_data) == 0:
            ax.set_title(f"{roi_titles.get(roi_name, roi_name)}\n(no data)")
            continue

        rel_vals_all = []

        for subject_id, vals in subject_data.items():
            vals = np.asarray(vals, float)

            if np.any(np.isnan(vals)):
                continue

            rel_vals = vals - vals[0]   # relative to phi1 / phi = 0
            rel_vals_all.append(rel_vals)

            ax.plot(
                phi_vals, rel_vals,
                color='lightgrey',
                lw=1.2,
                alpha=0.7,
                marker='o',
                ms=4,
                zorder=1
            )

            ax.text(
                phi_vals[-1] + 0.05,
                rel_vals[-1],
                subject_id,
                fontsize=7,
                color='grey',
                va='center'
            )

        rel_vals_all = np.array(rel_vals_all)

        mean_v = np.nanmean(rel_vals_all, axis=0)
        n_per = np.sum(~np.isnan(rel_vals_all), axis=0)
        sem_v = np.nanstd(rel_vals_all, axis=0, ddof=1) / np.sqrt(np.maximum(n_per, 1))

        color = roi_colors.get(roi_name, 'steelblue')

        ax.plot(
            phi_vals, mean_v,
            color=color,
            lw=2.5,
            marker='o',
            ms=7,
            zorder=3,
            label=f'Mean (n={rel_vals_all.shape[0]})'
        )

        ax.fill_between(
            phi_vals,
            mean_v - sem_v,
            mean_v + sem_v,
            color=color,
            alpha=0.2,
            zorder=2
        )

        ax.axhline(0, color='black', linestyle='--', linewidth=1)

        if phi_ticks is not None:
            ax.set_xticks(phi_ticks)
            ax.set_xticklabels(phi_labels if phi_labels is not None else phi_ticks, fontsize=13)

        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)

        ax.set_xlabel("Phase condition φ (rad)", fontsize=16)
        ax.set_ylabel(measure_label, fontsize=14)
        ax.set_title(roi_titles.get(roi_name, roi_name), fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Print how many participants go up/down relative to phi = 0
        print(f"\n{roi_name} ROI: change relative to phi=0")
        for i in range(1, len(phi_vals)):
            n_down = np.sum(rel_vals_all[:, i] < 0)
            n_up = np.sum(rel_vals_all[:, i] > 0)
            n_same = np.sum(rel_vals_all[:, i] == 0)
            print(
                f"  {phi_labels[i] if phi_labels else i}: "
                f"{n_down} down, {n_up} up, {n_same} unchanged"
            )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')

    plt.show()
    return fig

# def cosine_fit_table(all_data, phi_vals, measure_name):
#     """Per-subject cosine fit + group tests for a per-ROI measure.

#     Prints, for each ROI: a per-subject table of phi_opt / depth / R^2,
#     the Wilcoxon test that modulation depth > 0 across subjects, and the
#     circular mean + clustering R of the optima.

#     Parameters mirror plot_measure_vs_phase. `measure_name` is just a label
#     used in the printout (e.g. 'absolute power', 'SNR').
#     """
#     for roi_name, subject_data in all_data.items():
#         if len(subject_data) < 3:
#             print(f"\n{roi_name}: not enough subjects for cosine fit.")
#             continue

#         print(f"\n{'='*70}")
#         print(f"Per-subject cosine fit ({measure_name}) — {roi_name} ROI")
#         print(f"{'='*70}")
#         print(f"{'Subject':<18}{'φ_opt (rad)':>14}{'φ_opt (deg)':>14}"
#               f"{'depth':>10}{'R²':>8}")
#         print("-" * 70)

#         depths, phi_opts = [], []
#         for subject_id, vals in subject_data.items():
#             if np.any(np.isnan(vals)):
#                 continue
#             phi_opt, depth, r2 = fit_cosine(phi_vals, vals)
#             depths.append(depth)
#             phi_opts.append(phi_opt)
#             print(f"{subject_id:<18}{phi_opt:>14.3f}{np.rad2deg(phi_opt):>14.1f}"
#                   f"{depth:>10.3f}{r2:>8.3f}")

#         depths = np.array(depths)
#         phi_opts = np.array(phi_opts)

#         # (a) any phase tuning?  depth > 0 across subjects
#         try:
#             _, w_p = wilcoxon(depths)
#             print(f"\n  Modulation depth > 0 (Wilcoxon): "
#                   f"median={np.median(depths):.3f}, p={w_p:.4f}")
#         except ValueError:
#             print(f"\n  Modulation depth: median={np.median(depths):.3f} "
#                   f"(too few for test)")

#         # (b) clustering of optima
#         mvec = np.mean(np.exp(1j * phi_opts))
#         circ_mean = np.angle(mvec)
#         R = np.abs(mvec)
#         print(f"  Optimal φ: circular mean = {np.rad2deg(circ_mean):.1f}°, "
#               f"clustering R = {R:.3f}  (N={len(phi_opts)})")
#         print("  (R near 1 = consistent optimum across subjects; "
#               "near 0 = scattered)")
#         print("  NOTE: φ sampled only over [0, π]; φ_opt outside this range "
#               "is an edge/extrapolated estimate, not a precise optimum.")


def rm_anova_table(all_data, event_id, measure_name="measure"):
    """Repeated-measures ANOVA testing the effect of phase condition.
 
    Runs a one-way within-subjects ANOVA (phase condition as the factor) for
    each ROI, with Greenhouse-Geisser correction, and Bonferroni-corrected
    pairwise post-hoc tests when the omnibus test is significant.
 
    Parameters
    ----------
    all_data : dict
        {roi_name: {subject_id: array over the phi conditions}}.
        e.g. all_snr or all_pow.
    event_id : dict
        {condition_name: trigger_code}; only the keys (order) are used as the
        phase-condition labels, e.g. {'phi1':10, ...}.
    measure_name : str
        Label for the dependent variable in the printout (e.g. 'SNR', 'power').
    """
    import pandas as pd
    import pingouin as pg
 
    cond_names = list(event_id.keys())
 
    print(f"\n{'='*70}")
    print(f"Repeated-measures ANOVA — effect of phase on {measure_name}")
    print(f"{'='*70}")
 
    for roi_name, subject_data in all_data.items():
        if len(subject_data) < 3:
            print(f"\n{roi_name}: not enough subjects for ANOVA.")
            continue
 
        print(f"\n{roi_name} ROI (N={len(subject_data)}):")
 
        # long format for pingouin
        rows = []
        for subject_id, vals in subject_data.items():
            for i, cond_name in enumerate(cond_names):
                rows.append({'subject': subject_id,
                             'phi': cond_name,
                             'value': vals[i]})
        df = pd.DataFrame(rows)
 
        aov = pg.rm_anova(data=df, dv='value', within='phi', subject='subject', correction=True, detailed=True)
        print(aov.columns.tolist())
        phi_row = aov[aov['Source'] == 'phi'].iloc[0]
        err_df = aov[aov['Source'] == 'Error'].iloc[0]['DF']
 
        print(f"  F({int(phi_row['DF'])}, {err_df}) = {phi_row['F']:.3f}, "
              f"p = {phi_row['p_unc']:.4f}, "
              f"p-GG = {phi_row['p_GG_corr']:.4f}, "
              f"η²g = {phi_row['ng2']:.3f}")
 
        p_gg = phi_row['p_GG_corr']
        sig = ('***' if p_gg < 0.001 else '**' if p_gg < 0.01
               else '*' if p_gg < 0.05 else 'n.s.')
        print(f"  Result: {sig}")
 
        if p_gg < 0.05:
            print("\n  Post-hoc pairwise t-tests (Bonferroni corrected):")
            posthoc = pg.pairwise_tests(data=df, dv='value', within='phi',
                                        subject='subject', padjust='bonf')
            print(f"  {'Comparison':<18}{'t':>8}{'p (raw)':>11}"
                  f"{'p (corr)':>11}{'sig':>6}")
            print("  " + "-" * 54)
            for _, row in posthoc.iterrows():
                label = f"{row['A']} vs {row['B']}"
                pc = row['p_corr']
                s = ('***' if pc < 0.001 else '**' if pc < 0.01
                     else '*' if pc < 0.05 else 'n.s.')
                print(f"  {label:<18}{row['T']:>8.3f}{row['p_unc']:>11.4f}"
                      f"{pc:>11.4f}{s:>6}")
        else:
            print("  ANOVA not significant — post-hoc tests not warranted.")
 
    print(f"\n{'='*70}")
    print("p-GG = Greenhouse-Geisser corrected p (sphericity correction)")
    print("η²g  = generalized eta squared (small=0.02, medium=0.13, large=0.26)")
    print(f"{'='*70}")

def linear_trend_test(all_data, event_id, measure_name="measure"):
    """Within-subjects linear trend test across ordered phase conditions.

    For each ROI, applies linear contrast weights to each subject's values
    across the (ordered) phase conditions, then tests whether the resulting
    contrast differs from zero across subjects with a one-sample t-test
    (and a Wilcoxon signed-rank test as a non-parametric backup).

    A significant positive contrast = response increases with φ;
    negative = response decreases with φ.

    Parameters
    ----------
    all_data : dict
        {roi_name: {subject_id: array over the phi conditions}}.
    event_id : dict
        {condition_name: trigger_code}; only the number/order of keys is used,
        to set the number of levels. Conditions are assumed equally spaced and
        in ascending φ order.
    measure_name : str
        Label for the printout (e.g. 'SNR', 'absolute power').
    """
    from scipy.stats import ttest_1samp, wilcoxon

    n_levels = len(event_id)
    # linear contrast weights for equally-spaced levels, centred and summing to 0
    # e.g. 5 levels -> [-2, -1, 0, 1, 2]
    weights = np.arange(n_levels) - (n_levels - 1) / 2

    print(f"\n{'='*70}")
    print(f"Linear trend test — effect of phase on {measure_name}")
    print(f"Contrast weights (φ ascending): {weights}")
    print(f"{'='*70}")

    for roi_name, subject_data in all_data.items():
        if len(subject_data) < 3:
            print(f"\n{roi_name}: not enough subjects for trend test.")
            continue

        # per-subject linear contrast value
        contrasts = []
        for subject_id, vals in subject_data.items():
            vals = np.asarray(vals, float)
            if np.any(np.isnan(vals)):
                continue
            contrasts.append(np.dot(weights, vals))
        contrasts = np.array(contrasts)

        if len(contrasts) < 3:
            print(f"\n{roi_name}: not enough complete subjects.")
            continue

        mean_c = contrasts.mean()
        sd_c = contrasts.std(ddof=1)
        t_stat, p_t = ttest_1samp(contrasts, 0.0)
        try:
            _, p_w = wilcoxon(contrasts)
        except ValueError:
            p_w = np.nan

        direction = "increasing" if mean_c > 0 else "decreasing"
        sig = ('***' if p_t < 0.001 else '**' if p_t < 0.01
               else '*' if p_t < 0.05 else 'n.s.')

        print(f"\n{roi_name} ROI (N={len(contrasts)}):")
        print(f"  Mean contrast = {mean_c:.3f} ± {sd_c:.3f} (SD)  "
              f"[{direction} with φ]")
        print(f"  One-sample t-test:  t({len(contrasts)-1}) = {t_stat:.3f}, "
              f"p = {p_t:.4f}  {sig}")
        print(f"  Wilcoxon (backup):  p = {p_w:.4f}")

    print(f"\n{'='*70}")
    print("Positive contrast = response rises with φ; negative = falls with φ.")
    print("Weights are linear & equally spaced; assumes conditions ordered by φ.")
    print(f"{'='*70}")

