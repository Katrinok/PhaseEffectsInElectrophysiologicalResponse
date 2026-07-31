
import numpy as np
import mne
from typing import Optional, Tuple, List
from specparam import SpectralModel

def compute_power(
    epochs: mne.Epochs,
    picks: Optional[mne.channels.channels._pick_data_channels] = None,
    n_fft: Optional[int] = None,
    apply_hann: bool = False,
    demean: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute power spectrum of the trial-averaged signal and ITPC per channel
    from an MNE Epochs object.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data. epochs.get_data() yields array of shape (n_trials, n_channels, n_times).
    picks : array-like | str | None
        Channels to include. Uses mne.pick_types / mne.pick_channels logic.
        If None, use all channels present in the Epochs object.
    n_fft : int | None
        Length of the FFT. If None, uses n_times (no zero-padding).
        If provided and > n_times, performs zero-padding.
    apply_hann : bool
        If True, apply a Hann window to each epoch before FFT (per channel).
    demean : bool
        If True, remove the mean (over time) from each epoch before FFT.

    Returns
    -------
    pow_sub : ndarray, shape (n_channels, n_pos_freqs)
        Power spectrum of the *trial-average* per channel.
    itpc_sub : ndarray, shape (n_channels, n_pos_freqs)
        Inter-Trial Phase Coherence (0..1) per channel.
    f : ndarray, shape (n_pos_freqs,)
        Frequency vector in Hz corresponding to columns of pow_sub / itpc_sub.
    ch_names : list of str
        Names of the channels corresponding to the first dimension.

    Notes
    -----
    -Computes:
        * power of the *trial-mean* signal per channel
        * ITPC computed from complex FFT phases per trial, then:
              itpc = |sum( X/|X| over trials )| / N
    - Only non-negative frequencies are returned (DC..Nyquist).
    - FFT scaling: fft(x) / (len(x)/2).
    """
    # Extract data: (n_trials, n_channels, n_times)
    data = epochs.get_data(copy=True)

    # Select channels
    if picks is None:
        picks = np.arange(data.shape[1])
        ch_names = [epochs.ch_names[p] for p in picks]
    else:
        picks = mne.pick_channels(epochs.ch_names, include=picks) if isinstance(picks, (list, tuple)) else picks
        ch_names = [epochs.ch_names[p] for p in picks]
        data = data[:, picks, :]

    n_trials, n_channels, n_times = data.shape
    fs = float(epochs.info["sfreq"])
    N = n_trials

    # Determine FFT length and number of positive frequency bins
    if n_fft is None:
        n_fft = n_times
    if n_fft < n_times:
        raise ValueError(f"n_fft ({n_fft}) must be >= n_times ({n_times}).")

    n_pos = n_fft // 2 + 1  # include Nyquist if even

    # Optional preprocessing: demean and/or Hann window
    if demean:
        data = data - data.mean(axis=-1, keepdims=True)

    if apply_hann:
        # Broadcast Hann window over trials and channels
        hann = np.hanning(n_times)[None, None, :]
        data = data * hann

    # Prepare outputs
    pow_sub = np.zeros((n_channels, n_pos), dtype=float)
    itpc_sub = np.zeros((n_channels, n_pos), dtype=float)

    # Work in (channels, time, trials)
    data_cc = np.transpose(data, (1, 2, 0))  # (ch, time, trial)

    # Per-channel loop
    for cc in range(n_channels):
        # --- Trial-mean signal (over trials), then FFT ---
        M = data_cc[cc].mean(axis=-1)  # shape: (n_times,)
        # Zero-pad if needed
        if n_fft > n_times:
            Mz = np.zeros(n_fft, dtype=float)
            Mz[:n_times] = M
        else:
            Mz = M

        f_fft = np.fft.rfft(Mz)  # rfft already returns non-negative freqs
        f_fft = f_fft / (n_fft / 2.0)
        pow_tmp = np.abs(f_fft ** 2)
        pow_sub[cc, :] = pow_tmp  # already truncated to non-negative freqs

        # --- ITPC from per-trial phase ---
        itpc_accum = np.zeros((N, n_pos), dtype=complex)
        for it in range(N):
            x = data_cc[cc, :, it]
            if n_fft > n_times:
                xz = np.zeros(n_fft, dtype=float)
                xz[:n_times] = x
            else:
                xz = x

            X = np.fft.rfft(xz) / (n_fft / 2.0)
            itpc_accum[it, :] = X

        # Normalize to unit vectors (phase only), sum, take magnitude, divide by N
        # Add small epsilon to avoid division by zero at DC
        eps = np.finfo(float).eps
        itpc_norm = itpc_accum / (np.abs(itpc_accum) + eps)
        itpc_mean = np.abs(np.sum(itpc_norm, axis=0)) / N
        itpc_sub[cc, :] = itpc_mean

    # Frequency vector (non-negative frequencies)
    f = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    return pow_sub, itpc_sub, f, ch_names

def compute_snr(input_pow, freqs, target_freq=40, noise_band=(35, 45),center=False):
    """
    Compute SNR in dB for a given target frequency relative to a noise band.

    Parameters
    ----------
    input_pow : ndarray, shape (n_channels, n_freqs)
        Power spectrum per channel.
    freqs : ndarray, shape (n_freqs,)
        Frequency vector corresponding to columns of input_pow.
    target_freq : float
        Frequency of interest for SNR calculation (e.g., 40 Hz).
    noise_band : tuple of (float, float)
        Frequency range to use as noise baseline (e.g., (35, 45) Hz).
    center : bool
        Whether to center the SNR values across channels.

    Returns
    -------
    snr_db : ndarray, shape (n_channels,)
        SNR in dB for each channel at the target frequency.
    snr : ndarray, shape (n_channels,)
        SNR as a linear ratio (not in dB).
    signal_power_db : ndarray, shape (n_channels,)
        Power at the target frequency in dB.
    signal_power : ndarray, shape (n_channels,)
        Power at the target frequency in linear scale.
    rel_signal_power : ndarray, shape (n_channels,)
        Relative power in dB: target frequency power minus noise baseline power.
    """
    idx_target = np.argmin(np.abs(freqs - target_freq))
    idx_noise = np.where((freqs >= noise_band[0]) & (freqs <= noise_band[1]) &
                         (np.arange(len(freqs)) != idx_target))[0]

    signal_power = input_pow[:, idx_target]
    noise_power = input_pow[:, idx_noise].mean(axis=1)

    signal_power_db = 10 * np.log10(signal_power)
    noise_power_db = 10 * np.log10(noise_power)

    rel_signal_power = signal_power_db - noise_power_db

    snr = signal_power / noise_power
    snr_db = 10 * np.log10(snr)
    if center:
        snr_db = snr_db - np.mean(snr_db)
        rel_signal_power = rel_signal_power - np.mean(rel_signal_power)

    return snr_db, snr, signal_power_db, signal_power, rel_signal_power

def compute_snr_floorfit(input_pow, freqs, target_freq=40, noise_band=(35, 45),center=False, fit_range=(np.nan, np.nan)):
    """
    Compute SNR in dB for a given target frequency relative to a noise band.
    Noise floor is estimated by fitting a 1/f model to the power spectrum and evaluating the fitted model at the noise frequencies.

    Parameters
    ----------
    input_pow : ndarray, shape (n_channels, n_freqs)
        Power spectrum per channel.
    freqs : ndarray, shape (n_freqs,)
        Frequency vector corresponding to columns of input_pow.
    target_freq : float
        Frequency of interest for SNR calculation (e.g., 40 Hz).
    noise_band : tuple of (float, float)
        Frequency range to use as noise baseline (e.g., (35, 45) Hz).
    center : bool
        Whether to center the SNR values across channels.
    fit_range : tuple of (float, float)
        Frequency range to use for floor fitting (e.g., (2, 120) Hz).

    Returns
    -------
    snr_db : ndarray, shape (n_channels,)
        SNR in dB for each channel at the target frequency.
    snr : ndarray, shape (n_channels,)
        SNR as a linear ratio (not in dB).
    signal_power_db : ndarray, shape (n_channels,)
        Power at the target frequency in dB.
    signal_power : ndarray, shape (n_channels,)
        Power at the target frequency in linear scale.
    rel_signal_power : ndarray, shape (n_channels,)
        Relative power in dB: target frequency power minus noise baseline power.
    """
    idx_target = np.argmin(np.abs(freqs - target_freq))
    idx_noise = np.where((freqs >= noise_band[0]) & (freqs <= noise_band[1]) &
                        (np.arange(len(freqs)) != idx_target))[0]
    signal_power = input_pow[:, idx_target]
    if fit_range[0] is np.nan:
            fit_range = (freqs[1], freqs[-1])  # exclude DC by default
    noise_powers = []
    for ii in range(input_pow.shape[0]):
        fm = SpectralModel()
        fm.fit(freqs, input_pow[ii, :].T, freq_range=fit_range)
        if fm.results.params.aperiodic.params.shape[0] == 2:
            offset, exponent = fm.results.params.aperiodic.params
            noise_power = offset - exponent * np.log10(freqs[idx_noise])
        elif fm.results.params.aperiodic.params.shape[0] == 3:
            offset, knee, exponent = fm.results.params.aperiodic.params
            noise_power = offset - np.log10(knee + freqs[idx_noise]**exponent)
        noise_power = noise_power.mean()
        noise_powers.append(noise_power)
    noise_power = np.array(noise_powers)
    signal_power_db = 10 * np.log10(signal_power)
    noise_power_db = 10 * np.log10(noise_power)

    rel_signal_power = signal_power_db - noise_power_db

    snr = signal_power / noise_power
    snr_db = 10 * np.log10(snr)
    if center:
        snr_db = snr_db - np.mean(snr_db)
        rel_signal_power = rel_signal_power - np.mean(rel_signal_power)

    return snr_db, snr, signal_power_db, signal_power, rel_signal_power


