from scipy.signal import welch
from scipy.stats import chi2
import numpy as np

def compute_psd(index, fs=1, nperseg=None, alpha=0.95):
    """
    Compute Welch PSD with AR(1) and white-noise significance,
    and normalized power.

    Returns
    -------
    periods : ndarray
        Periods corresponding to PSD.
    psd_norm : ndarray
        Normalized PSD.
    psd_ar1_norm : ndarray
        Normalized AR(1) significance spectrum.
    psd_wn_norm : float
        Normalized white-noise significance.
    """
    index = np.asarray(index).flatten()
    N = len(index)
    if nperseg is None:
        nperseg = N // 2

    f, Pxx = welch(index, fs=fs, window="hann", nperseg=nperseg, scaling="density")
    mask = f > 0
    f = f[mask]
    psd = Pxx[mask]
    periods = 1 / f

    # normalize PSD by variance
    psd_norm = psd / np.var(index)

    # AR(1 significance
    r1 = np.corrcoef(index[:-1], index[1:])[0,1]
    psd_ar1 = (1 - r1**2) / (1 + r1**2 - 2*r1*np.cos(2*np.pi*f/fs))
    psd_ar1 *= chi2.ppf(alpha, df=2) / 2
    psd_ar1_norm = psd_ar1  # already relative to variance=1

    # White-noise significance
    psd_wn_norm = chi2.ppf(alpha, df=2) / 2

    return periods, psd_norm, psd_ar1_norm, psd_wn_norm




def morlet_wavelet_spectrum_coi(x, dt=1, dj=0.25, s0=2, J=None):
    """
    Compute Morlet wavelet spectrum with COI and normalized power.
    """
    x = np.asarray(x).flatten()
    N = len(x)
    t = np.arange(N) * dt
    x = (x - np.mean(x)) / np.std(x)

    if J is None:
        J = int(np.log2(N*dt/s0) / dj)

    scales = s0 * 2 ** (np.arange(0, J+1) * dj)
    periods = scales.copy()

    # Fourier frequencies
    k = np.fft.fftfreq(N, d=dt) * 2*np.pi
    f = np.fft.fft(x)

    wave = np.zeros((len(scales), N), dtype=complex)

    for i, s in enumerate(scales):
        norm = (dt / s**0.5) * (np.pi**-0.25)
        # Morlet daughter in Fourier space
        expnt = -0.5 * (s * k - 6)**2
        daughter = norm * np.exp(expnt)
        wave[i, :] = np.fft.ifft(f * daughter)

    power = np.abs(wave)**2
    # power_norm = power / np.var(x)
    power_norm = power / np.max(power)

    # COI (1D, length N)
    coi = dt * np.minimum(np.arange(N), N - np.arange(N))
    coi = coi * 2**0.5

    return power_norm, scales, periods, t, coi



def generate_spectrum_ensemble(nino34_ds):
    # Define common grids
    common_periods = np.logspace(np.log10(0.5), np.log10(100), 200)  # PSD
    common_scales = np.logspace(np.log10(2), np.log10(50), 100)      # Morlet
    
    # Precompute max time length across all models
    max_time = max(len(nino34_ds.isel(model=i).time.dropna('time')) 
                   for i in range(nino34_ds.model.size))

    # Initialize lists
    psd_list, psd_ar1_list, psd_wn_list = [], [], []
    morlet_list = []
    coi_list = []

    for i in range(nino34_ds.model.size):
        # Extract 1D NumPy array of the time series
        ts = nino34_ds.isel(model=i).dropna('time').values.flatten()
        N = len(ts)
        if N < 5:
            continue  # skip very short series

        # ---- PSD ----
        periods, psd_norm, psd_ar1_norm, psd_wn_norm = compute_psd(ts)

        # Interpolate PSD to common periods
        psd_interp = np.interp(common_periods, periods[::-1], psd_norm[::-1])
        psd_ar1_interp = np.interp(common_periods, periods[::-1], psd_ar1_norm[::-1])
        psd_wn_interp = np.full_like(common_periods, psd_wn_norm)

        psd_list.append(psd_interp)
        psd_ar1_list.append(psd_ar1_interp)
        psd_wn_list.append(psd_wn_interp)

        # ---- Morlet ----
        power_norm, scales, periods_m, t, coi = morlet_wavelet_spectrum_coi(ts)

        # Interpolate Morlet power to common scales
        power_interp_scales = np.zeros((len(common_scales), power_norm.shape[1]))
        for j, s in enumerate(common_scales):
            idx = np.argmin(np.abs(scales - s))
            power_interp_scales[j, :] = power_norm[idx, :]

        # Interpolate Morlet power to common max_time
        x_old = np.linspace(0, 1, power_interp_scales.shape[1])
        x_new = np.linspace(0, 1, max_time)
        power_interp = np.zeros((len(common_scales), max_time))
        for j in range(len(common_scales)):
            power_interp[j, :] = np.interp(x_new, x_old, power_interp_scales[j, :])
        morlet_list.append(power_interp)

    # Interpolate COI to common max_time
    x_old = np.linspace(0, 1, len(coi))
    coi_interp = np.interp(x_new, x_old, coi)
    coi_list.append(coi_interp)

    psd_ensemble = np.array(psd_list)        # shape: (n_models, n_periods)
    psd_ar1_ensemble = np.array(psd_ar1_list)
    psd_wn_ensemble = np.array(psd_wn_list)

    morlet_ensemble = np.array(morlet_list)  # shape: (n_models, n_scales, n_time_max)
    coi_ensemble = np.array(coi_list)
    return psd_ensemble, psd_ar1_ensemble, psd_wn_ensemble, morlet_ensemble, coi_ensemble, common_periods, common_scales, max_time

    # PSD ensemble mean and spread
    # psd_mean_pic = np.mean(psd_ensemble, axis=0)
    # psd_std_pic = np.std(psd_ensembl, axis=0)

    # # Morlet ensemble mean (ignoring varying lengths for now)
    # morlet_mean_pic = np.mean(morlet_ensemble, axis=0)
    # morlet_std_pic = np.std(morlet_ensemble, axis=0)

    # # Average COI across ensemble (make sure you collected this earlier!)
    # coi_mean_pic = np.mean(np.vstack(coi_list), axis=0)