
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import welch
from scipy.stats import entropy, iqr


def _as_float_array(s: pd.Series) -> NDArray[np.float64]:
    # Coerce to numeric (NaNs for bad values), then expose as float64 ndarray
    s_num = pd.to_numeric(s, errors="coerce")
    return s_num.to_numpy(dtype=np.float64, copy=False)

def extract_features_from_window(window: pd.DataFrame, fs: float = 28.5) -> pd.Series:
    """Extract rich time- and frequency-domain features from a single accelerometer window.

    This function computes both classical and robust statistical features that describe
    the motion captured in a 3-axis accelerometer signal over a short window
    (typically 3 seconds). The resulting features summarize the magnitude, variability,
    smoothness, and periodicity of the movement, providing an orientation-robust
    representation useful for human activity recognition (HAR) models.

    Parameters
    ----------
    window : pd.DataFrame
        A DataFrame containing accelerometer samples for one time window.
        Must include numeric columns ['x', 'y', 'z'] representing the three axes.
    fs : float, optional
        Sampling rate in Hertz (samples per second). Defaults to 28.5 Hz
        (the typical rate for the HTAD dataset).

    Returns
    -------
    pd.Series
        A one-dimensional Series containing extracted features with names such as:
        - Statistical: mean, std, mad, iqr, energy, min, max
        - Dynamic: jerk_mean, jerk_std, zero-crossing rate (zcr)
        - Spectral: dominant frequency, spectral entropy, spectral centroid
        - Cross-axis: correlations (corr_xy, corr_yz, corr_xz)
        - Magnitude-based: features on |a| combining all axes (orientation-robust)
        - Overall: signal magnitude area (sma)

    Notes
    -----
    Feature rationale:
    - **Statistical features** capture amplitude and variability of motion.
    - **MAD** and **IQR** are robust to user-specific intensity differences.
    - **Jerk** quantifies smoothness (first derivative of acceleration).
    - **ZCR** counts sign changes, representing rhythmic movement.
    - **Spectral features** (from Welch PSD) describe periodicity, e.g. gait cadence.
    - **Magnitude and correlation** features reduce sensitivity to phone orientation.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> t = np.linspace(0, 3, 90)  # 3 seconds @ 30 Hz
    >>> df = pd.DataFrame({'x': np.sin(2*np.pi*1*t),
    ...                    'y': np.sin(2*np.pi*1*t + np.pi/2),
    ...                    'z': np.zeros_like(t)})
    >>> feats = extract_features_from_window(df, fs=30)
    >>> feats[['x_dominant_freq', 'mag_spectral_entropy']]
    x_dominant_freq       1.0
    mag_spectral_entropy  0.5
    dtype: float64s

    """

    # --- Helper functions (for readability and static typing) ---
    def as_f64(s: pd.Series) -> NDArray[np.float64]:
        """Coerce Series to float64 ndarray, replacing invalid entries with NaN."""
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, copy=False)

    def zcr(sig: NDArray[np.float64]) -> float:
        """Zero-crossing rate: rhythm measure based on sign changes."""
        if sig.size < 2:
            return 0.0
        return float(np.mean((sig[:-1] * sig[1:]) < 0))

    def spectral_feats(sig: NDArray[np.float64]) -> dict[str, float]:
        """Welch PSD → frequency features: entropy, dominant freq, centroid."""
        f, Pxx = welch(sig, fs=fs)
        total = float(np.sum(Pxx))
        if total <= 0.0:
            return {"spectral_entropy": 0.0, "dominant_freq": 0.0, "spectral_centroid": 0.0}
        P = Pxx / (total + 1e-12)
        return {
            "spectral_entropy": float(-(P * np.log2(P + 1e-12)).sum()),
            "dominant_freq": float(f[int(np.argmax(P))]),
            "spectral_centroid": float((f * P).sum()),
        }

    feats: dict[str, float] = {}

    # --- Per-axis features ---
    for axis in ["x", "y", "z"]:
        sig = as_f64(window[axis])
        valid = np.isfinite(sig)
        sigv = sig[valid]
        n = int(sigv.size)

        # Time-domain (statistical + robust)
        feats[f"{axis}_mean"] = float(np.nanmean(sig))
        feats[f"{axis}_std"] = float(np.nanstd(sig))
        feats[f"{axis}_mad"] = float(np.median(np.abs(sigv - np.median(sigv))) if n else 0.0)
        feats[f"{axis}_iqr"] = float(iqr(sigv) if n else 0.0)
        feats[f"{axis}_min"] = float(np.nanmin(sig)) if n else 0.0
        feats[f"{axis}_max"] = float(np.nanmax(sig)) if n else 0.0
        feats[f"{axis}_energy"] = float(np.nansum(sigv**2) / n) if n else 0.0

        # Dynamics: smoothness and rhythm
        jerk = np.diff(sigv) if n > 1 else np.array([], dtype=np.float64)
        feats[f"{axis}_jerk_mean"] = float(jerk.mean()) if jerk.size else 0.0
        feats[f"{axis}_jerk_std"] = float(jerk.std()) if jerk.size else 0.0
        feats[f"{axis}_zcr"] = zcr(sigv) if n else 0.0

        # Frequency-domain (Welch PSD)
        feats.update({f"{axis}_{k}": v for k, v in spectral_feats(sigv).items()})

        # Histogram entropy (distribution complexity)
        if n:
            h, _ = np.histogram(sigv, bins=10, density=True)
            feats[f"{axis}_entropy"] = float(entropy(h + 1e-6))
        else:
            feats[f"{axis}_entropy"] = 0.0

    # --- Cross-axis correlations (orientation relationships) ---
    x, y, z = as_f64(window["x"]), as_f64(window["y"]), as_f64(window["z"])

    def safe_corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        """Return correlation or 0.0 if one vector is constant/empty."""
        av, bv = a[np.isfinite(a)], b[np.isfinite(b)]
        if av.size < 2 or bv.size < 2 or av.std() < 1e-9 or bv.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(av, bv)[0, 1])

    feats["corr_xy"] = safe_corr(x, y)
    feats["corr_yz"] = safe_corr(y, z)
    feats["corr_xz"] = safe_corr(x, z)

    # --- Magnitude-based features (orientation-robust) ---
    mag = np.sqrt(x**2 + y**2 + z**2)
    magv = mag[np.isfinite(mag)]
    feats["mag_mean"] = float(np.nanmean(mag))
    feats["mag_std"] = float(np.nanstd(mag))
    feats["mag_mad"] = float(np.median(np.abs(magv - np.median(magv))) if magv.size else 0.0)
    feats["mag_iqr"] = float(iqr(magv) if magv.size else 0.0)
    feats["mag_energy"] = float(np.nansum(magv**2) / magv.size) if magv.size else 0.0
    feats["mag_zcr"] = zcr(magv) if magv.size else 0.0
    mag_j = np.diff(magv) if magv.size > 1 else np.array([], dtype=np.float64)
    feats["mag_jerk_mean"] = float(mag_j.mean()) if mag_j.size else 0.0
    feats["mag_jerk_std"] = float(mag_j.std()) if mag_j.size else 0.0

    # Overall motion magnitude (SMA)
    feats["sma"] = float(np.nanmean(np.abs(x) + np.abs(y) + np.abs(z)))

    # Spectral features on magnitude
    feats.update({f"mag_{k}": v for k, v in spectral_feats(magv).items()})

    return pd.Series(feats)