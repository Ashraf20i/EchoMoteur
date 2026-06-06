import logging
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

from utils.noise_reduction import preprocess_audio
from utils.audio_validator import extract_context_profile, compute_signal_quality


# Pondérations V2.2 :
# On donne plus de poids aux indices mécaniques : impulsions, harmonicité, énergie par bandes.
# MFCC reste secondaire.
BLOCK_WEIGHTS = {
    "impulses": 0.22,
    "harmonicity": 0.17,
    "frequency_bands": 0.15,
    "spectral_shape": 0.14,
    "energy": 0.15,
    "cepstrum": 0.10,
    "mfcc": 0.07,
}

WINDOW_DURATION = 1.0
HOP_DURATION = 0.5
MIN_WINDOW_RMS = 1e-5

# Seuil plus prudent que 1.5 pour éviter le faux positif massif avec référence unique.
WINDOW_ANOMALY_THRESHOLD = 2.75


def safe_float(value, default=0.0) -> float:
    try:
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except Exception:
        return default


def ensure_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)

    if arr.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    if arr.ndim == 0:
        return arr.reshape(1, 1)

    if arr.ndim == 1:
        return arr.reshape(1, -1)

    if arr.ndim == 2:
        return arr

    return arr.reshape(arr.shape[0], -1)


def split_into_windows(
    signal: np.ndarray,
    sr: int,
    window_duration: float = WINDOW_DURATION,
    hop_duration: float = HOP_DURATION,
) -> list[np.ndarray]:
    signal = np.asarray(signal, dtype=np.float32)

    window_size = int(window_duration * sr)
    hop_size = int(hop_duration * sr)

    if signal.size < window_size:
        if signal.size < int(0.5 * sr):
            return []
        padded = np.zeros(window_size, dtype=np.float32)
        padded[: signal.size] = signal
        return [padded]

    windows = []
    for start in range(0, signal.size - window_size + 1, hop_size):
        window = signal[start : start + window_size].astype(np.float32)
        rms = float(np.sqrt(np.mean(window ** 2)) + 1e-12)
        if rms >= MIN_WINDOW_RMS:
            windows.append(window)

    return windows


def _band_energy_ratios(power: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    total = float(np.sum(power) + 1e-12)

    bands = [
        (0, 200),
        (200, 800),
        (800, 2000),
        (2000, 4000),
        (4000, 8000),
    ]

    ratios = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        ratios.append(float(np.sum(power[mask, :]) / total) if np.any(mask) else 0.0)

    very_low, low_mid, mid, high_mid, high = ratios

    low_total = very_low + low_mid
    high_total = high_mid + high

    extra = [
        low_total,
        high_total,
        low_total / (high_total + 1e-12),
        high / (low_total + high_total + 1e-12),
        mid / (total + 1e-12),
    ]

    return np.asarray(ratios + extra, dtype=np.float32)


def extract_energy_features(window: np.ndarray, sr: int) -> np.ndarray:
    rms_frames = librosa.feature.rms(y=window, frame_length=1024, hop_length=256)[0]

    rms_mean = safe_float(np.mean(rms_frames))
    rms_std = safe_float(np.std(rms_frames))
    rms_p90 = safe_float(np.percentile(rms_frames, 90))
    rms_cv = safe_float(rms_std / (rms_mean + 1e-12))

    abs_w = np.abs(window)
    peak = safe_float(np.max(abs_w))
    rms = safe_float(np.sqrt(np.mean(window ** 2)) + 1e-12)
    crest = safe_float(peak / (rms + 1e-12))

    zcr = librosa.feature.zero_crossing_rate(window, hop_length=256)[0]

    return np.asarray([
        rms_mean,
        rms_std,
        rms_p90,
        rms_cv,
        crest,
        safe_float(np.mean(zcr)),
        safe_float(np.std(zcr)),
        safe_float(kurtosis(window, fisher=False, nan_policy="omit")),
        safe_float(skew(window, nan_policy="omit")),
    ], dtype=np.float32)


def extract_frequency_band_features(window: np.ndarray, sr: int) -> np.ndarray:
    n_fft = 2048
    stft = np.abs(librosa.stft(window, n_fft=n_fft, hop_length=512)) + 1e-12
    power = stft ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    return _band_energy_ratios(power, freqs)


def extract_spectral_shape_features(window: np.ndarray, sr: int) -> np.ndarray:
    n_fft = 2048
    stft = np.abs(librosa.stft(window, n_fft=n_fft, hop_length=512)) + 1e-12

    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)

    features = [
        np.mean(centroid), np.std(centroid),
        np.mean(bandwidth), np.std(bandwidth),
        np.mean(rolloff), np.std(rolloff),
        np.mean(flatness), np.std(flatness),
    ]

    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    return np.asarray([safe_float(x) for x in features], dtype=np.float32)


def extract_harmonicity_features(window: np.ndarray, sr: int) -> np.ndarray:
    # Autocorrélation : force de la périodicité temporelle
    x = window - np.mean(window)
    if np.max(np.abs(x)) > 0:
        x = x / (np.max(np.abs(x)) + 1e-12)

    autocorr = np.correlate(x, x, mode="full")[len(x)-1:]
    autocorr = autocorr / (autocorr[0] + 1e-12)

    min_lag = max(1, int(sr / 500.0))
    max_lag = min(len(autocorr) - 1, int(sr / 20.0))

    if max_lag > min_lag:
        search = autocorr[min_lag:max_lag]
        peak_idx = int(np.argmax(search))
        peak_value = safe_float(search[peak_idx])
        peak_lag = float(min_lag + peak_idx)
        estimated_f0 = float(sr / (peak_lag + 1e-12))
    else:
        peak_value = 0.0
        peak_lag = 0.0
        estimated_f0 = 0.0

    # Harmonic/percussive separation
    try:
        harmonic, percussive = librosa.effects.hpss(window)
        h_energy = float(np.sum(harmonic ** 2) + 1e-12)
        p_energy = float(np.sum(percussive ** 2) + 1e-12)
        harmonic_ratio = h_energy / (h_energy + p_energy)
        harmonic_percussive_ratio = h_energy / p_energy
    except Exception:
        harmonic_ratio = 0.0
        harmonic_percussive_ratio = 0.0

    # Pics spectraux : estimation du nombre de pics harmoniques
    spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window)))) + 1e-12
    freqs = np.fft.rfftfreq(len(window), d=1.0 / sr)

    mask = (freqs >= 20) & (freqs <= 5000)
    spec = spectrum[mask]

    if spec.size > 10:
        threshold = np.percentile(spec, 75)
        peaks, props = find_peaks(spec, height=threshold, distance=5, prominence=np.std(spec))
        peak_count = float(len(peaks))
        peak_prom_mean = float(np.mean(props.get("prominences", [0.0]))) if len(peaks) else 0.0
    else:
        peak_count = 0.0
        peak_prom_mean = 0.0

    return np.asarray([
        peak_value,
        peak_lag,
        estimated_f0,
        harmonic_ratio,
        harmonic_percussive_ratio,
        peak_count,
        peak_prom_mean,
    ], dtype=np.float32)


def extract_impulse_features(window: np.ndarray, sr: int) -> np.ndarray:
    abs_w = np.abs(window)

    # Enveloppe lissée
    frame = max(64, int(0.01 * sr))
    kernel = np.ones(frame, dtype=np.float32) / frame
    envelope = np.convolve(abs_w, kernel, mode="same")

    env_mean = safe_float(np.mean(envelope))
    env_std = safe_float(np.std(envelope))
    env_kurt = safe_float(kurtosis(envelope, fisher=False, nan_policy="omit"))

    threshold = env_mean + 2.5 * env_std
    min_distance = max(1, int(0.05 * sr))

    peaks, props = find_peaks(
        envelope,
        height=threshold,
        distance=min_distance,
        prominence=max(env_std, 1e-8),
    )

    duration = len(window) / sr
    peak_rate = float(len(peaks) / (duration + 1e-12))

    prominences = props.get("prominences", np.asarray([], dtype=np.float32))
    mean_prom = safe_float(np.mean(prominences)) if prominences.size else 0.0
    max_prom = safe_float(np.max(prominences)) if prominences.size else 0.0

    rms = safe_float(np.sqrt(np.mean(window ** 2)) + 1e-12)
    crest = safe_float(np.max(abs_w) / (rms + 1e-12))

    # Flux spectral : variation de spectre d'une trame à l'autre
    stft = np.abs(librosa.stft(window, n_fft=1024, hop_length=256)) + 1e-12
    stft_norm = stft / (np.sum(stft, axis=0, keepdims=True) + 1e-12)
    flux = np.sqrt(np.sum(np.diff(stft_norm, axis=1) ** 2, axis=0)) if stft_norm.shape[1] > 1 else np.array([0.0])

    # Ratio énergie impulsionnelle
    impulse_energy = float(np.sum(envelope[peaks] ** 2)) if len(peaks) else 0.0
    total_energy = float(np.sum(envelope ** 2) + 1e-12)
    impulse_energy_ratio = impulse_energy / total_energy

    return np.asarray([
        peak_rate,
        mean_prom,
        max_prom,
        env_kurt,
        crest,
        safe_float(np.mean(flux)),
        safe_float(np.std(flux)),
        impulse_energy_ratio,
    ], dtype=np.float32)


def extract_cepstral_features(window: np.ndarray, sr: int) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window)))) + 1e-12
    log_spectrum = np.log(spectrum)
    cepstrum = np.abs(np.fft.irfft(log_spectrum))

    quefrencies = np.arange(len(cepstrum)) / sr

    # Zone utile approximative pour périodicités 20-500 Hz
    mask = (quefrencies >= 1 / 500.0) & (quefrencies <= 1 / 20.0)

    if not np.any(mask):
        return np.zeros(4, dtype=np.float32)

    c = cepstrum[mask]
    peak = safe_float(np.max(c))
    mean = safe_float(np.mean(c))
    std = safe_float(np.std(c))
    ratio = safe_float(peak / (mean + 1e-12))

    peak_idx = int(np.argmax(c))
    q_values = quefrencies[mask]
    peak_quefrency = safe_float(q_values[peak_idx])

    return np.asarray([peak, mean, std, ratio, peak_quefrency], dtype=np.float32)


def extract_mfcc_features(window: np.ndarray, sr: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)

    features = []
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    features.extend(np.mean(delta, axis=1))

    return np.asarray([safe_float(x) for x in features], dtype=np.float32)


def extract_features_from_window(window: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    return {
        "energy": extract_energy_features(window, sr),
        "frequency_bands": extract_frequency_band_features(window, sr),
        "spectral_shape": extract_spectral_shape_features(window, sr),
        "harmonicity": extract_harmonicity_features(window, sr),
        "impulses": extract_impulse_features(window, sr),
        "cepstrum": extract_cepstral_features(window, sr),
        "mfcc": extract_mfcc_features(window, sr),
    }


def extract_features_by_windows(signal: np.ndarray, sr: int) -> dict:
    windows = split_into_windows(signal, sr)

    if not windows:
        raise ValueError("Aucune fenêtre exploitable pour l'extraction de features.")

    block_values = {block: [] for block in BLOCK_WEIGHTS}

    for window in windows:
        features = extract_features_from_window(window, sr)

        for block_name in BLOCK_WEIGHTS:
            block_values[block_name].append(features[block_name])

    blocks = {
        block_name: ensure_2d(np.vstack(values))
        for block_name, values in block_values.items()
        if values
    }

    duration_sec = float(len(signal) / sr)
    quality = compute_signal_quality(signal, sr)

    logging.info(f"Features extraites : {len(windows)} fenêtres.")

    return {
        "blocks": blocks,
        "n_windows": len(windows),
        "duration_sec": duration_sec,
        "quality": quality,
    }


def combine_feature_sets(feature_sets: list[dict]) -> dict[str, np.ndarray]:
    combined = {}

    for block_name in BLOCK_WEIGHTS:
        matrices = []

        for fs in feature_sets:
            blocks = fs.get("blocks", fs)

            if block_name not in blocks:
                continue

            matrix = ensure_2d(blocks[block_name])
            if matrix.size > 0:
                matrices.append(matrix)

        if matrices:
            try:
                combined[block_name] = np.concatenate(matrices, axis=0).astype(np.float32)
            except ValueError as exc:
                shapes = [m.shape for m in matrices]
                raise ValueError(
                    f"Erreur de concaténation dans le bloc '{block_name}'. Shapes : {shapes}"
                ) from exc
        else:
            combined[block_name] = np.empty((0, 0), dtype=np.float32)

    return combined


def build_reference_features(reference_paths: list[str], target_sr: int = 22050) -> dict:
    feature_sets = []
    context_profiles = []
    durations = []
    qualities = []

    for ref_path in reference_paths:
        logging.info(f"  Chargement référence : {ref_path}")
        signal, sr = preprocess_audio(ref_path, target_sr=target_sr)

        context_profiles.append(extract_context_profile(signal, sr))
        features = extract_features_by_windows(signal, sr)

        feature_sets.append(features)
        durations.append(features["duration_sec"])
        qualities.append(features["quality"])

    combined_blocks = combine_feature_sets(feature_sets)

    return {
        "blocks": combined_blocks,
        "reference_paths": reference_paths,
        "context_profiles": context_profiles,
        "durations": durations,
        "qualities": qualities,
        "n_reference_files": len(reference_paths),
    }


def fit_scaler(reference_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference_matrix = ensure_2d(reference_matrix)

    mean = np.mean(reference_matrix, axis=0)
    std = np.std(reference_matrix, axis=0)

    std = np.where(std < 1e-6, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)


def transform(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    matrix = ensure_2d(matrix)
    return ((matrix - mean) / std).astype(np.float32)


def pairwise_euclidean_normalized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = ensure_2d(a)
    b = ensure_2d(b)

    if a.size == 0 or b.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    diff = a[:, None, :] - b[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    dist = dist / np.sqrt(max(a.shape[1], 1))

    return dist.astype(np.float32)


def compute_reference_baseline(ref_scaled: np.ndarray, neighbor_exclusion: int = 3) -> dict:
    """
    Calcule la variabilité interne de la référence.

    Correction V2.2 :
    on ignore les fenêtres très voisines, car elles se chevauchent à 50 %
    et rendent le baseline artificiellement trop petit.
    """
    ref_scaled = ensure_2d(ref_scaled)
    n = ref_scaled.shape[0]

    if n < 3:
        return {
            "median": 0.75,
            "p90": 1.00,
            "min_allowed": 0.35,
            "note": "Référence très courte : baseline de secours utilisé.",
        }

    dist = pairwise_euclidean_normalized(ref_scaled, ref_scaled)

    for i in range(n):
        start = max(0, i - neighbor_exclusion)
        end = min(n, i + neighbor_exclusion + 1)
        dist[i, start:end] = np.inf

    nearest = np.min(dist, axis=1)
    nearest = nearest[np.isfinite(nearest)]

    if nearest.size == 0:
        nearest = dist[np.isfinite(dist)]

    if nearest.size == 0:
        return {
            "median": 0.75,
            "p90": 1.00,
            "min_allowed": 0.35,
            "note": "Baseline impossible : valeurs de secours utilisées.",
        }

    median = safe_float(np.median(nearest), default=0.75)
    p90 = safe_float(np.percentile(nearest, 90), default=1.00)

    # Plancher pour éviter ratios explosifs avec une seule référence trop homogène.
    p90 = max(p90, 0.35)
    median = max(median, 0.25)

    return {
        "median": median,
        "p90": p90,
        "min_allowed": 0.35,
        "note": "Baseline calculé avec exclusion des fenêtres voisines.",
    }


def compare_block(test_matrix: np.ndarray, ref_matrix: np.ndarray) -> dict:
    test_matrix = ensure_2d(test_matrix)
    ref_matrix = ensure_2d(ref_matrix)

    if test_matrix.size == 0 or ref_matrix.size == 0:
        return {
            "score": 0.0,
            "median_ratio": 0.0,
            "p90_ratio": 0.0,
            "worst_window_ratio_raw": 0.0,
            "worst_window_score": 0.0,
            "window_ratios": [],
            "baseline": {},
        }

    if test_matrix.shape[1] != ref_matrix.shape[1]:
        raise ValueError(
            f"Dimensions incompatibles : test={test_matrix.shape}, reference={ref_matrix.shape}"
        )

    mean, std = fit_scaler(ref_matrix)
    ref_scaled = transform(ref_matrix, mean, std)
    test_scaled = transform(test_matrix, mean, std)

    dist_matrix = pairwise_euclidean_normalized(test_scaled, ref_scaled)
    min_distances = np.min(dist_matrix, axis=1)

    baseline = compute_reference_baseline(ref_scaled)
    baseline_p90 = max(float(baseline["p90"]), 0.35)

    ratios = min_distances / (baseline_p90 + 1e-12)

    median_ratio = safe_float(np.median(ratios))
    p90_ratio = safe_float(np.percentile(ratios, 90))
    worst_raw = safe_float(np.max(ratios))
    worst_capped = min(worst_raw, 10.0)

    score = (
        0.50 * median_ratio
        + 0.35 * p90_ratio
        + 0.15 * worst_capped
    )

    return {
        "score": safe_float(score),
        "median_ratio": median_ratio,
        "p90_ratio": p90_ratio,
        "worst_window_ratio_raw": worst_raw,
        "worst_window_score": worst_capped,
        "window_ratios": ratios.astype(float).tolist(),
        "baseline": baseline,
    }


def compute_confidence(context_result: dict, test_quality: dict, n_reference_files: int) -> float:
    confidence = 100.0

    context_score = float(context_result.get("score", 50.0))
    if context_score < 40:
        confidence -= 35
    elif context_score < 65:
        confidence -= 15

    quality_score = float(test_quality.get("quality_score", 70.0))
    confidence = 0.70 * confidence + 0.30 * quality_score

    if n_reference_files <= 1:
        confidence -= 20.0

    return float(np.clip(confidence, 0.0, 100.0))


def compare_to_reference(test_features: dict, reference_features: dict, context_result: dict | None = None) -> dict:
    test_blocks = test_features.get("blocks", test_features)
    reference_blocks = reference_features.get("blocks", reference_features)

    block_details = {}
    block_scores = {}

    total_windows = int(test_features.get("n_windows", 0))
    global_window_scores = np.zeros(total_windows, dtype=np.float32)

    used_weight = 0.0

    for block_name, weight in BLOCK_WEIGHTS.items():
        if block_name not in test_blocks or block_name not in reference_blocks:
            continue

        result = compare_block(test_blocks[block_name], reference_blocks[block_name])
        block_details[block_name] = result
        block_scores[block_name] = result["score"]

        ratios = np.asarray(result["window_ratios"], dtype=np.float32)

        if ratios.size == total_windows and total_windows > 0:
            global_window_scores += weight * ratios
            used_weight += weight

    if used_weight > 0:
        global_window_scores /= used_weight

    if global_window_scores.size:
        median_ratio = safe_float(np.median(global_window_scores))
        p90_ratio = safe_float(np.percentile(global_window_scores, 90))
        worst_raw = safe_float(np.max(global_window_scores))
        worst_score = safe_float(min(worst_raw, 10.0))
        anomalous_mask = global_window_scores > WINDOW_ANOMALY_THRESHOLD
        anomalous_windows = int(np.sum(anomalous_mask))
        anomalous_ratio = float(anomalous_windows / len(global_window_scores))
    else:
        median_ratio = 0.0
        p90_ratio = 0.0
        worst_raw = 0.0
        worst_score = 0.0
        anomalous_windows = 0
        anomalous_ratio = 0.0

    global_score = 0.0
    for block_name, score in block_scores.items():
        global_score += BLOCK_WEIGHTS.get(block_name, 0.0) * score

    n_reference_files = int(reference_features.get("n_reference_files", 1))
    confidence = compute_confidence(
        context_result or {"score": 50.0},
        test_features.get("quality", {}),
        n_reference_files,
    )

    return {
        "duration_sec": float(test_features.get("duration_sec", 0.0)),
        "global_score": safe_float(global_score),
        "median_ratio": median_ratio,
        "p90_ratio": p90_ratio,
        "worst_window_ratio_raw": worst_raw,
        "worst_window_score": worst_score,
        "anomalous_windows": anomalous_windows,
        "total_windows": total_windows,
        "anomalous_ratio": anomalous_ratio,
        "window_anomaly_threshold": WINDOW_ANOMALY_THRESHOLD,
        "block_scores": block_scores,
        "block_details": block_details,
        "confidence_score": confidence,
        "n_reference_files": n_reference_files,
        "quality": test_features.get("quality", {}),
    }
