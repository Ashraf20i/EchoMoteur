from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def validate_audio_file(file_path: str) -> dict:
    """
    Validation légère du fichier audio.
    Ne valide pas la santé du moteur, seulement la lisibilité du fichier.
    """
    path = Path(file_path)

    result = {
        "is_valid": False,
        "path": str(path),
        "exists": path.exists(),
        "extension": path.suffix.lower(),
        "warnings": [],
        "errors": [],
    }

    if not path.exists():
        result["errors"].append("Fichier introuvable.")
        return result

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        result["warnings"].append("Extension audio non standard pour le projet.")

    if path.suffix.lower() == ".mp3":
        result["warnings"].append(
            "Fichier MP3 compressé : les hautes fréquences peuvent être modifiées. "
            "La confiance de comparaison fréquentielle est réduite."
        )

    try:
        try:
            info = sf.info(str(path))
            result["sample_rate"] = info.samplerate
            result["channels"] = info.channels
            result["duration_sec"] = float(info.duration)
            result["format"] = info.format
            result["subtype"] = info.subtype
        except Exception:
            duration = librosa.get_duration(path=str(path))
            result["duration_sec"] = float(duration)
            result["sample_rate"] = None
            result["channels"] = None
            result["format"] = "unknown"
            result["subtype"] = "unknown"

        if result["duration_sec"] < 2.0:
            result["errors"].append("Durée trop courte pour une analyse fiable.")

        result["is_valid"] = len(result["errors"]) == 0
        return result

    except Exception as exc:
        result["errors"].append(f"Impossible de lire le fichier audio : {exc}")
        return result


def compute_signal_quality(signal: np.ndarray, sr: int) -> dict:
    signal = np.asarray(signal, dtype=np.float32)

    duration_sec = float(signal.size / sr) if sr else 0.0
    abs_signal = np.abs(signal)

    rms = float(np.sqrt(np.mean(signal ** 2)) + 1e-12)
    peak = float(np.max(abs_signal) + 1e-12)

    clipping_ratio = float(np.mean(abs_signal >= 0.98))

    frame_length = max(512, int(0.05 * sr))
    hop_length = max(256, frame_length // 2)

    rms_frames = librosa.feature.rms(
        y=signal,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    silence_threshold = max(1e-5, 0.10 * float(np.median(rms_frames) + 1e-12))
    silence_ratio = float(np.mean(rms_frames < silence_threshold))

    p95 = float(np.percentile(rms_frames, 95))
    p05 = float(np.percentile(rms_frames, 5))
    dynamic_range_db = float(20 * np.log10((p95 + 1e-12) / (p05 + 1e-12)))

    quality_score = 100.0

    if duration_sec < 3.0:
        quality_score -= 30.0
    elif duration_sec < 6.0:
        quality_score -= 10.0

    if clipping_ratio > 0.01:
        quality_score -= 30.0
    elif clipping_ratio > 0.001:
        quality_score -= 10.0

    if silence_ratio > 0.60:
        quality_score -= 25.0
    elif silence_ratio > 0.35:
        quality_score -= 10.0

    if rms < 1e-4:
        quality_score -= 40.0

    quality_score = float(np.clip(quality_score, 0.0, 100.0))

    return {
        "duration_sec": duration_sec,
        "rms": rms,
        "peak": peak,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
        "dynamic_range_db": dynamic_range_db,
        "quality_score": quality_score,
    }


def extract_context_profile(signal: np.ndarray, sr: int) -> dict:
    """
    Profil global pour savoir si l'audio ressemble à un contexte moteur.
    Ce n'est PAS un verdict de normalité.
    """
    signal = np.asarray(signal, dtype=np.float32)

    n_fft = 2048
    hop_length = 512

    stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)) + 1e-12
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power = stft ** 2
    total_energy = float(np.sum(power) + 1e-12)

    def band_ratio(low, high):
        mask = (freqs >= low) & (freqs < high)
        return float(np.sum(power[mask, :]) / total_energy) if np.any(mask) else 0.0

    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    zcr = librosa.feature.zero_crossing_rate(signal, hop_length=hop_length)[0]

    low = band_ratio(0, 800)
    mid = band_ratio(800, 2500)
    high = band_ratio(2500, 8000)

    return {
        "low_ratio": low,
        "mid_ratio": mid,
        "high_ratio": high,
        "low_high_ratio": float(low / (high + 1e-12)),
        "centroid": float(np.mean(centroid)),
        "rolloff": float(np.mean(rolloff)),
        "flatness": float(np.mean(flatness)),
        "zcr": float(np.mean(zcr)),
        "quality": compute_signal_quality(signal, sr),
    }


def _relative_difference(a: float, b: float) -> float:
    return abs(a - b) / (abs(a) + abs(b) + 1e-12)


def compare_audio_context(reference_profiles: list[dict], test_signal: np.ndarray, sr: int) -> dict:
    """
    Compare le contexte global.

    But :
    - compatible : ressemble à un son moteur analysable
    - suspect : probablement moteur, mais conditions différentes
    - out_of_context : possiblement voix/musique/bruit hors sujet

    Important : compatible ≠ normal.
    """
    test_profile = extract_context_profile(test_signal, sr)

    if not reference_profiles:
        return {
            "status": "suspect",
            "score": 50.0,
            "message": "Aucun profil de référence disponible.",
            "test_profile": test_profile,
        }

    numeric_keys = [
        "low_ratio",
        "mid_ratio",
        "high_ratio",
        "low_high_ratio",
        "centroid",
        "rolloff",
        "flatness",
        "zcr",
    ]

    distances = []

    for ref in reference_profiles:
        diffs = []
        for key in numeric_keys:
            if key in ref and key in test_profile:
                diffs.append(_relative_difference(float(ref[key]), float(test_profile[key])))

        if diffs:
            distances.append(float(np.mean(diffs)))

    if not distances:
        score = 50.0
    else:
        distance = min(distances)
        score = float(np.clip(100.0 * (1.0 - distance), 0.0, 100.0))

    quality_score = test_profile["quality"]["quality_score"]
    score = float(0.75 * score + 0.25 * quality_score)

    if score >= 65.0:
        status = "compatible"
    elif score >= 40.0:
        status = "suspect"
    else:
        status = "out_of_context"

    return {
        "status": status,
        "score": score,
        "message": "Contexte audio évalué. Ce score ne signifie pas que le moteur est normal.",
        "test_profile": test_profile,
    }
