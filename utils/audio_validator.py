import numpy as np
import librosa


def extract_audio_profile(signal, sr):
    """
    Extrait un profil large du signal.
    Pas pour diagnostiquer, juste pour vérifier
    s'il est globalement cohérent avec la référence.
    """
    rms = librosa.feature.rms(y=signal)[0]
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=signal)[0]

    harmonic, percussive = librosa.effects.hpss(signal)
    harmonic_energy = float(np.mean(np.abs(harmonic)))
    percussive_energy = float(np.mean(np.abs(percussive)))
    harmonic_ratio = harmonic_energy / (harmonic_energy + percussive_energy + 1e-10)

    stft = np.abs(librosa.stft(signal, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mean_spectrum = np.mean(stft, axis=1) + 1e-10
    total_energy = np.sum(mean_spectrum)

    low_freq_ratio = float(np.sum(mean_spectrum[freqs < 2000]) / total_energy)
    very_low_freq_ratio = float(np.sum(mean_spectrum[freqs < 500]) / total_energy)

    return {
        "rms_mean": float(np.mean(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "rolloff_mean": float(np.mean(rolloff)),
        "flatness_mean": float(np.mean(flatness)),
        "harmonic_ratio": float(harmonic_ratio),
        "low_freq_ratio": float(low_freq_ratio),
        "very_low_freq_ratio": float(very_low_freq_ratio),
    }


def compare_profiles(reference_profile, test_profile):
    """
    Compare le profil large du test à celui de la référence.
    Retourne une décision de compatibilité.
    """

    rules = []

    # Différences absolues pour les ratios [0,1]
    rules.append(abs(test_profile["low_freq_ratio"] - reference_profile["low_freq_ratio"]) <= 0.20)
    rules.append(abs(test_profile["very_low_freq_ratio"] - reference_profile["very_low_freq_ratio"]) <= 0.10)
    rules.append(abs(test_profile["harmonic_ratio"] - reference_profile["harmonic_ratio"]) <= 0.20)
    rules.append(abs(test_profile["flatness_mean"] - reference_profile["flatness_mean"]) <= 0.10)
    rules.append(abs(test_profile["zcr_mean"] - reference_profile["zcr_mean"]) <= 0.12)

    # Différences relatives pour les grandeurs plus grandes
    centroid_rel_diff = abs(test_profile["centroid_mean"] - reference_profile["centroid_mean"]) / max(reference_profile["centroid_mean"], 1.0)
    rolloff_rel_diff = abs(test_profile["rolloff_mean"] - reference_profile["rolloff_mean"]) / max(reference_profile["rolloff_mean"], 1.0)
    rms_rel_diff = abs(test_profile["rms_mean"] - reference_profile["rms_mean"]) / max(reference_profile["rms_mean"], 1e-6)

    rules.append(centroid_rel_diff <= 0.40)
    rules.append(rolloff_rel_diff <= 0.35)
    rules.append(rms_rel_diff <= 0.70)

    score = 100.0 * sum(rules) / len(rules)

    is_compatible = score >= 62.5

    return {
        "is_compatible": is_compatible,
        "score": float(score),
        "details": {
            "centroid_rel_diff": float(centroid_rel_diff),
            "rolloff_rel_diff": float(rolloff_rel_diff),
            "rms_rel_diff": float(rms_rel_diff),
        }
    }


def build_validation_message(validation_result):
    if validation_result["is_compatible"]:
        return (
            f"Audio globalement cohérent avec la référence "
            f"(score compatibilité : {validation_result['score']:.2f} %)."
        )

    return (
        f"Analyse non fiable : audio trop éloigné du profil large de la référence "
        f"(score compatibilité : {validation_result['score']:.2f} %)."
    )