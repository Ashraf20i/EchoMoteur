import numpy as np
import librosa
from scipy.stats import kurtosis, skew


# ─────────────────────────────────────────────
#  DÉCOUPAGE DU SIGNAL
# ─────────────────────────────────────────────

def trim_and_fix_length(signal, sr, target_duration=5.0):
    """
    Supprime le silence au début/à la fin puis force une durée fixe.
    """
    trimmed_signal, _ = librosa.effects.trim(signal, top_db=20)
    target_length = int(target_duration * sr)

    if len(trimmed_signal) >= target_length:
        return trimmed_signal[:target_length]

    padding = target_length - len(trimmed_signal)
    return np.pad(trimmed_signal, (0, padding), mode="constant")


def split_into_windows(signal, sr, window_duration=1.0, hop_duration=0.5):
    """
    Découpe le signal en fenêtres chevauchantes.
    """
    window_length = int(window_duration * sr)
    hop_length    = int(hop_duration * sr)

    windows = []
    start   = 0

    while start + window_length <= len(signal):
        windows.append(signal[start:start + window_length])
        start += hop_length

    return windows


# ─────────────────────────────────────────────
#  EXTRACTION DES FEATURES PAR FENÊTRE
# ─────────────────────────────────────────────

def _mfcc_block(signal, sr):
    """
    MFCC + Delta + Delta²
    Capture le timbre et son évolution temporelle.
    Delta  = vitesse de changement du timbre   → détecte rupture progressive
    Delta² = accélération du changement         → détecte rupture brutale
    Dimensions : 13 * 6 = 78
    """
    mfcc    = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    delta   = librosa.feature.delta(mfcc)
    delta2  = librosa.feature.delta(mfcc, order=2)

    return np.concatenate([
        np.mean(mfcc,   axis=1), np.std(mfcc,   axis=1),
        np.mean(delta,  axis=1), np.std(delta,  axis=1),
        np.mean(delta2, axis=1), np.std(delta2, axis=1),
    ])


def _spectral_shape_block(signal, sr):
    """
    Forme spectrale classique : centroid, bandwidth, rolloff, flatness, ZCR, RMS.
    Dimensions : 12
    """
    rms       = librosa.feature.rms(y=signal)
    zcr       = librosa.feature.zero_crossing_rate(signal)
    centroid  = librosa.feature.spectral_centroid(y=signal, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    rolloff   = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    flatness  = librosa.feature.spectral_flatness(y=signal)

    return np.array([
        np.mean(rms),      np.std(rms),
        np.mean(zcr),      np.std(zcr),
        np.mean(centroid), np.std(centroid),
        np.mean(bandwidth),np.std(bandwidth),
        np.mean(rolloff),  np.std(rolloff),
        np.mean(flatness), np.std(flatness),
    ])


def _spectral_contrast_block(signal, sr):
    """
    [NOUVEAU] Contraste spectral sur 6 sous-bandes.
    Mesure la différence entre les PICS et les VALLÉES du spectre dans chaque bande.
    Physique moteur : un moteur sain a des harmoniques bien marqués (pics élevés)
    séparés par des creux — une anomalie "aplatit" ou "déplace" ces harmoniques.
    Dimensions : 7 * 2 = 14
    """
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr, n_bands=6)
    return np.concatenate([
        np.mean(contrast, axis=1),
        np.std(contrast,  axis=1),
    ])


def _mel_spectrogram_block(signal, sr, n_mels=32):
    """
    [NOUVEAU] Mel-spectrogram avec 32 bandes mel.
    Représentation fréquentielle adaptée à la perception humaine.
    Bien meilleure résolution dans les basses fréquences (là où vit le moteur)
    que les MFCC qui compriment cette info en 13 coefficients.
    Dimensions : 32 * 2 = 64
    """
    mel    = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.concatenate([
        np.mean(mel_db, axis=1),
        np.std(mel_db,  axis=1),
    ])


def _subband_energy_block(signal, sr):
    """
    [NOUVEAU] Énergie dans 5 sous-bandes fréquentielles spécifiques.
    Physique moteur :
      0–200 Hz   → vibrations mécaniques lourdes, balourd, bancale
      200–800 Hz → explosions moteur, combustion fondamentale
      800–2000Hz → harmoniques moteur, soupapes
      2–4 kHz    → bruits auxiliaires, courroies, accessoires
      4–8 kHz    → bruit haute fréquence, fuites, anormalités
    Un déplacement d'énergie entre bandes = signature d'anomalie.
    Dimensions : 5
    """
    stft       = np.abs(librosa.stft(signal, n_fft=2048, hop_length=512))
    freqs      = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mean_spec  = np.mean(stft, axis=1) + 1e-10
    total      = np.sum(mean_spec)

    bands = [
        (0,   200),
        (200, 800),
        (800, 2000),
        (2000, 4000),
        (4000, 8000),
    ]

    ratios = []
    for low, high in bands:
        mask  = (freqs >= low) & (freqs < high)
        ratio = float(np.sum(mean_spec[mask]) / total)
        ratios.append(ratio)

    return np.array(ratios)


def _crest_factor_block(signal, sr):
    """
    [NOUVEAU] Facteur de crête et kurtosis.
    Crest factor = max(|signal|) / RMS
      → Un claquement, un cognement moteur (knocking) fait exploser ce ratio.
      → Valeur normale moteur : 4–8. Anomalie : >12.
    Kurtosis = "à quel point les valeurs extrêmes sont fréquentes"
      → Distribution normale → kurtosis ≈ 3
      → Choc, impact, knock → kurtosis >> 3
    Skewness = asymétrie de la distribution des amplitudes
      → Détecte une dissymétrie dans les explosions (combustion inégale)
    Dimensions : 3
    """
    rms_val     = float(np.sqrt(np.mean(signal ** 2))) + 1e-10
    peak_val    = float(np.max(np.abs(signal)))
    crest       = peak_val / rms_val
    kurt        = float(kurtosis(signal))
    skewness    = float(skew(signal))

    return np.array([crest, kurt, skewness])


def _spectral_flux_block(signal, sr):
    """
    [NOUVEAU] Flux spectral : vitesse de changement du spectre frame par frame.
    Physique moteur : un moteur sain tourne régulièrement →
    le spectre change peu entre deux frames consécutives (flux faible et stable).
    Une anomalie (raté d'allumage, irrégularité) → pic de flux localisé.
    On retourne mean et std du flux + le 90ème percentile (pour capturer les pics).
    Dimensions : 3
    """
    stft       = np.abs(librosa.stft(signal, n_fft=2048, hop_length=512))
    # Différence entre frames consécutives, normalisée
    flux       = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))

    return np.array([
        float(np.mean(flux)),
        float(np.std(flux)),
        float(np.percentile(flux, 90)),
    ])


def _autocorrelation_block(signal, sr):
    """
    [NOUVEAU] Périodicité du signal via autocorrélation.
    Un moteur = machine cyclique. Chaque rotation produit un pattern qui se répète.
    L'autocorrélation cherche "est-ce que ce signal se ressemble à lui-même
    décalé de T secondes ?" Si oui → fort pic → moteur régulier.
    Une anomalie brise la périodicité → pic plus faible, ou décalé.
    On extrait :
      - peak_ratio : force du pic de périodicité (0 = apériodique, 1 = parfait)
      - peak_lag_normalized : à quelle fréquence de répétition (lié au RPM)
    Dimensions : 2
    """
    # On limite à 0.5s pour la performance
    max_lag    = min(len(signal), int(sr * 0.5))
    signal_cut = signal[:max_lag]

    autocorr   = np.correlate(signal_cut, signal_cut, mode="full")
    autocorr   = autocorr[len(autocorr) // 2:]  # partie positive uniquement
    autocorr   = autocorr / (autocorr[0] + 1e-10)  # normaliser à 1 en lag=0

    # Cherche le premier pic après lag=0 (exclure le pic trivial)
    min_lag    = int(sr * 0.01)  # ignore les lags < 10ms (bruit)
    search     = autocorr[min_lag:]

    if len(search) == 0:
        return np.array([0.0, 0.0])

    peak_idx   = int(np.argmax(search)) + min_lag
    peak_ratio = float(autocorr[peak_idx])
    peak_lag_normalized = float(peak_idx / max_lag)

    return np.array([peak_ratio, peak_lag_normalized])


def _harmonic_ratio_block(signal):
    """
    [NOUVEAU] Ratio harmonique/percussif via HPSS par fenêtre.
    Physique moteur : les sons moteur sont principalement harmoniques
    (combustion cyclique régulière). Un claquement ou grippement
    injecte de l'énergie percussive anormale.
    Dimensions : 3
    """
    harmonic, percussive = librosa.effects.hpss(signal)

    h_energy = float(np.mean(np.abs(harmonic)))
    p_energy = float(np.mean(np.abs(percussive)))
    total    = h_energy + p_energy + 1e-10

    return np.array([
        h_energy / total,  # ratio harmonique
        p_energy / total,  # ratio percussif
        h_energy / (p_energy + 1e-10),  # rapport H/P brut
    ])


# ─────────────────────────────────────────────
#  ASSEMBLAGE FINAL DU VECTEUR
# ─────────────────────────────────────────────

def extract_window_features(signal, sr):
    """
    Vecteur de features complet par fenêtre.
    Total : 78 + 12 + 14 + 64 + 5 + 3 + 3 + 2 + 3 = 184 dimensions
    """
    blocks = [
        _mfcc_block(signal, sr),            # 78  — timbre + évolution
        _spectral_shape_block(signal, sr),  # 12  — forme spectrale globale
        _spectral_contrast_block(signal, sr),# 14 — pics vs vallées harmoniques
        _mel_spectrogram_block(signal, sr), # 64  — basses fréquences détaillées
        _subband_energy_block(signal, sr),  # 5   — énergie par bande physique
        _crest_factor_block(signal, sr),    # 3   — chocs, knocking, asymétrie
        _spectral_flux_block(signal, sr),   # 3   — régularité temporelle
        _autocorrelation_block(signal, sr), # 2   — périodicité / RPM
        _harmonic_ratio_block(signal),      # 3   — ratio harmonique/percussif
    ]

    return np.concatenate(blocks).astype(np.float32)


def extract_windowed_features(signal, sr, window_duration=1.0, hop_duration=0.5):
    """
    Extrait les features de toutes les fenêtres du signal.
    Retourne une matrice (n_windows, n_features).
    """
    windows = split_into_windows(signal, sr, window_duration, hop_duration)
    if len(windows) == 0:
        return np.empty((0, 0), dtype=np.float32)

    features = [extract_window_features(w, sr) for w in windows]
    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────
#  STANDARDISATION
# ─────────────────────────────────────────────

def fit_reference_scaler(reference_features):
    """
    Calcule moyenne et écart-type à partir de la référence.
    """
    mean       = np.mean(reference_features, axis=0)
    std        = np.std(reference_features,  axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def transform_features(features, mean, std):
    """
    Standardise les features avec les stats de la référence.
    """
    return (features - mean) / std


# ─────────────────────────────────────────────
#  COMPARAISON ET SCORING
# ─────────────────────────────────────────────

def pairwise_euclidean(a, b):
    """
    Distances euclidiennes entre toutes les paires (ligne de a, ligne de b).
    """
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def compute_reference_baseline(reference_scaled):
    """
    Mesure la variabilité interne normale de la référence.
    Sert de règle pour calibrer le score d'anomalie.
    """
    if len(reference_scaled) < 2:
        return {"median": 0.0, "p90": 1.0, "max": 1.0, "nearest_distances": []}

    dist_matrix = pairwise_euclidean(reference_scaled, reference_scaled)
    np.fill_diagonal(dist_matrix, np.inf)
    nearest_distances = np.min(dist_matrix, axis=1)

    return {
        "median": float(np.median(nearest_distances)),
        "p90":    float(np.percentile(nearest_distances, 90)),
        "max":    float(np.max(nearest_distances)),
        "nearest_distances": nearest_distances.tolist(),
    }


def compare_to_reference(reference_features, test_features):
    """
    Compare le test à la référence.
    Retourne un score d'anomalie basé sur la distance euclidienne normalisée,
    enrichi par des indicateurs sur les fenêtres les plus déviantes.
    """
    if len(reference_features) == 0 or len(test_features) == 0:
        return {
            "anomaly_score":      float("inf"),
            "anomaly_ratio":      float("inf"),
            "distance_median":    float("inf"),
            "distance_p90":       float("inf"),
            "worst_window_ratio": float("inf"),
            "local_distances":    [],
        }

    mean, std = fit_reference_scaler(reference_features)

    reference_scaled = transform_features(reference_features, mean, std)
    test_scaled      = transform_features(test_features,      mean, std)

    baseline = compute_reference_baseline(reference_scaled)

    dist_matrix    = pairwise_euclidean(test_scaled, reference_scaled)
    min_distances  = np.min(dist_matrix, axis=1)

    distance_median  = float(np.median(min_distances))
    distance_p90     = float(np.percentile(min_distances, 90))
    distance_max     = float(np.max(min_distances))

    baseline_ref     = max(baseline["p90"], 1e-6)
    anomaly_ratio    = distance_median / baseline_ref

    # Fenêtres les plus anormales (au-dessus du p90 de la référence)
    threshold        = baseline["p90"]
    anomalous_mask   = min_distances > threshold
    anomalous_count  = int(np.sum(anomalous_mask))
    anomalous_ratio  = float(anomalous_count / len(min_distances))

    # Fenêtre la pire (ratio local)
    worst_window_ratio = float(distance_max / baseline_ref)

    return {
        "anomaly_score":       distance_median,
        "anomaly_ratio":       anomaly_ratio,
        "distance_median":     distance_median,
        "distance_p90":        distance_p90,
        "distance_max":        distance_max,
        "baseline_median":     baseline["median"],
        "baseline_p90":        baseline["p90"],
        "baseline_max":        baseline["max"],
        "anomalous_windows":   anomalous_count,
        "anomalous_ratio":     anomalous_ratio,
        "worst_window_ratio":  worst_window_ratio,
        "local_distances":     min_distances.tolist(),
    }