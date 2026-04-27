import logging
import os

import librosa
import numpy as np
import soundfile as sf


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# ─────────────────────────────────────────────
#  VALIDATION DU CHEMIN
# ─────────────────────────────────────────────

def validate_audio_path(file_path: str) -> None:
    """
    Vérifie que le chemin existe, pointe vers un fichier,
    et que l'extension est supportée.
    Lève une exception explicite à la première anomalie trouvée.
    """
    if not file_path:
        raise ValueError("Le chemin du fichier audio est vide.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"Ce chemin ne pointe pas vers un fichier : {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Extension non supportée : '{ext}'. "
            f"Extensions autorisées : {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    logging.debug(f"Fichier audio validé : {file_path}")


# ─────────────────────────────────────────────
#  INFORMATIONS FICHIER (sans charger le signal)
# ─────────────────────────────────────────────

def get_file_info(file_path: str) -> dict:
    """
    Récupère des métadonnées sur le fichier audio
    SANS charger le signal en mémoire (rapide).
    Utile pour un pré-diagnostic avant traitement.
    """
    info = sf.info(file_path)

    metadata = {
        "samplerate": info.samplerate,
        "channels":   info.channels,
        "duration":   round(info.duration, 3),
        "frames":     info.frames,
        "format":     info.format,
        "subtype":    info.subtype,
    }

    logging.debug(
        f"Infos fichier [{file_path}] : "
        f"{metadata['duration']}s — {metadata['samplerate']} Hz — "
        f"{metadata['channels']} canal(aux)"
    )

    return metadata


# ─────────────────────────────────────────────
#  CHARGEMENT DU SIGNAL
# ─────────────────────────────────────────────

def load_audio(file_path: str, target_sr: int = 22050, mono: bool = True):
    """
    Charge le signal audio avec librosa.

    target_sr : fréquence d'échantillonnage cible (rééchantillonnage automatique).
    mono=True : convertit en mono — obligatoire pour la comparaison par features.

    Retourne : (signal: np.ndarray, sr: int)
    """
    logging.debug(f"Chargement audio : {file_path} (target_sr={target_sr}, mono={mono})")
    signal, sr = librosa.load(file_path, sr=target_sr, mono=mono)
    logging.debug(f"Signal chargé : {len(signal)} samples — {sr} Hz — durée {len(signal)/sr:.2f}s")
    return signal, sr


# ─────────────────────────────────────────────
#  INFORMATIONS SIGNAL (après chargement)
# ─────────────────────────────────────────────

def get_signal_info(signal: np.ndarray, sr: int) -> dict:
    """
    Retourne des statistiques simples sur le signal chargé.
    Utile pour vérifier visuellement qu'un signal n'est pas silencieux
    ou saturé avant de lancer le pipeline complet.
    """
    if len(signal) == 0:
        logging.warning("Signal vide détecté dans get_signal_info.")
        return {
            "nb_samples":       0,
            "loaded_samplerate": sr,
            "loaded_duration":  0.0,
            "min_amplitude":    0.0,
            "max_amplitude":    0.0,
        }

    duration = len(signal) / sr if sr > 0 else 0.0

    return {
        "nb_samples":        len(signal),
        "loaded_samplerate": sr,
        "loaded_duration":   round(duration, 3),
        "min_amplitude":     float(signal.min()),
        "max_amplitude":     float(signal.max()),
    }