import logging
from pathlib import Path

import librosa
import numpy as np

try:
    import noisereduce as nr
except Exception:
    nr = None


DEFAULT_TARGET_SR = 22050
DEFAULT_TARGET_RMS = 0.08

# Important : par défaut, on ne débruite pas.
# Pour un moteur, un "bruit" peut être l'information utile : claquement, choc, frottement.
USE_DENOISING = False


def load_audio(file_path: str, target_sr: int = DEFAULT_TARGET_SR) -> tuple[np.ndarray, int]:
    """
    Charge un fichier audio en mono avec librosa.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier audio introuvable : {file_path}")

    signal, sr = librosa.load(str(path), sr=target_sr, mono=True)
    signal = np.asarray(signal, dtype=np.float32)

    if signal.size == 0:
        raise ValueError("Signal audio vide.")

    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal, sr


def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
    """
    Supprime la composante continue.
    """
    signal = np.asarray(signal, dtype=np.float32)
    return signal - float(np.mean(signal))


def normalize_rms(signal: np.ndarray, target_rms: float = DEFAULT_TARGET_RMS) -> np.ndarray:
    """
    Normalisation RMS prudente.

    On évite la normalisation par amplitude max seule, car elle rend un seul pic trop influent.
    """
    signal = np.asarray(signal, dtype=np.float32)
    rms = float(np.sqrt(np.mean(signal ** 2)) + 1e-12)

    if rms < 1e-8:
        return signal

    normalized = signal * (target_rms / rms)

    peak = float(np.max(np.abs(normalized)) + 1e-12)
    if peak > 0.98:
        normalized = normalized / peak * 0.98

    return normalized.astype(np.float32)


def trim_silence(signal: np.ndarray, sr: int, top_db: float = 35.0) -> np.ndarray:
    """
    Supprime les grands silences au début et à la fin, sans imposer une durée fixe.
    """
    signal = np.asarray(signal, dtype=np.float32)

    try:
        trimmed, _ = librosa.effects.trim(signal, top_db=top_db)
    except Exception:
        return signal

    if trimmed.size < int(0.5 * sr):
        # Si le trim détruit presque tout, on garde le signal original.
        return signal

    return trimmed.astype(np.float32)


def reduce_background_noise(signal: np.ndarray, sr: int, prop_decrease: float = 0.35) -> np.ndarray:
    """
    Réduction de bruit optionnelle.

    Désactivée par défaut dans preprocess_audio, car elle peut supprimer ou déformer des indices mécaniques.
    """
    if nr is None:
        logging.warning("noisereduce indisponible. Signal gardé sans débruitage.")
        return signal

    try:
        cleaned = nr.reduce_noise(
            y=signal,
            sr=sr,
            prop_decrease=prop_decrease,
            stationary=False,
        )
        return np.asarray(cleaned, dtype=np.float32)
    except Exception as exc:
        logging.warning(f"Réduction de bruit échouée : {exc}. Signal original conservé.")
        return signal


def preprocess_audio(
    file_path: str,
    target_sr: int = DEFAULT_TARGET_SR,
    use_denoising: bool = USE_DENOISING,
    trim: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Prétraitement V2.2 :
    - chargement mono
    - suppression DC
    - normalisation RMS
    - trim des silences
    - débruitage optionnel uniquement
    - aucune coupure fixe à 5 secondes
    """
    signal, sr = load_audio(file_path, target_sr=target_sr)

    signal = remove_dc_offset(signal)
    signal = normalize_rms(signal)

    if trim:
        signal = trim_silence(signal, sr)

    if use_denoising:
        signal = reduce_background_noise(signal, sr)
        signal = normalize_rms(remove_dc_offset(signal))

    return signal.astype(np.float32), sr


def trim_and_fix_length(signal: np.ndarray, sr: int, target_duration: float = 5.0) -> np.ndarray:
    """
    Fonction gardée seulement pour compatibilité avec l'ancien code.
    Elle est dépréciée : ne pas l'utiliser dans le pipeline principal.
    """
    logging.warning("trim_and_fix_length est dépréciée. Utiliser trim_silence() sans durée fixe.")

    trimmed = trim_silence(signal, sr)
    target_length = int(target_duration * sr)

    if trimmed.size >= target_length:
        return trimmed[:target_length].astype(np.float32)

    padded = np.zeros(target_length, dtype=np.float32)
    padded[: trimmed.size] = trimmed
    return padded
