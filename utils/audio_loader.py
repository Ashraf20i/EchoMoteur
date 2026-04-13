import os
import librosa
import soundfile as sf


ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def validate_audio_path(file_path: str) -> None:
    """
    Vérifie que le chemin existe, pointe vers un fichier,
    et que l'extension est supportée.
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
            f"Extension non supportée : {ext}. "
            f"Extensions autorisées : {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )


def get_file_info(file_path: str) -> dict:
    """
    Récupère des informations générales sur le fichier audio
    sans encore charger complètement le signal dans la mémoire.
    """
    info = sf.info(file_path)

    return {
        "samplerate": info.samplerate,
        "channels": info.channels,
        "duration": round(info.duration, 3),
        "frames": info.frames,
        "format": info.format,
        "subtype": info.subtype,
    }


def load_audio(file_path: str, target_sr: int = 22050, mono: bool = True):
    """
    Charge le signal audio avec librosa.
    target_sr : fréquence d'échantillonnage cible
    mono=True : convertit en mono pour simplifier le projet
    """
    signal, sr = librosa.load(file_path, sr=target_sr, mono=mono)
    return signal, sr


def get_signal_info(signal, sr: int) -> dict:
    """
    Retourne des informations simples sur le signal chargé.
    """
    duration = len(signal) / sr if sr > 0 else 0

    return {
        "nb_samples": len(signal),
        "loaded_samplerate": sr,
        "loaded_duration": round(duration, 3),
        "min_amplitude": float(signal.min()) if len(signal) > 0 else 0.0,
        "max_amplitude": float(signal.max()) if len(signal) > 0 else 0.0,
    }