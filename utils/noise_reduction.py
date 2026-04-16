import os
import numpy as np
import noisereduce as nr
import matplotlib.pyplot as plt
import soundfile as sf


def normalize_audio(signal):
    """
    Normalise le signal entre -1 et 1.
    """
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal / max_val


def reduce_background_noise(signal, sr, prop_decrease=0.4):
    """
    Réduit le bruit de fond.
    prop_decrease plus faible = traitement moins agressif.
    """
    cleaned_signal = nr.reduce_noise(
        y=signal,
        sr=sr,
        prop_decrease=prop_decrease
    )
    return cleaned_signal


def save_audio(signal, sr, output_path):
    """
    Sauvegarde un signal audio dans un fichier WAV.
    """
    sf.write(output_path, signal, sr)


def save_waveform_comparison(original_signal, normalized_signal, cleaned_signal, sr, output_path):
    """
    Sauvegarde une image avec 3 formes d'onde :
    original / normalisé / nettoyé.
    """
    duration_original = len(original_signal) / sr
    duration_normalized = len(normalized_signal) / sr
    duration_cleaned = len(cleaned_signal) / sr

    time_original = np.linspace(0, duration_original, len(original_signal))
    time_normalized = np.linspace(0, duration_normalized, len(normalized_signal))
    time_cleaned = np.linspace(0, duration_cleaned, len(cleaned_signal))

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time_original, original_signal)
    plt.title("Signal original")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 2)
    plt.plot(time_normalized, normalized_signal)
    plt.title("Signal normalisé")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 3)
    plt.plot(time_cleaned, cleaned_signal)
    plt.title("Signal après réduction du bruit")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def ensure_results_dir(results_dir="results"):
    """
    Crée le dossier results s'il n'existe pas.
    """
    os.makedirs(results_dir, exist_ok=True)