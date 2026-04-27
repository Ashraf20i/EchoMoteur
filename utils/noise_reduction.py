import logging

import numpy as np
import noisereduce as nr
import soundfile as sf


# ─────────────────────────────────────────────
#  NORMALISATION
# ─────────────────────────────────────────────

def normalize_audio(signal: np.ndarray) -> np.ndarray:
    """
    Normalise le signal entre -1 et 1.
    Indispensable pour comparer des enregistrements
    faits avec des micros ou des niveaux d'entrée différents.
    """
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        logging.warning("Signal silencieux détecté (amplitude nulle). Normalisation ignorée.")
        return signal
    return signal / max_val


# ─────────────────────────────────────────────
#  RÉDUCTION DU BRUIT
# ─────────────────────────────────────────────

def reduce_background_noise(signal: np.ndarray, sr: int, prop_decrease: float = 0.4) -> np.ndarray:
    """
    Réduit le bruit de fond ambiant du signal audio.

    prop_decrease : intensité du traitement
      - 0.0 → aucune réduction (signal intact)
      - 0.4 → réduction modérée (recommandé pour audio moteur en extérieur)
      - 1.0 → réduction maximale (risque d'artefacts sur le signal utile)

    La valeur 0.4 est volontairement conservative : on préfère garder
    un peu de bruit plutôt que de déformer les harmoniques du moteur.
    """
    logging.debug(f"Réduction du bruit (prop_decrease={prop_decrease})")
    cleaned_signal = nr.reduce_noise(
        y=signal,
        sr=sr,
        prop_decrease=prop_decrease,
    )
    return cleaned_signal


# ─────────────────────────────────────────────
#  UTILITAIRES OPTIONNELS (non utilisés dans le pipeline principal)
# ─────────────────────────────────────────────

def save_audio(signal: np.ndarray, sr: int, output_path: str) -> None:
    """
    Sauvegarde un signal audio dans un fichier WAV.
    Utile pour inspecter manuellement un signal après prétraitement.
    """
    sf.write(output_path, signal, sr)
    logging.info(f"Audio sauvegardé : {output_path}")


def save_waveform_comparison(
    original_signal: np.ndarray,
    normalized_signal: np.ndarray,
    cleaned_signal: np.ndarray,
    sr: int,
    output_path: str,
) -> None:
    """
    Sauvegarde une image avec 3 formes d'onde :
    original / normalisé / nettoyé.
    Utile pour déboguer visuellement le prétraitement.
    Import matplotlib ici pour ne pas alourdir le pipeline si non utilisé.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    signals = [
        (original_signal,   "Signal original"),
        (normalized_signal, "Signal normalisé"),
        (cleaned_signal,    "Signal après réduction du bruit"),
    ]

    for ax, (sig, title) in zip(axes, signals):
        duration = len(sig) / sr
        time     = np.linspace(0, duration, len(sig))
        ax.plot(time, sig)
        ax.set_title(title)
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Graphe de formes d'onde sauvegardé : {output_path}")