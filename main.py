import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from utils.audio_loader import validate_audio_path, load_audio
from utils.noise_reduction import normalize_audio, reduce_background_noise
from utils.audio_validator import extract_audio_profile, compare_profiles, build_validation_message
from utils.comparator import (
    trim_and_fix_length,
    extract_windowed_features,
    compare_to_reference,
    save_reference_cache,
    load_reference_cache,
)
from utils.verdict import get_verdict, build_result_message

# ─────────────────────────────────────────────
#  CONFIGURATION CENTRALE
# ─────────────────────────────────────────────

REFERENCE_DATABASE = {
    "V_MERCEDES": [
        r"data/reference/Mercedes Benz E200 M271 Cold Start.wav",
    ]
}

CACHE_DIR       = Path("data/cache")
RESULTS_DIR     = Path("results")
TARGET_SR       = 22050
TARGET_DURATION = 5.0
NOISE_DECREASE  = 0.4


# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────

def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )


# ─────────────────────────────────────────────
#  PRÉTRAITEMENT AUDIO
# ─────────────────────────────────────────────

def preprocess_audio(file_path: str, target_sr: int = TARGET_SR):
    signal, sr = load_audio(file_path, target_sr=target_sr, mono=True)
    signal = normalize_audio(signal)
    signal = reduce_background_noise(signal, sr, prop_decrease=NOISE_DECREASE)
    signal = trim_and_fix_length(signal, sr, target_duration=TARGET_DURATION)
    return signal, sr


# ─────────────────────────────────────────────
#  RÉFÉRENCE — CACHE ET MULTI-FICHIERS
# ─────────────────────────────────────────────

def build_reference_features(vehicle_id: str, reference_paths: list, sr: int):
    import numpy as np
    all_features = []
    for path in reference_paths:
        logging.info(f"  Chargement référence : {path}")
        validate_audio_path(path)
        ref_signal, ref_sr = preprocess_audio(path, target_sr=sr)
        feats = extract_windowed_features(ref_signal, ref_sr)
        all_features.append(feats)
    combined = np.concatenate(all_features, axis=0)
    logging.info(f"Référence totale : {len(combined)} fenêtres ({len(reference_paths)} fichier(s))")
    return combined


def get_reference_features(vehicle_id: str, reference_paths: list, sr: int):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{vehicle_id}_features.npy"
    cached = load_reference_cache(str(cache_path))
    if cached is not None:
        logging.info(f"Cache trouvé → {len(cached)} fenêtres chargées")
        return cached
    logging.info("Pas de cache → calcul des features de référence...")
    features = build_reference_features(vehicle_id, reference_paths, sr)
    save_reference_cache(str(cache_path), features)
    return features


# ─────────────────────────────────────────────
#  EXPORT JSON
# ─────────────────────────────────────────────

def export_results(vehicle_id, test_path, comparison_result, verdict, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_stem  = Path(test_path).stem
    output_path = output_dir / f"{vehicle_id}_{audio_stem}_report.json"

    report = {
        "vehicle_id":         vehicle_id,
        "test_audio":         test_path,
        "anomaly_ratio":      round(comparison_result["anomaly_ratio"], 4),
        "verdict":            verdict,
        "baseline_median":    round(comparison_result["baseline_median"], 4),
        "baseline_p90":       round(comparison_result["baseline_p90"], 4),
        "distance_median":    round(comparison_result["distance_median"], 4),
        "distance_p90":       round(comparison_result["distance_p90"], 4),
        "distance_max":       round(comparison_result["distance_max"], 4),
        "anomalous_windows":  comparison_result["anomalous_windows"],
        "anomalous_ratio":    round(comparison_result["anomalous_ratio"], 4),
        "worst_window_ratio": round(comparison_result["worst_window_ratio"], 4),
        "total_windows":      len(comparison_result["local_distances"]),
        "local_distances":    [round(d, 4) for d in comparison_result["local_distances"]],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return output_path


# ─────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def main():
    setup_logging()

    print("=" * 60)
    print("Bienvenue dans EchoMoteur")
    print("=" * 60)

    vehicle_id      = input("Entrez l'identifiant du véhicule : ").strip()
    test_audio_path = input("Entrez le chemin du fichier audio à analyser : ").strip()

    progress = None

    try:
        if vehicle_id not in REFERENCE_DATABASE:
            raise ValueError(
                f"Identifiant inconnu : '{vehicle_id}'. "
                f"Disponibles : {', '.join(REFERENCE_DATABASE.keys())}"
            )

        reference_paths = REFERENCE_DATABASE[vehicle_id]
        progress = tqdm(total=7, desc="Analyse", unit="étape")

        # Étape 1 — Validation fichier test
        validate_audio_path(test_audio_path)
        progress.update(1)

        # Étape 2 — Référence (cache ou calcul)
        reference_features = get_reference_features(vehicle_id, reference_paths, TARGET_SR)
        progress.update(1)

        # Étape 3 — Prétraitement signal test
        test_signal, sr_test = preprocess_audio(test_audio_path, target_sr=TARGET_SR)
        progress.update(1)

        # Étape 4 — Gate de compatibilité
        ref_signal_sample, sr_ref = preprocess_audio(reference_paths[0], target_sr=TARGET_SR)
        ref_profile  = extract_audio_profile(ref_signal_sample, sr_ref)
        test_profile = extract_audio_profile(test_signal, sr_test)
        validation   = compare_profiles(ref_profile, test_profile)
        logging.info(build_validation_message(validation))

        if not validation["is_compatible"]:
            progress.close()
            print(
                f"\nAnalyse arrêtée : signal trop éloigné du profil de référence "
                f"(score : {validation['score']:.1f}%). "
                f"Vérifiez que l'audio vient bien du véhicule {vehicle_id}."
            )
            return
        progress.update(1)

        # Étape 5 — Features test
        test_features = extract_windowed_features(test_signal, sr_test)
        progress.update(1)

        # Étape 6 — Comparaison
        comparison_result = compare_to_reference(reference_features, test_features)
        progress.update(1)

        # Étape 7 — Verdict + export
        anomaly_ratio = comparison_result["anomaly_ratio"]
        verdict       = get_verdict(anomaly_ratio)
        report_path   = export_results(vehicle_id, test_audio_path, comparison_result, verdict, RESULTS_DIR)
        progress.update(1)
        progress.close()

        print("\n" + "=" * 60)
        print("Résultat de comparaison")
        print("=" * 60)
        print(f"Véhicule                : {vehicle_id}")
        print(f"Test                    : {test_audio_path}")
        print(f"Compatibilité globale   : {validation['score']:.1f} %")
        print()
        print(f"Baseline médian réf.    : {comparison_result['baseline_median']:.4f}")
        print(f"Baseline P90 réf.       : {comparison_result['baseline_p90']:.4f}")
        print(f"Distance médiane test   : {comparison_result['distance_median']:.4f}")
        print(f"Distance P90 test       : {comparison_result['distance_p90']:.4f}")
        print(f"Fenêtres anormales      : {comparison_result['anomalous_windows']} / "
              f"{len(comparison_result['local_distances'])} "
              f"({comparison_result['anomalous_ratio']*100:.1f} %)")
        print(f"Rapport d'anomalie      : {anomaly_ratio:.4f}")
        print(f"Verdict                 : {verdict}")
        print()
        print(build_result_message(vehicle_id, anomaly_ratio, verdict))
        print()
        print(f"Rapport JSON sauvegardé : {report_path}")

    except Exception as e:
        if progress is not None:
            progress.close()
        print(f"\nErreur : {e}")


if __name__ == "__main__":
    main()