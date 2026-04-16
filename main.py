from tqdm import tqdm

from utils.audio_loader import validate_audio_path, load_audio
from utils.noise_reduction import normalize_audio
from utils.comparator import (
    trim_and_fix_length,
    extract_windowed_features,
    compare_to_reference,
)
from utils.verdict import get_verdict, build_result_message


REFERENCE_DATABASE = {
    "V_MERCEDES": r"data/reference/Mercedes Benz E200 M271 Cold Start.wav"
}


def preprocess_audio(file_path, target_sr=22050):
    signal, sr = load_audio(file_path, target_sr=target_sr, mono=True)
    signal = normalize_audio(signal)
    signal = trim_and_fix_length(signal, sr, target_duration=5.0)
    return signal, sr


def main():
    print("=" * 60)
    print("Bienvenue dans EchoMoteur")
    print("=" * 60)

    vehicle_id = input("Entrez l'identifiant du véhicule : ").strip()
    test_audio_path = input("Entrez le chemin du fichier audio à analyser : ").strip()

    progress = None

    try:
        if vehicle_id not in REFERENCE_DATABASE:
            raise ValueError(
                f"Identifiant inconnu : {vehicle_id}. "
                f"Identifiants disponibles : {', '.join(REFERENCE_DATABASE.keys())}"
            )

        reference_audio_path = REFERENCE_DATABASE[vehicle_id]

        progress = tqdm(total=6, desc="Analyse", unit="étape")

        validate_audio_path(test_audio_path)
        progress.update(1)

        validate_audio_path(reference_audio_path)
        progress.update(1)

        reference_signal, sr_ref = preprocess_audio(reference_audio_path)
        progress.update(1)

        test_signal, sr_test = preprocess_audio(test_audio_path)
        progress.update(1)

        if sr_ref != sr_test:
            raise ValueError("Les fréquences d'échantillonnage ne correspondent pas après traitement.")

        reference_features = extract_windowed_features(reference_signal, sr_ref)
        test_features = extract_windowed_features(test_signal, sr_test)
        progress.update(1)

        comparison_result = compare_to_reference(reference_features, test_features)
        progress.update(1)

        progress.close()

        anomaly_ratio = comparison_result["anomaly_ratio"]
        verdict = get_verdict(anomaly_ratio)
        result_message = build_result_message(vehicle_id, anomaly_ratio, verdict)

        print("\n" + "=" * 60)
        print("Résultat de comparaison")
        print("=" * 60)
        print(f"Véhicule                : {vehicle_id}")
        print(f"Référence               : {reference_audio_path}")
        print(f"Test                    : {test_audio_path}")
        print(f"Baseline médian réf.    : {comparison_result['baseline_median']:.4f}")
        print(f"Baseline P90 réf.       : {comparison_result['baseline_p90']:.4f}")
        print(f"Distance médiane test   : {comparison_result['distance_median']:.4f}")
        print(f"Distance P90 test       : {comparison_result['distance_p90']:.4f}")
        print(f"Rapport d'anomalie      : {anomaly_ratio:.4f}")
        print(f"Verdict                 : {verdict}")
        print(f"Fenêtres comparées      : {len(comparison_result['local_distances'])}")
        print()
        print(result_message)

    except Exception as e:
        if progress is not None:
            progress.close()
        print("\nErreur :", e)


if __name__ == "__main__":
    main()