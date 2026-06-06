import json
import logging
import pickle
from pathlib import Path

from tqdm import tqdm

from utils.audio_validator import validate_audio_file, compare_audio_context
from utils.noise_reduction import preprocess_audio
from utils.comparator import (
    build_reference_features,
    extract_features_by_windows,
    compare_to_reference,
)
from utils.verdict import get_verdict


# ============================================================
# Configuration générale
# ============================================================

TARGET_SR = 22050
CACHE_DIR = Path("data/cache")
RESULTS_DIR = Path("results")

# Base actuelle.
# Plus tard, on la déplacera dans config/vehicles.json.
REFERENCE_DATABASE = {
    "V_MERCEDES": [
        r"data/reference/Mercedes Benz E200 M271 Cold Start.wav",
    ]
}


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# ============================================================
# Outils
# ============================================================

def sanitize_filename(name: str) -> str:
    forbidden = '<>:"/\\|?*'
    cleaned = "".join("_" if c in forbidden else c for c in name)
    return cleaned.strip().replace(" ", "_")


def get_cache_path(vehicle_id: str) -> Path:
    """
    Cache V2.2 séparé du cache V2 pour éviter les mélanges de formats.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{vehicle_id}_features_v22.pkl"


def load_reference_cache(vehicle_id: str):
    cache_path = get_cache_path(vehicle_id)

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as file:
            data = pickle.load(file)

        if not isinstance(data, dict):
            logging.warning("Cache invalide : format inattendu.")
            return None

        if data.get("cache_version") != "2.2":
            logging.warning("Cache ancien ou incompatible. Recalcul nécessaire.")
            return None

        logging.info(f"Cache chargé : {cache_path}")
        return data

    except Exception as exc:
        logging.warning(f"Cache illisible : {exc}. Recalcul nécessaire.")
        return None


def save_reference_cache(vehicle_id: str, reference_features: dict):
    cache_path = get_cache_path(vehicle_id)
    reference_features["cache_version"] = "2.2"

    with open(cache_path, "wb") as file:
        pickle.dump(reference_features, file)

    logging.info(f"Cache sauvegardé : {cache_path}")


def export_report(vehicle_id: str, test_audio_path: str, result: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    audio_name = Path(test_audio_path).stem
    report_name = f"{sanitize_filename(vehicle_id)}_{sanitize_filename(audio_name)}_report_v22.json"
    report_path = RESULTS_DIR / report_name

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)

    return report_path


def is_reference_file(test_audio_path: str, reference_paths: list[str]) -> bool:
    try:
        test_resolved = Path(test_audio_path).resolve()
        return any(Path(ref).resolve() == test_resolved for ref in reference_paths)
    except Exception:
        return False


def print_block_scores(block_scores: dict):
    if not block_scores:
        print("Aucun score par bloc disponible.")
        return

    ordered = sorted(block_scores.items(), key=lambda item: item[1], reverse=True)
    for block_name, score in ordered:
        print(f"- {block_name:<16} : {score:.4f}")


# ============================================================
# Pipeline principal
# ============================================================

def run_pipeline(vehicle_id: str, test_audio_path: str) -> dict:
    if vehicle_id not in REFERENCE_DATABASE:
        available = ", ".join(REFERENCE_DATABASE.keys())
        raise ValueError(f"Véhicule inconnu : {vehicle_id}. Véhicules disponibles : {available}")

    reference_paths = REFERENCE_DATABASE[vehicle_id]

    if is_reference_file(test_audio_path, reference_paths):
        logging.warning(
            "Le fichier testé appartient à la base de référence. "
            "Ce test vérifie le pipeline mais ne valide pas la capacité de diagnostic."
        )

    # 1. Validation fichier
    validation = validate_audio_file(test_audio_path)
    if not validation.get("is_valid", False):
        return {
            "status": "error",
            "message": "Fichier audio invalide.",
            "validation": validation,
        }

    # 2. Charger ou calculer la référence
    reference_features = load_reference_cache(vehicle_id)

    if reference_features is None:
        logging.info("Pas de cache valide → calcul des features de référence...")
        reference_features = build_reference_features(reference_paths, target_sr=TARGET_SR)
        save_reference_cache(vehicle_id, reference_features)

    # 3. Prétraitement audio test
    test_signal, sr = preprocess_audio(test_audio_path, target_sr=TARGET_SR)

    # 4. Comparaison de contexte audio
    context_result = compare_audio_context(reference_features.get("context_profiles", []), test_signal, sr)

    if context_result["status"] == "compatible":
        logging.info(f"Audio compatible avec la référence (score : {context_result['score']:.1f} %).")
    elif context_result["status"] == "suspect":
        logging.warning(
            f"Audio moteur suspect mais analysable (score : {context_result['score']:.1f} %)."
        )
    else:
        logging.warning(
            f"Audio possiblement hors contexte (score : {context_result['score']:.1f} %). "
            "L'analyse continue mais la confiance sera réduite."
        )

    # 5. Extraction features test
    test_features = extract_features_by_windows(test_signal, sr)

    # 6. Comparaison fine à la référence
    comparison = compare_to_reference(
        test_features=test_features,
        reference_features=reference_features,
        context_result=context_result,
    )

    # 7. Verdict composite
    verdict = get_verdict(comparison)

    result = {
        "status": "success",
        "vehicle_id": vehicle_id,
        "test_audio_path": str(test_audio_path),
        "is_reference_file": is_reference_file(test_audio_path, reference_paths),
        "validation": validation,
        "context": context_result,
        "comparison": comparison,
        "verdict": verdict,
    }

    report_path = export_report(vehicle_id, test_audio_path, result)
    result["report_path"] = str(report_path)

    return result


def main():
    print("=" * 60)
    print("Bienvenue dans EchoMoteur V2.2")
    print("=" * 60)

    vehicle_id = input("Entrez l'identifiant du véhicule : ").strip()
    test_audio_path = input("Entrez le chemin du fichier audio à analyser : ").strip()

    try:
        with tqdm(total=1, desc="Analyse", unit="pipeline") as progress:
            result = run_pipeline(vehicle_id, test_audio_path)
            progress.update(1)

    except Exception as exc:
        print(f"\nErreur : {exc}")
        return

    if result["status"] != "success":
        print("\nAnalyse impossible.")
        print(json.dumps(result, indent=4, ensure_ascii=False))
        return

    comparison = result["comparison"]
    verdict = result["verdict"]
    context = result["context"]

    print("\n" + "=" * 60)
    print("Résultat de comparaison V2.2")
    print("=" * 60)

    print(f"Véhicule                  : {result['vehicle_id']}")
    print(f"Test                      : {result['test_audio_path']}")
    print(f"Durée analysée             : {comparison.get('duration_sec', 0):.2f} s")
    print(f"Contexte audio             : {context.get('status')} ({context.get('score', 0):.1f} %)")

    if result.get("is_reference_file"):
        print("\n⚠️  Attention : le fichier testé est aussi utilisé comme référence.")
        print("Ce résultat vérifie le pipeline, mais ne valide pas la capacité de diagnostic.")

    print("\n--- Scores globaux ---")
    print(f"Score global anomalie      : {comparison.get('global_score', 0):.4f}")
    print(f"Ratio médian               : {comparison.get('median_ratio', 0):.4f}")
    print(f"Ratio P90                  : {comparison.get('p90_ratio', 0):.4f}")
    print(f"Pire fenêtre brute         : {comparison.get('worst_window_ratio_raw', 0):.4f}")
    print(f"Pire fenêtre pondérée      : {comparison.get('worst_window_score', 0):.4f}")
    print(
        f"Fenêtres anormales         : {comparison.get('anomalous_windows', 0)} / "
        f"{comparison.get('total_windows', 0)} "
        f"({comparison.get('anomalous_ratio', 0) * 100:.1f} %)"
    )
    print(f"Confiance analyse          : {comparison.get('confidence_score', 0):.1f} %")
    print(f"Verdict                    : {verdict.get('label')}")

    print("\nScores par bloc :")
    print_block_scores(comparison.get("block_scores", {}))

    print("\nExplication :")
    print(verdict.get("summary", ""))

    explanations = verdict.get("explanations", [])
    if explanations:
        print("Détails :")
        for explanation in explanations:
            print(f"- {explanation}")

    print(f"\nRapport JSON sauvegardé : {result.get('report_path')}")


if __name__ == "__main__":
    main()
