def _score(block_scores: dict, name: str) -> float:
    try:
        return float(block_scores.get(name, 0.0))
    except Exception:
        return 0.0


def _top_blocks(block_scores: dict, n: int = 3):
    return sorted(block_scores.items(), key=lambda item: item[1], reverse=True)[:n]


def _block_metric(comparison: dict, block_name: str, metric_name: str) -> float:
    """
    Lit une métrique locale dans comparison["block_details"].

    Exemple :
    block_details["impulses"]["worst_window_ratio_raw"]

    Cette information est capitale : un défaut mécanique peut être court,
    donc invisible dans la médiane globale, mais très visible dans une fenêtre.
    """
    try:
        return float(
            comparison
            .get("block_details", {})
            .get(block_name, {})
            .get(metric_name, 0.0)
        )
    except Exception:
        return 0.0


def _local_extreme_summary(comparison: dict) -> dict:
    """
    Cherche les blocs où une fenêtre locale s'écarte fortement.

    Pourquoi ?
    - Un knocking peut être intermittent.
    - La médiane peut rester faible.
    - Le verdict ne doit pas ignorer un pic local très violent.
    """
    block_details = comparison.get("block_details", {})

    local_scores = {}
    for block_name, details in block_details.items():
        try:
            local_scores[block_name] = float(details.get("worst_window_ratio_raw", 0.0))
        except Exception:
            local_scores[block_name] = 0.0

    if not local_scores:
        return {
            "local_extreme_score": 0.0,
            "local_extreme_block": None,
            "local_scores": {},
            "top_local_blocks": [],
        }

    top_local_blocks = sorted(local_scores.items(), key=lambda item: item[1], reverse=True)
    block, value = top_local_blocks[0]

    return {
        "local_extreme_score": value,
        "local_extreme_block": block,
        "local_scores": local_scores,
        "top_local_blocks": top_local_blocks[:3],
    }


def get_verdict(comparison: dict) -> dict:
    """
    Verdict V2.3.

    Correction majeure par rapport à V2.2 :
    V2.2 évitait les faux positifs, mais elle écrasait les anomalies courtes
    avec la médiane. Résultat : un knocking local pouvait sortir "normal".

    V2.3 garde la prudence sur les signatures différentes, mais ajoute
    une règle de sécurité :
    une fenêtre locale très anormale, surtout dans les blocs impulsions /
    cepstre / énergie, doit déclencher une alerte.
    """
    global_score = float(comparison.get("global_score", 0.0))
    median_ratio = float(comparison.get("median_ratio", 0.0))
    p90_ratio = float(comparison.get("p90_ratio", 0.0))
    worst_global = float(comparison.get("worst_window_ratio_raw", 0.0))
    anomalous_ratio = float(comparison.get("anomalous_ratio", 0.0))
    confidence = float(comparison.get("confidence_score", 100.0))
    n_reference_files = int(comparison.get("n_reference_files", 1))

    block_scores = comparison.get("block_scores", {})

    impulses = _score(block_scores, "impulses")
    frequency_bands = _score(block_scores, "frequency_bands")
    energy = _score(block_scores, "energy")
    spectral_shape = _score(block_scores, "spectral_shape")
    harmonicity = _score(block_scores, "harmonicity")
    cepstrum = _score(block_scores, "cepstrum")
    mfcc = _score(block_scores, "mfcc")

    local = _local_extreme_summary(comparison)
    local_extreme_score = local["local_extreme_score"]
    local_extreme_block = local["local_extreme_block"]

    impulse_worst = _block_metric(comparison, "impulses", "worst_window_ratio_raw")
    cepstrum_worst = _block_metric(comparison, "cepstrum", "worst_window_ratio_raw")
    energy_worst = _block_metric(comparison, "energy", "worst_window_ratio_raw")
    spectral_worst = _block_metric(comparison, "spectral_shape", "worst_window_ratio_raw")
    band_worst = _block_metric(comparison, "frequency_bands", "worst_window_ratio_raw")

    explanations = []

    for block, value in _top_blocks(block_scores, 3):
        if value < 2.0:
            continue

        if block == "impulses":
            explanations.append(
                f"présence moyenne de pics, claquements ou composantes percussives (score={value:.2f})"
            )
        elif block == "frequency_bands":
            explanations.append(
                f"déplacement d'énergie entre bandes fréquentielles (score={value:.2f})"
            )
        elif block == "energy":
            explanations.append(
                f"instabilité moyenne d'énergie temporelle (score={value:.2f})"
            )
        elif block == "spectral_shape":
            explanations.append(
                f"changement de forme spectrale globale (score={value:.2f})"
            )
        elif block == "harmonicity":
            explanations.append(
                f"modification de périodicité ou d'harmoniques (score={value:.2f})"
            )
        elif block == "cepstrum":
            explanations.append(
                f"modification cepstrale de la signature cyclique (score={value:.2f})"
            )
        elif block == "mfcc":
            explanations.append(
                f"différence de timbre global MFCC (score={value:.2f})"
            )

    if local_extreme_score >= 8.0:
        explanations.append(
            f"anomalie locale forte détectée : bloc '{local_extreme_block}' "
            f"avec un ratio local max de {local_extreme_score:.2f}"
        )

    if impulse_worst >= 6.0:
        explanations.append(
            f"pic impulsionnel local important : impulses_worst={impulse_worst:.2f}"
        )

    if cepstrum_worst >= 6.0:
        explanations.append(
            f"rupture locale de signature cyclique : cepstrum_worst={cepstrum_worst:.2f}"
        )

    if n_reference_files <= 1:
        explanations.append(
            "base de référence limitée : une seule référence normale, diagnostic à confirmer"
        )

    # ============================================================
    # Logique de verdict
    # ============================================================

    if confidence < 45:
        label = "⚫ Analyse peu fiable"
        level = "unreliable"
        message = (
            "La qualité ou le contexte audio ne permet pas un diagnostic fiable. "
            "Le résultat doit être confirmé avec un meilleur enregistrement."
        )

    # Cas critique V2.3 :
    # Knocking/local anomaly : même si la médiane est faible, une fenêtre violente compte.
    elif (
        worst_global >= 12.0
        and (
            impulse_worst >= 5.0
            or cepstrum_worst >= 6.0
            or energy_worst >= 8.0
            or spectral_worst >= 8.0
        )
    ):
        label = "🔴 Anomalie locale forte / claquement suspect"
        level = "localized_anomaly"
        message = (
            "Le signal contient au moins une fenêtre très anormale. "
            "Même si la moyenne globale reste faible, ce type de pic local peut correspondre "
            "à un claquement, un choc mécanique ou un événement bref au démarrage."
        )

    # Cas knocking plus étendu.
    elif impulses >= 4.5 and (global_score >= 2.5 or p90_ratio >= 2.5 or impulse_worst >= 7.0):
        label = "🔴 Knocking / claquement probable"
        level = "knocking_probable"
        message = (
            "Le signal présente des indices impulsionnels compatibles avec un claquement moteur."
        )

    # Signature différente : surtout bandes fréquentielles, sans preuve locale forte de claquement.
    elif (
        frequency_bands >= 7.0
        and impulse_worst < 5.0
        and cepstrum_worst < 6.0
        and harmonicity < 4.0
    ):
        label = "🟠 Signature différente à confirmer"
        level = "signature_shift"
        message = (
            "Le son est très différent de la référence, surtout dans la répartition fréquentielle. "
            "Cela peut venir d'une condition d'enregistrement différente, d'une phase moteur différente, "
            "ou d'une anomalie à confirmer."
        )

    # Anomalie générale forte.
    elif global_score >= 4.5 and (anomalous_ratio >= 0.40 or p90_ratio >= 4.0):
        label = "🔴 Anomalie probable"
        level = "anomaly_probable"
        message = (
            "Plusieurs familles de caractéristiques s'écartent fortement de la référence."
        )

    # Dérive modérée.
    elif global_score >= 2.4 or p90_ratio >= 2.6 or anomalous_ratio >= 0.25:
        label = "🟠 Dérive / anomalie légère possible"
        level = "drift_possible"
        message = (
            "Le signal s'écarte modérément de la référence. "
            "Il faut comparer avec d'autres références normales avant de conclure."
        )

    # Normal, mais avec garde-fou : on n'appelle pas normal si un pic local énorme existe.
    elif worst_global >= 8.0:
        label = "🟡 Événement local suspect"
        level = "local_event_suspect"
        message = (
            "Le signal est globalement proche de la référence, mais une fenêtre locale est suspecte. "
            "Il faut écouter cette zone ou afficher les fenêtres anormales."
        )

    else:
        label = "✅ Son normal"
        level = "normal"
        message = "Aucun écart significatif par rapport à la référence."

    if not explanations:
        explanations.append("Aucun bloc ne s'écarte fortement de la référence.")

    summary = (
        f"Score global : {global_score:.2f} — "
        f"Médiane : {median_ratio:.2f} — "
        f"P90 : {p90_ratio:.2f} — "
        f"Pire fenêtre : {worst_global:.2f} — "
        f"Fenêtres anormales : {anomalous_ratio * 100:.1f}% — "
        f"Confiance : {confidence:.1f}% — "
        f"Verdict : {label}"
    )

    return {
        "label": label,
        "level": level,
        "message": message,
        "summary": summary,
        "explanations": explanations,
        "diagnostic_values": {
            "global_score": global_score,
            "median_ratio": median_ratio,
            "p90_ratio": p90_ratio,
            "worst_window_ratio_raw": worst_global,
            "anomalous_ratio": anomalous_ratio,
            "confidence_score": confidence,
            "impulses": impulses,
            "frequency_bands": frequency_bands,
            "energy": energy,
            "spectral_shape": spectral_shape,
            "harmonicity": harmonicity,
            "cepstrum": cepstrum,
            "mfcc": mfcc,
            "local_extreme_score": local_extreme_score,
            "local_extreme_block": local_extreme_block,
            "impulse_worst": impulse_worst,
            "cepstrum_worst": cepstrum_worst,
            "energy_worst": energy_worst,
            "spectral_worst": spectral_worst,
            "band_worst": band_worst,
            "top_local_blocks": local["top_local_blocks"],
        },
    }
