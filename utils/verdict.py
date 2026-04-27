# ─────────────────────────────────────────────
#  CONFIGURATION DES SEUILS
# ─────────────────────────────────────────────
#
#  Ces seuils définissent à partir de quel rapport d'anomalie
#  on change de verdict. Ils sont ici et nulle part ailleurs.
#  Pour les recalibrer sur de nouvelles données : modifier uniquement ce dict.
#
#  anomaly_ratio = distance_médiane_test / baseline_p90_référence
#
#    ≤ 1.25  → le test est dans la zone de normalité de la référence
#    ≤ 2.00  → légère dérive, à surveiller
#    ≤ 3.50  → anomalie probable, inspection recommandée
#    > 3.50  → son très éloigné ou analyse non fiable

VERDICT_THRESHOLDS = {
    "normal":   1.25,
    "drift":    2.00,
    "anomaly":  3.50,
}

VERDICT_LABELS = {
    "normal":      "✅ Son normal",
    "drift":       "⚠️  Légère dérive",
    "anomaly":     "🔴 Anomalie probable",
    "unreliable":  "❌ Analyse non fiable ou son très éloigné de la référence",
}


# ─────────────────────────────────────────────
#  LOGIQUE DE VERDICT
# ─────────────────────────────────────────────

def get_verdict(anomaly_ratio: float) -> str:
    """
    Retourne un verdict lisible en fonction du rapport d'anomalie.
    Les seuils sont lus depuis VERDICT_THRESHOLDS — jamais hardcodés ici.
    """
    if anomaly_ratio <= VERDICT_THRESHOLDS["normal"]:
        return VERDICT_LABELS["normal"]
    elif anomaly_ratio <= VERDICT_THRESHOLDS["drift"]:
        return VERDICT_LABELS["drift"]
    elif anomaly_ratio <= VERDICT_THRESHOLDS["anomaly"]:
        return VERDICT_LABELS["anomaly"]
    else:
        return VERDICT_LABELS["unreliable"]


def build_result_message(vehicle_id: str, anomaly_ratio: float, verdict: str) -> str:
    """
    Construit le message final affiché à l'utilisateur.
    """
    return (
        f"Véhicule [{vehicle_id}] — "
        f"Rapport d'anomalie : {anomaly_ratio:.2f} "
        f"(seuil normal ≤ {VERDICT_THRESHOLDS['normal']}) — "
        f"Verdict : {verdict}"
    )