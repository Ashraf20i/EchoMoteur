def get_verdict(anomaly_ratio):
    """
    Verdict basé sur le rapport à la variabilité normale de la référence.
    """
    if anomaly_ratio <= 1.25:
        return "son normal"
    elif anomaly_ratio <= 2.0:
        return "légère dérive"
    elif anomaly_ratio <= 3.5:
        return "anomalie probable"
    else:
        return "analyse non fiable ou son très éloigné de la référence"


def build_result_message(vehicle_id, anomaly_ratio, verdict):
    return (
        f"Pour ce véhicule connu, le test présente un rapport d'anomalie de "
        f"{anomaly_ratio:.2f} par rapport à la variabilité normale de la référence. "
        f"Verdict : {verdict}."
    )