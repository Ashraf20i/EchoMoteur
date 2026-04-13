from utils.audio_loader import validate_audio_path, get_file_info, load_audio, get_signal_info

def main():
    print("=" * 50)
    print("Bienvenue dans EchoMoteur")
    print("=" * 50)

    vehicle_id = input("Entrez l'identifiant du véhicule : ").strip()
    audio_path = input("Entrez le chemin du fichier audio à analyser : ").strip()

    try:
        print("\n[1/3] Vérification du fichier audio...")
        validate_audio_path(audio_path)

        print("[2/3] Lecture des informations du fichier...")
        file_info = get_file_info(audio_path)

        print("[3/3] Chargement du signal audio...")
        signal, sr = load_audio(audio_path, target_sr=22050, mono=True)
        signal_info = get_signal_info(signal, sr)

        print("\n" + "=" * 50)
        print("Informations du fichier")
        print("=" * 50)
        print(f"Véhicule                : {vehicle_id}")
        print(f"Fréquence originale     : {file_info['samplerate']} Hz")
        print(f"Canaux                  : {file_info['channels']}")
        print(f"Durée originale         : {file_info['duration']} s")
        print(f"Nombre de frames        : {file_info['frames']}")
        print(f"Format                  : {file_info['format']}")
        print(f"Sous-type               : {file_info['subtype']}")

        print("\n" + "=" * 50)
        print("Informations du signal chargé")
        print("=" * 50)
        print(f"Fréquence chargée       : {signal_info['loaded_samplerate']} Hz")
        print(f"Durée chargée           : {signal_info['loaded_duration']} s")
        print(f"Nombre d'échantillons   : {signal_info['nb_samples']}")
        print(f"Amplitude min           : {signal_info['min_amplitude']:.5f}")
        print(f"Amplitude max           : {signal_info['max_amplitude']:.5f}")

        print("\nChargement terminé avec succès.")

    except Exception as e:
        print("\nErreur :", e)

if __name__ == "__main__":
    main()