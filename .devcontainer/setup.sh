#!/usr/bin/env bash
set -e

echo "=== [EchoMoteur] Installation des dependances systeme ==="
sudo apt-get update
# libsndfile1 : requis par soundfile/librosa pour lire/ecrire le WAV
# ffmpeg      : requis par audioread/librosa pour mp3, m4a, ogg, etc.
# python3-tk  : requis par l'interface desktop (Tkinter) - PAS installe par defaut
sudo apt-get install -y libsndfile1 ffmpeg python3-tk

echo "=== [EchoMoteur] Normalisation de requirements.txt (UTF-16 -> UTF-8) ==="
# Le requirements.txt du repo est encode en UTF-16 + CRLF (genere sous Windows).
# pip sous Linux echoue a le lire. On le convertit en UTF-8 sans CR.
if file requirements.txt | grep -qi "UTF-16"; then
  iconv -f UTF-16 -t UTF-8 requirements.txt | tr -d '\r' > requirements.utf8.txt
  mv requirements.utf8.txt requirements.txt
  echo "requirements.txt converti en UTF-8."
else
  echo "requirements.txt deja en UTF-8, rien a faire."
fi

echo "=== [EchoMoteur] Installation des dependances Python ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== [EchoMoteur] Recreation de l'arborescence data/ (ignoree par git) ==="
# data/ est dans .gitignore : les dossiers n'existent pas apres clone.
mkdir -p data/reference data/test data/cache results

echo ""
echo "=== Setup termine. ==="
echo "MODE CONSOLE : python main.py"
echo "MODE DESKTOP : DISPLAY=:1 python run_desktop.py   (puis ouvrir le port 6080)"
echo ""
echo "RAPPEL : place tes fichiers audio dans data/reference/ et data/test/"
echo "         (ils ne sont PAS dans le repo car data/ est gitignore)."
