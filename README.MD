# EchoMoteur

EchoMoteur est une application Python de diagnostic acoustique moteur basée sur le **traitement du signal audio**.  
Le projet compare un son moteur à une ou plusieurs références normales afin de détecter des écarts de signature : claquements, instabilité énergétique, déplacement fréquentiel, perte de périodicité ou événement local suspect.

> Le projet n’utilise pas d’intelligence artificielle. Il repose sur des méthodes classiques de traitement du signal : fenêtrage, analyse fréquentielle, énergie par bandes, impulsions, autocorrélation, cepstre, MFCC et comparaison normalisée.

---

## Objectif du projet

L’objectif est de fournir un outil simple permettant de :

- charger un audio de moteur ;
- comparer cet audio à une référence saine ;
- extraire des caractéristiques acoustiques ;
- produire un score de similarité/anomalie ;
- afficher un verdict compréhensible ;
- sauvegarder un rapport JSON détaillé.

Le projet vise une utilisation pédagogique, expérimentale et démonstrative.

---

## Fonctionnalités principales

- Chargement audio mono avec fréquence cible.
- Prétraitement prudent du signal.
- Suppression des silences inutiles.
- Normalisation RMS.
- Analyse complète du signal, sans coupure fixe à 5 secondes.
- Découpage en fenêtres temporelles.
- Extraction de caractéristiques par familles physiques.
- Comparaison à une signature de référence.
- Détection d’anomalie globale et locale.
- Verdict explicatif.
- Rapport JSON.
- Interface desktop simple avec Tkinter.

---

## Structure du projet

```text
EchoMoteur/
│
├── app/
│   ├── __init__.py
│   └── desktop_app.py
│
├── data/
│   ├── reference/
│   ├── test/
│   └── cache/
│
├── docs/
│   ├── ANALYSIS_PIPELINE.md
│   └── TEST_PROTOCOL.md
│
├── results/
│
├── utils/
│   ├── audio_validator.py
│   ├── comparator.py
│   ├── noise_reduction.py
│   └── verdict.py
│
├── main.py
├── run_desktop.py
├── requirements.txt
└── README.md
```

---

## Installation

Créer et activer un environnement virtuel :

```bash
python -m venv .venv
```

Sous Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## Utilisation en mode console

Depuis la racine du projet :

```bash
python main.py
```

Le programme demande :

```text
Entrez l'identifiant du véhicule
Entrez le chemin du fichier audio à analyser
```

Exemple :

```text
V_MERCEDES
data/test/Mercedes-Benz m271 engine knocking noise on startup.wav
```

---

## Utilisation en mode desktop

Depuis la racine du projet :

```bash
python run_desktop.py
```

L’interface permet de :

- choisir le véhicule ;
- choisir le fichier audio ;
- lancer l’analyse ;
- afficher le verdict ;
- consulter les scores ;
- consulter le JSON complet.

---

## Principe d’analyse

Le signal audio est traité selon les étapes suivantes :

```text
audio
→ validation fichier
→ prétraitement
→ découpage en fenêtres
→ extraction de caractéristiques
→ comparaison à la référence
→ score global + score local
→ verdict
→ rapport JSON
```

Les caractéristiques extraites sont organisées en blocs :

| Bloc | Rôle |
|---|---|
| `energy` | stabilité énergétique et dynamique temporelle |
| `frequency_bands` | répartition de l’énergie dans les bandes fréquentielles |
| `spectral_shape` | forme globale du spectre |
| `harmonicity` | périodicité, harmoniques, autocorrélation |
| `impulses` | claquements, pics, événements courts |
| `cepstrum` | signature cyclique cachée dans le spectre |
| `mfcc` | timbre global secondaire |

---

## Types de verdict

EchoMoteur peut produire plusieurs verdicts :

| Verdict | Signification |
|---|---|
| ✅ Son normal | pas d’écart significatif |
| 🟡 Événement local suspect | une fenêtre locale semble anormale |
| 🟠 Signature différente à confirmer | signature audio différente, mais pas forcément panne |
| 🟠 Dérive / anomalie légère possible | écart modéré |
| 🔴 Knocking / claquement probable | indices impulsionnels compatibles avec un claquement |
| 🔴 Anomalie probable | écart fort sur plusieurs familles |
| ⚫ Analyse peu fiable | qualité ou contexte audio insuffisant |

---

## Limites actuelles

Le projet reste expérimental.

Les principales limites sont :

- base de référence encore faible ;
- forte sensibilité aux conditions d’enregistrement ;
- comparaison dépendante du micro, de la distance et de la compression audio ;
- pas encore de calibration statistique sur une vraie base de données ;
- pas de localisation temporelle visuelle des fenêtres anormales dans l’interface ;
- pas de diagnostic mécanique médical/garage garanti.

Un verdict ne doit pas être interprété comme une décision mécanique définitive. Il indique une **signature acoustique suspecte** à confirmer.

---

## Bonnes pratiques pour les tests

Pour obtenir des résultats plus fiables, il faut :

- utiliser des fichiers WAV si possible ;
- enregistrer dans des conditions similaires ;
- garder la même distance micro-moteur ;
- éviter musique, parole et bruit extérieur ;
- utiliser plusieurs références normales ;
- comparer froid/froid et chaud/chaud séparément ;
- ne pas mélanger ralenti et accélération.

---

## Améliorations futures

- Configuration externe des véhicules avec `config/vehicles.json`.
- Ajout de plusieurs références par véhicule.
- Affichage graphique de la forme d’onde.
- Affichage des fenêtres anormales.
- Export PDF du rapport.
- Historique des analyses.
- Packaging `.exe` avec PyInstaller.
- Calibration des seuils avec une base de données plus riche.

---

## Exemple de description courte

EchoMoteur est un outil Python de diagnostic acoustique moteur qui compare un enregistrement audio à une signature de référence à l’aide de méthodes de traitement du signal. Il extrait des caractéristiques temporelles, fréquentielles, impulsionnelles et périodiques afin de détecter des écarts compatibles avec une anomalie ou un claquement moteur.
