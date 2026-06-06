# Pipeline d’analyse EchoMoteur

Ce document décrit le noyau de traitement du signal utilisé dans EchoMoteur.

---

## 1. Validation du fichier audio

Le programme commence par vérifier :

- existence du fichier ;
- extension ;
- durée ;
- lisibilité ;
- format ;
- présence éventuelle de compression MP3.

Un fichier MP3 n’est pas refusé, mais le programme ajoute un avertissement car la compression peut modifier les hautes fréquences.

---

## 2. Prétraitement

Le prétraitement est volontairement prudent.

Étapes :

```text
chargement mono
→ suppression de la composante continue
→ normalisation RMS
→ suppression des silences début/fin
→ débruitage optionnel désactivé par défaut
```

Le programme ne coupe plus le signal à 5 secondes.  
Cette correction est importante car une anomalie moteur peut apparaître après les premières secondes.

---

## 3. Découpage en fenêtres

Le signal est découpé en fenêtres temporelles :

```text
durée fenêtre : 1 seconde
chevauchement : 50 %
```

Chaque fenêtre est analysée séparément.

Cette approche est nécessaire parce qu’une anomalie peut être :

- permanente ;
- intermittente ;
- courte ;
- localisée au démarrage.

---

## 4. Extraction des caractéristiques

Les caractéristiques sont regroupées par familles.

---

### 4.1 Bloc `energy`

Ce bloc mesure l’énergie temporelle.

Exemples de mesures :

- RMS moyen ;
- RMS écart-type ;
- RMS P90 ;
- coefficient de variation RMS ;
- facteur de crête ;
- zero crossing rate ;
- kurtosis ;
- skewness.

Interprétation :

- énergie stable → moteur plus régulier ;
- variation forte → instabilité, micro qui bouge ou régime instable ;
- facteur de crête élevé → présence de pics.

---

### 4.2 Bloc `frequency_bands`

Ce bloc mesure la répartition de l’énergie dans plusieurs bandes.

Bandes utilisées :

```text
0–200 Hz
200–800 Hz
800–2000 Hz
2000–4000 Hz
4000–8000 Hz
```

Interprétation :

- hausse des basses fréquences → vibration lourde ou condition d’enregistrement différente ;
- hausse des hautes fréquences → frottement, fuite, bruit aigu ou compression ;
- déplacement global des bandes → signature différente.

Attention : ce bloc est utile, mais il peut produire des faux positifs si les conditions d’enregistrement changent.

---

### 4.3 Bloc `spectral_shape`

Ce bloc décrit la forme générale du spectre.

Mesures :

- centroïde spectral ;
- largeur spectrale ;
- rolloff ;
- flatness ;
- contraste spectral.

Interprétation :

- spectre plus aigu ;
- spectre plus plat ;
- changement de timbre ;
- modification de la distribution fréquentielle.

---

### 4.4 Bloc `harmonicity`

Ce bloc cherche la périodicité et les harmoniques.

Mesures :

- autocorrélation ;
- force du pic périodique ;
- estimation de fréquence fondamentale ;
- ratio harmonique/percussif ;
- nombre de pics spectraux.

Interprétation :

- moteur régulier → signature périodique plus claire ;
- perte de périodicité → instabilité possible ;
- harmoniques déplacées → changement de régime ou anomalie.

---

### 4.5 Bloc `impulses`

Ce bloc est essentiel pour détecter les claquements.

Méthode :

```text
signal
→ enveloppe d’amplitude
→ lissage
→ détection de pics
→ mesure du taux de pics et de leur proéminence
```

Mesures :

- nombre de pics par seconde ;
- proéminence moyenne ;
- proéminence maximale ;
- kurtosis de l’enveloppe ;
- facteur de crête ;
- flux spectral ;
- ratio d’énergie impulsionnelle.

Interprétation :

- pics courts et forts → claquement, choc, raté ou événement mécanique ;
- pics localisés → événement bref au démarrage ;
- pics répétés → knocking plus probable.

---

### 4.6 Bloc `cepstrum`

Le cepstre permet d’observer une périodicité cachée dans le spectre.

Principe simplifié :

```text
FFT
→ log du spectre
→ IFFT
```

Interprétation :

- changement de signature cyclique ;
- modification du motif harmonique ;
- indice complémentaire pour les sons périodiques de moteur.

---

### 4.7 Bloc `mfcc`

Les MFCC donnent une description globale du timbre.

Ils sont utiles, mais secondaires dans ce projet, car ils ne sont pas spécifiquement mécaniques.  
Ils peuvent réagir à une différence de micro ou de compression audio.

---

## 5. Comparaison avec la référence

Pour chaque bloc :

1. les features de référence sont normalisées ;
2. les features du test sont transformées avec la même moyenne et le même écart-type ;
3. chaque fenêtre test est comparée aux fenêtres de référence ;
4. la distance minimale est retenue ;
5. un ratio est calculé par rapport à la variabilité normale de la référence.

---

## 6. Baseline de référence

La baseline représente la variabilité normale entre fenêtres de référence.

Correction importante :

Les fenêtres voisines sont ignorées dans le calcul de baseline, car elles se chevauchent.  
Sans cette correction, la référence paraît artificiellement trop stable, ce qui augmente les faux positifs.

---

## 7. Score global et événement local

Le programme calcule :

- score global ;
- ratio médian ;
- ratio P90 ;
- pire fenêtre brute ;
- pourcentage de fenêtres anormales ;
- scores par bloc ;
- score de confiance.

La pire fenêtre brute est importante.  
Une anomalie courte ne doit pas être écrasée par la médiane.

Exemple :

```text
médiane faible
mais pire fenêtre très élevée
→ événement local suspect
```

---

## 8. Verdict

Le verdict est basé sur :

- écart global ;
- événement local ;
- blocs dominants ;
- confiance ;
- nombre de références ;
- contexte audio.

Le système distingue :

- normal ;
- signature différente ;
- événement local suspect ;
- knocking probable ;
- anomalie probable ;
- analyse peu fiable.

---

## 9. Interprétation correcte

Le programme ne dit pas :

```text
le moteur est mécaniquement cassé
```

Il dit plutôt :

```text
la signature acoustique s’écarte de la référence selon certaines familles de signal
```

Cette nuance est importante.

---

## 10. Point faible principal

Le plus grand point faible actuel reste la base de référence.

Avec une seule référence, le système peut confondre :

```text
anomalie réelle
```

et :

```text
condition d’enregistrement différente
```

La priorité scientifique est donc d’ajouter plusieurs références normales.
