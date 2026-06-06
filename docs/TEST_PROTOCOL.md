# Protocole de test EchoMoteur

Ce document décrit comment tester EchoMoteur proprement.

---

## 1. Objectif

Le but n’est pas seulement de voir si le programme affiche un verdict.  
Le but est de vérifier si le verdict est cohérent avec le signal.

---

## 2. Jeux de tests minimaux

Pour chaque véhicule, il faut idéalement :

```text
1 référence normale
1 test normal différent
1 test knocking / claquement
1 test hors contexte
```

Avec un seul fichier de référence, les conclusions restent limitées.

---

## 3. Test A — référence contre elle-même

Entrée :

```text
V_MERCEDES
data/reference/Mercedes Benz E200 M271 Cold Start.wav
```

Résultat attendu :

```text
✅ Son normal
score proche de 0
fenêtres anormales proches de 0 %
```

Interprétation :

Ce test vérifie que le pipeline fonctionne.  
Il ne valide pas la capacité de diagnostic, car le fichier est comparé à lui-même.

---

## 4. Test B — son normal différent

Entrée :

```text
autre son normal du même moteur
```

Résultat souhaité :

```text
✅ Son normal
```

ou :

```text
🟠 Signature différente à confirmer
```

Résultat suspect :

```text
🔴 Anomalie probable
```

si le son est censé être normal.

Interprétation :

Si le programme donne une anomalie forte sur un son normal, la référence est trop pauvre ou les seuils sont trop sévères.

---

## 5. Test C — knocking / claquement

Entrée :

```text
son moteur avec claquement
```

Résultat attendu :

```text
🔴 Knocking / claquement probable
```

ou :

```text
🔴 Anomalie locale forte / claquement suspect
```

Indices attendus :

```text
impulses élevé
pire fenêtre brute élevée
cepstrum ou energy parfois élevés
```

---

## 6. Test D — hors contexte

Entrée :

```text
voix, musique, bruit non moteur
```

Résultat attendu :

```text
⚫ Analyse peu fiable
```

ou contexte audio faible.

---

## 7. Grille de lecture des scores

### Score global

Indique l’écart moyen pondéré.

- faible : proche de la référence ;
- moyen : dérive ou signature différente ;
- fort : anomalie probable.

### Ratio médian

Mesure le comportement typique.

Attention : il peut cacher une anomalie courte.

### Ratio P90

Mesure les fenêtres hautes mais pas forcément extrêmes.

### Pire fenêtre brute

Très important pour les claquements.

Si la pire fenêtre est très élevée, il ne faut pas conclure trop vite à normal, même si la médiane est faible.

### Fenêtres anormales

Indique si l’anomalie est locale ou généralisée.

- faible pourcentage : événement local ;
- fort pourcentage : anomalie globale ou signature très différente.

---

## 8. Règles d’interprétation

### Cas 1 : impulses très élevé

Interprétation probable :

```text
claquement, choc, raté, knocking
```

### Cas 2 : frequency_bands très élevé seul

Interprétation prudente :

```text
condition d’enregistrement différente
ou phase moteur différente
ou anomalie à confirmer
```

### Cas 3 : energy élevé seul

Interprétation possible :

```text
moteur instable
ou micro qui bouge
ou volume variable
```

### Cas 4 : tout est élevé

Interprétation :

```text
anomalie probable ou audio très différent
```

---

## 9. Conclusion de test

Pour chaque audio testé, noter :

```text
nom fichier
rôle supposé
verdict attendu
verdict obtenu
score global
pire fenêtre
bloc dominant
commentaire
```

Exemple :

| Fichier | Rôle | Verdict attendu | Verdict obtenu | Commentaire |
|---|---|---|---|---|
| reference.wav | normal référence | normal | normal | test pipeline |
| cold_start_2.mp3 | normal différent | normal/signature différente | signature différente | cohérent |
| knocking.wav | claquement | knocking probable | knocking probable | cohérent |
