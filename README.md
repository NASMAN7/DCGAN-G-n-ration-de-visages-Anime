# DCGAN — Génération de visages Anime

Projet de Deep Learning utilisant un **DCGAN (Deep Convolutional Generative Adversarial Network)** pour générer des visages de style anime.

## Présentation

Le modèle est entraîné sur l'**Anime Face Dataset** afin d'apprendre à générer de nouvelles images de visages anime à partir d'un vecteur de bruit aléatoire.

Le projet utilise plusieurs techniques de stabilisation de l'entraînement :

- **Label smoothing** avec des labels réels à `0.9`
- **Bruit d'instance décroissant** sur les images réelles
- **Learning rates différenciés** :
  - Discriminateur : `0.0001`
  - Générateur : `0.0002`
- **Spectral Normalization** dans le discriminateur
- Suivi de `D(x)` et `D(G(z))` : D(x) mesure la confiance du discriminateur lorsqu’il reçoit une image réelle, tandis que D(G(z)) mesure sa confiance lorsqu’il reçoit une image générée par le générateur.
- Sauvegarde de checkpoints pendant l'entraînement

## Architecture

### Générateur

Le générateur transforme un vecteur latent de taille `100` en une image RGB de `128×128`.

Architecture principale :

`1×1 → 4×4 → 8×8 → 16×16 → 32×32 → 64×64 → 128×128`

Chaque étape utilise principalement :

- `ConvTranspose2d`
- `BatchNorm2d`
- `ReLU`

La dernière couche utilise `Tanh()`.

### Discriminateur

Le discriminateur prend une image `128×128` et prédit si elle est réelle ou générée.

Il utilise :

- `Conv2d`
- `LeakyReLU`
- **Spectral Normalization**
- `Sigmoid` en sortie

## Dataset

Le notebook télécharge automatiquement l'**Anime Face Dataset** avec `kagglehub`.

Environ 60 000 images sont utilisées comme base d'entraînement.

Le notebook redimensionne les images en `128×128` et normalise les pixels dans l'intervalle `[-1, 1]` afin de correspondre à la sortie `Tanh()` du générateur.

## Entraînement

Paramètres principaux :

- Taille du bruit latent : `100`
- Batch size : `64`
- Nombre d'epochs : `150`
- Learning rate du discriminateur : `0.0001`
- Learning rate du générateur : `0.0002`
- Optimiseur : `Adam`
- `betas = (0.5, 0.999)`

Le modèle sauvegarde les poids finaux dans :

```text
generator.pth
discriminator.pth
```

Des checkpoints intermédiaires sont également créés tous les 10 epochs pendant l'entraînement.

## Évaluation

La qualité des images générées est évaluée avec le **FID (Fréchet Inception Distance)**.

Le notebook :

1. sélectionne 500 images réelles ;
2. génère 500 images avec le générateur ;
3. uniformise leur résolution ;
4. calcule le FID avec `pytorch-fid`.

Le notebook affiche également les courbes de :

- `Loss D`
- `Loss G`
- `D(x)`
- `D(G(z))`

## Installation

Cloner le projet :

```bash
git clone https://github.com/NASMAN7/DCGAN-Generation-de-visages-Anime.git
cd DCGAN-Generation-de-visages-Anime
```

Installer les dépendances (si vous choisissez d'exécuter le projet hors de Colab) :
```bash
pip install -r requirements.txt
```

## Utilisation

Le projet est conçu pour être exécuté dans **Google Colab** avec un GPU.

Ouvrir ensuite :

```text
DCGAN_Anime.ipynb
```

Puis exécuter les cellules dans l'ordre.

Le notebook utilise Google Drive pour sauvegarder les modèles et les résultats lorsqu'il est exécuté sur Colab.

## Résultats

Les résultats du projet comprennent :

- des images générées pendant l'entraînement ;
- les modèles `generator.pth` et `discriminator.pth` ;
- le score FID ;
- les courbes des pertes du générateur et du discriminateur ;
- le suivi de `D(x)` et `D(G(z))`.

## Technologies utilisées

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow
- KaggleHub
- PyTorch FID
- Google Colab

## Auteur

Réalisé par **NASMAN7** (Projet de Deep Learning — DCGAN)
