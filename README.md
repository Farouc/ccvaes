# CCVAE : Characteristic Capturing Variational Autoencoder

[cite\_start]Ce projet est une implémentation **PyTorch** du modèle **CCVAE**, basée sur le papier de recherche *"Capturing Label Characteristics in VAEs"* (Joy et al., ICLR 2021)[cite: 1, 11].

L'objectif est de structurer l'espace latent d'un VAE pour séparer le **style** (fond, forme globale) des **caractéristiques spécifiques** (attributs étiquetés comme la couleur de cheveux, les lunettes, etc.), permettant ainsi des manipulations précises de l'image.

## 📂 Structure du Projet

  * [cite\_start]`model.py` : Architecture du CCVAE (Encodeur, Décodeur, Classifieur Latent)[cite: 11, 14].
  * [cite\_start]`loss.py` : Fonction de coût spécifique (Reconstruction + KL Divergence + Perte de supervision)[cite: 153, 154].
  * `dataset.py` : Dataloader personnalisé pour le **Google Cartoon Set**. Gère la normalisation des labels.
  * `train_cartoon.py` : Script d'entraînement principal.
  * `visualize.py` : Script de génération de "Latent Traversals" (modification progressive d'un attribut).

## ⚙️ Installation

1.  **Pré-requis** : Python 3.8+, PyTorch (avec support CUDA recommandé).
2.  **Installation des dépendances** :
    ```bash
    pip install torch torchvision pandas tqdm matplotlib pillow
    ```

## 🎨 Dataset : Google Cartoon Set

Nous utilisons une version réduite (10k images) du Google Cartoon Set pour démontrer la capacité du modèle à capturer des caractéristiques visuelles variées.

1.  Téléchargez le dataset (version 10k).
2.  Placez le dossier décompressé dans `cartoonset10k/`.
3.  L'arborescence doit ressembler à : `./cartoonset10k/cartoonset10k/*.png`

## 🚀 Utilisation

### 1\. Entraînement

Pour lancer l'entraînement du modèle :

```bash
python train_cartoon.py
```

  * Le script détecte automatiquement le nombre d'attributs (généralement 18).
  * Les poids du modèle sont sauvegardés dans `ccvae_cartoon.pth`.

### 2\. Visualisation (Latent Traversal)

Pour générer des images montrant l'échange de caractéristiques :

```bash
python visualize.py
```

  * Modifiez la variable `ATTRIBUTE_INDEX_TO_VARY` dans le script pour choisir quel attribut modifier (ex: couleur de peau, lunettes).

## 💡 Choix Techniques & Implémentation

### Régression vs Classification

Contrairement à l'approche classique de classification (Cross-Entropy) pour les attributs catégoriels, nous avons opté pour une approche de **Régression (MSE)** sur les étiquettes normalisées entre `[0, 1]`.

**Pourquoi ?**

  * **Continuité :** Le CCVAE vise à effectuer des transitions douces ("smooth traversals") dans l'espace latent. La régression force le modèle à apprendre une relation continue entre les variantes d'un attribut (ex: morphing progressif d'une coupe de cheveux à une autre) plutôt que des sauts discrets.
  * **Efficacité :** Cela permet de condenser l'information de chaque attribut (qui peut avoir \~10 variantes) en **un seul neurone latent** ($z_c^i$), rendant l'espace latent plus compact et interprétable.

### Architecture Latente

[cite\_start]L'espace latent $z$ est divisé en deux parties[cite: 112]:

  * **$z_c$ (Characteristic Latents)** : Dimensions supervisées, chacune dédiée à un attribut spécifique du dataset.
  * **$z_{\setminus c}$ (Contextual Latents)** : Dimensions non-supervisées capturant le reste de l'information (style, fond).

## 👥 Auteurs

  * [Ton Prénom] [Ton Nom]
  * [Prénom Partenaire] [Nom Partenaire]

-----

*Projet réalisé dans le cadre du cours [Nom du Cours], Décembre 2025.*