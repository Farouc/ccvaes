# CCVAE : Characteristic Capturing Variational Autoencoder

Ce projet est une implémentation **PyTorch** du modèle **CCVAE**, basée sur le papier de recherche *"Capturing Label Characteristics in VAEs"* (Joy et al., ICLR 2021)[cite: 1, 11].

L'objectif est de structurer l'espace latent d'un VAE pour séparer le **style** (fond, forme globale) des **caractéristiques spécifiques** (attributs étiquetés comme la couleur de cheveux, les lunettes, etc.), permettant ainsi des manipulations précises de l'image.

## 📂 Structure du Projet

  * `model.py` : Architecture du CCVAE (Encodeur, Décodeur, Classifieur Latent)[cite: 11, 14].
  * `loss.py` : Fonction de coût spécifique (Reconstruction + KL Divergence + Perte de supervision)[cite: 153, 154].
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

1.  Téléchargez le dataset (version 10k): https://google.github.io/cartoonset/download.html
2.  Placez le dossier décompressé dans `cartoonset10k/`.
3.  L'arborescence doit ressembler à : `./cartoonset10k/cartoonset10k/*.png`

## 🚀 Utilisation

### 1\. Entraînement

Pour lancer l'entraînement du modèle :

```bash
python train.py
```

  * Les poids du modèle sont sauvegardés dans `ccvae_haircolor.pth`.
  * Des reconstructions de test sont sauvegardées à chaque époque dans le dossier results/.

### 2\.Inférence & Démo (CLI)

Pour tester le modèle sur des images spécifiques (Classification, Génération, Style Swapping) :

```Bash

python inference.py
```

Note : Vous pouvez modifier les chemins d'images directement dans le main du script.

### 2\.Analyse approfondie (Notebook)
```bash
jupter notebook demo_ccvae.ipynb
```


## 💡 Choix Techniques & Implémentation

Contrairement à certaines approches qui traitent les attributs comme des valeurs continues, nous avons opté pour une approche de **Classification Supervisée (Cross-Entropy)**.

Pourquoi ?
- **Nature des Données :** La couleur des cheveux est une donnée catégorielle distincte (10 classes). 
- L'utilisation de vecteurs One-Hot combinée à une CrossEntropyLoss permet une séparation plus nette des clusters dans l'espace latent qu'une régression MSE.
- **Auxiliary Loss ($\gamma$) :** Pour forcer le modèle à structurer l'espace latent $z_c$ dès le début de l'entraînement (et éviter le "posterior collapse"), nous avons ajouté une perte de classification auxiliaire avec un poids $\gamma = 20$. Cela garantit que $z_c$ capture explicitement l'information de classe.

### Reconstruction : BCE vs MSE

Nous utilisons la Binary Cross Entropy (BCE) plutôt que la Mean Squared Error (MSE) pour la reconstruction des images.

**Pourquoi ?** Les images de type "Cartoon" possèdent des aplats de couleurs et des contours nets. La MSE tend à produire des résultats flous (moyenne des couleurs, grisâtre). La BCE pénalise fortement les pixels "hésitants", produisant des images aux traits nets et au fond blanc pur.

Architecture Latente (Disentanglement)L'espace latent total $z$ est scindé en deux sous-espaces distincts :$z_c$ (Characteristic Latents) : Dimensions supervisées (dim=16). Elles sont forcées d'encoder la Couleur des Cheveux via le Conditional Prior $p(z_c|y)$.$z_{\neg c}$ (Contextual Latents) : Dimensions non-supervisées (dim=64). Elles capturent tout le reste de l'information (forme du visage, lunettes, style) et suivent un prior gaussien standard $\mathcal{N}(0, I)$.C'est cette séparation qui permet le Style Swapping : on peut conserver le $z_{\neg c}$ d'une image A (son visage) et lui injecter le $z_c$ d'une image B (sa couleur).

## 👥 Auteurs

  * Farouk YARTAOU
  * Rida ASSALOUH
  * El Mehdi NEZAHI

-----

*Projet réalisé dans le cadre du cours Introduction to Probabilistic Graphical Models and Deep Generative Models , Décembre 2025.*