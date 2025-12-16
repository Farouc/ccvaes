# CCVAE-Probing-Benchmark: Conditional VAE for Disentangled Representation and Downstream Tasks

This project is part of the MVA course on **Introduction to Graphical Models and Probabilistic Generative Models**. It implements and benchmarks a **Conditional Contrastive Variational Autoencoder (CCVAE)** framework designed to learn disentangled representations for image data, focusing on applying the learned features to downstream tasks: One-label Classification, Multi-label Classification and Regression.

---

## 🚀 Getting Started

1.  **Clone the repository:**
    
```bash
git clone https://github.com/Farouc/ccvaes.git
cd CCVAES
```

2.  **Set up the environment:**

```bash
conda create -n ccvae_env python=3.11.9
conda activate ccvae_env
pip install -r requirements.txt
```

3.  **Data Setup:**
    
    * **Classification:** Download the [CartoonSet](https://google.github.io/cartoonset/)  dataset  and place the images in `data/cartoonset10k/cartoonset10k`.
    
    * **Regression:** Download the [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new/data) dataset and place the images in `data/UTKFace`.

---

## 📂 Project Structure

The project is organized into distinct directories for each task, ensuring clear separation between classification and regression experiments, reflecting the CCVAE's application across different supervised settings.

CCVAES/
├── ccvae_env/                      # Python environment / dependencies
│
├── classification_mono/            # Single-label classification & probing
│   ├── dataset.py                  # Dataset loaders
│   ├── model.py                    # CNN / CCVAE models
│   └── train.py                    # Training & evaluation script
│
├── classification_multi/           # Multi-label classification benchmarks
│   ├── benchmark_multilabel.py     # Classical models & CNN benchmark
│   ├── dataset.py                  # Multi-label dataset loader
│   ├── loss.py                     # Multi-label / contrastive losses
│   └── model.py                    # Model definition
│
├── regression/                     # Regression benchmarks (age prediction)
│   ├── benchmark_regression.py     # Classical regressors & CNN benchmark
│   ├── dataset.py                  # UTKFace dataset loader (normalized age)
│   ├── loss.py                     # CCVAE loss adapted for regression
│   └── model.py                    # Regression models
│
├── data/                           # Raw datasets (not versioned)
│   ├── UTKFace/
│   └── cartoonset10k/
│
└── notebooks/                      # Interactive analysis & demos
    ├── demo_ccvae.ipynb
    ├── demo_multilabel.ipynb
    └── demo_regression.ipynb



---

## 🔬 Experiments and Benchmarks

To explore the different models, tasks, probing experiments we realized, please look at the notebooks in the folder `notebooks`.

## 👥 Project Team

This project was developed for the MVA course on Introduction to Graphical Models and Probabilistic Generative Models by:

* **Farouk Yartaoui**
* **Elmehdi Nezahi**
* **Rida Assalouh**