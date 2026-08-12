# DG-PBs: A web server for diffusion-enhanced protein–DNA binding site prediction under class imbalance

DG-PBs is a web server for residue-level protein–DNA binding site prediction. The system integrates ESM-2 residue embeddings, cosine similarity Top-k residue graph construction, and a diffusion-enhanced graph neural network model to improve prediction under class imbalance.

---

## Overview

Protein-DNA binding site prediction is a highly imbalanced residue-level classification task. In most protein sequences, only a small fraction of residues are DNA-binding residues, while the majority are non-binding residues.

To better model this problem, DG-PBs combines sequence representation learning, generative positive-sample augmentation, and graph-based residue classification within a unified framework.

The main components include:

- residue-level protein representation using ESM-2;
- K-nearest-neighbor graph construction in the embedding space;
- conditional diffusion-based positive residue generation;
- hybrid GNN classification with GAT, GCN, and GraphSAGE;
- robust training strategies for evaluation across DNA and PDNA benchmark datasets.

---

## Workflow

The overall workflow is as follows:

1. Extract residue-level embeddings from protein sequences using ESM-2.
2. Construct residue graphs using KNN connectivity in the embedding space.
3. Train a conditional diffusion model on positive DNA-binding residues.
4. Generate additional positive residue samples to reduce class imbalance.
5. Merge generated positive samples back into the residue graph.
6. Reconstruct augmented residue graphs.
7. Train a hybrid GNN classifier for residue-level prediction.
8. Evaluate the model on independent DNA and PDNA test datasets.

---

## Method

### 1. Protein Representation

Each protein sequence is encoded using ESM-2:

```python
esm.pretrained.esm2_t33_650M_UR50D()
```

ESM-2 produces residue-level embeddings, which are used as node features for downstream graph learning.

---

### 2. Graph Construction

Each protein is converted into a residue graph:

- nodes represent amino acid residues;
- node features are ESM-2 residue embeddings;
- edges are constructed using K-nearest-neighbor connectivity in the embedding space.

This provides a lightweight way to introduce neighborhood relationships for residue-level classification without requiring experimentally determined 3D structures.

---

### 3. Diffusion-Based Positive Residue Augmentation

DNA-binding residues are usually much fewer than non-binding residues. To alleviate this imbalance, DG-PBs trains a conditional DDPM-style diffusion model on positive residue embeddings.

The diffusion model learns the feature distribution of positive DNA-binding residues and generates additional positive residue samples conditioned on protein-level context.

The generated positive samples are then merged back into the residue graph to form augmented training graphs.

---

### 4. Hybrid Graph Neural Network

The residue classifier is based on a hybrid GNN architecture that combines:

- GAT;
- GCN;
- GraphSAGE.

The outputs of different graph convolution branches are fused and passed through residual and fully connected layers to predict the DNA-binding probability of each residue.

---

### 5. Robust Training Pipeline

The robust training pipeline provides a more complete training and evaluation procedure, including:

- quality control for generated samples;
- diversity filtering;
- imbalance-aware optimization;
- stronger regularization;
- cross-validation-based training;
- evaluation on multiple external test files.

The recommended entry point is:

```bash
python robust_pipeline.py
```

---

## Repository Structure

```text
.
├── Raw_data/                         # Training and test datasets
├── README.md                         # Project description
├── balanced_training_config.py       # Balanced and robust training configuration
├── config.py                         # Baseline configuration
├── data_loader.py                    # Dataset loader with ESM-2 embeddings
├── data_loader_from_raw.py           # Alternative raw-data loader
├── ddpm_diffusion_model.py           # Conditional diffusion model
├── gnn_model.py                      # Baseline hybrid GNN model
├── improved_gnn_model.py             # Improved GNN model
├── main.py                           # Baseline training pipeline
├── robust_pipeline.py                # Robust training and evaluation pipeline
├── best_gnn_model.pt                 # Saved baseline checkpoint
└── best_improved_gnn_model.pt        # Saved improved checkpoint
```

---

## Data Format

All input files are stored in the `Raw_data/` directory as `.txt` files.

Each protein entry follows a three-line format:

```text
>protein_name
AMINO_ACID_SEQUENCE
BINARY_LABEL_STRING
```

Example:

```text
>4JBM0
MDPLVVTVLKAINPFECETQEGRQEIFHATVATETDFFFVKVLNAQFKDKFIPKRTI...
000000000000000000000000000000000000000000000000000001...
```

Label convention:

```text
1 = DNA-binding residue
0 = non-binding residue
```

---

## Included Datasets

The repository includes several DNA and PDNA training/test files, such as:

```text
DNA-573_Train.txt
DNA-646_Train.txt
DNA-129_Test.txt
DNA-181_Test.txt
DNA-46_Test.txt
```

Additional PDNA datasets can be placed in the `Raw_data/` directory using the same three-line format.

---

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- PyTorch Geometric
- fair-esm
- scikit-learn
- scipy
- numpy
- tqdm

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install fair-esm scikit-learn scipy numpy tqdm
```

---

## Quick Start

### Baseline Pipeline

Run the standard diffusion + GNN workflow:

```bash
python main.py
```

This pipeline performs:

- dataset loading;
- ESM-2 embedding extraction;
- diffusion model training on positive residues;
- graph augmentation;
- GNN training;
- evaluation.

---

### Robust Pipeline

Run the recommended robust training pipeline:

```bash
python robust_pipeline.py
```

This version includes:

- robust positive residue augmentation;
- sample quality filtering;
- diversity control;
- stronger regularization;
- cross-validation;
- multi-file evaluation.

---

### Ratio Test Mode

To test different augmentation ratios:

```bash
python robust_pipeline.py --ratio-test
```

---

## Outputs

### Baseline Outputs

Baseline results are saved to:

```text
Augmented_data/
```

Typical outputs include:

- diffusion model checkpoints;
- GNN checkpoints;
- training metadata;
- test result JSON files.

### Robust Outputs

Robust pipeline results are saved to:

```text
Augmented_data_balanced/
```

Typical outputs include:

- improved GNN checkpoints;
- robust evaluation summaries;
- full pipeline result files;
- ratio-test experiment outputs.

---

## Notes

### ESM Version

Although some early descriptions mentioned ESM-3, the current implementation uses ESM-2:

```python
esm.pretrained.esm2_t33_650M_UR50D()
```

Therefore, the current implementation is based on ESM-2.

---

### GPU Configuration

Some configuration files contain hard-coded CUDA settings. Before running the code on your own machine, please check and modify the GPU selection lines, such as:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
```

If you only have one GPU, you may change it to:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

Alternatively, you may remove this line and allow PyTorch to select the available device automatically.

---

## Highlights

- Diffusion-based augmentation for rare DNA-binding residues.
- Residue-level graph learning with ESM-2 embeddings.
- Hybrid GAT/GCN/GraphSAGE architecture.
- Robust training pipeline for improved generalization.
- Support for DNA and PDNA benchmark-style datasets.
- Suitable for residue-level protein-DNA binding site prediction.

---

## Citation

If you use this repository or build upon this work, please cite or acknowledge:

```text
DG-PBs: Diffusion-Augmented Graph Learning for Protein-DNA Binding Site Prediction
Hanqing Zhang
```

A formal citation will be added after publication.

---

## Contact

Hanqing Zhang  
Email: 3165619783@qq.com
