# Diffusion-Augmented Graph Learning for Protein–DNA Binding Site Prediction

This repository presents a diffusion-augmented graph learning framework for **residue-level protein–DNA binding site prediction**.  
The method integrates **protein language model embeddings**, **conditional diffusion-based positive sample generation**, and **graph neural networks** to address the severe class imbalance that commonly appears in binding residue identification.

The codebase includes both a **baseline training pipeline** and a more advanced **robust pipeline** designed for stronger generalization across multiple DNA/PDNA benchmark-style test sets.

---

## Overview

Protein–DNA binding site prediction is a highly imbalanced node classification problem: only a small fraction of residues are binding residues, while the vast majority are non-binding. To better model this setting, this project combines sequence representation learning, generative augmentation, and graph-based classification into a unified framework.

### Core idea

The overall workflow is:

1. **Extract residue-level embeddings** from protein sequences using **ESM-2**
2. **Construct residue graphs** with KNN-based connectivity
3. **Train a conditional diffusion model** on positive binding-site residues
4. **Generate additional positive samples** to alleviate imbalance
5. **Train a hybrid GNN classifier** for residue-level prediction
6. **Evaluate on held-out DNA and PDNA test sets**

---

## Method

### 1. Protein representation
Each protein sequence is encoded using **ESM-2 (`esm2_t33_650M_UR50D`)**, producing residue-level embeddings that serve as node features for downstream graph learning.

### 2. Graph construction
Each protein is converted into a graph in which:
- **nodes** represent residues
- **edges** are constructed through **K-nearest neighbors (KNN)** in embedding space

This provides a lightweight way to inject neighborhood structure into residue-level classification.

### 3. Diffusion-based positive augmentation
To mitigate the scarcity of positive binding residues, the repository trains a **conditional DDPM-style diffusion model** on positive residue embeddings.  
The generator uses protein-level context to synthesize additional positive samples, which are then merged back into the residue graph.

### 4. Graph neural prediction
The classifier is built on a **hybrid GNN architecture** combining:
- **GAT**
- **GCN**
- **GraphSAGE**

The repository also includes an improved variant with stronger regularization and more robust training behavior.

### 5. Robust training strategy
The advanced pipeline further introduces:
- **quality control for generated samples**
- **diversity filtering**
- **imbalance-aware optimization**
- **domain-adaptive regularization**
- **cross-validation-based training**
- **evaluation across multiple external test files**

## Repository Structure

```text
.
├── main.py                          # Baseline training pipeline
├── robust_pipeline.py               # Robust training / evaluation pipeline
├── config.py                        # Baseline configuration
├── balanced_training_config.py      # Balanced / robust training configuration
├── data_loader.py                   # Main dataset loader with ESM-2 embeddings
├── data_loader_from_raw.py          # Alternative raw-data loader
├── ddpm_diffusion_model.py          # Conditional diffusion model
├── gnn_model.py                     # Baseline hybrid GNN
├── improved_gnn_model.py            # Improved GNN for robust training
├── modules/
│   └── edge_predictor.py            # Auxiliary edge prediction utilities
├── Raw_data/                        # Training / test txt files
├── Augmented_data/                  # Baseline outputs and checkpoints
├── Augmented_data_balanced/         # Robust / balanced outputs and checkpoints
├── best_gnn_model.pt                # Saved baseline checkpoint
├── best_improved_gnn_model.pt       # Saved improved checkpoint
└── README.md
```

## Data Format

All input files are stored in Raw_data/ as .txt files.

Each protein entry uses a three-line format:

>protein_name
SEQUENCE
LABELS

Example:

>4JBM0
MSEQUENCE...
001000000100...
Label convention
1 = DNA-binding residue
0 = non-binding residue
Files included in the repository

The current repository already contains multiple train/test files, including:

DNA-573_Train.txt
DNA-646_Train.txt
DNA-129_Test.txt
DNA-181_Test.txt
DNA-46_Test.txt
PDNA-543-train.txt
PDNA-316-test.txt
PDNA-335-test.txt
PDNA-41-test.txt
PDNA-52-test.txt
Installation
Requirements
Python >= 3.8
PyTorch >= 2.0
PyTorch Geometric
fair-esm
scikit-learn
scipy
numpy
tqdm
Install dependencies
pip install torch torchvision torchaudio
pip install torch-geometric
pip install fair-esm scikit-learn scipy numpy tqdm
Quick Start
Baseline pipeline

Run the standard diffusion + GNN workflow:

python main.py

This pipeline performs:

dataset loading
ESM-2 embedding extraction
diffusion model training on positive residues
graph augmentation
GNN training
held-out evaluation
Robust pipeline

Run the more advanced and recommended pipeline:

python robust_pipeline.py

This version adds:

robust positive augmentation
sample quality filtering
diversity control
stronger regularization
cross-validation
broader multi-file evaluation
Ratio-test mode

To test different augmentation ratios:

python robust_pipeline.py --ratio-test
Outputs
Baseline outputs

Results from the baseline pipeline are saved to:

Augmented_data/

Typical contents include:

diffusion checkpoints
GNN checkpoints
training metadata
test result JSON files
Robust outputs

Results from the robust pipeline are saved to:

Augmented_data_balanced/

Typical contents include:

improved GNN checkpoints
robust evaluation summaries
complete pipeline result files
ratio-test experiment outputs
Notes
ESM model

Although the original README mentioned ESM-3, the actual code in this repository uses:

esm.pretrained.esm2_t33_650M_UR50D()

So the current implementation is based on ESM-2.

GPU configuration

config.py and balanced_training_config.py currently contain hard-coded CUDA settings.
Before running on your own machine, you may need to modify the GPU selection lines, for example:

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
Practical recommendation

If you are using this repository for experiments or for a project page, the best entry point is usually:

python robust_pipeline.py

because it reflects the more complete training and evaluation workflow contained in the repository.

Highlights
Diffusion-based augmentation for rare positive binding residues
Residue-level graph learning with ESM-2 embeddings
Hybrid GAT/GCN/GraphSAGE architecture
Robust training pipeline for improved generalization
Support for both DNA and PDNA benchmark-style datasets
Included checkpoints and experiment outputs for reproducibility
Citation

If you use this repository or build on its ideas, please cite or acknowledge:

Diffusion-Augmented Graph Learning for Protein–DNA Binding Site Prediction
Hanqing Zhang

You can further adapt the citation format to your paper, thesis, or report style.

Contact

Hanqing Zhang
Email: 3165619783@qq.com
