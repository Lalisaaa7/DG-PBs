# DG-PBs: Diffusion-Augmented Graph Learning for Protein-DNA Binding Site Prediction

DG-PBs is a diffusion-augmented graph learning framework for residue-level protein-DNA binding site prediction.

The framework integrates protein language model embeddings, conditional diffusion-based positive residue augmentation, and hybrid graph neural networks to address the severe class imbalance problem commonly observed in DNA-binding residue prediction.

---

## Overview

Protein-DNA binding site prediction is a highly imbalanced residue-level classification task. In most protein sequences, only a small fraction of residues are DNA-binding residues, while the majority are non-binding residues.

To better model this setting, DG-PBs combines:

- residue-level protein representation learning using ESM-2;
- K-nearest-neighbor graph construction in the embedding space;
- conditional diffusion-based positive residue generation;
- hybrid GNN classification using GAT, GCN, and GraphSAGE;
- robust training strategies for improved generalization across DNA and PDNA benchmark datasets.

---

## Workflow

The overall workflow is as follows:

1. Extract residue-level embeddings from protein sequences using ESM-2.
2. Construct residue graphs using KNN connectivity in the embedding space.
3. Train a conditional diffusion model on positive DNA-binding residues.
4. Generate additional positive residue samples to reduce class imbalance.
5. Reconstruct augmented graphs with generated positive nodes.
6. Train a hybrid GNN classifier for residue-level prediction.
7. Evaluate the model on independent DNA and PDNA test datasets.

---

## Method

### 1. Protein Representation

Each protein sequence is encoded using ESM-2:

```python
esm.pretrained.esm2_t33_650M_UR50D()
Email: 3165619783@qq.com
The model produces residue-level embeddings, which are used as node features for downstream graph learning.

2. Graph Construction

Each protein is converted into a residue graph:

nodes represent amino acid residues;
node features are ESM-2 residue embeddings;
edges are constructed using K-nearest-neighbor connectivity in the embedding space.

This provides a lightweight way to introduce neighborhood relationships for residue-level classification.

3. Diffusion-Based Positive Residue Augmentation

To alleviate the scarcity of DNA-binding residues, DG-PBs trains a conditional DDPM-style diffusion model on positive residue embeddings.

The diffusion model learns the feature distribution of positive DNA-binding residues and generates additional positive residue samples conditioned on protein-level context.

The generated positive samples are then merged back into the residue graph to form an augmented training graph.

4. Hybrid Graph Neural Network

The residue classifier is based on a hybrid GNN architecture that combines:

GAT;
GCN;
GraphSAGE.

The outputs of different graph convolution branches are fused and passed through residual and fully connected layers for residue-level binary classification.

5. Robust Training Pipeline

The robust pipeline includes:

quality control for generated samples;
diversity filtering;
imbalance-aware optimization;
stronger regularization;
cross-validation-based training;
evaluation on multiple external test files.

The recommended entry point is:

python robust_pipeline.py
Repository Structure
.
├── Raw_data/                         # Training and test datasets
├── README.md                         # Project description
├── balanced_training_config.py       # Robust training configuration
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
Data Format

All input files are stored in Raw_data/ as .txt files.

Each protein entry follows a three-line format:

>protein_name
AMINO_ACID_SEQUENCE
BINARY_LABEL_STRING

Example:

>4JBM0
MDPLVVTVLKAINPFECETQEGRQEIFHATVATETDFFFVKVLNAQFKDKFIPKRTI...
000000000000000000000000000000000000000000000000000001...

Label convention:

1 = DNA-binding residue
0 = non-binding residue
Included Datasets

The repository includes several DNA and PDNA training/test files, such as:

DNA-573_Train.txt
DNA-646_Train.txt
DNA-129_Test.txt
DNA-181_Test.txt
DNA-46_Test.txt

Additional PDNA datasets can also be placed in Raw_data/ using the same three-line format.

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
Install Dependencies
pip install torch torchvision torchaudio
pip install torch-geometric
pip install fair-esm scikit-learn scipy numpy tqdm
Quick Start
Baseline Pipeline

Run the standard diffusion + GNN workflow:

python main.py

This pipeline performs:

dataset loading;
ESM-2 embedding extraction;
diffusion model training on positive residues;
graph augmentation;
GNN training;
evaluation.
Robust Pipeline

Run the recommended robust training pipeline:

python robust_pipeline.py

This version includes:

robust positive residue augmentation;
sample quality filtering;
diversity control;
stronger regularization;
cross-validation;
multi-file evaluation.
Ratio Test Mode

To test different augmentation ratios:

python robust_pipeline.py --ratio-test
Outputs
Baseline Outputs

Baseline results are saved to:

Augmented_data/

Typical outputs include:

diffusion checkpoints;
GNN checkpoints;
training metadata;
test result JSON files.
Robust Outputs

Robust pipeline results are saved to:

Augmented_data_balanced/

Typical outputs include:

improved GNN checkpoints;
robust evaluation summaries;
full pipeline result files;
ratio-test experiment outputs.
Notes
ESM Version

Although some early descriptions mentioned ESM-3, the current implementation uses:

esm.pretrained.esm2_t33_650M_UR50D()

Therefore, the current implementation is based on ESM-2.

GPU Configuration

Some configuration files contain hard-coded CUDA settings. Before running on your own machine, please check and modify the GPU selection lines, such as:

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

If you only have one GPU, you may change it to:

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

or remove this line and let PyTorch select the available device automatically.

Highlights
Diffusion-based augmentation for rare DNA-binding residues.
Residue-level graph learning with ESM-2 embeddings.
Hybrid GAT/GCN/GraphSAGE architecture.
Robust training pipeline for improved generalization.
Support for DNA and PDNA benchmark-style datasets.
Suitable for residue-level protein-DNA binding site prediction.
Citation

If you use this repository or build upon this work, please cite or acknowledge:

DG-PBs: Diffusion-Augmented Graph Learning for Protein-DNA Binding Site Prediction
Hanqing Zhang

A formal citation will be added after publication.

Contact

Hanqing Zhang
Email: 3165619783@qq.com
