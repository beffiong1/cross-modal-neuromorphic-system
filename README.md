# Modality-Dependent Memory Mechanisms in Cross-Modal Neuromorphic Computing
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **Official PyTorch implementation of "Modality-Dependent Memory Mechanisms in Cross-Modal Neuromorphic Computing"**  
> Effiong Blessing | Saint Louis University  
> *IEEE Computer Magazine Special Issue: Convergence of Neuromorphic and Adaptive Cognition (2025)*

## 📄 Abstract

We present the first comprehensive cross-modal ablation study of memory mechanisms in spiking neural networks (SNNs), revealing striking modality-dependent performance patterns. Hopfield networks achieve 97.68% accuracy on visual tasks but only 76.15% on auditory tasks (21.53% gap), while supervised contrastive learning achieves best cross-modal performance (89.44%). Our unified model achieves 88.78% average accuracy with 50% memory reduction, demonstrating practical multi-sensory neuromorphic processing with 603× energy efficiency.

## 🔑 Key Findings

- **21.53% performance gap** between Hopfield networks on visual vs. auditory tasks
- **First evidence** of modality-dependent memory mechanism preferences in SNNs
- **Unified model** achieves -0.66% degradation vs. parallel with 50% memory savings
- **603× energy efficiency** over traditional neural networks with >97% sparsity
- **Biological validation** through quantitative engram analysis (0.871 silhouette score)

## 🎯 Main Contributions

1. **Discovery of modality-dependent architectural preferences** - Hopfield for vision, SCL for audio, HGRN for balanced performance
2. **Unified model validation** - Single model processes both modalities with minimal degradation
3. **Quantitative engram analysis** - Weak cross-modal alignment (0.038) validates parallel architecture design
4. **Design principles** - Clear guidelines for neuromorphic system development
5. **Energy efficiency** - 603× reduction while maintaining >97% sparsity

## 📊 Results Summary

| Model | Visual (N-MNIST) | Audio (SHD) | Average | Energy |
|-------|------------------|-------------|---------|---------|
| Baseline | 96.77% | 80.04% | 88.40% | 603× |
| **+SCL** | 96.72% | **82.16%** | **89.44%** | 603× |
| +Hopfield | **97.68%** | 76.15% | 86.91% | 603× |
| +HGRN | 97.48% | 80.08% | 88.78% | 603× |
| Full Hybrid | 97.58% | 76.94% | 87.26% | 603× |
| **Joint Training** | 94.41% | 79.37% | 88.78% | 603× |

## 🏗️ Repository Structure

```
cross-modal-neuromorphic/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── environment.yml                     # Conda environment
├── paper/                              # Paper and supplementary materials
│   ├── paper_corrected_final.tex      # LaTeX source
│   ├── references.bib                  # Bibliography
│   └── figures/                        # Paper figures
├── models/                             # Model architectures
│   ├── __init__.py
│   ├── base_snn.py                    # Base SNN architecture
│   ├── model_scl.py                   # Model 2: +SCL
│   ├── model_hopfield.py              # Model 3: +Hopfield
│   ├── model_hgrn.py                  # Model 4: +HGRN
│   ├── model_hybrid.py                # Model 5: Full Hybrid
│   └── dual_input_snn.py              # Joint training model
├── data/                               # Data loading and preprocessing
│   ├── __init__.py
│   ├── nmnist_loader.py               # N-MNIST dataset
│   └── shd_loader.py                  # SHD dataset
├── training/                           # Training scripts
│   ├── __init__.py
│   ├── train_parallel.py              # Parallel training (5 models × 2 modalities)
│   ├── train_joint.py                 # Joint multi-modal training
│   └── utils.py                       # Training utilities
├── analysis/                           # Analysis scripts
│   ├── __init__.py
│   ├── engram_analysis.py             # Engram formation analysis
│   ├── energy_analysis.py             # Energy efficiency calculation
│   └── visualization.py               # Result visualization
├── experiments/                        # Experiment configurations
│   ├── config_nmnist.yaml             # N-MNIST experiment config
│   ├── config_shd.yaml                # SHD experiment config
│   └── config_joint.yaml              # Joint training config
├── checkpoints/                        # Saved model checkpoints
│   └── README.md                      # Checkpoint organization
├── results/                            # Experimental results
│   ├── cross_modal_results.csv        # Main results (Table I)
│   ├── joint_training_results.csv     # Joint training (Table II)
│   ├── engram_analysis_results.csv    # Engram analysis (Table III)
│   └── figures/                       # Generated figures
├── notebooks/                          # Jupyter notebooks
│   ├── 01_data_exploration.ipynb      # Dataset visualization
│   ├── 02_model_training.ipynb        # Training walkthrough
│   ├── 03_results_analysis.ipynb      # Results visualization
│   └── 04_engram_analysis.ipynb       # Engram formation analysis
├── docs/                               # Documentation
│   ├── INSTALL.md                     # Installation instructions
│   ├── DATASETS.md                    # Dataset preparation guide
│   ├── TRAINING.md                    # Training guide
│   └── REPRODUCING.md                 # Reproducing paper results
└── tests/                              # Unit tests
    ├── test_models.py
    ├── test_data.py
    └── test_training.py
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cross-modal-neuromorphic.git
cd cross-modal-neuromorphic

# Option A: Conda
conda env create -f environment.yml
conda activate cross-modal-neuromorphic

# Option B: venv
python -m venv .venv
source .venv/bin/activate

# Install project requirements (CPU/GPU agnostic)
pip install -r requirements.txt

# Install PyTorch (pick one)
# CUDA (A100/GPUs; adjust cu121 if your driver targets another CUDA build)
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CPU-only
# pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Download Datasets

```bash
# Download N-MNIST
python data/download_nmnist.py --output data/nmnist/

# Download SHD
python data/download_shd.py --output data/shd/
```

### Train Models

```bash
# Train Model 2 (+SCL) on N-MNIST
python training/train_parallel.py --model scl --dataset nmnist --config experiments/config_nmnist.yaml

# Train Model 2 (+SCL) on SHD
python training/train_parallel.py --model scl --dataset shd --config experiments/config_shd.yaml

# Train unified model (joint training)
python training/train_joint.py --config experiments/config_joint.yaml
```

### Evaluate Models

```bash
# Evaluate cross-modal performance
python analysis/evaluate_crossmodal.py --checkpoints checkpoints/

# Run engram analysis
python analysis/engram_analysis.py --model-path checkpoints/model4_nmnist_best.pth

# Generate paper figures
python analysis/visualization.py --results results/ --output results/figures/
```

## 📈 Reproducing Paper Results

See [docs/REPRODUCING.md](docs/REPRODUCING.md) for detailed instructions on reproducing all experiments from the paper.

**Quick reproduction:**
```bash
# Run all experiments (5 models × 2 modalities + joint training)
bash scripts/reproduce_all.sh

# Expected runtime: ~24 hours on A100 GPU
# Outputs: All tables and figures from paper
```

## 🔬 Key Components

### Memory Mechanisms

1. **Supervised Contrastive Learning (SCL)** - Best average cross-modal performance (89.44%)
2. **Hopfield Networks** - Best for visual tasks (97.68%), struggles on audio (76.15%)
3. **Hierarchical Gated Recurrent Networks (HGRN)** - Balanced cross-modal performance
4. **Full Hybrid** - Combines all mechanisms

### Architectures

- **Visual (N-MNIST)**: Conv2D layers exploit spatial structure
- **Auditory (SHD)**: 3 Linear layers process temporal frequency patterns
- **Joint Model**: Modality-specific encoders + shared HGRN processing

### Rate Encoding

Features extracted via rate encoding: mean firing rate (proportion of timesteps each neuron fires) computed across temporal dimension, converting spike trains to continuous representations suitable for analysis.

## 📊 Datasets

### N-MNIST (Neuromorphic MNIST)
- **Source**: Event-camera recordings of handwritten digits
- **Sensor size**: 34 × 34 × 2 (H × W × polarity)

### DVS-Gesture
- **Source**: Event-camera recordings of hand/arm gestures (DVS128 Gesture)
- **Sensor size**: 128 × 128 × 2 (H × W × polarity)

### SHD (Spiking Heidelberg Digits)
- **Source**: Spoken digits encoded through artificial cochlea
- **Sensor size**: 700 channels

### SSC (Spiking Speech Commands)
- **Source**: Spoken commands encoded through artificial cochlea
- **Sensor size**: 700 × 1 × 1
- **Samples**: 60K train / 10K test
- **Classes**: 10 (digits 0-9)
- **Dimensions**: (25, 2, 34, 34) [time, polarity, height, width]
- **Download**: [Link](https://www.garrickorchard.com/datasets/n-mnist)

### SHD (Spiking Heidelberg Digits)
- **Source**: Spoken digits via artificial cochlea
- **Samples**: 8.1K train / 2.3K test
- **Classes**: 20 (mapped to 10 via modulo for comparison)
- **Dimensions**: (100, 700) [time, cochlear channels]
- **Download**: [Link](https://zenkelab.org/datasets/)

## 🎓 Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{effiong2025modality,
  title={Modality-Dependent Memory Mechanisms in Cross-Modal Neuromorphic Computing},
  author={Effiong, Blessing},
  journal={IEEE Computer Magazine},
  year={2025},
  publisher={IEEE}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## 📧 Contact

- **Effiong Blessing** - blessing.effiong@slu.edu
- **Institution**: Saint Louis University, Department of Computer Science
- **Project Link**: [https://github.com/yourusername/cross-modal-neuromorphic](https://github.com/yourusername/cross-modal-neuromorphic)

## 🙏 Acknowledgments

- Computational resources provided by Kaggle and RunPod
- snnTorch library for SNN implementations
- Tonic library for neuromorphic dataset handling

## 📚 Related Work

- [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217)
- [Surrogate Gradient Learning in SNNs](https://ieeexplore.ieee.org/document/8891809)
- [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
- [snnTorch](https://github.com/jeshraghian/snntorch)
- [Tonic](https://github.com/neuromorphs/tonic)

## 🔄 Updates

- **2025-01**: Repository created
- **2025-01**: Paper submitted to IEEE Computer Magazine
- **2025-XX**: Paper accepted
- **2025-XX**: Pre-trained models released

---

**Star ⭐ this repository if you find it useful!**
