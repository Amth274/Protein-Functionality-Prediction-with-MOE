# Protein Functionality Prediction with Mixture-of-Experts

A deep learning framework for predicting protein functionality using meta-learning and Mixture-of-Experts (MoE) architectures on ProteinGym benchmark datasets.

## 🧬 Overview

This project implements a novel approach to protein functionality prediction by combining:
- **Protein Language Models (ESM2)** for sequence embeddings
- **Mixture-of-Experts** for specialized protein regime handling
- **Meta-learning** for few-shot adaptation to new protein families
- **Ranking-based optimization** for fitness landscape prediction

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Support Set   │───▶│  Gating Network │───▶│   Expert 1-N    │
│  (128 samples)  │    │                 │    │   (Specialized) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Query Set     │───▶│   Aggregation   │◀───│  Fitness Score  │
│  (72 samples)   │    │                 │    │   Prediction    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Dataset

The project uses **ProteinGym DMS (Deep Mutational Scanning)** datasets:
- **Training**: 173 protein assay tasks
- **Testing**: 44 protein assay tasks
- **Input**: Amino acid sequences with fitness scores
- **Target**: Ranking-based fitness prediction

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd protein-functionality

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers pandas numpy scikit-learn scipy matplotlib seaborn tqdm pyyaml tensorboard
```

### Data Preparation

```bash
# 1. Preprocess raw data
python scripts/preprocess_data.py \
    --input-dir data/raw/Train_split \
    --output-dir data/processed \
    --embeddings-dir data/embeddings

# 2. Process test data
python scripts/preprocess_data.py \
    --input-dir data/raw/Test_split \
    --output-dir data/processed \
    --embeddings-dir data/embeddings
```

### Training

```bash
# Local training
python scripts/train_hpc.py --config configs/moe_config.yaml

# HPC training with SLURM
sbatch scripts/submit_job.sh
```

### Evaluation

```bash
python scripts/evaluate_model.py \
    --model-path outputs/best_model.pth \
    --test-dir data/embeddings/test \
    --config configs/moe_config.yaml \
    --create-plots
```

## 📁 Project Structure

```
protein-functionality/
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   │   ├── moe.py               # Mixture-of-Experts implementation
│   │   ├── experts.py           # Expert network variants
│   │   └── encoders.py          # Sequence encoders
│   ├── data/                    # Data processing
│   │   ├── dataset.py           # Dataset classes
│   │   ├── preprocessing.py     # Data preprocessing
│   │   └── utils.py             # Data utilities
│   ├── training/                # Training modules
│   │   ├── losses.py            # Loss functions
│   │   └── utils.py             # Training utilities
│   └── utils/                   # General utilities
│       ├── config.py            # Configuration handling
│       ├── logging.py           # Logging utilities
│       └── metrics.py           # Evaluation metrics
├── scripts/                     # Command-line scripts
│   ├── train_hpc.py            # HPC training script
│   ├── preprocess_data.py      # Data preprocessing
│   ├── evaluate_model.py       # Model evaluation
│   └── submit_job.sh           # SLURM job submission
├── configs/                     # Configuration files
│   └── moe_config.yaml         # Default MoE configuration
├── data/                        # Data directories
│   ├── raw/                    # Original CSV files
│   ├── processed/              # Preprocessed data
│   └── embeddings/             # Pre-computed embeddings
└── outputs/                     # Training outputs
    ├── checkpoints/            # Model checkpoints
    ├── logs/                   # Training logs
    └── tensorboard/            # TensorBoard logs
```

## ⚙️ Configuration

Key configuration options in `configs/moe_config.yaml`:

```yaml
model:
  input_dim: 320                 # ESM2 embedding dimension
  hidden_dim: 512                # MoE hidden dimension
  num_experts: 4                 # Number of expert networks
  top_k: 2                       # Active experts per input

training:
  epochs: 100                    # Training epochs
  learning_rate: 1e-4            # Initial learning rate
  batch_size: 32                 # Training batch size

data:
  support_size: 128              # Meta-learning support set size
  query_size: 72                 # Meta-learning query set size
```

## 🎯 Key Features

### 1. Mixture-of-Experts Architecture
- **Specialized Experts**: Different experts for viral, structural, and general proteins
- **Gating Mechanism**: Learned routing to appropriate experts
- **Load Balancing**: Prevents expert collapse during training

### 2. Meta-Learning Framework
- **Support/Query Split**: Each protein task split into support and query sets
- **In-Context Learning**: Model conditions on support set for query predictions
- **Task Adaptation**: Quick adaptation to new protein families

### 3. Ranking-Based Training
- **Pairwise Ranking Loss**: Optimizes relative fitness ordering
- **Spearman Correlation**: Primary evaluation metric
- **Multiple Metrics**: Comprehensive evaluation suite

### 4. HPC Support
- **SLURM Integration**: Native support for cluster computing
- **Distributed Training**: Multi-GPU training capabilities
- **Checkpointing**: Robust checkpoint and resume functionality

## 📈 Performance Metrics

The model is evaluated using multiple ranking and regression metrics:

- **Spearman Correlation**: Primary ranking metric
- **Pearson Correlation**: Linear correlation assessment
- **NDCG@10**: Normalized Discounted Cumulative Gain
- **Top-k Accuracy**: Overlap in top-k predictions
- **MSE/MAE**: Regression error metrics

## 🔬 Research Context

This work builds upon recent advances in:
- **Protein Language Models**: ESM-2 representations
- **Meta-Learning**: Few-shot learning for proteins
- **Mixture-of-Experts**: Specialized protein modeling
- **ProteinGym Benchmark**: Standardized evaluation framework

## 📝 Usage Examples

### Custom Training Configuration

```python
# Create custom configuration
from src.utils.config import get_default_config, save_config

config = get_default_config()
config['model']['num_experts'] = 8
config['training']['learning_rate'] = 2e-4

save_config(config, 'configs/custom_config.yaml')
```

### Loading and Using Trained Model

```python
import torch
from src.models import MoEModel

# Load model
model = MoEModel(input_dim=320, num_experts=4)
checkpoint = torch.load('outputs/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict on new data
with torch.no_grad():
    predictions, aux_loss = model(protein_embeddings)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ESM Team** for protein language model development
- **ProteinGym** for benchmark datasets
- **Meta AI** for foundational protein modeling research
- **HPC Community** for distributed training frameworks

## 📚 References

1. Lin, Z. et al. "Evolutionary-scale prediction of atomic level protein structure with a language model." Science (2023)
2. Notin, P. et al. "ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction." NeurIPS (2023)
3. Fedus, W. et al. "Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." JMLR (2022)

## 📞 Contact

For questions and support, please open an issue on GitHub or contact the development team.

---

**Note**: This project is designed for research purposes and protein functionality prediction benchmarking. Ensure appropriate computational resources for training and evaluation.