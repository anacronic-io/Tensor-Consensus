# Tensor-Consensus System for Scalable Multi-Agent Systems

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.25+-orange.svg)](https://github.com/google/jax)

**A TPU-Based Framework for Differentiable Coordination in Large-Scale State Space Models**

This repository contains the complete implementation of the Tensor-Consensus system described in our paper. Tensor-Consensus is the first framework to achieve sub-linear coordination overhead for multi-agent systems via fully differentiable tensor operations on TPU accelerators.

## 🎯 Key Features

- **Sub-linear Complexity**: Amortized O(N) complexity vs traditional O(N²) approaches
- **18-45× Speedup**: Compared to state-of-the-art GPU implementations
- **Scales to 5000+ Agents**: Linear memory growth enables unprecedented scalability
- **Fully Differentiable**: End-to-end gradient flow through all coordination operators
- **Three Core Operators**:
  1. **Adaptive Projection**: Dynamic resource allocation based on task complexity
  2. **Differentiable Strategic Assignment**: Sinkhorn-based task assignment
  3. **Robust Latent Space Consensus**: Outlier-resistant consensus with exponential weighting

## 📊 Performance Highlights

| Metric | Tensor-Consensus | QMIX | MADDPG | HAMT | Speedup |
|--------|------------------|------|--------|------|---------|
| Latency (N=100) | **2.8ms** | 31.5ms | 29.2ms | 37.5ms | **11.2×** |
| Throughput | **45k tokens/s** | 8.3k | 9.1k | 6.7k | **5.4×** |
| Consensus Accuracy (N=1000) | **94.2%** | 76.8% | 71.2% | 80.1% | **+17.4%** |
| Memory (N=1000) | **2.1 GB** | OOM | OOM | OOM | **>10×** |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/anacronic-io/tensor-consensus.git
cd tensor-consensus

# Run setup script
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# Or manually with conda
conda env create -f environment.yml
conda activate tensor-consensus
pip install -e .
```

### Basic Usage

```python
import jax
from src.operators.pipeline import TensorConsensusPipeline

# Create pipeline
pipeline = TensorConsensusPipeline(
    complexity_dim=8,
    embedding_dim=64,
    consensus_sigma=1.0,
)

# Initialize
rng = jax.random.PRNGKey(0)
agent_states = jax.random.normal(rng, (32, 100, 64))  # [batch, agents, dim]
params = pipeline.init(rng, agent_states)

# Forward pass
consensus_output, metrics = pipeline.apply(params, agent_states)

print(f"Consensus shape: {consensus_output.shape}")
print(f"Metrics: {metrics}")
```

### Running Experiments

```bash
# Reproduce all paper experiments
python scripts/reproduce.py --all --hardware tpu

# Run specific environment
python scripts/reproduce.py --env smac --scenario 3s_vs_5z

# Benchmark scalability
python scripts/benchmark.py --agents 100 500 1000 2000 5000
```

## 📁 Repository Structure

```
tensor-consensus/
├── src/                          # Main source code
│   ├── operators/                # Core tensor operators
│   │   ├── projection.py         # Adaptive projection
│   │   ├── assignment.py         # Differentiable assignment
│   │   ├── consensus.py          # Robust consensus
│   │   └── pipeline.py           # Integrated pipeline
│   ├── environments/             # Benchmark integrations
│   │   ├── smac_integration.py   # StarCraft Multi-Agent Challenge
│   │   ├── mpe_integration.py    # Multi-Particle Environments
│   │   └── football_integration.py # Google Research Football
│   ├── models/                   # Base models
│   │   ├── mamba_ssm.py          # Mamba state space model
│   │   └── transformer.py        # Transformer baselines
│   └── training/                 # Training scripts
│       ├── train.py              # Main training loop
│       ├── evaluation.py         # Evaluation utilities
│       └── curriculum.py         # Curriculum learning
├── experiments/                  # Experiment configurations
├── scripts/                      # Automation scripts
│   ├── setup_environment.sh      # Environment setup
│   ├── reproduce.py              # Full reproduction
│   ├── benchmark.py              # Performance benchmarking
│   └── plot_results.py           # Figure generation
├── tests/                        # Unit and integration tests
├── results/                      # Experimental results
├── notebooks/                    # Analysis notebooks
├── pyproject.toml                # Project configuration
├── environment.yml               # Conda environment
└── README.md                     # This file
```

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📈 Reproducing Paper Results

Our paper includes comprehensive benchmarks across 3 environments and 11 baselines. To reproduce:

```bash
# Full reproduction (requires TPU, 24-48 hours)
python scripts/reproduce.py --all --hardware tpu --num-runs 25

# Quick test on CPU/GPU
python scripts/reproduce.py --env smac --scenario 3s_vs_5z --num-runs 5

# Generate paper figures
python scripts/plot_results.py --paper-figures
```

## 🔬 Core Operators

### 1. Adaptive Projection

```python
from src.operators.projection import AdaptiveProjection

projection = AdaptiveProjection(complexity_dim=8)
resources, metrics = projection(agent_states)
```

### 2. Differentiable Assignment

```python
from src.operators.assignment import DifferentiableAssignment

assignment = DifferentiableAssignment(
    embedding_dim=64,
    temperature=1.0,
    num_iterations=10,
)
assignment_matrix, metrics = assignment(agent_states, task_states)
```

### 3. Robust Consensus

```python
from src.operators.consensus import RobustConsensus

consensus = RobustConsensus(sigma=1.0, max_iters=10)
centroid, metrics = consensus(agent_responses)
```

## 🌍 Environments

### SMAC (StarCraft II)

```python
from src.environments.smac_integration import SMACEnvironment

env = SMACEnvironment(map_name="3s_vs_5z", batch_size=32)
states, info = env.reset()
next_states, rewards, dones, info = env.step(actions)
```

### MPE (Multi-Particle)

```python
from src.environments.mpe_integration import MPEEnvironment

env = MPEEnvironment(scenario="simple_spread", n_agents=5, batch_size=32)
states, info = env.reset()
```

### Google Research Football

```python
from src.environments.football_integration import FootballEnvironment

env = FootballEnvironment(
    scenario="academy_3_vs_1_with_keeper",
    num_players=3,
    batch_size=16
)
```

## 📊 Hardware Requirements

### Minimum (for testing)
- **CPU**: 4+ cores
- **RAM**: 16 GB
- **GPU**: Optional, any CUDA-capable GPU
- **Disk**: 10 GB

### Recommended (for full reproduction)
- **TPU**: Google Cloud TPU v5e-64 or v4-32
- **GPU**: 8× NVIDIA A100 80GB (for GPU baselines)
- **RAM**: 512 GB
- **Disk**: 100 GB SSD

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{duran2024tensorconsensus,
  title={Tensor-Consensus System for Scalable Multi-Agent Systems: A TPU-Based Framework for Differentiable Coordination},
  author={Durán Cabobianco, Marco and Tensor-Consensus Research Team},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024},
  url={https://github.com/anacronic-io/tensor-consensus}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Cloud team for TPU resources via the TRC program
- Developers of JAX, Mamba, SMAC, MPE, and Google Research Football
- Barcelona Supercomputing Center for valuable feedback
- Anachroni S.Coop for supporting this research

## 📧 Contact

- **Marco Durán Cabobianco** - marco@anachroni.co
- **Project Homepage**: https://github.com/anacronic-io/tensor-consensus
- **Issues**: https://github.com/anacronic-io/tensor-consensus/issues

## 🔗 Links

- [Documentation](https://tensor-consensus.readthedocs.io)
- [Blog Post](https://anachroni.co/blog/tensor-consensus)

---

**Built with ❤️ by the Anachroni Research Team**
