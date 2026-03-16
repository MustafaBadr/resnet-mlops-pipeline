# ResNet MLOps Pipeline

End-to-end MLOps pipeline for image classification, built from scratch as a portfolio project.

## What's here

- **ResNet-18** built from scratch in PyTorch (adapted for CIFAR-10)
- **Training pipeline** with configurable hyperparameters via YAML
- **MLflow experiment tracking** — log params, metrics, and model artifacts
- **Colab notebook** for GPU training with MLflow integration

## Project roadmap

- [x] Phase 1: Core model + experiment tracking (MLflow)
- [ ] Phase 2: Model serving (FastAPI) + containerization (Docker)
- [ ] Phase 3: CI/CD (GitHub Actions) + data versioning (DVC)
- [ ] Phase 4: Monitoring + drift detection
- [ ] Phase 5: Agentic AI capstone (LangGraph)

## Quick start
```bash
# Local (CPU)
pip install torch torchvision mlflow pyyaml
python src/train.py --epochs 5

# With MLflow tracking
python src/train.py --tracker mlflow --epochs 10
mlflow ui  # Open http://localhost:5000
```

## Tech stack

PyTorch · MLflow · CIFAR-10 · YAML configs

## Results

| Experiment | LR | Optimizer | Test Acc |
|---|---|---|---|
| Run 1 | 0.001 | Adam | ~76% |
| Run 2 | 0.0001 | Adam | ~70% |

*Tracked with MLflow. 5 epochs on T4 GPU.*
