# CIFAR-10 Classification Framework 🚀

- Provide a flexible and modular deep learning framework for image classification tasks
- Enable easy experimentation with different neural network architectures (e.g., MLP, CNN)
- Allow seamless tuning of training hyperparameters (learning rate, batch size, epochs, etc.)
- Facilitate comparison between optimization strategies (optimizers, schedulers)
- Integrate data augmentation techniques to improve model generalization
- Support reproducible experiments through controlled randomness (seed setting)
- Offer a clean structure for research-oriented workflows and rapid prototyping
  
## ✨ Goals

- Modular Architecture Selection
- Easily switch between different models via configuration (MLP, CNN, etc.)
- Hydra-based Configuration System
- Hyperparameter Optimization with Optuna
- Automated search for optimal parameters (learning rate, architecture size, etc.)
- Early Stopping Mechanism
- Learning Rate Scheduling
- Data Augmentation Module
- Experiment Tracking with Weights & Biases
- Reproducibility

## 🧱 Features

- **Modular Architecture Selection:** Easily switch between different models via configuration (MLP, CNN, etc.)
- **Configuration Management:** Centralized declarative config via [Hydra](https://hydra.cc/).
- **Hyperparameter Sweeps:** Automated search for optimal parameters using [Optuna](https://optuna.org/).
- **Experiment Tracking:** Real-time logging of metrics, losses, and system stats via [Weights & Biases (W&B)](https://wandb.ai/).
- **Data Augmentation:** Integrated PyTorch transformations (Random Crop, Horizontal Flip).
- **Automated Checkpointing:** Saves the best model weights based on validation performance.

## 📂 Project Structure

```bash
pytorch-training-framework/
├─ conf/
│  └─ config.yaml               # Main Hydra/Optuna configuration
├─ data/                        # CIFAR-10 dataset (downloaded automatically)
├─ src/                         # Source code
│  ├─ __init__.py
│  ├─ data.py                   # Dataloaders and data augmentation
│  ├─ model.py                  # CNN and MLP architectures
│  ├─ optimize.py               # Optimizer and loss functions
│  ├─ trainer.py                # Core training and validation loops
│  └─ utils.py                  # Helpers (logging, checkpointing)
├─ wandb/                       # W&B local sync logs
├─ multirun/                    # Hydra logs and configs for Optuna sweeps
├─ checkpoints/
│  └─ best_model.pt             # Best model weights
├─ train.py                     # Main execution script
├─ predictions_exemples.png     # Visual preview of predictions
├─ .gitignore
└─ README.md
```

## ⚙️ Requirements
To run this framework, you need Python 3.9+ and the following core libraries:

- torch, torchvision (PyTorch)
- hydra-core (Configuration)
- hydra-optuna-sweeper (Optuna plugin for Hydra)
- wandb (Weights & Biases)

## 🚀 Installation
```bash
# 1) Clone the repository
git clone [https://github.com/OxmoGambino/pytorch-training-framework.git](https://github.com/OxmoGambino/pytorch-training-framework.git)
cd pytorch-training-framework

# 2) Create and activate a virtual environment
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install torch torchvision hydra-core hydra-optuna-sweeper wandb
# (Alternatively: pip install -r requirements.txt if available)
```

## ▶️ Usage & Training

1. Running a Hyperparameter Sweep (Default)
By default, the project is set up to run an Optuna sweep (mode: MULTIRUN). It will search for the best learning rate, batch size, channel dimensions, and weight decay based on the search space defined in conf/config.yaml.
```bash
python train.py
```
Note: Results for each trial are saved in the multirun/ directory and synced to your W&B dashboard.

2. Overriding Parameters via CLI
Hydra allows you to modify the configuration dynamically without touching the YAML file. This is perfect for quick tests.

Examples:
Change training parameters:
```bash
python train.py training.epochs=50 training.batch_size=128 training.lr=0.005
```
Change model architecture:
```bash
python train.py model.cnn.nb_channels1=64 model.cnn.dropout=0.5 model.mlp.hidden_dim=1024
```
Toggle data augmentation:

```bash
python train.py augmentation.rotation.enabled=True augmentation.rotation.degrees=20
```

## 🧠 Dataset: CIFAR-10
- 60,000 color images (32x32 pixels)
- 10 distinct classes
- Standard split: 50,000 training images / 10,000 test images
- Data is downloaded automatically via torchvision on the first run.

## 📈 Tracking & Logs
This framework heavily relies on Weights & Biases. Make sure you are logged in to your W&B account:
```bash
wandb login
```
# Tracked metrics include:
- Training and Validation Loss
- Accuracy
- Hyperparameters for each run

## 📊 Visual Results

Below are sample predictions made by the best model on the test dataset:
("The optimal configuration achieved X% accuracy on the test set.")

## 🧩 Configuration Highlight (YAML)
All logic is driven by conf/config.yaml. Here is a snippet of how the Optuna search space is defined:
```bash
YAML
hydra:
  mode: MULTIRUN
  sweeper:
    direction: maximize
    study_name: cifar10_GTG_NDP
    params:
      training.lr: tag(log, interval(1e-4, 1e-2))
      training.batch_size: choice(32, 64, 128)
      model.cnn.nb_channels1: choice(16, 32, 64)
      optimizer.weight_decay: tag(log, interval(1e-6, 1e-2))
```

📜 License
This project is licensed under the MIT License.

👤 Authors
[Hydra](https://hydra.cc/)
GitHub: [@OxmoGambino](https://github.com/OxmoGambino)
GitHub: [@gtritzguden](https://github.com/gtritzguden)
