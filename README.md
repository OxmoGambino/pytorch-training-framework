# pytorch-training-framework

A minimal **PyTorch training framework** for image classification on **CIFAR-10**.
This project provides a clean, modular, and reproducible baseline to train, evaluate, and iterate on computer vision models.

## ✨ Goals

- Provide a simple and maintainable deep learning training structure.
- Standardize the key steps: data loading, model definition, training, evaluation, and checkpointing.
- Serve as a learning-friendly template and a solid starting point for more advanced projects.

## 🧱 Features

- CIFAR-10 data loading and preprocessing
- PyTorch training loop (train/validation)
- Core metrics tracking (loss, accuracy)
- Model checkpoint saving (best + last)
- Centralized hyperparameter configuration
- CPU / GPU support (CUDA if available)
- Clear epoch-by-epoch logging

## 📂 Project Structure (example)

> Adjust this section to match the real repository tree if needed.

```bash
pytorch-training-framework/
├─ README.md
├─ requirements.txt
├─ train.py
├─ evaluate.py
├─ config/
│  └─ default.yaml
├─ src/
│  ├─ data/
│  │  └─ cifar10.py
│  ├─ models/
│  │  └─ cnn.py
│  ├─ engine/
│  │  ├─ trainer.py
│  │  └─ evaluator.py
│  ├─ utils/
│  │  ├─ metrics.py
│  │  ├─ seed.py
│  │  └─ checkpoint.py
└─ outputs/
   ├─ checkpoints/
   └─ logs/
```

## ⚙️ Requirements

- Python 3.9+
- pip
- (Optional) NVIDIA GPU with CUDA support

## 🚀 Installation

```bash
# 1) Clone the repository
git clone https://github.com/OxmoGambino/pytorch-training-framework.git
cd pytorch-training-framework

# 2) Create and activate a virtual environment
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt
```

## ▶️ Training

Basic run:

```bash
python train.py
```

Example with CLI arguments (if implemented in your code):

```bash
python train.py --epochs 20 --batch-size 128 --lr 0.001 --device cuda
```

## 🧪 Evaluation

```bash
python evaluate.py --checkpoint outputs/checkpoints/best.pt
```

## 🧠 Dataset: CIFAR-10

- 60,000 color images (32x32)
- 10 classes
- Standard split: 50,000 train / 10,000 test

Data is typically downloaded automatically using `torchvision.datasets.CIFAR10`.

## 📈 Tracked Metrics

- **Training loss**
- **Validation loss**
- **Top-1 accuracy**
- (Optional) precision/recall/F1 per class

## 💾 Checkpoints & Logs

- `best` checkpoint (best validation performance)
- `last` checkpoint (most recent epoch)
- Training history logs (text/JSON/CSV depending on implementation)

Example output convention:

```text
outputs/checkpoints/best.pt
outputs/checkpoints/last.pt
outputs/logs/train.log
```

## 🔁 Reproducibility

To improve reproducibility:

- Set a global random seed (`torch`, `numpy`, `random`)
- Pin dependency versions
- Log full experiment configuration (hyperparameters, seed, device)

## 🛠️ Customization

You can easily:

- Swap model architectures in `src/models/`
- Add augmentations in `src/data/`
- Change optimizer/scheduler in `train.py` or `trainer.py`
- Add custom metrics in `src/utils/metrics.py`

## 🧩 Configuration Example (YAML)

```yaml
seed: 42
device: "cuda"
training:
  epochs: 20
  batch_size: 128
  learning_rate: 0.001
  weight_decay: 0.0
data:
  dataset: "CIFAR10"
  num_workers: 4
model:
  name: "SimpleCNN"
checkpoint:
  dir: "outputs/checkpoints"
  save_best: true
```

## ✅ Roadmap

- [ ] Add TensorBoard / Weights & Biases integration
- [ ] Add early stopping
- [ ] Add multi-GPU support (DDP)
- [ ] Add unit tests
- [ ] Add GitHub Actions CI
- [ ] Add ONNX/TorchScript export

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Commit your changes (`git commit -m "feat: add ..."`)
4. Push your branch (`git push origin feat/my-feature`)
5. Open a Pull Request

## 📜 License

Add a license file (`LICENSE`).
MIT is recommended for educational starter frameworks.

## 👤 Author(s)

- GitHub: [@OxmoGambino](https://github.com/OxmoGambino)
- GitHub: [@gtritzguden](https://github.com/gtritzguden)