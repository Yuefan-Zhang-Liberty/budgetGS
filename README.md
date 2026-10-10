# BudgetGS

Train 3D Gaussian Splatting scenes onto an explicit byte budget.

BudgetGS is a training framework that accepts an explicit storage, memory, or frame-time budget
and optimizes rendering quality under that constraint during training, without post-training
correction.

## ✨ Features

- **Budget as input:** name a byte budget once, before training starts.
- **Byte ledger:** predicts the serialized size of a checkpoint within 1 % of the encoded file.
- **Fit without retraining:** lands a scene on a budget by bisecting the global quantization step.

## 📖 Background

The representation is [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) trained with
[gsplat](https://github.com/nerfstudio-project/gsplat). The budget-as-input baselines are
[SizeGS](https://arxiv.org/abs/2412.05808) and [MesonGS++](https://arxiv.org/abs/2604.26799),
which reach a size after training rather than during it.

## 📦 Install

```bash
pip install -e .
```

### Dependencies

Python >= 3.10, PyTorch and CUDA. Install PyTorch first from https://pytorch.org.

## 🚀 Usage

```bash
budgetgs train --scene <colmap-scene> --steps 30000
budgetgs fit --ckpt <checkpoint> --budget-mb 10
budgetgs ledger --ckpt <checkpoint>
budgetgs serialize --ckpt <checkpoint> --out scene.splat
```

## 📁 Repository layout

```
.
├── budgetgs                  # Package source code
│   ├── data                  # Data processing and loaders
│   ├── diagnostics           # Diagnostic tools
│   ├── modules               # Main modules implementing logic
│   ├── evaluation            # Evaluation scripts and protocols
│   ├── utils                 # Utility functions
│   └── cli.py                # Command-line entry point
├── slurm                     # SLURM job scripts
├── tests                     # Automated tests
└── README.md
```

## 🤝 Contributing

Questions and bug reports go to the GitHub issues. Pull requests are welcome. Run `pytest` and
`ruff check .` before pushing.

### Testing

```bash
pytest
```

## 📄 License

[MIT](LICENSE) (`MIT`) © 2026 Yuefan Zhang
