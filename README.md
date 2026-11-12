# BudgetGS

Quantization-aware 3D Gaussian Splatting compression under a hard byte budget.

BudgetGS trains a 3D Gaussian Splatting scene and optimizes the **decoded (landed)** rendering
quality under a fixed file-size budget — *during* training, not via post-training correction.
The quantity we minimize is the **quantization gap** (live PSNR − landed PSNR).

## Why landed, not live

The live model can look strong while the compressed model collapses after quantization. On the
room scene, pure L1 fine-tuning pushed live PSNR to **28.45 dB** yet the landed model fell to
**24.35 dB** — a **4.10 dB** quantization gap. BudgetGS's quantization-aware fine-tuning (P0-A)
closes that gap and keeps the landed model on top.

## Key result (P0-A)

| landed metric | P02D baseline | pure L1 (v2) | **P0-A QAT** |
|---|---|---|---|
| PSNR (dB) | 25.08 | 24.35 | **25.24** |
| SSIM | 0.772 | 0.754 | **0.789** |
| LPIPS | 0.435 | 0.420 | **0.417** |
| size (MB) | 7.78 | 7.45 | 7.48 |
| quantization gap (dB) | 0.59 | 4.10 | **1.05** |

*room / factor=2 / 455,555 Gaussians, `real_landing` metric.*

## Pipeline

```
Scene → 3DGS Training → Joint Compression → Entropy Coding → Landing Eval
                         ├── per-attribute quantization (progressive levels)
                         ├── learnable μ-law companding
                         ├── Lagrangian dual λ (budget as hard constraint)
                         └── MCMC Gaussian budget + SGLD relocation
```

## Method sources (source tracking)

| Module | Source | Our change |
|---|---|---|
| 3DGS backbone | Kerbl et al., SIGGRAPH 2023 | base framework |
| Per-attribute quantization | QAT (Jacob et al. 2018) | sigmoid soft-bound + staged unlock |
| Learnable μ-law | ITU-T G.711 (μ-law companding) | per-attribute learnable μ |
| Lagrangian dual λ | rate-distortion optimization (classic) | dual ascent, adaptive λ on budget |
| Range coder | arithmetic coding / rANS | replaces zlib |
| MCMC budget + SGLD | 3DGS-MCMC (arXiv:2404.09591) | relocation + regularization |
| QAT fine-tuning (P0-A) | QAT (Jacob et al. 2018) | tail noise + light regularization |

## Install

```bash
pip install -e .
```

Python >= 3.10, PyTorch and CUDA. Install PyTorch first from https://pytorch.org.

## Usage

The package exposes a CLI (see `budgetgs/cli.py`):

```bash
budgetgs train --scene <colmap-scene> --steps 30000
budgetgs fit --ckpt <checkpoint> --budget-mb 10
budgetgs ledger --ckpt <checkpoint>
budgetgs serialize --ckpt <checkpoint> --out scene.splat
```

Experiment-level entry points (top-level scripts):

| Script | Purpose |
|---|---|
| `train_joint.py` | joint training + progressive per-attribute quantization (P0-2) |
| `quant_aware_ft.py` | quantization-aware fine-tuning with tail noise (P0-A) |
| `pure_l1_ft.py` / `pure_l1_ft2.py` | pure-L1 continuation fine-tuning baseline |
| `land_qa.py` / `land_v2.py` / `land_only.py` | landing evaluation (decode → render → metrics) |
| `sensitivity.py` | single-attribute quantization sensitivity (P1-a) |
| `p1a_errors.py` / `p1a_geom.py` | P1-a fragility / geometry diagnosis |
| `cmp_qa_dist.py` / `cmp_qa_perview.py` | P0-A per-view comparison |
| `read_mu.py` | inspect learned μ-law coefficients |
| `render_compare.py` | side-by-side render comparison |
| `directionA_offline.py` | offline per-attribute bit-allocation (direction A) |
| `directionB_room.py` / `directionB_scene.py` | direction-B scene experiments |

## Repository layout

```
.
├── budgetgs/            # package source (data, modules, evaluation, utils, cli)
├── tests/               # automated tests
├── slurm/               # smoke-test entry (smoke_gsplat.py)
├── runs/analysis/       # experiment results: json metrics + figures
├── *.py                 # experiment-level entry points (see table above)
├── pyproject.toml
└── README.md
```

## Findings so far

- **Compression channel is not the bottleneck.** Three channel runs (P02B/C/D) all plateau at
  landed 25.0–25.1 dB.
- **8 collapsed views** (`168, 288, 272, 40, 232, 280, 224, 296`) break on the *uncompressed*
  live model — a base-model defect, exposed but not caused by compression.
- **Means and high-order SH are the fragile attributes.** Quantizing them alone costs
  −2.67 dB and −2.22 dB respectively (P1-a).
- **QAT is the winning direction.** P0-A narrows the gap 4.10 → 1.05 dB and lands the best
  PSNR / SSIM / LPIPS at the same ~7.5 MB size.

## License

[MIT](LICENSE) © 2026 Yuefan Zhang
