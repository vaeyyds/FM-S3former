# FM-S³former

**Feature-Modulated Sequence-to-Sequence Successive Cancellation Transformer for
Digital In-Band Full-Duplex Self-Interference Cancellation**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c)

FM-S³former is a neural canceller for the **digital stage** of self-interference
cancellation (SIC) in in-band full-duplex (IBFD) radios. It reconstructs the
nonlinear self-interference (SI) that remains after linear cancellation, directly
from the transmitted baseband signal. The design unifies three ideas:

1. **Patch-based sequence-to-sequence modeling** — reconstructs multiple
   consecutive SI samples in one forward pass, exploiting correlations among
   neighboring outputs and amortizing inference cost. Consecutive input samples
   are grouped into non-overlapping patches to keep the token sequence compact.
2. **FiLM feature modulation** — repeatedly conditions the signal representation
   on self-derived features via feature-wise scaling and shifting, giving expressive multiplicative
   feature–signal interactions instead of one-off input concatenation.
3. **Successive residual cancellation** — each encoder layer emits its own
   cancellation component and is supervised on the residual left by preceding
   layers, so dominant distortion is captured first, and weaker structure is
   refined later.


---

## Repository structure

```
fm_s3former/
├── __init__.py
├── components.py  # FullAttention, AttentionLayer, EncoderLayer, Encoder
├── film.py        # FiLM
├── patch.py       # PatchEmbedding, PatchOutputHead
├── model.py       # FMS3former
├── run.py         # training / evaluation entry point
├── README.md
└── .gitignore
```

---

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- h5py *(only for MATLAB v7.3 `.mat` files)*

```bash
pip install torch numpy scipy h5py
```

---

## Data format

`run.py` reads a single `.mat` file containing the temporally aligned transmit
and receive baseband waveforms. Both MATLAB legacy and v7.3 (HDF5) formats are
supported. The transmit and receive arrays are located by trying common key
names:

| Role | Accepted keys (first match wins) |
|------|----------------------------------|
| Transmit | `txSamples`, `tx_signal`, `tx_samples`, `tx_data`, `tx`, `Tx`, `input`, `x`, `X` |
| Receive  | `analogResidual`, `rx_signal`, `rxSamples`, `rx_samples`, `rx_data`, `rx`, `Rx`, `output`, `y`, `Y` |

Complex I/Q may be stored as a native complex array or as a struct with
`real`/`imag` fields. The signals are mean-removed and the transmit signal is
peak-normalized before processing.

---

## Training and evaluation

`run.py` performs the full pipeline: data loading, least-squares linear
cancellation, patch construction, training with successive residual supervision,
and nonlinear-cancellation (NLC) evaluation.

```bash
# LFM waveform (amplitude + instantaneous frequency conditioning)
python run.py --data path/to/lfm.mat  --film_cond amp_fre --device cuda --save lfm.pt

# OFDM waveform (amplitude-only conditioning)
python run.py --data path/to/ofdm.mat --film_cond amp     --device cuda --save ofdm.pt
```

Each epoch prints the training loss and the test-set NLC. Run
`python run.py -h` for the complete flag list.

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | *(required)* | Path to the `.mat` dataset |
| `--film_cond` | `amp_fre` | `amp` (OFDM) or `amp_fre` (LFM) |
| `--chan_len` | `20` | Memory length used by linear cancellation and valid-sample alignment |
| `--chunk_size` | `96` | Samples per input chunk |
| `--patch_size` | `4` | Samples per patch |
| `--d_model` | `96` | Model width |
| `--num_heads` | `4` | Attention heads |
| `--num_layers` | `5` | Encoder depth |
| `--dim_feedforward` | `128` | Feed-forward width |
| `--epochs` | `60` | Training epochs |
| `--lr` | `5e-4` | Adam learning rate |
| `--batch_size` | `256` | Batch size |
| `--aux_weight` | `0.5` | Weight of the successive-residual supervision |
| `--no_film` | *(off)* | Disable FiLM conditioning |
| `--save` | *(none)* | Path to save the trained weights |

---

## Using the model directly

```python
import torch
from fm_s3former import FMS3former

model = FMS3former(patch_size=4, chunk_size=96, d_model=96, num_heads=4,
                   num_layers=5, dim_feedforward=128, film_cond="amp_fre")

x = torch.randn(8, 96, 2)                      # (batch, chunk_size, [real, imag])
real, imag = model(x)                          # nonlinear SI estimate

# per-layer cancellation components (successive residual)
final_pred, layer_preds = model(x, return_layer_outputs=True)
```

### FiLM conditioning (`film_cond`)

- `"amp"` — per-patch mean amplitude (1 feature); used for OFDM.
- `"amp_fre"` — per-patch mean amplitude and mean instantaneous frequency
  (2 features); used for LFM. Instantaneous frequency is the per-patch mean of
  the adjacent-sample phase increment `∠(x[n]·x*[n-1])`.

---

## Method

**PA-induced SI.** The power amplifier introduces amplitude-dependent, memory-dependent
distortion, modeled as

```
u[n] = Σ_p Σ_m  a_{p,m} · x[n-m] · |x[n-m]|^{p-1}
```

**Linear preprocessing.** An `M`-tap FIR channel is fit by least squares and its
output subtracted, leaving the residual `y_res = y − ŷ_L` that the network targets.

**Successive residual objective.** With per-layer components `Ŷ^(d)` and cumulative
estimate `S^(d) = Σ_{k≤d} Ŷ^(k)`, deep supervision is applied at every depth:

```
L = Σ_d  w_d · ‖ y_res − S^(d) ‖²
```

so each layer refines what the previous layers left unexplained.

**Metric.** Nonlinear cancellation (NLC), in dB, on the concatenated valid test
predictions:

```
NLC = 10·log10( ‖y_res‖² / ‖y_res − Ŷ_NL‖² )
```

---

## Results

Reported nonlinear cancellation on the measured datasets (higher is better):

| Method | LFM (dB) | OFDM (dB) |
|--------|:--------:|:---------:|
| Strongest baseline | 7.11 | 20.00 |
| **FM-S³former** | **7.67** | **22.54** |

FM-S³former also attains near-best per-sample inference latency among the
compared methods.

---

## Citation

```bibtex
@article{fms3former,
  title   = {FM-S3former: Feature-Modulated Sequence-to-Sequence Learning with
             Successive Residual Refinement for Digital In-Band Full-Duplex
             Self-Interference Cancellation},
  author  = {Anonymous},
  year    = {2025}
}
```
