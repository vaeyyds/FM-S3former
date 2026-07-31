# FM-S³former

**Feature-Modulated Sequence-to-Sequence Successive Cancellation Transformer for
Digital In-Band Full-Duplex Self-Interference Cancellation**


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
├── components.py  
├── film.py        # FiLM
├── patch.py       # PatchEmbedding, PatchOutputHead
├── model.py       # FMS3former
├── run.py         # training / evaluation
├── README.md
└── .gitignore
```

---

## Requirements

- Python 3.8+
- PyTorch 1.8+
- NumPy 1.17+
- SciPy 1.4+
- h5py 2.10+ *(only for MATLAB v7.3 `.mat` files)*

```bash
pip install torch numpy scipy h5py
```

---


## Training and evaluation

`run.py` performs the full pipeline: data loading, least-squares linear
cancellation, patch construction, training with successive residual supervision,
and nonlinear-cancellation (NLC) evaluation.

```bash
# LFM waveform 
python run.py --data path/to/lfm.mat  --film_cond amp_fre --device cuda --save lfm.pt

# OFDM waveform 
python run.py --data path/to/ofdm.mat --film_cond amp     --device cuda --save ofdm.pt
```

Each epoch prints the training loss and the test-set NLC. Run
`python run.py -h` for the complete flag list.



---

### FiLM conditioning (`film_cond`)

- `"amp"` — per-patch mean amplitude (1 feature); used for OFDM.
- `"amp_fre"` — per-patch mean amplitude and mean instantaneous frequency
  (2 features); used for LFM. Instantaneous frequency is the per-patch mean of
  the adjacent-sample phase increment `∠(x[n]·x*[n-1])`.


