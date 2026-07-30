import argparse
import pathlib
import sys

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from fm_s3former import FMS3former


TX_KEYS = ["txSamples", "tx_signal", "tx_samples", "tx_data", "tx", "Tx", "input", "x", "X"]
RX_KEYS = ["analogResidual", "rx_signal", "rxSamples", "rx_samples", "rx_data", "rx", "Rx", "output", "y", "Y"]


def _to_complex(value):
    arr = np.asarray(value).squeeze()
    if arr.dtype.names:
        real = next((n for n in arr.dtype.names if "real" in n.lower()), None)
        imag = next((n for n in arr.dtype.names if "imag" in n.lower()), None)
        if real and imag:
            arr = arr[real] + 1j * arr[imag]
    return np.asarray(arr, dtype=np.complex128).reshape(-1)


def _first_key(data, keys):
    for k in keys:
        if k in data and data[k] is not None:
            return data[k]
    raise KeyError("no matching tx/rx key in {}".format([k for k in data if not k.startswith("__")]))


def _load_mat_file(path):
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        import h5py

        def read(node):
            if hasattr(node, "keys"):
                return {k: read(node[k]) for k in node.keys() if k != "#refs#"}
            return node[()]

        with h5py.File(path, "r") as f:
            return read(f)


def load_mat(path):
    data = _load_mat_file(path)
    if "aligned_data" in data:
        a = data["aligned_data"]
        if isinstance(a, dict):
            data = a
        elif hasattr(a, "tx_signal"):
            data = {"tx_signal": a.tx_signal, "rx_signal": a.rx_signal}
    x = _to_complex(_first_key(data, TX_KEYS))
    y = _to_complex(_first_key(data, RX_KEYS))
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    x = x - x.mean()
    y = y - y.mean()
    x = x / (np.abs(x).max() + np.finfo(float).eps)
    return x, y


def linear_estimate(x, y, chan_len):
    n = x.size - chan_len
    idx = np.arange(n)[:, None] + (chan_len - np.arange(chan_len))[None, :]
    A = x[idx]
    h, _, _, _ = np.linalg.lstsq(A, y[chan_len:], rcond=None)
    return h


def linear_cancel(x, h):
    return np.convolve(x, h, mode="full")[:x.size]


def prepare_problem(path, chan_len, train_ratio):
    x, y = load_mat(path)
    n_train = int(np.floor(x.size * train_ratio))
    x_tr, y_tr = x[:n_train], y[:n_train]
    x_te, y_te = x[n_train:], y[n_train:]
    h = linear_estimate(x_tr, y_tr, chan_len)
    y_tr_lin = linear_cancel(x_tr, h)
    y_te_lin = linear_cancel(x_te, h)
    r_tr = y_tr - y_tr_lin
    res_mean = r_tr.mean()
    r_tr = r_tr - res_mean
    y_var = r_tr.var()
    r_tr = r_tr / np.sqrt(y_var)
    r_te = (y_te - y_te_lin - res_mean) / np.sqrt(y_var)
    return x_tr, r_tr, x_te, r_te, y_te, y_te_lin, res_mean, y_var


def make_patches(x, r, chan_len, chunk_size, stride):
    xs, yr, yi = [], [], []
    start = 0
    while start + chunk_size <= len(x) and start + chunk_size <= len(r) + (chan_len - 1):
        x_seg = x[start:start + chunk_size]
        r_seg = r[start + chan_len - 1:start + chunk_size]
        xs.append(np.stack([x_seg.real, x_seg.imag], axis=1))
        yr.append(r_seg.real)
        yi.append(r_seg.imag)
        start += stride
    return (torch.tensor(np.array(xs), dtype=torch.float32),
            torch.tensor(np.array(yr), dtype=torch.float32),
            torch.tensor(np.array(yi), dtype=torch.float32))


def evaluate_nlc(model, x_te, r_te, y_te, y_te_lin, res_mean, y_var,
                 chan_len, chunk_size, device):
    n_valid = chunk_size - chan_len + 1
    xt, _, _ = make_patches(x_te, r_te, chan_len, chunk_size, n_valid)
    model.eval()
    preds_r, preds_i = [], []
    with torch.no_grad():
        for i in range(0, xt.shape[0], 512):
            xb = xt[i:i + 512].to(device)
            out_r, out_i = model(xb)
            preds_r.append(out_r[:, chan_len - 1:].cpu().numpy())
            preds_i.append(out_i[:, chan_len - 1:].cpu().numpy())
    pred = (np.concatenate(preds_r).ravel() + 1j * np.concatenate(preds_i).ravel())
    n = pred.size
    y_res = (y_te - y_te_lin)[chan_len - 1:chan_len - 1 + n]
    y_hat = pred * np.sqrt(y_var) + res_mean
    num = np.mean(np.abs(y_res) ** 2)
    den = np.mean(np.abs(y_res - y_hat) ** 2) + np.finfo(float).eps
    return 10.0 * np.log10(num / den)


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_tr, r_tr, x_te, r_te, y_te, y_te_lin, res_mean, y_var = prepare_problem(
        args.data, args.chan_len, args.train_ratio)

    xt, yr, yi = make_patches(x_tr, r_tr, args.chan_len, args.chunk_size, 1)
    loader = DataLoader(TensorDataset(xt, yr, yi), batch_size=args.batch_size, shuffle=True)

    model = FMS3former(
        patch_size=args.patch_size,
        chunk_size=args.chunk_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        film_hidden=args.film_hidden,
        use_film=not args.no_film,
        film_cond=args.film_cond,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                     patience=3, min_lr=1e-6)
    criterion = nn.MSELoss()
    aux_groups = [1] * args.num_layers

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for xb, yrb, yib in loader:
            xb, yrb, yib = xb.to(device), yrb.to(device), yib.to(device)
            optimizer.zero_grad()
            (out_r, out_i), layer_preds = model(xb, return_layer_outputs=True,
                                                layer_groups=aux_groups)
            out_r = out_r[:, args.chan_len - 1:]
            out_i = out_i[:, args.chan_len - 1:]
            loss = criterion(out_r, yrb) + criterion(out_i, yib)
            tgt_r, tgt_i = yrb, yib
            aux = 0.0
            for pred_r, pred_i in layer_preds:
                pred_r = pred_r[:, args.chan_len - 1:]
                pred_i = pred_i[:, args.chan_len - 1:]
                aux = aux + criterion(pred_r, tgt_r) + criterion(pred_i, tgt_i)
                tgt_r = tgt_r - pred_r.detach()
                tgt_i = tgt_i - pred_i.detach()
            loss = loss + args.aux_weight * aux
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        scheduler.step(avg)
        nlc = evaluate_nlc(model, x_te, r_te, y_te, y_te_lin, res_mean, y_var,
                           args.chan_len, args.chunk_size, device)
        print("epoch {:03d}  train_loss {:.5f}  test_NLC {:.2f} dB".format(epoch + 1, avg, nlc))

    nlc = evaluate_nlc(model, x_te, r_te, y_te, y_te_lin, res_mean, y_var,
                       args.chan_len, args.chunk_size, device)
    print("final test NLC: {:.2f} dB".format(nlc))
    if args.save:
        torch.save(model.state_dict(), args.save)
        print("saved model to {}".format(args.save))
    return model


def build_parser():
    p = argparse.ArgumentParser(description="Train and evaluate FM-S3former for digital SIC.")
    p.add_argument("--data", required=True, help="path to a .mat file with tx/rx signals")
    p.add_argument("--film_cond", default="amp_fre", choices=["amp", "amp_fre"],
                   help="amp for OFDM, amp_fre for LFM")
    p.add_argument("--chan_len", type=int, default=20)
    p.add_argument("--chunk_size", type=int, default=96)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--dim_feedforward", type=int, default=128)
    p.add_argument("--film_hidden", type=int, default=16)
    p.add_argument("--no_film", action="store_true")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--aux_weight", type=float, default=0.5)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", default="")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
