import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utils
# -------------------------
def choose_device(force_cpu: bool = False) -> torch.device:
    """Pick best available device: cuda > mps > cpu."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def paper_batch_size(train_set_size: int) -> int:
    """
    Paper: minibatch size 512 OR half of training dataset size, whichever is smaller. 
    """
    return max(1, min(512, train_set_size // 2))


def apply_preset(args: argparse.Namespace) -> None:
    """
    Two paper-aligned presets:

    - paper_default: "most experiments" setting (AdamW, wd=1, budget=1e5). 
    - paper_late: Section 3.1 emphasize-late-generalization (Adam, no wd, budget=1e6). 
    """
    if args.preset == "custom":
        return

    # Model: 2 layers, width 128, 4 heads. 
    args.d_model = 128
    args.n_layers = 2
    args.n_heads = 4
    args.mlp_mult = 4
    args.dropout = 0.0

    # Common optimizer bits
    args.lr = 1e-3
    args.beta1 = 0.9
    args.beta2 = 0.98
    args.warmup_steps = 10

    if args.preset == "paper_default":
        args.optimizer = "adamw"
        args.weight_decay = 1.0
        args.steps = 100_000
    elif args.preset == "paper_late":
        args.optimizer = "adam"
        args.weight_decay = 0.0
        args.steps = 1_000_000
    else:
        raise ValueError(f"Unknown preset: {args.preset}")


# -------------------------
# Model (decoder-only with causal mask)
# -------------------------
class GrokkingTransformer(nn.Module):
    """
    Paper: "standard decoder-only transformer with causal attention masking" 
    We implement it as TransformerEncoder blocks + causal mask (self-attn only).

    IMPORTANT (paper-aligned):
    - Dataset is full equations: [x, op, y, '=', ans]. 
    - We compute loss/accuracy ONLY on the answer token position. 
      Concretely: predict ans at position 4 using logits at position 3 ('=' position),
      i.e., next-token prediction with shift.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        mlp_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(self.seq_len, d_model)

        # "Standard Transformer" defaults are post-norm + ReLU.
        # Paper doesn't specify norm placement / activation beyond "standard". 
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=mlp_mult * d_model,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.register_buffer("_causal_mask", torch.empty(0), persistent=False)

    def causal_mask(self, device: torch.device) -> torch.Tensor:
        """
        Bool mask: True means "cannot attend".
        Causal mask blocks attending to future tokens.
        """
        if self._causal_mask.numel() == 0 or self._causal_mask.device != device:
            m = torch.triu(
                torch.ones(self.seq_len, self.seq_len, device=device),
                diagonal=1,
            ).bool()
            self._causal_mask = m
        return self._causal_mask

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, S) with S == seq_len
        S = tokens.shape[1]
        if S != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {S}")

        pos = torch.arange(S, device=tokens.device)
        x = self.tok_embed(tokens) + self.pos_embed(pos)
        x = self.encoder(x, mask=self.causal_mask(tokens.device))
        return self.lm_head(x)


# -------------------------
# Data (equations of length 5)
# -------------------------
def mod_inverse(value: int, p: int) -> int:
    """Multiplicative inverse mod p (requires value != 0 mod p)."""
    value = value % p
    if value == 0:
        raise ValueError("Division by zero in modular inverse.")
    return pow(value, -1, p)


def build_division_dataset(p: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Paper division operation: x / y (mod p), for 0 ≤ x < p, 0 < y < p. 

    Equation tokens: [x, op, y, '=', ans] (length 5). 
    Loss/acc computed only on answer token. 

    Vocab:
      0..p-1 : residues
      p      : op_id ('÷')
      p+1    : eq_id ('=')
    """
    op_id = p
    eq_id = p + 1

    n = p * (p - 1)
    tokens = np.empty((n, 5), dtype=np.int64)
    labels = np.empty((n,), dtype=np.int64)

    idx = 0
    for x in range(p):
        for y in range(1, p):
            ans = (x * mod_inverse(y, p)) % p
            tokens[idx] = (x, op_id, y, eq_id, ans)
            labels[idx] = ans
            idx += 1

    metadata = {"op_id": int(op_id), "eq_id": int(eq_id)}
    return tokens, labels, metadata


def split_dataset_pair(
    tokens: np.ndarray,
    labels: np.ndarray,
    train_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Paper: choose a fraction of all available equations at random as training,
    rest as validation. 
    """
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0,1).")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(tokens))
    rng.shuffle(idx)
    split = int(len(tokens) * train_frac)
    train_idx = idx[:split]
    val_idx = idx[split:]

    # Sanity: ensure no overlapping (x,y) pairs
    train_pairs = set(map(tuple, tokens[train_idx][:, [0, 2]].tolist()))
    val_pairs = set(map(tuple, tokens[val_idx][:, [0, 2]].tolist()))
    if len(train_pairs & val_pairs) != 0:
        raise RuntimeError("Train/val leakage detected: overlapping (x,y) pairs.")

    return tokens[train_idx], labels[train_idx], tokens[val_idx], labels[val_idx]


# -------------------------
# Eval / Train
# -------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    tokens: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """
    Compute loss/acc ONLY on the answer token.
    For [x, op, y, '=', ans], predict ans using logits at '=' position (index 3).
    This matches next-token prediction with shift and avoids label leakage.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    eq_pos = 3  # '=' position
    for start in range(0, len(tokens), batch_size):
        bt = torch.as_tensor(tokens[start : start + batch_size], device=device)
        bl = torch.as_tensor(labels[start : start + batch_size], device=device)

        logits = model(bt)
        pred_logits = logits[:, eq_pos, :]  # predict answer token

        loss = F.cross_entropy(pred_logits, bl, reduction="sum")
        total_loss += float(loss.item())

        preds = pred_logits.argmax(dim=-1)
        total_correct += int((preds == bl).sum().item())
        total += int(bl.numel())

    return total_loss / total, total_correct / total


def train(args: argparse.Namespace) -> None:
    # Apply paper presets (overrides args unless preset=custom)
    apply_preset(args)

    run_dir = Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tokens, labels, metadata = build_division_dataset(args.p)
    train_tokens, train_labels, val_tokens, val_labels = split_dataset_pair(
        tokens=tokens,
        labels=labels,
        train_frac=args.train_frac,
        seed=args.seed,
    )

    device = choose_device(force_cpu=args.cpu)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Paper minibatch rule 
    batch_size = paper_batch_size(len(train_tokens))

    vocab_size = args.p + 2
    model = GrokkingTransformer(
        vocab_size=vocab_size,
        seq_len=train_tokens.shape[1],  # 5
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_mult=args.mlp_mult,
        dropout=args.dropout,
    ).to(device)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    config = {
        "paper": "arXiv:2201.02177",
        "preset": args.preset,
        "task": "division_mod_p",
        "p": args.p,
        "train_frac": args.train_frac,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size_effective": batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "optimizer": args.optimizer,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "mlp_mult": args.mlp_mult,
        "dropout": args.dropout,
        "eval_every": args.eval_every,
        "device": str(device),
        "metadata": metadata,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    metrics_path = run_dir / "metrics.jsonl"
    rng = np.random.default_rng(args.seed)
    start_time = time.time()

    for step in range(1, args.steps + 1):
        model.train()

        # Linear warmup over first 10 updates (paper) 
        if args.warmup_steps > 0:
            lr = args.lr * min(step / args.warmup_steps, 1.0)
            for g in optimizer.param_groups:
                g["lr"] = lr

        batch_idx = rng.integers(0, len(train_tokens), size=batch_size)
        bt = torch.as_tensor(train_tokens[batch_idx], device=device)
        bl = torch.as_tensor(train_labels[batch_idx], device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(bt)
        pred_logits = logits[:, 3, :]  # '=' position predicts answer
        loss = F.cross_entropy(pred_logits, bl)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            train_loss, train_acc = evaluate(model, train_tokens, train_labels, batch_size, device)
            val_loss, val_acc = evaluate(model, val_tokens, val_labels, batch_size, device)
            elapsed = time.time() - start_time

            record = {
                "step": step,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "elapsed_sec": elapsed,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            print(
                f"step {step:7d} | train acc {train_acc:.4f} | val acc {val_acc:.4f} | "
                f"train loss {train_loss:.4f} | val loss {val_loss:.4f}"
            )


# -------------------------
# Plot
# -------------------------
def load_metrics(path: Path) -> dict[str, list[float]]:
    steps, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            steps.append(record["step"])
            train_loss.append(record["train_loss"])
            val_loss.append(record["val_loss"])
            train_acc.append(record["train_acc"])
            val_acc.append(record["val_acc"])
    return {
        "steps": steps,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
    }


def plot_metrics(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(args.run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(run_dir / "metrics.jsonl")
    steps = np.array(metrics["steps"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, metrics["train_acc"], label="train acc")
    ax.plot(steps, metrics["val_acc"], label="val acc")
    ax.set_xlabel("steps")
    ax.set_ylabel("accuracy")
    if not args.linear_x:
        ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "accuracy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, metrics["train_loss"], label="train loss")
    ax.plot(steps, metrics["val_loss"], label="val loss")
    ax.set_xlabel("steps")
    ax.set_ylabel("loss")
    if not args.linear_x:
        ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "loss.png", dpi=160)
    plt.close(fig)


# -------------------------
# CLI
# -------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper-aligned grokking (arXiv:2201.02177) on modular division.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train on modular division.")
    train_p.add_argument("--preset", type=str, choices=["paper_default", "paper_late", "custom"], default="paper_late")
    train_p.add_argument("--p", type=int, default=97)
    train_p.add_argument("--train-frac", type=float, default=0.5)
    train_p.add_argument("--seed", type=int, default=0)

    # These are overridden unless preset=custom.
    train_p.add_argument("--steps", type=int, default=1_000_000)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--weight-decay", type=float, default=0.0)
    train_p.add_argument("--warmup-steps", type=int, default=10)
    train_p.add_argument("--optimizer", type=str, choices=["adam", "adamw"], default="adam")
    train_p.add_argument("--beta1", type=float, default=0.9)
    train_p.add_argument("--beta2", type=float, default=0.98)

    # Model (overridden unless preset=custom)
    train_p.add_argument("--d-model", type=int, default=128)
    train_p.add_argument("--n-heads", type=int, default=4)
    train_p.add_argument("--n-layers", type=int, default=2)
    train_p.add_argument("--mlp-mult", type=int, default=4)
    train_p.add_argument("--dropout", type=float, default=0.0)

    train_p.add_argument("--eval-every", type=int, default=1000)
    train_p.add_argument("--out-dir", type=str, default="runs/division_mod_97")
    train_p.add_argument("--cpu", action="store_true")

    plot_p = subparsers.add_parser("plot", help="Plot training curves.")
    plot_p.add_argument("--run-dir", type=str, default="runs/division_mod_97")
    plot_p.add_argument("--linear-x", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "plot":
        plot_metrics(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
