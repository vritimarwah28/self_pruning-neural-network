%%writefile /content/train.py
"""
================================================================================
train.py  —  Training Loop, Evaluation, Plotting & Main Entry Point
================================================================================
Part of: Self-Pruning Neural Network on CIFAR-10
Project: Tredence Analytics — AI Engineer Case Study

Overview
--------
Implements:
  - sparsity_loss()      : L1 penalty on all gate values (Part 2)
  - train_one_epoch()    : single training pass with Total Loss = CE + λ * SP
  - evaluate()           : test accuracy + sparsity level measurement
  - run_experiment()     : full train/eval pipeline for one lambda value
  - plot_gate_distributions() : gate histogram plots (bimodal = successful pruning)
  - plot_training_curves()    : CE loss + sparsity curves
  - main()               : lambda sweep over [1e-5, 5e-4, 2e-3] on real CIFAR-10

How to run
----------
  In Google Colab, run all three %%writefile cells first, then:
      !python train.py

  Or run directly in a Colab cell after importing:
      from train import main
      results = main()

Expected results (40 epochs, real CIFAR-10, GPU):
  lambda=1e-5  ->  ~65% accuracy,  ~20-35% sparsity   (low pruning)
  lambda=5e-4  ->  ~60% accuracy,  ~55-70% sparsity   (balanced)
  lambda=2e-3  ->  ~52% accuracy,  ~80-92% sparsity   (aggressive)
================================================================================
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from prunable_layer import PrunableLinear
from model import SelfPruningNet


# ==============================================================================
# PART 2 — Sparsity Regularisation Loss
# ==============================================================================

def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """
    L1 penalty on ALL gate values across every PrunableLinear layer.

    Formula
    -------
    Total Loss  = CrossEntropyLoss  +  lambda * SparsityLoss
    SparsityLoss = sum of sigmoid(gate_scores) over all PrunableLinear layers

    Why L1 drives gates to EXACTLY zero
    ------------------------------------
    After sigmoid, each gate g = sigmoid(s) is in (0, 1).
    The gradient of the L1 term w.r.t. raw score s is:

        d/ds [ sigmoid(s) ] = sigmoid(s) * (1 - sigmoid(s))  >  0  always

    This gradient is ALWAYS positive, so it consistently pushes every gate
    score downward (toward -inf, meaning gate -> 0).

    Crucially, unlike L2 regularisation whose gradient vanishes as the
    parameter approaches zero, the L1 gradient does NOT vanish near zero.
    This is what achieves TRUE sparsity (gate = 0.000...) rather than just
    small values.

    Returns
    -------
    Scalar tensor = sum of all gate values (fully differentiable via sigmoid).
    """
    device = next(model.parameters()).device
    total  = torch.tensor(0.0, device=device)
    for layer in model.prunable_layers():
        # sigmoid is differentiable -> gradients flow back into gate_scores
        total = total + torch.sigmoid(layer.gate_scores).sum()
    return total


# ==============================================================================
# PART 3 — Training Loop & Evaluation
# ==============================================================================

def train_one_epoch(
    model:     SelfPruningNet,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    lam:       float,
    device:    torch.device,
) -> tuple:
    """
    One full pass over the training loader.

    Loss formula applied each batch:
        ce   = CrossEntropy(logits, labels)
        sp   = sum of all gate values  (L1 sparsity loss)
        loss = ce + lam * sp           (Total Loss)

    Gradients flow through:
      - Conv weights and biases   (via ce)
      - FC weights and biases     (via ce)
      - gate_scores               (via both ce and lam*sp)
      - BatchNorm parameters      (via ce)

    Returns
    -------
    (avg_total_loss, avg_ce_loss, avg_sparsity_loss) — averaged over batches
    """
    model.train()
    tot_sum = ce_sum = sp_sum = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(images)

        ce   = F.cross_entropy(logits, labels)       # classification loss
        sp   = sparsity_loss(model)                  # gate L1 penalty
        loss = ce + lam * sp                         # total loss

        loss.backward()
        # Clip gradients to prevent instability when gates collapse rapidly
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        tot_sum += loss.item()
        ce_sum  += ce.item()
        sp_sum  += sp.item()

    n = len(loader)
    return tot_sum / n, ce_sum / n, sp_sum / n


@torch.no_grad()
def evaluate(
    model:          SelfPruningNet,
    loader:         DataLoader,
    device:         torch.device,
    gate_threshold: float = 1e-2,
) -> tuple:
    """
    Compute test accuracy and network sparsity level.

    Sparsity Level
    --------------
    Defined as the percentage of gates whose value is below gate_threshold.
    A gate below 0.01 contributes < 1% of its corresponding weight to the
    output — considered effectively pruned.

    A high sparsity level means the pruning method is working correctly:
    most connections have been driven to near-zero by the L1 penalty.

    Returns
    -------
    accuracy     : float   - test set accuracy in percent (0-100)
    sparsity_pct : float   - percentage of pruned gates (0-100)
    all_gates    : ndarray - flat array of all gate values (for plotting)
    """
    model.eval()
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds   = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    accuracy = 100.0 * correct / total

    all_gates = torch.cat(
        [l.get_gates().flatten() for l in model.prunable_layers()]
    ).cpu().numpy()

    sparsity_pct = 100.0 * (all_gates < gate_threshold).mean()
    return accuracy, sparsity_pct, all_gates


# ==============================================================================
# Experiment Runner
# ==============================================================================

def run_experiment(
    lam:          float,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    device:       torch.device,
    epochs:       int = 40,
) -> dict:
    """
    Train one SelfPruningNet with the given lambda; return results dict.

    Separate Learning Rate Groups
    ------------------------------
    Gate parameters receive a 5x higher LR (5e-3) than weight parameters (1e-3).

    Reason: The sparsity gradient magnitude is proportional to lambda. For small
    lambda values, this gradient is tiny compared to the cross-entropy gradient.
    Adam's adaptive scaling normalises both, making the gate updates negligible.
    A higher gate LR ensures the sparsity signal can meaningfully compete with
    the classification signal, producing visible sparsity even at low lambda.

    Scheduler
    ---------
    CosineAnnealingLR smoothly reduces both LRs from their initial values to
    near-zero over `epochs` steps — prevents oscillation near convergence.

    Parameters
    ----------
    lam    : sparsity regularisation coefficient (lambda)
    epochs : number of training epochs (default 40; use 60 for higher sparsity)

    Returns
    -------
    dict with keys: lam, model, accuracy, sparsity, all_gates, ce_history
    """
    print(f"\n{'='*62}")
    print(f"  lambda = {lam}   ({epochs} epochs)")
    print(f"{'='*62}")

    model = SelfPruningNet().to(device)

    # Separate param groups: gates get higher LR for stronger pruning signal
    gate_params  = [p for n, p in model.named_parameters() if "gate_scores" in n]
    other_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]

    optimizer = Adam([
        {"params": other_params, "lr": 1e-3,  "weight_decay": 1e-4},
        {"params": gate_params,  "lr": 5e-3},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ce_history = []

    for epoch in range(1, epochs + 1):
        tot, ce, sp = train_one_epoch(model, train_loader, optimizer, lam, device)
        scheduler.step()
        ce_history.append(ce)

        if epoch % 10 == 0 or epoch == 1:
            acc, spar, _ = evaluate(model, test_loader, device)
            print(f"  Ep {epoch:3d}/{epochs}  |  CE={ce:.4f}  SP={sp:.1f}  |  "
                  f"Acc={acc:.2f}%  Sparsity={spar:.1f}%")

    final_acc, final_spar, all_gates = evaluate(model, test_loader, device)
    print(f"\n  FINAL  ->  Accuracy: {final_acc:.2f}%   Sparsity: {final_spar:.1f}%")

    return {
        "lam":        lam,
        "model":      model,
        "accuracy":   final_acc,
        "sparsity":   final_spar,
        "all_gates":  all_gates,
        "ce_history": ce_history,
    }


# ==============================================================================
# Plotting
# ==============================================================================

COLOURS = ["#4A90D9", "#E63946", "#2A9D8F"]
LABELS  = ["Low  (λ=1e-5)", "Mid  (λ=5e-4)", "High (λ=2e-3)"]


def plot_gate_distributions(results: list, save_path: str) -> None:
    """
    Gate value histograms for each lambda value.

    Interpretation
    --------------
    A SUCCESSFUL self-pruning run produces a bimodal distribution:
      - Large spike near 0   : pruned connections (gate ≈ 0)
      - Smaller cluster > 0  : active connections (gate ≈ 0.5-1.0)

    A failed run (lambda too low, or synthetic data) shows all gates
    clustered around 0.5-0.9 with no spike at 0.

    The dashed vertical line marks the pruning threshold (0.01).
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, res, col, lbl in zip(axes, results, COLOURS, LABELS):
        g     = res["all_gates"]
        n_prn = (g < 0.01).mean() * 100

        ax.hist(g, bins=80, color=col, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(x=0.01, color="black", linestyle="--", linewidth=1.8,
                   label=f"threshold 0.01\n{n_prn:.1f}% pruned")
        ax.set_title(
            f"{lbl}\nAcc = {res['accuracy']:.1f}%  |  Sparsity = {res['sparsity']:.1f}%",
            fontweight="bold", fontsize=11
        )
        ax.set_xlabel("Gate value  sigmoid(gate_score)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_xlim([0, 1])
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    fig.suptitle(
        "Gate Value Distributions — Self-Pruning CNN on Real CIFAR-10",
        fontweight="bold", fontsize=13, y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved gate distribution plot -> {save_path}")


def plot_training_curves(results: list, save_path: str) -> None:
    """
    Two-panel figure:
      Left  : Training cross-entropy loss over epochs for each lambda
      Right : Final sparsity level achieved by each lambda (reference lines)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    for res, col, lbl in zip(results, COLOURS, LABELS):
        ep = range(1, len(res["ce_history"]) + 1)
        ax1.plot(ep, res["ce_history"], color=col, linewidth=2, label=lbl)

    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.set_title("Training CE Loss vs Epoch", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    for res, col, lbl in zip(results, COLOURS, LABELS):
        ax2.axhline(res["sparsity"], color=col, linewidth=2.5, linestyle="--",
                    label=f"{lbl}  ->  {res['sparsity']:.1f}%")

    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 105])
    ax2.set_ylabel("Final Sparsity Level (%)", fontsize=11)
    ax2.set_title("Final Sparsity by Lambda", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="center right")
    ax2.set_xticks([])
    for sp in ["top", "right", "bottom"]:
        ax2.spines[sp].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved training curves -> {save_path}")


def print_results_table(results: list) -> None:
    """Print a formatted summary table to stdout."""
    sep = "-" * 58
    print(f"\n{sep}")
    print(f"  {'Lambda':<18} | {'Test Accuracy':^14} | {'Sparsity Level':^14}")
    print(sep)
    for r, lbl in zip(results, LABELS):
        print(f"  {lbl:<18} | {r['accuracy']:>10.2f}%    | {r['sparsity']:>10.1f}%    ")
    print(sep)


# ==============================================================================
# PART 4 — Main: Lambda Sweep on Real CIFAR-10
# ==============================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Real CIFAR-10 Data Loading ─────────────────────────────────────────────
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=512, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"Train : {len(train_set):,} samples")
    print(f"Test  : {len(test_set):,} samples")

    # ── Lambda Sweep ───────────────────────────────────────────────────────────
    #   1e-5 : gentle nudge — network stays mostly connected, best accuracy
    #   5e-4 : balanced    — meaningful pruning with small accuracy cost
    #   2e-3 : aggressive  — high sparsity, noticeable accuracy drop
    lambdas = [1e-5, 5e-4, 2e-3]
    EPOCHS  = 40     # increase to 60 for even higher sparsity on high lambda

    results = []
    for lam in lambdas:
        res = run_experiment(lam, train_loader, test_loader, device, epochs=EPOCHS)
        results.append(res)

    # ── Report ─────────────────────────────────────────────────────────────────
    print_results_table(results)

    out_dir = "/content"
    os.makedirs(out_dir, exist_ok=True)

    plot_gate_distributions(results, os.path.join(out_dir, "gate_distributions.png"))
    plot_training_curves(results,    os.path.join(out_dir, "training_curves.png"))

    # Save numeric summary as JSON
    summary = [
        {
            "lambda":             r["lam"],
            "test_accuracy_pct":  round(r["accuracy"], 2),
            "sparsity_pct":       round(r["sparsity"],  1),
        }
        for r in results
    ]
    with open(os.path.join(out_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved results summary -> {out_dir}/results_summary.json")

    # Save best model checkpoint
    best = max(results, key=lambda r: r["accuracy"])
    ckpt_path = os.path.join(out_dir, f"best_model_lam{best['lam']}.pt")
    torch.save(best["model"].state_dict(), ckpt_path)
    print(f"\n  Best model : lambda={best['lam']}, acc={best['accuracy']:.2f}%")
    print(f"  Checkpoint : {ckpt_path}")
    print("\n  Done.")

    return results


if __name__ == "__main__":
    results = main()
