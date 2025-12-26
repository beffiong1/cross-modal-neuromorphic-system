"""
Plotting utilities used by the experiment notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    accuracy: float,
    save_path: Path,
) -> None:
    """
    Plot a publication-quality confusion matrix and save to disk.
    """
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Count"},
        xticklabels=range(10),
        yticklabels=range(10),
        ax=ax1,
        square=True,
    )
    ax1.set_title(f"{model_name} - Raw Counts", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")

    ax2 = axes[1]
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Oranges",
        cbar_kws={"label": "Proportion"},
        xticklabels=range(10),
        yticklabels=range(10),
        ax=ax2,
        square=True,
    )
    ax2.set_title(f"{model_name} - Normalized", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")

    fig.suptitle(
        f"Confusion Matrix: {model_name} (Acc: {accuracy:.2f}%)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_final_accuracy(evaluation_results_file: Path, fig_path: Path) -> None:
    """
    Load evaluation results and plot a single bar chart of model accuracies.
    """
    print(f"Loading results from {evaluation_results_file}...")
    try:
        with open(evaluation_results_file, "r", encoding="utf-8") as f:
            evaluation_results = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: Results file not found at {evaluation_results_file}")
        return
    except Exception as exc:
        print(f"❌ ERROR: Could not load or parse JSON file: {exc}")
        return

    model_names = []
    accuracies = []

    for model_key, results in evaluation_results.items():
        model_names.append(results.get("model_name", model_key))
        accuracies.append(results["accuracy"])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        model_names,
        accuracies,
        color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"],
        edgecolor="black",
        linewidth=1.5,
    )

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title("Final Accuracy Comparison", fontsize=16, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim([0, 105])
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved: {fig_path.name}")


def analyze_assemblies_tsne(
    features: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    save_path: Path,
) -> Dict[str, Any]:
    """
    Analyze memory assemblies (engrams) using t-SNE and clustering metrics.
    """
    print(f"\n{'='*70}")
    print(f"Assembly Analysis: {model_name}")
    print(f"{'='*70}")

    n_samples = min(2000, len(features))
    idx = np.random.choice(len(features), n_samples, replace=False)
    features_sub = features[idx]
    targets_sub = targets[idx]

    if len(features_sub.shape) > 2:
        features_sub = features_sub.reshape(n_samples, -1)

    print(f"Analyzing {n_samples} samples...")
    print("Computing t-SNE representation...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        init="pca",
        learning_rate="auto",
    )
    features_2d = tsne.fit_transform(features_sub)

    silhouette = silhouette_score(features_2d, targets_sub)
    davies_bouldin = davies_bouldin_score(features_2d, targets_sub)
    calinski = calinski_harabasz_score(features_2d, targets_sub)

    print("\n📊 Cluster Quality Metrics:")
    print(f"   Silhouette Score:      {silhouette:.4f}  (higher is better)")
    print(f"   Davies-Bouldin Index:  {davies_bouldin:.4f}  (lower is better)")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax1 = axes[0]
    scatter = ax1.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=targets_sub,
        cmap="tab10",
        alpha=0.6,
        s=30,
        edgecolors="black",
        linewidth=0.5,
    )
    ax1.set_title(
        f"t-SNE Engram Visualization: {model_name}\nSilhouette Score: {silhouette:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.colorbar(scatter, ax=ax1, label="Class ID")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for class_idx in np.unique(targets_sub):
        mask = targets_sub == class_idx
        ax2.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            alpha=0.4,
            s=20,
            label=f"Class {int(class_idx)}",
        )

    ax2.set_title(f"Neural Assembly Density\n{model_name}", fontsize=14, fontweight="bold")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return {
        "features_2d": features_2d,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski,
    }
