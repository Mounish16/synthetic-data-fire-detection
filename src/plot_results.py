"""Plot F1 (fire class) by training mix for both models from results/*.csv.

Usage
    python src/plot_results.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MIXES = ["100% Real", "20% Synthetic / 80% Real", "50% Synthetic / 50% Real",
         "80% Synthetic / 20% Real", "100% Synthetic"]


def load(name: str) -> pd.Series:
    df = pd.read_csv(ROOT / "results" / name)
    df["Mix"] = (df["Split"].str.replace("DenseNet121 — ", "", regex=False)
                 .str.replace(r"\s*/\s*", " / ", regex=True))
    return df.set_index("Mix")["F1-Score (Fire)"].reindex(MIXES)


def main() -> None:
    data = pd.DataFrame({"MobileNetV2": load("mobilenetv2_metrics.csv"),
                         "DenseNet121": load("densenet121_metrics.csv")})
    ax = data.plot.bar(figsize=(9, 4.6), color=["#1F6FEB", "#9AA5B1"], width=0.75)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8, padding=2)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1 score (fire class), real test set")
    ax.set_xlabel("Training data mix")
    ax.set_title("Effect of synthetic data on fire detection")
    ax.set_xticklabels([m.replace(" / ", "\n") for m in data.index], rotation=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    out = ROOT / "docs" / "f1-by-training-mix.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
