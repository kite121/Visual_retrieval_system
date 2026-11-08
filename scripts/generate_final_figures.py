"""Plot the final canonical tables; preserve time units and component definitions."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from build_final_results import DEFAULT_OUTPUT, MODELS, resolve


def generate(output=DEFAULT_OUTPUT):
    output = resolve(output)
    quality = pd.read_csv(output / "final_quality_metrics.csv")
    test = quality[quality.split == "test"].set_index("model_id").loc[list(MODELS)]
    runtime = pd.read_csv(output / "runtime_summary.csv").set_index("model_id").loc[list(MODELS)]
    dest = output / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    names = [s.replace(" (supervised)", "\n(supervised)") for s in test.model]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    bars = ax.bar(names, test.precision_at_5, color=["#40788c", "#999999", "#40788c", "#999999"])
    ax.bar_label(bars, fmt="%.4f", padding=4)
    ax.set(ylim=(0, 1), ylabel="Precision@5", title="Test retrieval quality · Oxford-IIIT Pet")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(dest / "precision_at_5_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    med = runtime.gallery_build_seconds_median.to_numpy()
    axes[0].bar(names, med, color="#40788c", yerr=np.vstack([
        med - runtime.gallery_build_seconds_min, runtime.gallery_build_seconds_max - med]), capsize=4)
    axes[0].set(ylabel="Seconds", title="Gallery build · median and min/max of repeats")
    x = np.arange(4)
    for i, (key, label) in enumerate((("preprocess", "Preprocessing"), ("embed", "Embedding"), ("search", "Search"))):
        med = runtime[f"{key}_median_ms"].to_numpy()
        axes[1].bar(x + (i-1)*.25, med, .25, label=label, yerr=np.vstack([
            med-runtime[f"{key}_min_ms"], runtime[f"{key}_max_ms"]-med]), capsize=3)
    axes[1].set(xticks=x, xticklabels=names, ylabel="Milliseconds per query (batch 1)",
                title="Components · median of per-repeat medians")
    axes[1].legend()
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"{runtime.gpu.iloc[0]} · gallery batch {runtime.batch_size.iloc[0]} · {runtime.repeats.iloc[0]} repeats; whiskers = min/max")
    fig.savefig(dest / "runtime_comparison.png", dpi=180)
    plt.close(fig)
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    print(generate(p.parse_args().output))
