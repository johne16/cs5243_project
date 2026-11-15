import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root_dir = Path(__file__).resolve().parent.parent
results_dir = root_dir / "results"
figures_dir = root_dir / "docs" / "final_report" / "figures"

arm_names = ["ViT-Tiny", "ResNet-18"]
arm_titles = {"ViT-Tiny": "ViT-Tiny + MAE", "ResNet-18": "ResNet-18 + SparK"}
init_names = ["pretrained", "random-init"]
init_labels = {"pretrained": "Pretrained", "random-init": "Random-init"}
init_colors = {"pretrained": "#1f5fbf", "random-init": "#c2570c"}
init_linestyles = {"pretrained": "-", "random-init": "--"}
metric_names = ["AP@[.5:.95]", "AP50", "AP75", "APs", "APm", "APl"]
metric_labels = {"AP@[.5:.95]": "AP@[.5:.95]", "AP50": "AP50", "AP75": "AP75", "APs": "AP$_s$", "APm": "AP$_m$", "APl": "AP$_l$"}
fractions = list(range(10, 101, 10))
dpi = 200

plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e9ee",
    "axes.axisbelow": True,
})


def load_runs():
    runs = defaultdict(dict)
    with open(results_dir / "results.csv", newline="") as fp:
        for row in csv.DictReader(fp):
            key = (row["arm"], row["init"], int(row["fraction"]))
            runs[key][int(row["iter"])] = {m: float(row[m]) for m in metric_names}
    return runs


def final_metrics(runs):
    return {key: evals[max(evals)] for key, evals in runs.items()}


def gap(final, arm, fraction, metric):
    return final[(arm, "pretrained", fraction)][metric] - final[(arm, "random-init", fraction)][metric]


def format_fraction_axis(ax):
    ax.set_xticks(fractions)
    ax.set_xticklabels([f"{f}%" for f in fractions])
    ax.set_xlabel("Labeled fraction")


def save(fig, name):
    path = figures_dir / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(root_dir)}")


def plot_final_ap(final):
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True, layout="constrained")
    for ax, arm in zip(axs, arm_names):
        for init in init_names:
            ax.plot(fractions, [final[(arm, init, f)]["AP@[.5:.95]"] for f in fractions],
                    color=init_colors[init], linestyle=init_linestyles[init], marker="o", linewidth=2, label=init_labels[init])
        ax.set_title(arm_titles[arm], fontweight="bold")
        format_fraction_axis(ax)
    axs[0].set_ylabel("Final AP@[.5:.95]")
    axs[0].set_ylim(bottom=0)
    axs[0].legend(frameon=False)
    save(fig, "final_ap.png")


def plot_ap_gap_over_training(runs):
    line_colors = ["#b0306f", "#1f5fbf", "#c2570c", "#2a7a3f", "#6a3fb5"]
    fraction_rows = [fractions[0::2], fractions[1::2]]
    fig, axs = plt.subplots(2, 2, figsize=(11.4, 6.4), sharex=True, sharey=True, layout="constrained")
    for row, plot_fractions in enumerate(fraction_rows):
        for col, arm in enumerate(arm_names):
            ax = axs[row, col]
            ax.axhline(0, color="#9aa3ae", linewidth=1.2)
            for fraction, color in zip(plot_fractions, line_colors):
                pretrained_evals = runs[(arm, "pretrained", fraction)]
                random_init_evals = runs[(arm, "random-init", fraction)]
                iters = sorted(set(pretrained_evals) & set(random_init_evals))
                max_iter = max(iters)
                ax.plot([100 * it / max_iter for it in iters],
                        [pretrained_evals[it]["AP@[.5:.95]"] - random_init_evals[it]["AP@[.5:.95]"] for it in iters],
                        color=color, linewidth=1.6, label=f"{fraction}%")
            ax.set_xlim(0, 100)
            if row == 0:
                ax.set_title(arm_titles[arm], fontweight="bold")
            if row == 1:
                ax.set_xlabel("Training progress (% of iterations)")
        axs[row, 0].set_ylabel("AP@[.5:.95] gap")
        axs[row, 1].legend(title="Labeled fraction", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save(fig, "ap_gap_over_training.png")


def plot_metric_gap(final):
    metric_colors = dict(zip(metric_names, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.2), sharey=True, layout="constrained")
    for ax, arm in zip(axs, arm_names):
        ax.axhline(0, color="#9aa3ae", linewidth=1.2)
        for metric in metric_names:
            ax.plot(fractions, [gap(final, arm, f, metric) for f in fractions],
                    color=metric_colors[metric], marker="o", markersize=4, linewidth=1.8, label=metric_labels[metric])
        ax.set_title(arm_titles[arm], fontweight="bold")
        format_fraction_axis(ax)
    axs[0].set_ylabel("Gap (pretrained minus random-init)")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside right center", frameon=False)
    save(fig, "metric_gap.png")


def main():
    figures_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    final = final_metrics(runs)
    plot_final_ap(final)
    plot_ap_gap_over_training(runs)
    plot_metric_gap(final)


if __name__ == "__main__":
    main()
