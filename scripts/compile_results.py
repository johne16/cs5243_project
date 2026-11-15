import csv
import re
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
logs_dir = root_dir / "logs"
results_dir = root_dir / "results"

arm_names = {"vit": "ViT-Tiny", "resnet18": "ResNet-18"}
init_names = {"pretrained": "pretrained", "scratch": "random-init"}
metric_names = ["AP@[.5:.95]","AP50", "AP75", "APs", "APm", "APl"]

run_pattern = re.compile(r"(pretrained|scratch)_(\d+)")
iter_pattern = re.compile(r"d2\.utils\.events\]:.* iter: (\d+)")
result_pattern = re.compile(r"copypaste: ([\d.]+(?:,[\d.]+){5})\s*$")


def parse_log(log_path):
    text = log_path.read_text(errors="replace")
    run = run_pattern.search(text)
    init, fraction = init_names[run.group(1)], int(run.group(2))
    arm = arm_names[log_path.name.split("-")[1]]

    evals = []
    pending = None
    last_iter = None
    for line in text.splitlines():
        iter_match = iter_pattern.search(line)
        if iter_match:
            last_iter = int(iter_match.group(1))
            if pending is not None:
                evals.append((last_iter + 1, pending))
                pending = None
            continue
        result_match = result_pattern.search(line)
        if result_match:
            if pending is not None:
                evals.append((last_iter + 1, pending))
            pending = [float(v) for v in result_match.group(1).split(",")]
    if pending is not None:
        evals.append((last_iter + 1, pending))

    return (arm, init, fraction), evals


def main():
    runs = {}
    log_paths = sorted(logs_dir.glob("*/detect-*.log"))
    for log_path in log_paths:
        key, evals = parse_log(log_path)
        runs.setdefault(key, {})
        for iteration, metrics in evals:
            runs[key][iteration] = metrics
        print(f"parsed {log_path.relative_to(root_dir)}")

    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "results.csv"
    with open(results_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["arm", "init", "fraction", "iter"] + metric_names)
        for arm, init, fraction in sorted(runs):
            for iteration in sorted(runs[(arm, init, fraction)]):
                metrics = runs[(arm, init, fraction)][iteration]
                writer.writerow([arm, init, fraction, iteration] + [f"{v:.4f}" for v in metrics])

    print(f"wrote {len(runs)} runs from {len(log_paths)} logs to {results_path.relative_to(root_dir)}")


if __name__ == "__main__":
    main()
