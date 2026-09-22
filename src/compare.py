"""Print the leaky and honest scorecards side by side: `make compare`.

The contrast is the deliverable. A model that scores a perfect AUC because a
feature encodes the answer, next to the same three models on a feature set that
does not, is a clearer argument about trustworthy AI than either table alone.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config

VARIANTS = {
    "LEAKY (all features)": "config.yaml",
    "HONEST (HaveWorkedWith + ComputerSkills excluded)": "config.honest.yaml",
}

COLUMNS = [
    "model",
    "roc_auc",
    "profit",
    "optimal_threshold",
    "bootstrap_flip_rate",
    "perturbation_flip_rate",
    "min_disparate_impact",
]


def main() -> None:
    for title, config_name in VARIANTS.items():
        path = Path(config_name)
        if not path.is_file():
            print(f"\n{title}\n  config not found: {config_name}")
            continue

        cfg = load_config(path, mkdirs=False)
        scorecard = cfg.paths.results_dir / "scorecard.csv"
        print("\n" + "=" * 92)
        print(title.center(92))
        print("=" * 92)
        if not scorecard.is_file():
            command = "make pipeline" if config_name == "config.yaml" else "make pipeline-honest"
            print(f"  not run yet — `{command}`")
            continue

        frame = pd.read_csv(scorecard)
        present = [c for c in COLUMNS if c in frame.columns]
        print(frame[present].round(4).to_string(index=False))

    print(
        "\nRead the two together: if the leaky table shows a near-perfect AUC and no\n"
        "fairness or stability differences worth discussing, that is the point — the\n"
        "trade-offs only become visible once the leak is removed.\n"
    )


if __name__ == "__main__":
    main()
