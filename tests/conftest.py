"""Shared test fixtures.

The suite never touches Kaggle. :func:`make_synthetic_raw` produces a frame with
the same columns and dtypes as ``stackoverflow_full.csv``, carrying real signal
(so the models train to better than chance) and a deliberate group imbalance
(so the fairness metrics are non-degenerate and the assertions mean something).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PROJECT_ROOT, Config, load_config

TECHNOLOGIES = [
    "Python",
    "JavaScript",
    "SQL",
    "Java",
    "C++",
    "Go",
    "Rust",
    "TypeScript",
    "Docker",
    "Kubernetes",
    "AWS",
    "React",
    "PostgreSQL",
    "MongoDB",
    "Git",
]
COUNTRIES = [
    "United States of America",
    "Germany",
    "India",
    "United Kingdom",
    "France",
    "Canada",
    "Brazil",
    "Netherlands",
    "Poland",
    "Spain",
    "Australia",
    "Sweden",
]


def make_synthetic_raw(n_rows: int = 1200, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic stand-in for the Kaggle raw CSV."""
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Man", "Woman", "NonBinary"], size=n_rows, p=[0.78, 0.20, 0.02])
    age = rng.choice(["<35", ">35"], size=n_rows, p=[0.65, 0.35])
    accessibility = rng.choice(["No", "Yes"], size=n_rows, p=[0.92, 0.08])
    mental_health = rng.choice(["No", "Yes"], size=n_rows, p=[0.80, 0.20])
    ed_level = rng.choice(
        ["Undergraduate", "Master", "PhD", "NoHigherEd", "Other"],
        size=n_rows,
        p=[0.45, 0.30, 0.07, 0.13, 0.05],
    )
    main_branch = rng.choice(["Dev", "NotDev"], size=n_rows, p=[0.75, 0.25])
    employment = rng.integers(0, 2, size=n_rows)
    country = rng.choice(COUNTRIES, size=n_rows)

    years_code = rng.integers(0, 40, size=n_rows)
    years_code_pro = np.minimum(years_code, rng.integers(0, 30, size=n_rows))
    computer_skills = rng.integers(1, 20, size=n_rows)
    previous_salary = rng.lognormal(mean=11.0, sigma=0.5, size=n_rows).round(2)

    have_worked_with = [
        ";".join(rng.choice(TECHNOLOGIES, size=rng.integers(1, 9), replace=False))
        for _ in range(n_rows)
    ]

    # Real signal, plus a modest penalty on two protected groups so fairness
    # gaps exist to be measured and mitigated.
    logit = (
        -2.9
        + 0.08 * years_code_pro
        + 0.10 * computer_skills
        + 0.55 * (main_branch == "Dev")
        + 0.40 * np.isin(ed_level, ["Master", "PhD"])
        + 0.45 * employment
        + 0.30 * (np.log(previous_salary) - 11.0)
        - 0.45 * (gender == "Woman")
        - 0.35 * (accessibility == "Yes")
        + rng.normal(0, 0.5, size=n_rows)
    )
    employed = (1 / (1 + np.exp(-logit)) > rng.uniform(size=n_rows)).astype(int)

    return pd.DataFrame(
        {
            "Unnamed: 0": np.arange(n_rows),
            "Age": age,
            "Accessibility": accessibility,
            "EdLevel": ed_level,
            "Employment": employment,
            "Gender": gender,
            "MentalHealth": mental_health,
            "MainBranch": main_branch,
            "YearsCode": years_code,
            "YearsCodePro": years_code_pro,
            "Country": country,
            "PreviousSalary": previous_salary,
            "HaveWorkedWith": have_worked_with,
            "ComputerSkills": computer_skills,
            "Employed": employed,
        }
    )


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_synthetic_raw()


@pytest.fixture
def cfg(tmp_path) -> Config:
    """Project config with all outputs redirected into a temp directory.

    Everything is shrunk so the whole pipeline runs in seconds: fewer bootstrap
    re-fits, a smaller SHAP sample, and a TabPFN subsample well under its limit.
    """
    # Explicit path: the suite must not follow a JOBAPP_CONFIG someone has
    # exported in their shell.
    config = load_config(PROJECT_ROOT / "config.yaml", mkdirs=False)
    config.paths.raw_dir = tmp_path / "raw"
    config.paths.processed_dir = tmp_path / "processed"
    config.paths.models_dir = tmp_path / "models"
    config.paths.results_dir = tmp_path / "results"
    config.paths.figures_dir = tmp_path / "figures"
    config.paths.mkdirs()

    # Everything shrunk so the full pipeline runs in seconds. These are the
    # knobs that dominate runtime; leaving any of them at their production
    # value pushes the suite past a minute.
    config.stability.n_boot = 2
    config.stability.eval_subsample = 60
    config.stability.bootstrap_frac = 0.8
    config.stability.perturbation["n_repeats"] = 2
    config.stability.shift["min_group_size"] = 25
    config.interpretability.shap_sample = 60
    config.interpretability.local_examples = 2
    config.interpretability.permutation_sample = 50
    config.interpretability.permutation_max_features = 4
    config.interpretability.permutation_repeats = 1
    config.models["tabpfn"].train_subsample = 100
    return config
