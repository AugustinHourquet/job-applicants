"""Scoring analysis for the 70k+ Job Applicants dataset.

HEC Paris — Interpretability, Stability & Algorithmic Fairness, Fall 2026.
"""

import os
import platform
from pathlib import Path

# ---------------------------------------------------------------------------
# macOS OpenMP guard. Must run before numpy, sklearn, xgboost or torch is
# imported anywhere, which is why it lives in the package __init__ rather than
# in models.py.
#
# On macOS, XGBoost links Homebrew's libomp while PyTorch ships its own copy
# inside the wheel. With both loaded in one process and threading enabled, the
# run either deadlocks inside torch.nn.init (0% CPU, forever) or segfaults —
# observed here as exit 139 while building TabPFN's transformer, but only ever
# *after* XGBoost had already been fitted, which makes it look like a TabPFN
# bug when it is not.
#
# Neither KMP_DUPLICATE_LIB_OK=TRUE nor torch.set_num_threads(1) fixes it; both
# still crash. Pinning OpenMP to a single thread before either runtime
# initialises is what works. The cost is negligible at this data size, and the
# guard is scoped to macOS so Docker and CI keep full XGBoost parallelism.
#
# Override with OMP_NUM_THREADS=4 in your shell if you want to experiment, but
# expect the pipeline to hang when it reaches TabPFN.
# ---------------------------------------------------------------------------
if platform.system() == "Darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")


# ---------------------------------------------------------------------------
# Load .env into the environment.
#
# docker compose reads .env by itself, but `uv run` and `streamlit run` do not,
# so without this the Kaggle credentials sit in the file and the pipeline still
# reports them missing. Loaded here, before any module reads os.environ.
#
# override=False: a variable already exported in your shell wins over the file,
# which is what you want in CI and when temporarily overriding a value.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
    pass

__version__ = "0.1.0"
