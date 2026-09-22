# Trustworthy hiring scores

Binary employment scoring on the [70k+ Job Applicants dataset][dataset]. We compare a
logistic regression, XGBoost and a tabular foundation model (TabPFN) on **predictive
performance, interpretability, stability and fairness**, and recommend one for
production.

HEC Paris — *Interpretability, Stability & Algorithmic Fairness*, Fall 2026.
Prof. Christophe Pérignon.

[dataset]: https://www.kaggle.com/datasets/ayushtankha/70k-job-applicants-data-human-resource

---

## Quick start

You need **Docker Desktop** (running) or **[uv]** installed. Nothing else.

[uv]: https://docs.astral.sh/uv/getting-started/installation/

```bash
git clone https://github.com/AugustinHourquet/job-applicants.git
cd job-applicants
cp .env.example .env         # then add your Kaggle credentials (see below)
```

### Option A — Docker (identical for everyone; use this for the demo)

```bash
make docker-pipeline         # downloads data, trains, evaluates  (~20 min first run)
make docker-app              # http://localhost:8501
```

### Option B — uv (faster to iterate; use this while developing)

```bash
make setup                   # create the venv from uv.lock
make pipeline                # same as above
make app                     # http://localhost:8501
```

`make` on its own lists every target.

### Kaggle credentials

The dataset is pulled with the official `kagglehub` library, which needs a token:

1. Go to <https://www.kaggle.com/settings> → **API** → **Create New Token**.
   This downloads `kaggle.json`.
2. Copy the two values into your `.env`:
   ```
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=your_key
   ```

`.env` is gitignored. Never commit it. The test suite generates its own synthetic
data, so you can run `make test` before you ever set this up.

---

## Repository layout

```
config.yaml           # THE shared contract — sectioned by owner
src/
  config.py           # typed loader for config.yaml
  data.py             # cleaning, leakage + proxy checks, features, split   → Member 1
  models.py           # registry; LR, XGBoost, TabPFN; predictions table    → Members 1-3
  economics.py        # cost matrix, profit curve, optimal threshold        → Member 2
  interpretability.py # odds ratios, SHAP, local explanations               → Member 3
  stability.py        # bootstrap re-fits, flip rate, perturbation, shift    → Member 4
  fairness.py         # group metrics, reweighing, group thresholds         → Member 5
  plots.py            # shared chart style (Plotly)
  pipeline.py         # runs everything in order
app/                  # Streamlit, six pages                                → Member 6
artifacts/            # written by the pipeline, read by the app (gitignored)
data/                 # raw + processed (gitignored)
tests/                # full pipeline on synthetic data, under a minute
```

### How the pieces talk to each other

```
data.py  ──DataBundle──▶  models.py  ──predictions.parquet──▶  economics.py
                              │                                      │
                              │                               thresholds
                              ▼                                      ▼
                    artifacts/models/*.joblib               fairness.py
                              │                                      │
                              └──────────▶  app/  ◀──────────────────┘
```

**`artifacts/results/predictions.parquet` is the contract.** One row per
`(split, model, observation)` with `y_true`, `y_prob` and every protected
attribute. Anything that depends only on scores and a threshold — profit curves,
fairness gaps, flip rates — is a groupby over that one table. That is why the
app's sliders are instant, and why Members 2, 4 and 5 can work without waiting
for anyone's feature engineering to be final.

---

## Working in parallel

Each module has **one owner and one entry point**. Edit your own file and
`config.yaml`'s own section; you should almost never need to touch
`pipeline.py`.

| Module | Entry point |
|---|---|
| `data.py` | `build_datasets(cfg) -> DataBundle` |
| `models.py` | `train_all(cfg, bundle) -> dict[str, FittedModel]` |
| `economics.py` | `run(cfg, predictions) -> summary` |
| `interpretability.py` | `run(cfg, bundle, models)` |
| `stability.py` | `run(cfg, bundle, models)` |
| `fairness.py` | `run(cfg, predictions, thresholds)` |

**Adding a model** takes a builder and a config entry — no pipeline change:

```python
@register("random_forest")
def build_random_forest(cfg, spec):
    return RandomForestClassifier(random_state=cfg.seed, **spec.params)
```

**Branching.** `main` stays green. Work on `feat/<your-module>` and open a PR;
CI runs lint and the test suite on every one.

**Iterating fast.** Skip the expensive stages while you work on yours:

```bash
uv run python -m src.pipeline --skip stability interpretability
uv run python -m src.pipeline --models logistic_regression xgboost
```

---

## Things that will bite you

**XGBoost fails to import on macOS.** It needs the OpenMP runtime:
```bash
brew install libomp
```

**The pipeline hangs forever, or dies with exit 139, right after XGBoost trains.**
Already handled — but worth knowing why. On macOS, XGBoost links Homebrew's
`libomp` while PyTorch ships its own copy; with both loaded the process
deadlocks inside `torch.nn.init` or segfaults. `src/__init__.py` pins
`OMP_NUM_THREADS=1` on macOS before anything imports, which is the only fix that
works (`KMP_DUPLICATE_LIB_OK` and `torch.set_num_threads(1)` both still crash).
Don't override it locally unless you enjoy debugging this.

**TabPFN is slow, and that is not a bug.** It is a fixed-context transformer
whose CPU cost grows super-linearly with the training set, and it refuses more
than 1000 CPU rows unless overridden. We fit on a stratified subsample
(`models.tabpfn.train_subsample`, currently 3000) but **always evaluate on the
full test set**, so its scorecard row stays comparable. Measured timings are
recorded in `config.yaml` — quote them if the jury asks why we subsampled.

**The app says "No results yet".** You have not run the pipeline, or you ran it
inside Docker with a different `artifacts/` mount. Run `make pipeline`.

---

## Deliverables

| Deliverable | Where | Due |
|---|---|---|
| Dataset pre-validation | email to the instructor | **Thu 24 Sept, 9:40** |
| Slide deck | `slides/` | Mon 28 Sept, 9:40 |
| Code / notebook | this repo | Mon 28 Sept, 9:40 |
| Interactive application | `app/`, via `docker compose up app` | Mon 28 Sept, 9:40 |

Presentation Monday 28 September: 15 min + 10 min Q&A. **Every member must be
able to answer questions on any part of the analysis**, not just their own
module — so read the other five files.

---

## Methodology notes worth defending

These are deliberate choices, not defaults. Each one is a likely Q&A question.

- **The operating threshold is chosen on validation and applied to test.**
  Choosing it on test and reporting profit at that same cut-off is optimistic.
  The scorecard records `cost_of_honest_threshold`: what the cheat would have
  been worth.
- **Fairness is measured at the deployed threshold, not at 0.5.** Auditing a
  cut-off nobody uses describes a system that does not exist.
- **All four fairness criteria are reported.** They cannot be jointly satisfied
  when base rates differ across groups. Choosing one is a decision about which
  error the client is willing to make, and the deck argues for a specific choice.
- **The scorecard is not collapsed into a single score.** The four dimensions
  trade off, and choosing between them is the client's call.
- **Leakage and proxy findings are reported, never auto-applied.** Dropping a
  column is a modelling decision we defend, not something a script does silently.
- **Interpretability uses a different method per model** — exact odds ratios,
  exact TreeSHAP, and permutation importance for TabPFN. That TabPFN is the
  hardest of the three to explain is itself a finding.
