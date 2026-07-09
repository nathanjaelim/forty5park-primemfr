# Tutorial — Running the Rent AVM on Your Local Machine

> **Who this is for:** anyone comfortable enough with a computer to open a terminal and copy-paste commands. No prior Python, git, or ML knowledge assumed.
> **What you'll do:** install a Python package manager, install the project's dependencies, train the model on your laptop, and reproduce the headline result.
> **Time:** ~10-15 minutes total (~3 minutes of actual training compute; the rest is one-time setup).

---

## What is this thing?

This repository is a **rent prediction model** for Atlanta apartment buildings. Given the physical attributes of an apartment (bedrooms, square footage, submarket, etc.) and some historical rent context, it predicts what that apartment should rent for per month. The model achieves about **$77 average error** (out of a $1,800 average rent) — well within broker tolerance.

You're going to install it and train it. By the end you'll have a working ML pipeline running on your laptop and the same numbers a professional data-science team would get.

---

## Before you start — what you need

- **A computer** running macOS, Linux, or Windows (Windows works but examples below use macOS/Linux syntax).
- **A terminal.** This is the text-only "command line" window where you type commands and press Enter.
  - **macOS:** open Spotlight (⌘+Space), type "Terminal", press Enter.
  - **Linux:** press Ctrl+Alt+T, or search "Terminal" in your app menu.
  - **Windows:** open "PowerShell" or "Windows Terminal" from the Start menu.
- **About 2 GB of free disk space** (the model's Python dependencies are large — LightGBM, CatBoost, and scikit-learn together are ~1.3 GB).
- **This repository, downloaded to your computer.** If you don't have it yet:
  - Ask a teammate for a `.zip` and unzip it somewhere you can find it (e.g., `~/Documents/prime-mfr`), OR
  - If you have git installed: `git clone <URL>` into a folder of your choice.

You do **not** need Python installed in advance — the tool we're about to install (`uv`) will download the right Python version for you.

You do **not** need a GPU. Everything runs on your regular CPU.

---

## A quick primer on what we're about to do

A few concepts that show up over and over. Skim these once and you'll be fine.

**Terminal / Command Line:** the text window where you type commands. When you see `curl -LsSf ...` in this tutorial, you paste that into the terminal and press Enter.

**Directory (folder):** a folder on your computer. The terminal has a "current directory" — the folder it's currently looking at. You change it with `cd path/to/folder`. Print it with `pwd`.

**The "repo root":** the top-level folder of this project — the one that contains `pyproject.toml` and `README.md`. Almost every command in this tutorial must be run from there.

**Virtual environment:** an isolated Python installation that lives inside a folder called `.venv/`. It keeps this project's dependencies separate from your system Python (and from other projects). `uv` creates it for you.

**`uv`:** a Python package manager (the tool we're about to install). Think of it as `npm` for Python, if that helps. It handles everything: downloading Python itself, creating the virtual environment, and installing the project's dependencies.

**`uv run <command>`:** runs `<command>` inside this project's virtual environment. So `uv run pytest` means "run pytest using the tools installed for this project."

If any of that feels fuzzy, it'll click after you use it once. Keep going.

---

## Step 1 — Install `uv` (one-time)

`uv` is the tool that handles everything else. This step installs `uv` itself. You do it once per machine; if `uv --version` already works for you, skip to Step 2.

**macOS or Linux:** paste this into your terminal and press Enter:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows PowerShell:** paste this and press Enter:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Homebrew users (macOS):** if you already use Homebrew, you can install with:

```bash
brew install uv
```

The installer takes ~30 seconds. When it finishes, **close and reopen your terminal window** so the new `uv` command is on your PATH. Then verify:

```bash
uv --version
```

Expected output (yours may have a slightly newer version number):

```
uv 0.11.2
```

> ✅ **Checkpoint:** if `uv --version` prints a version number, Step 1 is done. If you get "command not found," close and reopen your terminal — the installer added `uv` to your PATH but only new terminal windows see the change.

---

## Step 2 — Navigate to the repo root

Every remaining command must be run from this project's **repo root** — the folder containing `pyproject.toml`. If your download is at `~/Documents/prime-mfr`, then the repo root is `~/Documents/prime-mfr`.

In the terminal, `cd` (change directory) into it:

```bash
cd ~/Documents/prime-mfr      # <-- adjust to wherever you put the repo
```

> **Windows PowerShell users:** paths use backslashes, e.g. `cd C:\Users\yourname\Documents\prime-mfr`.

**Confirm you're in the right place** by running:

```bash
ls pyproject.toml uv.lock README.md
```

You should see all three filenames listed back to you. If instead you see `ls: cannot access ...: No such file or directory`, you're in the wrong folder — `cd` up or into the correct one.

```
prime-mfr/                    ← YOU MUST BE INSIDE HERE for every remaining command
├── src/prime_mfr/            source code
├── configs/                  YAML configs for model variants
├── artifacts/                source data (Yardi parquets) + tuned hyperparameters
├── eda/                      curated reference data (Atlanta landmark coordinates)
├── tests/                    automated tests
├── docs/                     this tutorial + specs
├── pyproject.toml            ← seeing this file confirms you're in the right place
├── uv.lock                   pinned dependency versions
└── README.md
```

> ✅ **Checkpoint:** `ls pyproject.toml uv.lock README.md` prints all three filenames without errors. **Every** subsequent command assumes you stayed in this directory. If you close your terminal and reopen it tomorrow, `cd` back here first.

---

## Step 3 — Install the project's dependencies

From the repo root (see Step 2), run:

```bash
uv sync
```

This does a LOT of things in one command:
- Reads the file `.python-version` and downloads Python 3.10 if you don't have it
- Creates the virtual environment at `.venv/` (a hidden folder inside the repo)
- Reads `pyproject.toml` and `uv.lock`
- Downloads and installs every dependency (~1.3 GB — this includes the ML libraries LightGBM, CatBoost, scikit-learn, and lots more)
- Installs the `prime-mfr` package itself into the virtual environment

**On a first run this takes 2-4 minutes and prints a lot of progress bars.** Later runs are seconds because everything is cached.

**Verify the installation succeeded:**

```bash
uv run prime-mfr --version
```

Expected output:

```
prime-mfr 0.2.0
```

> ✅ **Checkpoint:** `prime-mfr 0.2.0` prints. If you get "command not found" or an error mentioning missing modules, `uv sync` didn't finish. Scroll up in the terminal output to find the error, and re-run `uv sync`.

---

## Step 4 — Get oriented before training

Before we train anything, let's see what's available. All commands still run from the repo root.

**List the three model variants:**

```bash
uv run prime-mfr list
```

Expected output:

```
Name            Features  Bases  Description
------------------------------------------------------------------------------
cold_start            39      4  Cold-start variant — 4-base stack, no hist
graceful              44      4  Graceful-degradation variant — random hist
primary               44      4  Repricing model — 4-base stack with full h
```

Three flavors of the model. `primary` is the one we're going to train.

**See all available commands:**

```bash
uv run prime-mfr --help
```

You'll see `train`, `status`, `evaluate`, `predict`, `ablate`, `tune`, `clean`, `list`. Ignore the ones you don't need for now.

> ✅ **Checkpoint:** both commands print output without errors.

---

## Step 5 — Run the tests (sanity check)

Before training, let's confirm the code works by running the automated tests:

```bash
uv run pytest
```

Expected: **49 passed in ~3s.** These tests cover data loading, feature engineering (especially the safety-critical checks that prevent target leakage), and metric computations.

> ✅ **Checkpoint:** you see something like `49 passed in 2.55s`. If any test fails, stop and ask for help — training won't produce correct results until tests pass.

---

## Step 6 — Train the model! (the main event)

This is the payoff step. From the repo root, run:

```bash
uv run prime-mfr train --model primary
```

The `--model primary` flag says "train the primary model variant."

**What you'll see happen for about 3 minutes:**

```cdlv sync
=> Training model: primary
   Repricing model — 4-base stack with full historical-rent lags.
   features: 31 numeric + 13 categorical
   bases: ['lgbm_l1', 'cat_q50', 'knn_geo', 'knn_lean']
   meta: aug_ridge (alpha=1.0)

=> Running prep step (feature engineering + CV splits)
   Wrote (6887, 209) static-feature df (2s)
   Wrote splits (5 folds, GroupKFold on property_id)

=> Fold 1/5
   fold 1/5 lgbm_l1  MAE=$77.65  R2=0.925  (13s)
   fold 1/5 cat_q50  MAE=$69.88  R2=0.930  (10s)
   fold 1/5 knn_geo  MAE=$186.17 R2=0.781  (<1s)
   fold 1/5 knn_lean MAE=$203.90 R2=0.722  (<1s)

... [Folds 2 through 5, similar output] ...

=> Fitting meta-learner
   [Stacked (Aug-Ridge meta on log-rent)]
     MAE       : $76.67
     MAPE      : 3.51%
     MedianAPE : 1.77%
     R^2       : 0.8690
```

**Read that final block carefully.** If your MAE is close to $76.67 (anywhere from $75 to $78), **you've successfully reproduced the model.**

Small drift (a dollar or two either way) is normal — LightGBM's parallel training isn't perfectly deterministic across machines. If your number is dramatically different (say $150 or $30), something is wrong with your setup; ask for help.

**Where the output landed:**

| File | What it is |
|---|---|
| `artifacts/metrics.json` | The MAE, MAPE, and other metrics you just saw, saved as JSON |
| `artifacts/oof_predictions.parquet` | The row-by-row predictions (audit trail) |
| `artifacts/feature_importance.csv` | Which features mattered most (from LightGBM) |
| `artifacts/stacking_scratch/` | Working files the pipeline uses; safe to ignore |

> ✅ **Checkpoint:** you see `MAE : $XX.XX` in the last few lines of output, and the number is roughly $76.67 (± a couple dollars).

---

## Step 7 — Inspect the results

**Read the metrics as JSON:**

```bash
cat artifacts/metrics.json
```

On macOS/Linux you'll see raw JSON. To make it readable, pipe through a formatter:

```bash
cat artifacts/metrics.json | python3 -m json.tool | head -30
```

**Check which fold-and-base combinations completed:**

```bash
uv run prime-mfr status
```

You'll see a matrix — every cell should say `done`.

---

## Step 8 — Iterate faster (single-model runs)

Full training is ~3 minutes. If you're experimenting with feature changes or hyperparameters, you probably don't want to wait that long every time. Here are three faster options.

### 8a. Train ONE base model across all folds (~15-60 seconds)

Curious how a single learner performs by itself, without the meta-blend?

```bash
uv run prime-mfr train --model primary --base lgbm_l1
```

This trains only LightGBM across all 5 folds. Skips the meta step. At the end you'll see:

```
=== Single-base OOF metrics for lgbm_l1 (n_rows=6887) ===
  MAE       : $87.13
  MAPE      : 4.00%
  MedianAPE : 2.32%
  R^2       : 0.837
```

Available base names: `lgbm_l1` (LightGBM), `cat_q50` (CatBoost), `knn_geo` (K-Nearest-Neighbors on full features), `knn_lean` (KNN on geography + size only).

Notice CatBoost alone gets $76.70 MAE — very close to the full stack's $76.67. The stack's gain comes from **combining diverse models**, not from any single one being much better.

### 8b. Train ONE fold with all bases (~30 seconds)

```bash
uv run prime-mfr train --model primary --fold 3
```

Trains all 4 base learners on fold 3 only. Skips meta. Useful when you want to check one fold's per-base metrics quickly.

### 8c. Train ONE (fold, base) — the fastest inner loop (~15 seconds)

```bash
uv run prime-mfr train --model primary --fold 3 --base lgbm_l1
```

The smallest unit of training. Trains one base on one fold. Use this when you're tweaking a hyperparameter or debugging a specific feature.

### 8d. Skip prep on repeat runs

The "prep" step (feature engineering + CV splits) takes ~2 seconds. If you've already run it, you can skip it:

```bash
uv run prime-mfr train --model primary --skip-prep --fold 3 --base lgbm_l1
```

Reuses the cached feature-engineered dataframe at `artifacts/stacking_scratch/df_with_static.parquet`.

---

## Step 9 — Clean up (reset to a fresh state)

If you want to wipe all the generated stuff and start over:

**Preview what would be deleted (safe, no-op):**

```bash
uv run prime-mfr clean --dry-run
```

**Actually delete:**

```bash
uv run prime-mfr clean --yes
```

This removes: OOF predictions, metrics JSONs, feature importance CSV, all working files under `artifacts/stacking_scratch/`, log files, and Python cache directories.

This keeps: the source Yardi parquets, the tuned hyperparameters (`best_params.json`, `best_catboost_params.json`), source code, configs, docs.

If you also want to wipe the tuned hyperparameters (only do this if you plan to re-tune, which takes 45-60 minutes per learner):

```bash
uv run prime-mfr clean --deep --yes
```

---

## Common problems and how to fix them

| You see this | It means | Do this |
|---|---|---|
| `uv: command not found` | The `uv` install didn't reach your shell yet | Close and reopen your terminal window. If that doesn't help, re-run the Step 1 installer. |
| `Error: no pyproject.toml found in this directory or any parent` | You're not in the repo root | `cd` to the folder containing `pyproject.toml`. See Step 2. |
| `prime-mfr: command not found` | You're not using `uv run` in front of the command | Prefix commands with `uv run` (e.g., `uv run prime-mfr list`), OR activate the venv once per session with `source .venv/bin/activate` and drop the `uv run`. |
| `ModuleNotFoundError: No module named 'lightgbm'` | `uv sync` didn't finish; some dependencies missing | Re-run `uv sync`. Scroll up in the output to find the first error. |
| `FileNotFoundError: pretraining_v2.parquet` | The training-input file isn't in the repo | Either ask a teammate for it, or generate it: `uv run prime-mfr-pretrain` (requires the four `artifacts/042026-*.parquet` source files) |
| `FileNotFoundError: 042026-*.parquet` | The source Yardi data isn't in the repo | Ask the data-engineering team for the source parquets |
| Any pytest failure (`FAILED tests/...`) | Something's broken in your setup | Don't try to train yet — the tests catch real problems. Post the failing test name and error to your team's channel |
| Training MAE is dramatically off ($30 or $200 instead of ~$77) | Corrupted or missing data, or wrong Python version | Run `uv run pytest` and check that all tests pass. If yes, look at the fold-by-fold output for the first anomalous number |

---

## Glossary

| Term | Plain-language explanation |
|---|---|
| **Terminal / shell / command line** | The text-only window where you type commands (`cd`, `ls`, `uv sync`, etc.) |
| **Directory** | Same as "folder" |
| **Repo root** | The top-level folder of this project — where `pyproject.toml` lives |
| **`cd path`** | "Change directory" — move the terminal's current folder to `path` |
| **`ls`** | "List" — show what's in the current folder |
| **`pwd`** | "Print working directory" — show what folder you're currently in |
| **Virtual environment / venv** | An isolated Python installation kept inside `.venv/` in this project. Doesn't affect the rest of your computer. |
| **Package manager** | A tool that downloads and installs libraries. `uv` for Python; think of it like an app store for code libraries. |
| **CLI** | Command-Line Interface — the way you interact with `prime-mfr` (via typed commands rather than clicking buttons) |
| **Fold** | One slice of the training data used for cross-validation. 5-fold means the data is split into 5 pieces and the model is trained/evaluated 5 times. |
| **Base learner / base model** | One of the four component models (LightGBM, CatBoost, KNN-Geo, KNN-Lean). They're combined by the "meta-learner" at the end. |
| **Meta-learner** | The final model that blends the 4 base learners' predictions into a single output. |
| **MAE** | Mean Absolute Error — average dollar amount the prediction is off by. Lower is better. |
| **MAPE** | Mean Absolute Percentage Error — average percent the prediction is off by. Lower is better. Broker tolerance is 3-5%. |
| **OOF** | "Out-of-fold" — predictions made on data the model didn't see at training. The honest way to measure model quality. |
| **stacked ensemble / stack** | An ML technique where multiple different models' predictions are combined by a "meta" model. The current pipeline is a 4-base stack + Aug-Ridge meta. |

---

## What to read next

Now that the model runs, dig into the details:

- [`README.md`](../README.md) — repo overview + architecture flowchart
- [`docs/ml_stack_spec.md`](ml_stack_spec.md) — deep dive on the 4-base stack and Augmented Ridge meta
- [`docs/feature_engineering_spec.md`](feature_engineering_spec.md) — every feature, its inputs, and its transformation logic
- [`docs/geospatial_features.md`](geospatial_features.md) — the geographic side of the pipeline (landmarks, H3 cells, POIs)
- [`docs/results.md`](results.md) — full technical results with ablations
- [`docs/feature_pipeline_contract.md`](feature_pipeline_contract.md) — DE-handoff contract for a Snowflake/Spark upstream pipeline
