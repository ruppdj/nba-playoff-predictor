# NBA Playoff Bracket Predictor

A reproducible machine learning experiment for predicting NBA playoff series winners and series lengths (1984–2025), with a 2026 bracket prediction.

---

## Results

| Model | Accuracy | Log-Loss | Brier Score | ECE |
|-------|----------|----------|-------------|-----|
| Naive (always higher seed) | ~71% | ~0.60 | ~0.21 | — |
| Logistic Regression | TBD | TBD | TBD | TBD |
| XGBoost (Optuna-tuned) | TBD | TBD | TBD | TBD |
| LightGBM (Optuna-tuned) | TBD | TBD | TBD | TBD |
| **Stacking Ensemble** | **TBD** | **TBD** | **TBD** | **TBD** |

*Results populated after running `make train evaluate`.*

### 2026 Bracket Prediction

> Run `make predict` to generate predictions for the current playoffs.
> See `data/predictions/2026/bracket_output.csv` and `reports/bracket_2026.md`.

---

## Quick Start

```bash
# 1. Create the conda environment
make env

# 2. Activate it
conda activate nba-predictor

# 3. Fetch all raw data (runs once; takes a few hours due to rate limiting)
make fetch

# 4. Build feature dataset
make process

# 5. Run EDA notebooks
make eda

# 6. Train all models
make train

# 7. Evaluate and backtest
make evaluate

# 8. Generate 2026 bracket predictions
make predict

# 9. View MLflow experiment results
make mlflow-ui
# Then open http://localhost:5000
```

### Smoke test (2 seasons, fast)

```bash
make smoke-test
```

---

## Project Structure

```
nba-playoff-predictor/
├── conf/config.yaml              # Central config: seasons, features, hyperparams
├── data/
│   ├── raw/                      # Scraped data (gitignored — run make fetch)
│   ├── processed/                # Feature parquets (gitignored — run make process)
│   └── predictions/2026/         # 2026 bracket predictions (committed)
├── notebooks/                    # EDA + modeling notebooks (outputs stripped)
│   ├── 00_data_audit.ipynb
│   ├── 01_eda_team_stats.ipynb
│   ├── ...
│   └── 09_bracket_prediction_2026.ipynb
├── src/nba_predictor/
│   ├── data/                     # Scrapers (Basketball Reference, NBA API, Kaggle)
│   ├── features/                 # Era normalization, team/player/injury/matchup features
│   ├── models/                   # Baseline, XGBoost, LightGBM, ensemble, series length
│   ├── evaluation/               # Walk-forward CV, metrics, backtesting
│   ├── tracking/                 # MLflow logging helpers
│   └── predict/                  # Bracket simulator + output formatting
├── tests/                        # Unit tests + API smoke tests
├── environment.yml               # Pinned conda dependencies
├── Makefile                      # Full pipeline automation
└── pyproject.toml                # Project metadata, ruff/mypy config
```

---

## Data Sources

| Source | Data | Years | Key Columns |
|--------|------|-------|-------------|
| Basketball Reference | Team advanced stats | 1984–2025 | ORtg, DRtg, NRtg, Pace, eFG%, VORP |
| Basketball Reference | Player per-game + advanced | 1984–2025 | BPM, VORP, WS/48, PER, TS% |
| Basketball Reference | Playoff series results | 1984–2025 | Round, teams, series winner, length |
| NBA Stats API | Team/player game logs | 1996–2025 | Per-game stats for momentum features |
| NBA Stats API | Roster data | 1996–2025 | Player availability for injury features |
| FiveThirtyEight | ELO ratings | 1984–2025 | Supplemental team strength signal |

**No API keys required** for Basketball Reference or NBA Stats API.
Kaggle requires `~/.kaggle/kaggle.json` (see `src/nba_predictor/data/kaggle_loader.py`).

---

## Feature Engineering

### Era Normalization
All rate stats are z-scored within each season (not across the full dataset) to ensure comparability across decades. The NBA has changed dramatically — a 0.36 3P% was elite in 2005 but mediocre by 2025. See `notebooks/05_era_normalization.ipynb` for the data-driven justification for era boundaries.

### Key Feature Groups

| Group | Examples | Notes |
|-------|---------|-------|
| Team (season) | NRtg_norm, eFG%, TOV%, SRS | Era-normalized |
| Momentum (L10/L20) | L10_NRtg_delta, current_win_streak | Trending up/down entering playoffs |
| Player (season) | team_VORP_sum, Star_player_BPM, Top3_VORP | Minutes-weighted aggregates |
| Player momentum | Star_PTS_L10_delta, Star_GP_L10 | Is the star hot? Managing injury? |
| Injury (simple) | adj_VORP_sum, Star_injured | Availability-weighted VORP |
| Injury (role-specific) | Lost_top_scorer, Lost_top_rebounder | Role gaps matter more than generic loss |
| Matchup | delta_NRtg, delta_VORP, seed_diff | Higher seed minus lower seed |
| H2H | H2H_win_pct, H2H_NRtg_avg | Regular season head-to-head |

---

## Model Architecture

```
Base Models:
  ├── LogisticRegression (L2, interpretable baseline)
  ├── XGBoostClassifier (Optuna-tuned, 100 trials)
  └── LGBMClassifier (Optuna-tuned, 100 trials)
        ↓ out-of-fold predictions
Meta-Learner:
  └── LogisticRegression (stacking, calibrated with isotonic regression)
        ↓
  Series Winner Probability

Series Length:
  └── LGBMClassifier (4-class: 4/5/6/7 games, class_weight='balanced')
        OR
  └── mord.LogisticIT (ordinal regression)

Bracket Simulation:
  └── Monte Carlo (10,000 simulations)
      Samples stochastically from calibrated win probabilities
      Propagates uncertainty through all 4 rounds
```

### Cross-Validation

Walk-forward by season (critical — standard k-fold would leak future data):
- Train on all seasons before year N, test on season N
- Minimum 10 training seasons before first test fold
- ~30 out-of-sample test folds (1994–2025)

---

## Reproducing Results

All randomness is seeded (`random_state=42`). Package versions are pinned in `environment.yml`. Data reproducibility is ensured by MD5 checksums in `data/processed/checksums.txt`.

```bash
# Verify processed data checksums
md5sum -c data/processed/checksums.txt
```

---

## 2026 Predictions

See `data/predictions/2026/bracket_output.csv` for per-series win probabilities and series length distributions, and `data/predictions/2026/champion_probabilities.csv` for championship odds from 10,000 Monte Carlo simulations.

Full rendered bracket: `reports/bracket_2026.md`

---

## License

MIT — see [LICENSE](LICENSE).
