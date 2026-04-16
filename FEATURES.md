# Feature Dictionary — Model Features

This document describes every feature used by the series winner and series length models, grouped by category. All features live in `data/processed/series_dataset.parquet` (345 series × 58 columns, 2003–2025).

**Feature lists are defined in `conf/config.yaml` under `features:` and resolved by `src/nba_predictor/models/ensemble.py:get_feature_cols()`.**

---

## Feature Anatomy

Each row in `series_dataset.parquet` represents one playoff series. Most features are **delta features** (higher-seed value − lower-seed value). Positive delta always means the higher-seeded team has more of that quantity.

**Pipeline:**
```
Raw data (data/raw/)
  → team_features (data/processed/team_season_features.parquet)
  → player_features (data/processed/player_season_features.parquet)
  → injury_features (data/processed/injury_adjusted.parquet)
  → series_dataset (data/processed/series_dataset.parquet)   ← model input
```

**Pipeline code:** `src/nba_predictor/features/matchup_features.py:build_matchup_dataset()`

---

## Target Variables

| Column | Type | Description |
|--------|------|-------------|
| `higher_seed_wins` | int (0/1) | 1 if the higher-seeded team won the series — **primary classification target** |
| `series_length` | int (4–7) | Number of games played — **series length regression/classification target** |

---

## Matchup Delta Features

Computed as `higher_seed_stat − lower_seed_stat`. Source: `team_season_features.parquet` and `player_season_features.parquet`.

| Feature | Source Column | Source File | Description |
|---------|--------------|-------------|-------------|
| `delta_NRtg` | `NRtg_norm` | `team_season_features.parquet` | Net rating differential (era-normalized). Primary discriminator. |
| `delta_ORtg` | `ORtg_norm` | `team_season_features.parquet` | Offensive rating differential (era-normalized) |
| `delta_DRtg` | `DRtg_norm` | `team_season_features.parquet` | Defensive rating differential (era-normalized; positive = higher seed has better defense) |
| `delta_Pace` | `Pace_norm` | `team_season_features.parquet` | Pace-of-play differential (era-normalized) |
| `delta_Win_pct` | `Win_pct` | `team_season_features.parquet` | Regular-season win% differential |
| `delta_Experience` | `Playoff_experience_years` | `team_season_features.parquet` | Playoff appearances in prior 5 seasons — experience gap |
| `delta_Prior_playoff_win_pct` | `Prior_playoff_win_pct` | `team_season_features.parquet` | Series win% over prior 3 playoff seasons — postseason track record |
| `delta_Prior_deepest_round` | `Prior_deepest_round` | `team_season_features.parquet` | Deepest round reached in last 2 playoff runs (0–5) |
| `delta_Recent_appearances` | `Prior_playoff_appearances_2yr` | `team_season_features.parquet` | Playoff appearances in prior 2 seasons (0–2) |
| `delta_L10_NRtg` | `L10_NRtg` | `team_season_features.parquet` | Net rating differential over last 10 games entering playoffs |
| `delta_L10_NRtg_trend` | `L10_NRtg_delta` | `team_season_features.parquet` | L10 NRtg vs season-average NRtg — hot/cold differential |
| `delta_streak` | `current_win_streak` | `team_season_features.parquet` | Win streak differential (positive = higher seed on hot streak) |
| `delta_VORP` | `team_VORP_sum` | `player_season_features.parquet` | Roster VORP sum differential |
| `delta_BPM` | `team_BPM_weighted_avg` | `player_season_features.parquet` | Minutes-weighted BPM differential |
| `delta_Top3_VORP` | `Top3_VORP_sum` | `player_season_features.parquet` | Top-3 VORP sum differential — star-tier depth |
| `delta_Star_BPM` | `Star_player_BPM` | `player_season_features.parquet` | Best player BPM differential — head-to-head star matchup |
| `delta_adj_VORP` | `adj_VORP_sum` | `injury_adjusted.parquet` | Availability-weighted VORP differential (accounts for injuries) |
| `delta_roster_health` | `Roster_VORP_available_pct` | `injury_adjusted.parquet` | % of roster VORP available differential |

---

## Seeding and Context Features

Computed directly in `matchup_features.py`; not from a separate parquet.

| Feature | Description |
|---------|-------------|
| `seed_diff` | Numeric seed gap (e.g. 4 for a 1v5 matchup). NaN before 2003 (seeds not stored in raw data). |
| `home_court_advantage` | Always 1 — higher seed always hosts. Present as an explicit feature signal. |
| `conference_East` | 1 if this is an East conference matchup |
| `conference_West` | 1 if this is a West conference matchup |
| `series_round` | String round name (e.g. `"Eastern Conference First Round"`) — not used as model input, for indexing only |

---

## Playoff History Flags

Absolute (not delta) flags included for both sides.

| Feature | Source Column | Source File | Description |
|---------|--------------|-------------|-------------|
| `higher_seed_Prior_champion_3yr` | `Prior_champion_3yr` | `team_season_features.parquet` | 1 if higher seed won championship in any of the prior 3 seasons |
| `lower_seed_Prior_champion_3yr` | `Prior_champion_3yr` | `team_season_features.parquet` | 1 if lower seed won championship in any of the prior 3 seasons |

---

## Injury Flags

Absolute (not delta) flags included for both higher and lower seed. All default to "healthy" (no live injury data fetched). See `DATA_DICTIONARY.md` for details.

Column pattern: `higher_{col}` and `lower_{col}` for each of the following.

| Base Column | Source File | Description |
|------------|-------------|-------------|
| `Star_injured` | `injury_adjusted.parquet` | 1 if highest-VORP player is below 50% availability |
| `Second_star_injured` | `injury_adjusted.parquet` | 1 if 2nd-highest-VORP player is injured |
| `Lost_top_scorer` | `injury_adjusted.parquet` | 1 if top scorer availability < 0.5 |
| `Lost_top_rebounder` | `injury_adjusted.parquet` | 1 if top rebounder availability < 0.5 |
| `Lost_top_playmaker` | `injury_adjusted.parquet` | 1 if top playmaker (AST%) availability < 0.5 |
| `Roster_VORP_available_pct` | `injury_adjusted.parquet` | % of total roster VORP that is available |
| `Injured_player_count` | `injury_adjusted.parquet` | Count of players below 0.5 availability |
| `has_injury_data` | `injury_adjusted.parquet` | 1 if live injury data was available (currently always 0) |

---

## Head-to-Head Features

Computed from `data/raw/nba_api/team_game_logs/team_game_logs_all.parquet` by `matchup_features.py:_compute_h2h()`. Available for seasons ≥ 1996.

| Feature | Description |
|---------|-------------|
| `H2H_win_pct` | Higher seed's win % in regular-season games vs lower seed (same season) |
| `H2H_NRtg_avg` | Higher seed's average PLUS_MINUS in those matchups |
| `H2H_games_played` | Number of regular-season matchups between the two teams |

---

## Era / Meta Features

Categorical era flags (one-hot encoded, not mutually exclusive across all rows).

| Feature | Config Key | Description |
|---------|-----------|-------------|
| `era_showtime` | `eras.showtime` | 1 if season in 1984–1994 (Physical/Showtime era) |
| `era_defensive` | `eras.defensive` | 1 if season in 1995–2004 (Defensive era) |
| `era_transition` | `eras.transition` | 1 if season in 2005–2014 (Transition era) |
| `era_analytics` | `eras.analytics` | 1 if season in 2015–present (Analytics/3-and-D era) |
| `season_flag` | `seasons.aberrant` | 1 for aberrant seasons: 2012 (lockout, 66 games), 2020 (bubble, no HCA) |

---

## Features Used by Each Model

### Series Winner Model (Stacking Ensemble)
**Code:** `src/nba_predictor/models/ensemble.py:get_feature_cols()`

Uses `features.matchup` + `features.meta` + `higher_*` and `lower_*` injury flags from `conf/config.yaml`:

```
Matchup deltas (22):
  delta_NRtg, delta_BPM, delta_VORP, delta_adj_VORP, delta_eFG_pct,
  delta_DRtg, delta_Pace, delta_Experience, delta_Prior_playoff_win_pct,
  delta_Prior_deepest_round, delta_Recent_appearances,
  higher_seed_Prior_champion_3yr, lower_seed_Prior_champion_3yr,
  delta_L10_NRtg, delta_Star_BPM,
  seed_diff, home_court_advantage, conference_East, conference_West,
  H2H_win_pct, H2H_NRtg_avg, H2H_games_played

Era/meta (5):
  era_showtime, era_defensive, era_transition, era_analytics, season_flag

Injury — higher and lower seed (16 × 2 = 32 but only present cols used):
  higher_adj_VORP_sum, higher_Star_injured, higher_Second_star_injured,
  higher_Top2_Scorer_availability_avg, higher_Lost_top_scorer, higher_Lost_top2_scorers,
  higher_Scoring_VORP_available_pct, higher_Top_Rebounder_available,
  higher_Lost_top_rebounder, higher_Rebounding_capacity_pct,
  higher_Top_Playmaker_available, higher_Lost_top_playmaker,
  higher_Top_Defender_available, higher_Roster_VORP_available_pct,
  higher_Injured_player_count, higher_has_injury_data
  (same set for lower_*)
```

**Base models:**
- `LogisticRegression` (L2, C=1.0, StandardScaler)
- `XGBClassifier` (300 estimators, max_depth=4, lr=0.05, Optuna-tuned)
- `LGBMClassifier` (300 estimators, 31 leaves, lr=0.05, Optuna-tuned)

**Meta-learner:** `LogisticRegression` (C=0.5) + isotonic calibration via `CalibratedClassifierCV`

---

### Series Length Model (LightGBM Multiclass)
**Code:** `src/nba_predictor/models/series_length.py`
**Saved as:** `models/trained/lgbm_length.pkl`

Trained on all features present in `series_dataset.parquet` that the model selects via `feature_name_()`. Primary discriminating features (from notebook 07 outputs):

| Feature | Importance (indicative) | Why it matters |
|---------|------------------------|----------------|
| `delta_NRtg` | High | Overall quality gap drives sweep likelihood |
| `seed_diff` | High | Large seed gap → shorter series |
| `delta_VORP` | Medium | Roster depth gap |
| `delta_Star_BPM` | Medium | Star player gap — dominant star = shorter series |
| `H2H_win_pct` | Medium | Familiarity can reduce variance |
| `delta_Win_pct` | Medium | Season record gap |
| `era_analytics` | Low | Modern era has fewer blowout sweeps |

**Classes:** 4, 5, 6, 7 games
**Class weighting:** `class_weight='balanced'` (7-game series are most common, sweeps are rare)

---

## Feature Engineering Pipeline

```
Step 1 — Raw team stats
  src: data/raw/bball_ref/team_stats/team_advanced_all.parquet
  builder: src/nba_predictor/features/team_features.py

Step 2 — Era normalization (z-score within season)
  builder: src/nba_predictor/features/era_normalizer.py
  output cols: *_norm (e.g. NRtg_norm, ORtg_norm, DRtg_norm, Pace_norm)

Step 3 — Momentum features (last 10/20 game rolling window)
  src: data/raw/nba_api/team_game_logs/team_game_logs_all.parquet
  builder: src/nba_predictor/features/team_features.py
  output cols: L10_*, L20_*, current_win_streak

Step 4 — Playoff history features
  src: data/raw/bball_ref/playoff_series/playoff_series_all.parquet (historical)
  builder: src/nba_predictor/features/team_features.py
  output cols: Playoff_experience_years, Prior_playoff_win_pct, Prior_deepest_round,
               Prior_champion_3yr, Prior_playoff_appearances_2yr

Step 5 — Player aggregates
  src: data/raw/bball_ref/player_stats/player_advanced_all.parquet
  builder: src/nba_predictor/features/player_features.py
  output: data/processed/player_season_features.parquet

Step 6 — Injury-adjusted features
  src: player_season_features.parquet + (optional) injury_reports/
  builder: src/nba_predictor/features/injury_features.py
  output: data/processed/injury_adjusted.parquet

Step 7 — Matchup construction (one row per series)
  src: team_season_features + player_season_features + injury_adjusted + game_logs
  builder: src/nba_predictor/features/matchup_features.py:build_matchup_dataset()
  output: data/processed/series_dataset.parquet
```

---

## Notes

- **Era normalization**: All `*_norm` features are z-scored *within each season*, not globally. This ensures comparability across the 1984–2025 range where league-wide stats shifted dramatically (pace, 3-point rate, scoring). See `notebooks/05_era_normalization.ipynb`.
- **Missing values**: Features that require game logs (momentum, H2H) are NaN for seasons before 1996 or teams not in the NBA API data. The models fill NaN with 0 (`fillna(0)`).
- **Injury features**: All availability values default to 1.0 (fully healthy) because live injury report data was not fetched. The `has_injury_data` flag is 0 for all rows. See `DATA_DICTIONARY.md` for details.
- **Feature selection**: Only columns that exist in the DataFrame are used (`[c for c in all_feat if c in df.columns]`). This makes the pipeline robust to missing or renamed columns.
