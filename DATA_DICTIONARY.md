# Data Dictionary — Stored Files

All paths are relative to the project root. Raw data is fetched by `make fetch`; processed files are built by `make process`. Neither is committed to git — both are fully reproducible.

---

## Raw Data

### `data/raw/bball_ref/team_stats/team_advanced_all.parquet`
**Source:** Basketball Reference team advanced stats pages (scraped via `src/nba_predictor/data/bball_ref_scraper.py`)
**Coverage:** 1984–2026, one row per team per season (~30 teams × 43 seasons = ~1,223 rows)

| Column | Type | Description |
|--------|------|-------------|
| Rk | int | Row rank on the source page |
| Team | str | Full team name (e.g. "Boston Celtics") |
| Age | float | Average player age |
| W | int | Regular-season wins |
| L | int | Regular-season losses |
| PW | int | Pythagorean expected wins |
| PL | int | Pythagorean expected losses |
| MOV | float | Margin of victory (points per game) |
| SOS | float | Strength of schedule |
| SRS | float | Simple Rating System (MOV adjusted for SOS) |
| ORtg | float | Offensive rating (points scored per 100 possessions) |
| DRtg | float | Defensive rating (points allowed per 100 possessions) |
| NRtg | float | Net rating (ORtg − DRtg) |
| Pace | float | Possessions per 48 minutes |
| FTr | float | Free throw rate (FTA / FGA) |
| 3PAr | float | Three-point attempt rate (3PA / FGA) |
| TS% | float | True shooting percentage |
| eFG% | float | Effective field-goal percentage |
| TOV% | float | Turnover rate (per 100 plays) |
| ORB% | float | Offensive rebounding percentage |
| FT/FGA | float | Free throws made per field-goal attempt |
| opp_eFG% | float | Opponent effective FG% |
| opp_TOV% | float | Opponent turnover rate |
| DRB% | float | Defensive rebounding percentage |
| opp_FT/FGA | float | Opponent FT/FGA |
| Arena | str | Home arena name |
| Attend. | int | Total season attendance |
| Attend./G | float | Average attendance per game |
| Team_abbrev | str | Canonical 3-letter abbreviation (bball-ref standard, e.g. "BOS") |
| season | int | NBA season end year (e.g. 2024 = 2023–24 season) |

---

### `data/raw/bball_ref/player_stats/player_advanced_all.parquet`
**Source:** Basketball Reference player advanced stats (per-season)
**Coverage:** 1984–2026, one row per player-season (~21,674 rows)

| Column | Type | Description |
|--------|------|-------------|
| Player | str | Player name |
| Pos | str | Position (PG/SG/SF/PF/C) |
| Age | int | Age at season start |
| Tm | str | Team abbreviation |
| G | int | Games played |
| MP | int | Minutes played |
| PER | float | Player Efficiency Rating |
| TS% | float | True shooting percentage |
| 3PAr | float | 3-point attempt rate |
| FTr | float | Free throw rate |
| ORB% | float | Offensive rebound % |
| DRB% | float | Defensive rebound % |
| TRB% | float | Total rebound % |
| AST% | float | Assist % |
| STL% | float | Steal % |
| BLK% | float | Block % |
| TOV% | float | Turnover % |
| USG% | float | Usage rate |
| OWS | float | Offensive win shares |
| DWS | float | Defensive win shares |
| WS | float | Win shares |
| WS/48 | float | Win shares per 48 minutes |
| OBPM | float | Offensive Box Plus/Minus |
| DBPM | float | Defensive Box Plus/Minus |
| BPM | float | Box Plus/Minus |
| VORP | float | Value Over Replacement Player |
| Team_abbrev | str | Canonical 3-letter abbreviation |
| season | int | Season end year |

---

### `data/raw/bball_ref/playoff_series/playoff_series_all.parquet`
**Source:** Basketball Reference playoff series results
**Coverage:** 1984–2025, one row per series (~345 series from 2003 in training set)

| Column | Type | Description |
|--------|------|-------------|
| season | int | Season end year |
| round | str | Round name (e.g. "Eastern Conference First Round") |
| team_a | str | Higher-seeded team abbreviation |
| team_b | str | Lower-seeded team abbreviation |
| series_winner | str | Winning team abbreviation |
| series_length | int | Number of games played (4–7) |

---

### `data/raw/nba_api/team_game_logs/team_game_logs_all.parquet`
**Source:** NBA Stats API (`LeagueGameLog` endpoint) via `src/nba_predictor/data/nba_api_fetcher.py`
**Coverage:** 1996–2026, one row per team per game (~73,472 rows)

| Column | Type | Description |
|--------|------|-------------|
| season | int | Season end year |
| SEASON_ID | str | NBA season string (e.g. "22024") |
| TEAM_ID | int | NBA internal team ID |
| TEAM_ABBREVIATION | str | NBA API abbreviation (may differ from bball-ref; normalized in pipeline) |
| TEAM_NAME | str | Full team name |
| GAME_ID | str | Unique game identifier |
| GAME_DATE | str | Date string (YYYY-MM-DD) |
| MATCHUP | str | Matchup string (e.g. "BOS vs. MIA" or "BOS @ MIA") |
| WL | str | Win/Loss ("W" or "L") |
| MIN | int | Minutes played |
| PTS | int | Points scored |
| FGM / FGA / FG_PCT | int/float | Field goals made/attempted/percentage |
| FG3M / FG3A / FG3_PCT | int/float | 3-point made/attempted/percentage |
| FTM / FTA / FT_PCT | int/float | Free throws made/attempted/percentage |
| OREB / DREB / REB | int | Offensive/defensive/total rebounds |
| AST | int | Assists |
| STL | int | Steals |
| BLK | int | Blocks |
| TOV | int | Turnovers |
| PF | int | Personal fouls |
| Team_abbrev | str | Normalized canonical abbreviation (post-processing) |

> **Note:** NBA API game logs do not include PLUS_MINUS. The pipeline derives it as `PTS − opponent_PTS` by self-joining on `GAME_ID`.

---

### `data/raw/nba_api/injury_reports/` *(not currently populated)*
**Source:** NBA Stats API injury reports
**Status:** Pipeline is wired but injury report fetching was not run. All `has_injury_data` flags are 0. All availability features default to 1.0 (fully healthy). See notebook 03 for details.

---

## Processed Data

### `data/processed/team_season_features.parquet`
**Built by:** `src/nba_predictor/features/feature_pipeline.py` → `src/nba_predictor/features/team_features.py`
**Coverage:** 1984–2026, one row per team per season (1,223 rows × 63 columns)

Includes all raw team stats plus the following engineered columns:

| Column | Type | Description |
|--------|------|-------------|
| Team_abbrev | str | Canonical 3-letter abbreviation |
| season | int | Season end year |
| Win_pct | float | W / (W + L) |
| ORtg_norm | float | ORtg z-scored within season |
| DRtg_norm | float | DRtg z-scored within season (higher = worse defense) |
| NRtg_norm | float | NRtg z-scored within season |
| Pace_norm | float | Pace z-scored within season |
| eFG%_norm | float | eFG% z-scored within season |
| TOV%_norm | float | TOV% z-scored within season |
| ORB%_norm | float | ORB% z-scored within season |
| DRB%_norm | float | DRB% z-scored within season |
| FT/FGA_norm | float | FT/FGA z-scored within season |
| opp_eFG%_norm | float | Opponent eFG% z-scored within season |
| opp_TOV%_norm | float | Opponent TOV% z-scored within season |
| SRS_norm | float | SRS z-scored within season |
| MOV_norm | float | MOV z-scored within season |
| era_showtime | int | 1 if season in 1984–1994 |
| era_defensive | int | 1 if season in 1995–2004 |
| era_transition | int | 1 if season in 2005–2014 |
| era_analytics | int | 1 if season in 2015–present |
| season_flag | int | 1 for aberrant seasons (2012 lockout, 2020 bubble) |
| Playoff_experience_years | int | Playoff appearances in prior 5 seasons |
| Prior_playoff_win_pct | float | Series win % over prior 3 playoff seasons |
| Prior_deepest_round | int | Max round reached in last 2 playoff appearances (0=none, 1=R1, 2=R2, 3=CF, 4=Finals loss, 5=champion) |
| Prior_champion_3yr | int | 1 if won championship in any of prior 3 seasons |
| Prior_playoff_appearances_2yr | int | Playoff appearances in prior 2 seasons (0–2) |
| L10_NRtg | float | Mean net rating (PLUS_MINUS proxy) in last 10 regular-season games |
| L10_Win_pct | float | Win % in last 10 regular-season games |
| L10_home_win_pct | float | Home win % in last 10 games |
| L10_away_win_pct | float | Away win % in last 10 games |
| current_win_streak | int | Current win streak (positive) or losing streak (negative) |
| L20_NRtg | float | Mean net rating in last 20 regular-season games |
| L20_Win_pct | float | Win % in last 20 regular-season games |
| L10_NRtg_delta | float | L10_NRtg minus season-average NRtg (trending up = positive) |
| L20_NRtg_delta | float | L20_NRtg minus season-average NRtg |

---

### `data/processed/player_season_features.parquet`
**Built by:** `src/nba_predictor/features/player_features.py`
**Coverage:** 1984–2026, one row per team per season (1,223 rows × 12 columns)

| Column | Type | Description |
|--------|------|-------------|
| season | int | Season end year |
| Team_abbrev | str | Canonical 3-letter abbreviation |
| team_VORP_sum | float | Sum of VORP for all rostered players |
| team_BPM_weighted_avg | float | BPM weighted by minutes played |
| team_WS48_weighted_avg | float | WS/48 weighted by minutes played |
| Top3_VORP_sum | float | Sum of VORP for the top-3 players by VORP |
| Star_player_BPM | float | BPM of the highest-BPM player |
| Has_AllNBA_player | int | 1 if any player has BPM > 5.0 |
| Top8_WS48_avg | float | Mean WS/48 of the top-8 players by minutes (rotation depth) |
| Guard_VORP | float | Sum of VORP for PG/SG players |
| Forward_VORP | float | Sum of VORP for SF/PF players |
| Center_VORP | float | Sum of VORP for C players |

---

### `data/processed/injury_adjusted.parquet`
**Built by:** `src/nba_predictor/features/injury_features.py`
**Coverage:** 1984–2026, one row per team per season (1,223 rows × 18 columns)
**Note:** All values are defaults (healthy) because live injury report data was not fetched.

| Column | Type | Description |
|--------|------|-------------|
| season | int | Season end year |
| Team_abbrev | str | Canonical 3-letter abbreviation |
| has_injury_data | int | 1 if live injury data was available for this team-season |
| adj_VORP_sum | float | Availability-weighted VORP sum (VORP × availability_fraction) |
| Star_injured | int | 1 if the highest-VORP player is below the `injured_threshold` (0.5) |
| Second_star_injured | int | 1 if the 2nd-highest-VORP player is injured |
| Top2_Scorer_availability_avg | float | Mean availability of the top-2 scorers |
| Lost_top_scorer | int | 1 if the top scorer is lost (availability < 0.5) |
| Lost_top2_scorers | int | 1 if both top-2 scorers are lost |
| Scoring_VORP_available_pct | float | % of scoring VORP that is available |
| Top_Rebounder_available | int | 1 if the top rebounder is available |
| Lost_top_rebounder | int | 1 if the top rebounder is lost |
| Rebounding_capacity_pct | float | % of rebounding capacity available |
| Top_Playmaker_available | int | 1 if the top AST% player is available |
| Lost_top_playmaker | int | 1 if the top playmaker is lost |
| Top_Defender_available | int | 1 if the top DBPM player is available |
| Roster_VORP_available_pct | float | Total available VORP / total roster VORP |
| Injured_player_count | int | Count of players below injured_threshold |

---

### `data/processed/series_dataset.parquet`
**Built by:** `src/nba_predictor/features/matchup_features.py`
**Coverage:** 2003–2025 (best-of-7 era), 345 series × 58 columns
**This is the primary training dataset for all models.**

See the [Feature Dictionary](FEATURES.md) for full column descriptions.

---

### `data/predictions/2026/`

| File | Description |
|------|-------------|
| `bracket_input.csv` | 8 R1 matchup feature rows used as model input |
| `bracket_teams.csv` | One row per playoff team with all raw feature values |
| `bracket_output.csv` | Full greedy bracket: 15 series (R1 × 8, R2 × 4, CF × 2, Finals × 1) with win probability and series length distribution |
| `champion_probabilities.csv` | Per-team advancement probabilities from 10,000 Monte Carlo simulations |

#### `bracket_output.csv` columns

| Column | Type | Description |
|--------|------|-------------|
| round | str | `first_round`, `second_round`, `conf_finals`, `nba_finals` |
| conference | str | `East`, `West`, or `Finals` |
| higher_seed | str | Lower seed number (better team) |
| lower_seed | str | Higher seed number |
| predicted_winner | str | Model's pick |
| p_winner | float | Win probability of the predicted winner |
| p_higher_seed_wins | float | P(higher-seeded team wins) |
| expected_length | float | Probability-weighted expected game count |
| modal_length | int | Most likely game count (4/5/6/7) — use this for bracket picks |
| p_length_4 | float | P(series ends in 4 games) |
| p_length_5 | float | P(series ends in 5 games) |
| p_length_6 | float | P(series ends in 6 games) |
| p_length_7 | float | P(series ends in 7 games) |

#### `champion_probabilities.csv` columns

| Column | Type | Description |
|--------|------|-------------|
| team | str | Team abbreviation |
| p_r2 | float | P(advances to Round 2) |
| p_conf_finals | float | P(reaches Conference Finals) |
| p_finals | float | P(reaches NBA Finals) |
| p_champion | float | P(wins championship) |

---

## Reports

### `reports/backtest_results.csv`
Walk-forward CV results written by `src/nba_predictor/evaluation/backtesting.py`.

| Column | Description |
|--------|-------------|
| fold | CV fold index (0-based) |
| test_season | Season used as test fold |
| n_train_series | Series in training window |
| n_test_series | Series in test window (always 15) |
| accuracy | Fraction of series correctly predicted |
| naive_accuracy | Accuracy of always-pick-higher-seed baseline |
| accuracy_vs_naive | accuracy − naive_accuracy |
| log_loss | Binary cross-entropy |
| brier_score | Mean squared probability error |
| upset_recall | Recall on actual upsets (lower-seed wins) |
| upset_precision | Precision on predicted upsets |
| n_upsets | Actual upsets in test fold |
| n_upset_predictions | Model upset predictions in test fold |
| ece | Expected Calibration Error |
