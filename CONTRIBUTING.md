# Contributing

## Setup

```bash
make env
conda activate nba-predictor
make fetch-smoke   # fetch 2 seasons for fast testing
make process
make test
```

## Adding a New Data Source

1. Create a loader in `src/nba_predictor/data/your_loader.py`
2. Add a `__main__` block for CLI usage
3. Add the fetch step to `Makefile` under `fetch:`
4. Update `feature_pipeline.py` to load your data
5. Add schema validation in `data_validator.py`

## Adding a New Feature

1. Add the feature computation to the appropriate module in `src/nba_predictor/features/`
2. Register the feature name in `conf/config.yaml` under `features:`
3. Run `make process` to rebuild `series_dataset.parquet`
4. Add a sanity check in `notebooks/04_feature_engineering_validation.ipynb`
5. Write a unit test in `tests/features/`

## Adding a New Model

1. Create a module in `src/nba_predictor/models/your_model.py`
2. Implement `main()` for CLI training
3. Add a `make train` step in `Makefile`
4. Log results to an MLflow experiment
5. Compare to the logistic regression baseline in notebook 06

## Code Style

```bash
make lint      # check style
make format    # auto-format
```

Notebooks must have outputs stripped before committing (enforced by `nbstripout` pre-commit hook).

## Testing

```bash
make test              # run all unit tests
make test-coverage     # with coverage report
```

Tests live in `tests/`. Each subpackage has a mirror: `tests/data/`, `tests/features/`, `tests/models/`, `tests/evaluation/`.
