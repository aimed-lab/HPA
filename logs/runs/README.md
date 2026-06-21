# Runs

Each completed training run is archived here by the agent as a timestamped folder.

Structure per run:
- `metrics.jsonl` — per-epoch metrics
- `config.json` — hyperparameters used
- `status.json` — final training status
- `meta.json` — summary (model, best_f1, epochs_done)
- `stdout.log` — raw training output
