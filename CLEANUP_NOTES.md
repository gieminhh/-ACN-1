# Cleanup Notes

## Active files

These files are part of the submitted project:

- `nazari_vrp/`
- `app.py`
- `run.py`
- `train_cvrp_final.py`
- `README.md`
- `PROJECT_SUMMARY_FOR_CLAUDE.md`
- `requirements.txt`
- `tests/test_nazari_vrp.py`

## Legacy files

The original repository was a full RL4CO source tree. It contains many
algorithms, configs, docs, notebooks, and tests that are not used by this CVRP
paper-aligned implementation.

The legacy folders are kept for reference only and are listed in `.claudeignore`
so an automated Claude review can focus on the active implementation.

## Removed from active flow

The previous RL4CO/PPO app was removed from the main flow because it made the
paper connection unclear. The active implementation now uses a Nazari-style
RNN + attention policy to learn a customer priority order, then applies Best
Insertion as the thesis method.
