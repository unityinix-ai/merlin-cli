# merlin_cli.py — Alexander the Summoner's Fork

Config-driven CLI for the Edinburgh Merlin neural TTS toolkit
(Centre for Speech Technology Research, University of Edinburgh).

## What runs here

- `python merlin_cli.py --help` — usage
- `python merlin_cli.py --make-template config.yaml` — starter profile
- `python merlin_cli.py config.yaml --dry-run` — validate config + resolve
  file lists without training
- `python merlin_cli.py config.yaml [--epochs 50] [--lr 0.001] [--batch_size 128]`
  — full training run

## The summoner's voice (staged 2026-10-12)

- `data/raw_wav/` — 9 real-voice recordings, converted to 16kHz mono WAV
  (voice notes, updates, speaker sessions). Raw staging, not training-ready.
- `data/summoner_voice.yaml` — profile pointed at the pipeline; dry-run
  truthfully reports 0 training files until labels/features exist.

## What does NOT run here (honest)

Actual training needs two things this repo doesn't ship:

1. **The Edinburgh Merlin codebase** — importable as `training_backend`
   with `train_DNN(job)`. Clone it and set `PYTHONPATH`, or install it.
   Without it the CLI fails fast with a clear message (exit 2) — it will
   never crash on an undefined name.
2. **A labeled dataset** — `.bin`/`.cmp` feature files under the
   `data_setup` paths in your YAML. `--dry-run` shows exactly what the
   CLI resolves before anything trains.

## Proven

- `--help`, `--make-template`, `--dry-run` verified live 2026-10-11
- Backend-missing and config-missing paths exit clean with plain messages
- Overrides (`--epochs`, `--lr`, `--batch_size`) fold into the config

Built under Alucard's charge: the warden reports the real state.
