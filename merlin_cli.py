#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merlin TTS Command-Line Interface (CLI) -- Alexander the Summoner's Fork.

Wraps the Edinburgh Merlin training backend
(Centre for Speech Technology Research, University of Edinburgh)
in a config-driven CLI: one YAML profile in, overrides on the command line.

Usage:
    python merlin_cli.py config.yaml [--epochs 50] [--lr 0.001] [--batch_size 128]
    python merlin_cli.py config.yaml --dry-run
    python merlin_cli.py --make-template my_config.yaml

Built under Alucard's charge. The warden reports the real state:
what runs, what is missing, and what it would take.
"""

import os
import sys
import argparse
import logging
import logging.config

try:
    import yaml
except ImportError:
    print("[ERROR] PyYAML is required: pip install pyyaml")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Backend contract
# ---------------------------------------------------------------------------
# The training backend is the Edinburgh Merlin codebase (not shipped here).
# It must be importable as `training_backend` and provide:
#
#     train_DNN(job) -> None
#
# where `job` is a dict with:
#     'config'   : full resolved config dict (CLI overrides already applied)
#     'train_x'  : list of resolved training input files
#     'train_y'  : list of resolved training target files
#     'valid_x'  : list of resolved validation input files
#     'valid_y'  : list of resolved validation target files
#     'work_dir' : output directory for models, logs, checkpoints
#
# Anything else the backend needs lives inside `job['config']`.
# ---------------------------------------------------------------------------


def init_global_logging():
    """Configures explicit standard console tracking layouts."""
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
        },
        'loggers': {
            '': {
                'handlers': ['console'],
                'level': 'INFO',
                'propagate': True
            },
            'merlin_cli.train': {
                'handlers': ['console'],
                'level': 'DEBUG',
                'propagate': False
            },
        }
    }
    logging.config.dictConfig(logging_config)


def parse_arguments(argv=None):
    """Parses the YAML profile path plus framework CLI parameter overrides."""
    parser = argparse.ArgumentParser(
        description="Alexander the Summoner's Merlin TTS Core Executor")
    parser.add_argument('config', nargs='?',
                        help="Path to the target configuration profile (YAML)")
    # Overrides popular inside the Summoner pipeline variants
    parser.add_argument('--epochs', type=int, default=None,
                        help="Override baseline training epochs count")
    parser.add_argument('--lr', type=float, default=None,
                        help="Override baseline global learning rate")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Override network mini-batch processing scale")
    # Warden's additions: see before you strike
    parser.add_argument('--dry-run', action='store_true',
                        help="Validate the config and resolve file lists without training")
    parser.add_argument('--make-template', metavar='PATH', default=None,
                        help="Write a starter config.yaml to PATH and exit")
    args = parser.parse_args(argv)

    if args.make_template:
        return args
    if not args.config:
        parser.error("config.yaml is required (unless --make-template is used)")
    if not os.path.exists(args.config):
        print("[ERROR] Configuration file not found: %s" % args.config)
        sys.exit(1)
    return args


def resolve_file_list(base_dir, scp_or_list_setting):
    """Framework-native dynamic file resolver.

    Handles raw lists, single .scp/.txt manifests, or direct directory scans.
    """
    if not scp_or_list_setting:
        return []
    # Case A: already a structured list of files
    if isinstance(scp_or_list_setting, list):
        return [f if os.path.isabs(f) else os.path.join(base_dir, f)
                for f in scp_or_list_setting]
    # Case B: points to a .scp or .txt manifest file
    if isinstance(scp_or_list_setting, str) and (
            scp_or_list_setting.endswith('.scp') or scp_or_list_setting.endswith('.txt')):
        manifest = (scp_or_list_setting if os.path.isabs(scp_or_list_setting)
                    else os.path.join(base_dir, scp_or_list_setting))
        resolved = []
        if os.path.exists(manifest):
            with open(manifest, 'r') as fh:
                for line in fh:
                    entry = line.strip()
                    if entry:
                        resolved.append(entry if os.path.isabs(entry)
                                        else os.path.join(base_dir, entry))
        return resolved
    # Case C: raw path -- directory scan (.bin/.cmp) or single file.
    # A path that does not exist resolves to nothing: the warden
    # reports the real state, never a phantom file.
    target = (scp_or_list_setting if os.path.isabs(scp_or_list_setting)
              else os.path.join(base_dir, scp_or_list_setting))
    if os.path.isdir(target):
        return sorted(os.path.join(target, f) for f in os.listdir(target)
                      if f.endswith('.bin') or f.endswith('.cmp'))
    if os.path.isfile(target):
        return [target]
    return []


def load_framework_config(config_path):
    """Safely extracts the parameters dictionary from the target YAML file."""
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("[ERROR] Failed to parse YAML config: %s" % exc)
            sys.exit(1)
    if not isinstance(config, dict):
        print("[ERROR] Config must be a YAML mapping at the top level: %s" % config_path)
        sys.exit(1)
    return config


def apply_overrides(config, args):
    """Folds CLI overrides into the resolved config dict (in place)."""
    training = config.setdefault('training', {})
    if args.epochs is not None:
        training['epochs'] = args.epochs
    if args.lr is not None:
        training['learning_rate'] = args.lr
    if args.batch_size is not None:
        training['batch_size'] = args.batch_size
    return config


def build_job(config):
    """Resolves file lists and assembles the training job dict."""
    data_setup = config.get('data_setup', {})
    base_dir = data_setup.get('base_data_dir', '')
    job = {
        'config': config,
        'train_x': resolve_file_list(base_dir, data_setup.get('train_x_files')),
        'train_y': resolve_file_list(base_dir, data_setup.get('train_y_files')),
        'valid_x': resolve_file_list(base_dir, data_setup.get('valid_x_files')),
        'valid_y': resolve_file_list(base_dir, data_setup.get('valid_y_files')),
        'work_dir': config.get('output', {}).get('work_dir', './merlin_run'),
    }
    return job


def report_job(job, logger):
    """Logs the resolved plan. The warden shows the field before the battle."""
    training = job['config'].get('training', {})
    logger.info("== Merlin training plan ==")
    logger.info("work_dir      : %s", job['work_dir'])
    logger.info("epochs        : %s", training.get('epochs', '(backend default)'))
    logger.info("learning_rate : %s", training.get('learning_rate', '(backend default)'))
    logger.info("batch_size    : %s", training.get('batch_size', '(backend default)'))
    for key in ('train_x', 'train_y', 'valid_x', 'valid_y'):
        files = job[key]
        logger.info("%-13s : %d file(s)", key, len(files))
        for f in files[:5]:
            logger.info("    - %s", f)
        if len(files) > 5:
            logger.info("    ... and %d more", len(files) - 5)


def check_job(job):
    """Validates the resolved job. Returns a list of problem strings (empty = clean)."""
    problems = []
    if not job['train_x']:
        problems.append("train_x_files resolved to zero files")
    if not job['train_y']:
        problems.append("train_y_files resolved to zero files")
    if job['train_x'] and job['train_y'] and len(job['train_x']) != len(job['train_y']):
        problems.append("train_x (%d) and train_y (%d) counts differ"
                        % (len(job['train_x']), len(job['train_y'])))
    if job['valid_x'] and job['valid_y'] and len(job['valid_x']) != len(job['valid_y']):
        problems.append("valid_x (%d) and valid_y (%d) counts differ"
                        % (len(job['valid_x']), len(job['valid_y'])))
    return problems


TEMPLATE = """\
# Merlin TTS -- Alexander the Summoner's Fork :: starter profile
# Usage: python merlin_cli.py this_file.yaml --dry-run

data_setup:
  # Base directory that relative file paths resolve against
  base_data_dir: "./data"
  # Each entry accepts: a list of files, a .scp/.txt manifest, or a directory
  # (directories are scanned for .bin / .cmp files)
  train_x_files: "train/x"
  train_y_files: "train/y"
  valid_x_files: "valid/x"
  valid_y_files: "valid/y"

training:
  epochs: 50
  learning_rate: 0.001
  batch_size: 128

output:
  work_dir: "./merlin_run"
"""


def write_template(path):
    with open(path, 'w') as fh:
        fh.write(TEMPLATE)


def import_backend():
    """Imports the Edinburgh training backend. Fails fast and says what's missing."""
    try:
        from training_backend import train_DNN
        return train_DNN
    except ImportError:
        print("[ERROR] training_backend not found.")
        print("        The Edinburgh Merlin codebase must be importable")
        print("        (clone it and set PYTHONPATH, or pip install it).")
        print("        Nothing was trained. Use --dry-run to validate configs without it.")
        sys.exit(2)


def main(argv=None):
    init_global_logging()
    logger = logging.getLogger("merlin_cli.main")
    args = parse_arguments(argv)

    if args.make_template:
        write_template(args.make_template)
        logger.info("Wrote starter config template to: %s", args.make_template)
        return 0

    logger.info("Loading runtime parameters from: %s", args.config)
    config = load_framework_config(args.config)
    apply_overrides(config, args)
    job = build_job(config)
    report_job(job, logger)

    problems = check_job(job)
    if problems:
        for p in problems:
            logger.error("Plan problem: %s", p)
        if not args.dry_run:
            print("[ERROR] Refusing to train with an invalid plan. "
                  "Fix the config or inspect with --dry-run.")
            return 1

    if args.dry_run:
        logger.info("Dry run complete -- no training started.")
        return 0

    train_DNN = import_backend()
    os.makedirs(job['work_dir'], exist_ok=True)
    train_logger = logging.getLogger("merlin_cli.train")
    train_logger.info("Handing the job to the training backend...")
    try:
        train_DNN(job)
    except Exception as exc:  # noqa: BLE001 -- report, don't swallow
        logger.error("Backend failed: %s: %s", type(exc).__name__, exc)
        return 3
    logger.info("Training finished. Models in: %s", job['work_dir'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
