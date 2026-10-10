"""Command-line interface for BudgetGS.

The CLI exposes four commands: ``train``, ``fit``, ``ledger`` and ``serialize``.
Phase 1 ships the argument contracts; the implementations land in later phases.
"""

from __future__ import annotations

import argparse
import sys

#---------------------------------------------------------------------
# Argument parsing
#---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level argument parser with its subcommands."""
    parser = argparse.ArgumentParser(
        prog='budgetgs',
        description='Train 3D Gaussian Splatting scenes onto an explicit byte budget.',
    )
    subcommands = parser.add_subparsers(dest='command', required=True)

    train_parser = subcommands.add_parser('train', help='train a gsplat scene and checkpoint it')
    train_parser.add_argument('--scene', required=True, help='path to a COLMAP scene directory')
    train_parser.add_argument('--steps', type=int, default=30000, help='number of training iterations')
    train_parser.add_argument('--ckpt-every', type=int, default=1000, help='checkpoint interval in iterations')
    train_parser.add_argument('--out', default='runs', help='output directory for checkpoints and logs')

    fit_parser = subcommands.add_parser('fit', help='fit a checkpoint onto a byte budget')
    fit_parser.add_argument('--ckpt', required=True, help='path to a trained checkpoint')
    fit_parser.add_argument('--budget-mb', type=float, required=True, help='target size in MB (1 MB = 10^6 bytes)')
    fit_parser.add_argument('--tolerance', type=float, default=0.01, help='hit tolerance, max(1 %, 256 KB)')

    ledger_parser = subcommands.add_parser('ledger', help='predict the serialized size of a checkpoint')
    ledger_parser.add_argument('--ckpt', required=True, help='path to a trained checkpoint')

    serialize_parser = subcommands.add_parser('serialize', help='write a checkpoint to disk and report bytes')
    serialize_parser.add_argument('--ckpt', required=True, help='path to a trained checkpoint')
    serialize_parser.add_argument('--out', required=True, help='output file path')

    return parser

#---------------------------------------------------------------------
# Command handlers
#---------------------------------------------------------------------


def _not_implemented(command_name: str) -> None:
    """Exit with an error stating that a command is not implemented yet."""
    print(f'budgetgs {command_name}: not implemented yet (Phase 1 skeleton)', file=sys.stderr)
    raise SystemExit(1)


def train(args: argparse.Namespace) -> None:
    """Train a gsplat scene, checkpointing every ``--ckpt-every`` iterations."""
    del args
    _not_implemented('train')


def fit(args: argparse.Namespace) -> None:
    """Land a checkpoint on ``--budget-mb`` bytes by bisecting the quantization step."""
    del args
    _not_implemented('fit')


def ledger(args: argparse.Namespace) -> None:
    """Print the predicted serialized size of a checkpoint, attribute by attribute."""
    del args
    _not_implemented('ledger')


def serialize(args: argparse.Namespace) -> None:
    """Write a checkpoint to disk and print the actual byte count."""
    del args
    _not_implemented('serialize')

#---------------------------------------------------------------------
# Entry point
#---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to the matching command handler."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        'train': train,
        'fit': fit,
        'ledger': ledger,
        'serialize': serialize,
    }
    handlers[args.command](args)


if __name__ == '__main__':
    main()
