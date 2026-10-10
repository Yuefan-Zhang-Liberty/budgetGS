"""Smoke tests for the BudgetGS package."""

import budgetgs


def test_version() -> None:
    """Return the package version as a non-empty string."""
    assert isinstance(budgetgs.__version__, str)
    assert len(budgetgs.__version__) > 0


def test_cli_parser_builds() -> None:
    """Build the CLI parser and enumerate its subcommands."""
    from budgetgs.cli import build_parser

    parser = build_parser()
    names = {action.dest for action in parser._subparsers._actions[1]._choices_actions}  # noqa: SLF001
    assert {'train', 'fit', 'ledger', 'serialize'} <= names
