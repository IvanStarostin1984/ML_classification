from __future__ import annotations

import argparse

from .models import logreg, cart


def main(args: list[str] | None = None) -> None:
    """CLI entry point training the logistic and tree models."""
    parser = argparse.ArgumentParser(description='Train model pipelines')
    parser.add_argument(
        '--model',
        '-m',
        action='append',
        choices=['logreg', 'cart'],
        help='models to train; defaults to both',
    )
    ns = parser.parse_args(args)
    models = ns.model or ['logreg', 'cart']

    if 'logreg' in models:
        logreg.main()
    if 'cart' in models:
        cart.main()


if __name__ == '__main__':
    main()
