#!/usr/bin/env python3
"""Run the existing reranker_main with an explicit YAML path.

The supplied reranker_main binds its default YAML path at import time and has
no command-line parser. This wrapper changes only config selection and leaves
the training/evaluation implementation untouched.
"""

from __future__ import annotations

import argparse

import src.reranker_main as reranker_main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    original_loader = reranker_main.load_structured_cfg

    def load_selected_config(path=None):
        return original_loader(args.config)

    reranker_main.load_structured_cfg = load_selected_config
    reranker_main.main()


if __name__ == "__main__":
    main()
