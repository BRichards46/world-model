"""Wait for the morning Norgate update to finish writing before the macro run reads it."""

from __future__ import annotations

import argparse

from world_model.config import load_yaml, project_root
from world_model.data.norgate import wait_for_settled_catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-roots", default=str(project_root() / "config" / "data_roots.yml"))
    parser.add_argument("--quiet-minutes", type=float, default=3.0)
    parser.add_argument("--timeout-minutes", type=float, default=60.0)
    args = parser.parse_args(argv)
    roots = load_yaml(args.data_roots)
    settled = wait_for_settled_catalog(
        roots["norgate_metadata"], quiet_seconds=args.quiet_minutes * 60.0, timeout_seconds=args.timeout_minutes * 60.0
    )
    # Proceed either way: a timed-out wait still leaves the per-read retries as a second line of defence.
    print("Norgate catalog settled." if settled else "Proceeding without a settled Norgate catalog.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
