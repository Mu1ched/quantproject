"""Driver — loops N rounds under a budget cap, writes PID + log files so
the dashboard can monitor."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config, paths, runner


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(log_path, encoding='utf-8'))
    except Exception:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s :: %(message)s',
        handlers=handlers,
        force=True,
    )


def _write_pid_file(pid_path: Path) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))


def _remove_pid_file(pid_path: Path) -> None:
    try:
        pid_path.unlink(missing_ok=True)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='agent.hypothesis_gen.driver')
    parser.add_argument('--max-rounds', type=int, default=config.MAX_ROUNDS)
    parser.add_argument('--budget',     type=float, default=config.BUDGET_USD,
                        help='Max cumulative cost in USD. Hard stop.')
    parser.add_argument('--model',      type=str, default=config.MODEL)
    parser.add_argument('--auto-sweep', action='store_true',
                        default=config.AUTO_SWEEP,
                        help='After each round, spawn a sweep on new strategies.')
    parser.add_argument('--dry-run',    action='store_true',
                        help='Build context + persist round folder, skip API.')
    args = parser.parse_args(argv)

    _setup_logging(paths.HYPGEN_LOG)
    log = logging.getLogger('driver')

    _write_pid_file(paths.HYPGEN_PID)
    log.info('hypothesis_gen.driver starting. pid=%d. project=%s. '
             'max_rounds=%d budget=$%.2f model=%s auto_sweep=%s dry_run=%s',
             os.getpid(), paths.THIS_PROJECT_NAME,
             args.max_rounds, args.budget, args.model,
             args.auto_sweep, args.dry_run)

    spent  = 0.0
    accepted_total = 0
    rounds_done    = 0
    try:
        for round_n in range(1, args.max_rounds + 1):
            if spent >= args.budget and not args.dry_run:
                log.info('Budget exhausted ($%.2f / $%.2f). Stopping.',
                         spent, args.budget)
                break

            log.info('--- ROUND %d / %d ---', round_n, args.max_rounds)
            try:
                result = runner.run_round(
                    round_n=round_n, model=args.model, dry_run=args.dry_run,
                )
            except Exception as e:
                log.exception('Round %d raised unhandled exception: %s',
                              round_n, e)
                continue

            spent += result.cost_usd
            rounds_done = round_n

            if result.status == 'ok':
                accepted_total += len(result.metas)
                log.info('Round %d OK. cost=$%.3f. cumulative=$%.2f/$%.2f. '
                         'accepted strategies: %s',
                         round_n, result.cost_usd, spent, args.budget,
                         [m['title'] for m in result.metas])
                if args.auto_sweep:
                    log.info('(auto-sweep requested — not implemented in this '
                             'version; run sweeps manually via the dashboard)')
            elif result.status == 'dry_run':
                log.info('Round %d dry-run OK. Round folder: %s',
                         round_n, result.round_dir)
            else:
                log.warning('Round %d FAILED (%s): %s',
                            round_n, result.status, result.error)

            time.sleep(config.ROUND_PAUSE_SEC)

        log.info('DONE. rounds=%d, accepted=%d, total cost=$%.2f',
                 rounds_done, accepted_total, spent)
        return 0
    finally:
        _remove_pid_file(paths.HYPGEN_PID)


if __name__ == '__main__':
    sys.exit(main())
