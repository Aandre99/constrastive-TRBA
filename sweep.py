#!/usr/bin/env python3
"""
sweep.py — Automated hyperparameter sweep for contrastive training.

Distributes experiments across GPUs (default: 0, 1, 2), running one
experiment per GPU at a time. Tracks progress in a CSV for resumability.

Usage:
    python sweep.py                    # Run full sweep (117 experiments)
    python sweep.py --dry-run          # Preview all runs without executing
    python sweep.py --gpus 0 1         # Use only GPUs 0 and 1
    python sweep.py --only-baseline    # Run only baseline (no contrastive)
    python sweep.py --only-contrastive # Run only contrastive experiments
"""

import argparse
import csv
import itertools
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


# ──────────────────────────────────────────────────────────────
# Configuration — edit these to change the sweep parameters
# ──────────────────────────────────────────────────────────────

NUM_ITERS = [5000, 15000, 30000]
BATCH_SIZES = [32, 64, 128]
MARGINS = [0.5, 0.8, 1.0, 1.2]
LAMBDAS = [0.1, 0.3, 0.6]

DEFAULT_GPUS = [0, 1, 2]
EXP_NAME = "CTRBA"
LOG_DIR = "sweep_logs"
PROGRESS_FILE = os.path.join(LOG_DIR, "progress.csv")

PROGRESS_FIELDS = ["timestamp", "run_name", "gpu", "status", "duration_min"]

# ──────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────


@dataclass
class Experiment:
    """Represents a single training experiment."""
    num_iter: int
    batch_size: int
    contrastive: bool
    contrastive_margin: Optional[float] = None
    contrastive_lambda: Optional[float] = None

    @property
    def run_name(self) -> str:
        """Generate a descriptive run name for MLflow identification."""
        iter_k = self.num_iter // 1000
        if not self.contrastive:
            return f"base_iter{iter_k}k_bs{self.batch_size}"
        return (
            f"ctr_iter{iter_k}k_bs{self.batch_size}"
            f"_m{self.contrastive_margin}_lam{self.contrastive_lambda}"
        )


# Thread-safe lock for CSV writes
_csv_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────
# Experiment generation
# ──────────────────────────────────────────────────────────────


def generate_experiments(
    include_baseline: bool = True,
    include_contrastive: bool = True,
) -> List[Experiment]:
    """Generate all experiment combinations.

    Baseline:     NUM_ITER × BATCH_SIZE = 9 runs
    Contrastive:  NUM_ITER × BATCH_SIZE × MARGIN × LAMBDA = 108 runs
    Total:        117 runs
    """
    experiments = []

    # Baseline runs (no contrastive loss)
    if include_baseline:
        for num_iter, batch_size in itertools.product(NUM_ITERS, BATCH_SIZES):
            experiments.append(Experiment(
                num_iter=num_iter,
                batch_size=batch_size,
                contrastive=False,
            ))

    # Contrastive runs
    if include_contrastive:
        for num_iter, batch_size, margin, lam in itertools.product(
            NUM_ITERS, BATCH_SIZES, MARGINS, LAMBDAS,
        ):
            experiments.append(Experiment(
                num_iter=num_iter,
                batch_size=batch_size,
                contrastive=True,
                contrastive_margin=margin,
                contrastive_lambda=lam,
            ))

    return experiments


# ──────────────────────────────────────────────────────────────
# Progress tracking (thread-safe, CSV-based)
# ──────────────────────────────────────────────────────────────


def load_completed_runs() -> set:
    """Load set of run_names that already finished successfully."""
    completed = set()
    if not os.path.exists(PROGRESS_FILE):
        return completed
    with open(PROGRESS_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "DONE":
                completed.add(row["run_name"])
    return completed


def log_progress(run_name: str, gpu: int, status: str, duration_s: float = 0):
    """Append a row to the progress CSV (thread-safe)."""
    with _csv_lock:
        file_exists = os.path.exists(PROGRESS_FILE)
        with open(PROGRESS_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "run_name": run_name,
                "gpu": gpu,
                "status": status,
                "duration_min": f"{duration_s / 60:.1f}",
            })


# ──────────────────────────────────────────────────────────────
# Experiment execution
# ──────────────────────────────────────────────────────────────


def build_command(gpu_id: int, exp: Experiment) -> List[str]:
    """Build the train.sh command line for an experiment."""
    cmd = ["./train.sh"]

    if exp.contrastive:
        cmd.append("--contrastive")
        cmd.extend(["--contrastive-margin", str(exp.contrastive_margin)])
        cmd.extend(["--contrastive-lambda", str(exp.contrastive_lambda)])

    cmd.extend([
        "--device", str(gpu_id),
        "--num-iter", str(exp.num_iter),
        "--batch-size", str(exp.batch_size),
        "--run_name", exp.run_name,
    ])

    return cmd


def run_experiment(gpu_id: int, exp: Experiment) -> bool:
    """Execute a single experiment via train.sh. Returns True on success."""
    cmd = build_command(gpu_id, exp)
    log_file = os.path.join(LOG_DIR, f"{exp.run_name}.log")

    tag = "CTR" if exp.contrastive else "BASE"
    print(f"  [GPU {gpu_id}] [{tag}] Starting: {exp.run_name}")

    start = time.time()
    try:
        with open(log_file, "w") as lf:
            # Write the command at the top of the log for reference
            lf.write(f"# Command: {' '.join(cmd)}\n")
            lf.write(f"# Started: {datetime.now().isoformat()}\n")
            lf.write(f"# GPU: {gpu_id}\n")
            lf.write("# " + "=" * 60 + "\n\n")
            lf.flush()

            result = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
            )

        elapsed = time.time() - start

        if result.returncode == 0:
            log_progress(exp.run_name, gpu_id, "DONE", elapsed)
            print(
                f"  [GPU {gpu_id}] ✓ Done: {exp.run_name} "
                f"({elapsed / 60:.1f} min)"
            )
            return True
        else:
            log_progress(
                exp.run_name, gpu_id,
                f"FAIL(rc={result.returncode})", elapsed,
            )
            print(
                f"  [GPU {gpu_id}] ✗ Failed: {exp.run_name} "
                f"(rc={result.returncode}, see {log_file})"
            )
            return False

    except Exception as e:
        elapsed = time.time() - start
        log_progress(exp.run_name, gpu_id, f"ERROR({e})", elapsed)
        print(f"  [GPU {gpu_id}] ✗ Error: {exp.run_name}: {e}")
        return False


def gpu_worker(gpu_id: int, queue: List[Experiment]):
    """Process a queue of experiments sequentially on a single GPU."""
    total = len(queue)
    done = 0
    failed = 0

    for i, exp in enumerate(queue, 1):
        print(f"  [GPU {gpu_id}] === Run {i}/{total} ===")
        success = run_experiment(gpu_id, exp)
        if success:
            done += 1
        else:
            failed += 1

    print(
        f"\n  [GPU {gpu_id}] Queue finished: "
        f"{done} done, {failed} failed out of {total}"
    )


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for contrastive text recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview all runs without executing",
    )
    parser.add_argument(
        "--gpus", type=int, nargs="+", default=DEFAULT_GPUS,
        help=f"GPU IDs to use (default: {DEFAULT_GPUS})",
    )
    parser.add_argument(
        "--only-baseline", action="store_true",
        help="Run only baseline experiments (no contrastive)",
    )
    parser.add_argument(
        "--only-contrastive", action="store_true",
        help="Run only contrastive experiments",
    )
    args = parser.parse_args()

    gpus = args.gpus
    os.makedirs(LOG_DIR, exist_ok=True)

    # Determine which experiments to include
    include_baseline = not args.only_contrastive
    include_contrastive = not args.only_baseline

    all_experiments = generate_experiments(include_baseline, include_contrastive)

    # Filter out already completed runs (resumability)
    completed = load_completed_runs()
    pending = [e for e in all_experiments if e.run_name not in completed]

    n_baseline = sum(1 for e in all_experiments if not e.contrastive)
    n_contrastive = sum(1 for e in all_experiments if e.contrastive)
    n_pending_baseline = sum(1 for e in pending if not e.contrastive)
    n_pending_contrastive = sum(1 for e in pending if e.contrastive)

    print(f"\n{'=' * 60}")
    print(f"  Hyperparameter Sweep — Experiment: {EXP_NAME}")
    print(f"{'=' * 60}")
    print(f"  Total combinations:  {len(all_experiments)}")
    print(f"    Baseline:          {n_baseline}")
    print(f"    Contrastive:       {n_contrastive}")
    print(f"  Already completed:   {len(completed)}")
    print(f"  Pending:             {len(pending)}")
    print(f"    Baseline:          {n_pending_baseline}")
    print(f"    Contrastive:       {n_pending_contrastive}")
    print(f"  GPUs:                {gpus}")
    print(f"  Runs per GPU:        ~{len(pending) // max(len(gpus), 1)}")
    print(f"{'=' * 60}")

    if args.dry_run:
        print("\n  [DRY RUN] Experiments that would be executed:\n")
        for i, exp in enumerate(pending):
            gpu = gpus[i % len(gpus)]
            tag = "CONTRASTIVE" if exp.contrastive else "BASELINE"
            print(f"    GPU {gpu} | [{tag:11s}] {exp.run_name}")
        print(f"\n  Total: {len(pending)} runs")
        print(f"  Logs would go to: {LOG_DIR}/")
        print(f"  Progress tracked in: {PROGRESS_FILE}")
        return

    if not pending:
        print("\n  All experiments already completed! Nothing to do.")
        return

    # Distribute experiments round-robin across GPUs
    gpu_queues = {gpu: [] for gpu in gpus}
    for i, exp in enumerate(pending):
        gpu = gpus[i % len(gpus)]
        gpu_queues[gpu].append(exp)

    print(f"\n  Distribution:")
    for gpu, queue in gpu_queues.items():
        bl = sum(1 for e in queue if not e.contrastive)
        ct = sum(1 for e in queue if e.contrastive)
        print(f"    GPU {gpu}: {len(queue)} runs (baseline={bl}, contrastive={ct})")

    start_time = datetime.now()
    print(f"\n  Starting sweep at {start_time.strftime('%Y-%m-%d %H:%M:%S')}...\n")

    # Launch one thread per GPU
    threads = []
    for gpu, queue in gpu_queues.items():
        t = threading.Thread(
            target=gpu_worker,
            args=(gpu, queue),
            name=f"GPU-{gpu}",
        )
        t.start()
        threads.append(t)

    # Wait for all to finish
    for t in threads:
        t.join()

    # Final summary
    end_time = datetime.now()
    elapsed = end_time - start_time
    completed_final = load_completed_runs()

    print(f"\n{'=' * 60}")
    print(f"  Sweep finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total elapsed: {elapsed}")
    print(f"  Completed: {len(completed_final)} / {len(all_experiments)}")
    print(f"  Progress file: {PROGRESS_FILE}")
    print(f"  Logs directory: {LOG_DIR}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
