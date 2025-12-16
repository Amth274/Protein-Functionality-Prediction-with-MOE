#!/usr/bin/env python3
"""
Run a single ablation experiment or a suite of experiments.

Usage:
    # Run single experiment
    python scripts/run_ablation.py --exp A1.2

    # Run a study (all experiments in A1)
    python scripts/run_ablation.py --study A1

    # Run Phase 1 (quick wins)
    python scripts/run_ablation.py --phase 1

    # Run specific experiments
    python scripts/run_ablation.py --exp A1.2 A1.4 A2.2
"""

import argparse
import subprocess
import yaml
import time
from pathlib import Path
from datetime import datetime


def load_experiment_index():
    """Load the experiment index."""
    index_path = Path('configs/ablations/index.yaml')
    with open(index_path) as f:
        return yaml.safe_load(f)


def run_experiment(exp_id, dry_run=False):
    """Run a single ablation experiment."""

    config_path = Path(f'configs/ablations/{exp_id}.yaml')
    output_dir = Path(f'outputs/ablations/{exp_id}')
    log_file = Path(f'logs/ablation_{exp_id}.log')

    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return False

    # Load config to display info
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*70}")
    print(f"Experiment: {exp_id}")
    print(f"Description: {config.get('experiment', {}).get('description', 'N/A')}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    if dry_run:
        print("(Dry run - not executing)\n")
        return True

    # Create command
    cmd = [
        'python3',
        'scripts/train_meta_learning_optimized.py',
        '--config', str(config_path),
        '--output-dir', str(output_dir)
    ]

    # Log start time
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f"Started: {timestamp}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        # Run training
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        # Calculate duration
        duration = time.time() - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)

        print(f"\n✅ Experiment {exp_id} completed successfully!")
        print(f"Duration: {hours}h {minutes}m {seconds}s")

        # Save stdout/stderr to log
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w') as f:
            f.write(f"=== Experiment {exp_id} ===\n")
            f.write(f"Started: {timestamp}\n")
            f.write(f"Duration: {hours}h {minutes}m {seconds}s\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)

        return True

    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"\n❌ Experiment {exp_id} FAILED!")
        print(f"Duration: {duration:.1f}s")
        print(f"Error: {e}")

        # Save error log
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w') as f:
            f.write(f"=== Experiment {exp_id} FAILED ===\n")
            f.write(f"Started: {timestamp}\n")
            f.write(f"Error: {str(e)}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(e.stdout if e.stdout else "None")
            f.write("\n=== STDERR ===\n")
            f.write(e.stderr if e.stderr else "None")

        return False


def get_phase_experiments(phase):
    """Get experiments for a given phase."""
    phases = {
        1: ['A1.2', 'A1.4', 'A1.5', 'A1.6', 'A4.3', 'A4.4', 'A4.5'],  # Quick wins
        2: ['A2.2', 'A2.3', 'A2.4', 'A6.2', 'A6.3', 'A6.4', 'A6.5', 'A7.2', 'A7.3', 'A7.4'],  # Capacity
        3: ['A3.2', 'A3.3', 'A3.4', 'A3.5', 'A3.6'],  # Backbone scaling
    }
    return phases.get(phase, [])


def main():
    parser = argparse.ArgumentParser(description='Run ablation experiments')
    parser.add_argument('--exp', nargs='+', help='Experiment ID(s) to run (e.g., A1.2)')
    parser.add_argument('--study', help='Run all experiments in a study (e.g., A1)')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], help='Run Phase 1, 2, or 3')
    parser.add_argument('--dry-run', action='store_true', help='Show what would run without executing')
    parser.add_argument('--continue-on-error', action='store_true', help='Continue even if an experiment fails')

    args = parser.parse_args()

    # Determine which experiments to run
    experiments = []

    if args.exp:
        experiments = args.exp
    elif args.study:
        # Get all experiments starting with study prefix
        index = load_experiment_index()
        experiments = [
            exp['id'] for exp in index['experiments']
            if exp['id'].startswith(args.study)
        ]
    elif args.phase:
        experiments = get_phase_experiments(args.phase)
    else:
        parser.print_help()
        return

    if not experiments:
        print("No experiments to run.")
        return

    # Display plan
    print(f"\n{'='*70}")
    print(f"Ablation Study Runner")
    print(f"{'='*70}")
    print(f"Experiments to run: {len(experiments)}")
    print(f"Estimated time: ~{len(experiments) * 2.7:.1f} hours")
    print(f"Dry run: {args.dry_run}")
    print(f"Continue on error: {args.continue_on_error}")
    print(f"{'='*70}\n")

    # Run experiments
    total_start = time.time()
    results = {}

    for i, exp_id in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Running experiment {exp_id}...")

        success = run_experiment(exp_id, dry_run=args.dry_run)
        results[exp_id] = success

        if not success and not args.continue_on_error and not args.dry_run:
            print(f"\nStopping due to failure in {exp_id}.")
            break

    # Summary
    total_duration = time.time() - total_start
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)

    print(f"\n{'='*70}")
    print("Ablation Study Complete")
    print(f"{'='*70}")
    print(f"Total time: {hours}h {minutes}m")
    print(f"Successful: {sum(results.values())}/{len(results)}")
    print(f"Failed: {len(results) - sum(results.values())}/{len(results)}")

    # Show failed experiments
    failed = [exp_id for exp_id, success in results.items() if not success]
    if failed:
        print(f"\nFailed experiments:")
        for exp_id in failed:
            print(f"  - {exp_id}")

    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
