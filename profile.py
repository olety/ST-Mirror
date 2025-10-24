#!/usr/bin/env python3
"""
Profile Generator - Simple Clean UI

Usage:
    uv run profile.py                 # Process all branches
    uv run profile.py --resume        # Skip existing profiles
    uv run profile.py --limit 10      # Process only first 10 branches
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from glob import glob
from typing import List, Optional
import time

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from toolkit.pipeline import TwoPhaseProfiler
from toolkit.segmentation import segment_by_arcs
from toolkit.classifier import heuristic_prefilter, classify_with_flash_lite

console = Console()


class SimpleStats:
    """Simple statistics tracker"""
    def __init__(self):
        self.completed = 0
        self.warnings = 0
        self.errors = 0
        self.filtered = 0
        self.start_time = time.time()
        self.error_log = []

    def add_error(self, branch: str, msg: str):
        self.errors += 1
        self.error_log.append(f"{branch}: {msg}")

    def add_filtered(self, branch: str, reason: str):
        self.filtered += 1
        self.error_log.append(f"{branch}: Filtered - {reason}")

    def elapsed(self) -> str:
        elapsed = int(time.time() - self.start_time)
        h, m = divmod(elapsed, 3600)
        m, s = divmod(m, 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        else:
            return f"{s}s"


async def process_branch(profiler, branch_file: str, segments_dir: Path,
                        stats: SimpleStats, progress, overall_task, branch_task,
                        skip_classification: bool = False, debug_dir: Optional[Path] = None) -> bool:
    """Process a single branch. Returns True if successful."""

    branch_name = Path(branch_file).stem
    branch_id = branch_name.replace(' ', '___').replace('-', '_')

    try:
        # Load messages from original branch file for classification
        messages = []
        with open(branch_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        messages.append(json.loads(line))
                    except:
                        pass

        # Tier 1: Heuristic filter (FREE - catches corrupted/empty/short files)
        heuristic_result = heuristic_prefilter(messages)
        if heuristic_result["skip"]:
            stats.add_filtered(branch_name, heuristic_result['reason'])
            progress.update(overall_task, advance=1)
            return False

        # Tier 2: LLM classification (costs $0.0001, saves $0.014 per test branch)
        if not skip_classification:
            classification = await classify_with_flash_lite(
                messages, branch_file, profiler.api_key,
                quiet=True, debug_dir=str(debug_dir) if debug_dir else None
            )
            if classification.get("skip"):
                reason = "test_branch" if classification.get("is_test") else "no_substance"
                stats.add_filtered(branch_name, reason)
                progress.update(overall_task, advance=1)
                return False

        # Step 1: Check for existing arc files
        arc_files = [
            str(f) for f in segments_dir.glob(f"{branch_id}_arc*.json")
            if not f.name.endswith('_arc_summary.json')
        ]

        if not arc_files:
            # Segment (suppress output)
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            arc_files = segment_by_arcs(branch_file, str(segments_dir),
                                       window_size=50, min_arc_length=100)

            sys.stdout = old_stdout

            if not arc_files:
                # Too short - copy as single arc
                arc_file = segments_dir / f"{branch_id}_arc01_whole.json"
                import shutil
                shutil.copy(branch_file, arc_file)
                arc_files = [str(arc_file)]
            else:
                arc_files = [str(segments_dir / Path(f).name) for f in arc_files]

        if not arc_files:
            stats.add_error(branch_name, "No arc files after segmentation")
            progress.update(branch_task,
                          description=f"[red]{branch_name[:40]} - No arcs",
                          visible=True)
            progress.update(overall_task, advance=1)
            return False

        # Calculate total steps: Segmentation + Phase1 (N chunks) + Phase2
        total_steps = 1 + len(arc_files) + 1
        current_step = 0

        # Show branch progress bar with actual step count
        progress.update(branch_task, description=f"[yellow]{branch_name[:40]} - Segmentation",
                       visible=True, completed=current_step, total=total_steps)
        current_step += 1
        progress.update(branch_task, completed=current_step)

        # Step 2: Phase 1 - Extract evidence with cumulative context
        evidence_list = []

        for i, arc_file in enumerate(arc_files, 1):
            # Build cumulative summary from previous chunks
            previous_summary = profiler._build_cumulative_summary(evidence_list) if evidence_list else None

            # Update progress
            progress.update(branch_task,
                          description=f"[yellow]{branch_name[:40]} - Phase 1 ({i}/{len(arc_files)})")

            # Extract evidence with context
            evidence = await profiler.phase1_extract_evidence(arc_file, previous_summary)
            if evidence:
                evidence_list.append(evidence)

            # Advance step
            current_step += 1
            progress.update(branch_task, completed=current_step)

        if not evidence_list:
            stats.add_error(branch_name, "No evidence extracted")
            progress.update(branch_task,
                          description=f"[red]{branch_name[:40]} - No evidence")
            progress.update(overall_task, advance=1)
            return False

        # Step 3: Phase 2 - Synthesize
        progress.update(branch_task, description=f"[yellow]{branch_name[:40]} - Phase 2")
        profile = await profiler.phase2_synthesize_profile(evidence_list, branch_id)

        if profile:
            current_step += 1
            progress.update(branch_task,
                          description=f"[green]{branch_name[:40]} - Complete",
                          completed=current_step)
            # Keep completed bars visible to show duration
            progress.update(overall_task, advance=1)
            return True
        else:
            stats.add_error(branch_name, "Synthesis failed")
            # Keep failed bars visible too
            progress.update(branch_task,
                          description=f"[red]{branch_name[:40]} - Failed")
            progress.update(overall_task, advance=1)
            return False

    except Exception as e:
        stats.add_error(branch_name, str(e))
        # Keep error bars visible
        progress.update(branch_task,
                      description=f"[red]{branch_name[:40]} - Error")
        progress.update(overall_task, advance=1)
        return False


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Process all RP branches into profiles",
        epilog="By default, skips branches that already have profiles. Use --force to reprocess."
    )
    parser.add_argument('--limit', type=int,
                       help='Process only first N branches')
    parser.add_argument('--parallel', type=int, default=5,
                       help='Number of branches to process in parallel (default: 5)')
    parser.add_argument('--skip-classification', action='store_true',
                       help='Skip LLM classification (only use heuristic filter)')
    parser.add_argument('--chats-dir', default='ST_DATA/chats',
                       help='Directory containing JSONL chat files (default: ST_DATA/chats)')
    parser.add_argument('--workspace', default=None,
                       help='Workspace directory (default: auto-derived from --chats-dir parent)')
    parser.add_argument('--force', action='store_true',
                       help='Reprocess all branches (skip existing profiles by default)')

    args = parser.parse_args()

    # Auto-derive workspace from chats directory if not specified
    chats_dir = Path(args.chats_dir)
    if args.workspace:
        workspace = Path(args.workspace)
    else:
        # Auto-derive: ST_DATA/chats → ST_DATA/workspace
        workspace = chats_dir.parent / 'workspace'

    # Setup directory structure
    profiles_dir = workspace / 'profiles'
    reports_dir = workspace / 'reports'
    segments_dir = workspace / 'segments'
    debug_dir = workspace / 'debug'

    # Create all debug subdirectories
    (debug_dir / 'classifier').mkdir(parents=True, exist_ok=True)
    (debug_dir / 'phase1').mkdir(parents=True, exist_ok=True)
    (debug_dir / 'phase2').mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[bold red]ERROR:[/bold red] OPENROUTER_API_KEY not set")
        sys.exit(1)

    profiler = TwoPhaseProfiler(api_key, str(workspace), quiet=True)
    stats = SimpleStats()

    # Find branches
    all_branches = sorted(glob(f"{chats_dir}/**/*.jsonl", recursive=True))

    # Default behavior: skip existing profiles (unless --force)
    if args.force:
        branches = all_branches
        console.print(f"[cyan]Force mode: Processing all {len(branches)} branches[/cyan]")
    else:
        # Filter out branches that already have profiles
        branches = []
        skipped_existing = 0

        for branch_file in all_branches:
            branch_id = Path(branch_file).stem.replace(' ', '___').replace('-', '_')
            profile_file = profiles_dir / f"{branch_id}_profile.json"

            if not profile_file.exists():
                branches.append(branch_file)
            else:
                skipped_existing += 1

        if skipped_existing > 0:
            console.print(f"[cyan]Skipped {skipped_existing} existing profiles (use --force to reprocess)[/cyan]")

    # Apply limit BEFORE pre-filtering (to match user expectation)
    if args.limit:
        branches = branches[:args.limit]
        console.print(f"[cyan]Limit mode: Processing first {args.limit} branches[/cyan]")
        console.print(f"[dim](Note: Some may be filtered during processing)[/dim]")
    else:
        # No limit - pre-filter with heuristic to save progress bar slots
        console.print(f"[cyan]Pre-filtering {len(branches)} branches with heuristic filter...[/cyan]")
        original_count = len(branches)
        filtered_branches = []

        for branch_file in branches:
            try:
                messages = []
                with open(branch_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                messages.append(json.loads(line))
                            except:
                                pass

                heuristic_result = heuristic_prefilter(messages)
                if not heuristic_result["skip"]:
                    filtered_branches.append(branch_file)
            except Exception:
                # If can't read file, skip it
                pass

        branches = filtered_branches
        filtered_count = original_count - len(branches)
        if filtered_count > 0:
            console.print(f"[yellow]Filtered out {filtered_count} branches (too short/corrupted)[/yellow]")

    if not branches:
        console.print("[green]All branches already processed or filtered![/green]")
        return

    total = len(branches)
    console.print(f"\n[bold]Processing {total} branches[/bold]\n")

    # Process with rolling window (as soon as one finishes, start next)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=2,
    ) as progress:

        overall_task = progress.add_task("[cyan]Overall progress", total=total)

        # Create branch task pool (reuse task IDs)
        max_concurrent = args.parallel
        branch_task_pool = []
        for i in range(max_concurrent):
            task_id = progress.add_task(f"[dim]Slot {i+1}", total=100, visible=False)
            branch_task_pool.append(task_id)

        # Rolling window implementation
        async def worker(branch_file: str, task_id: int):
            """Process one branch, returns result"""
            result = await process_branch(profiler, branch_file, segments_dir, stats, progress,
                                         overall_task, task_id, args.skip_classification,
                                         debug_dir / 'classifier')

            # Count successes (progress already updated in process_branch)
            if result is True:
                stats.completed += 1
            elif isinstance(result, Exception):
                stats.add_error("unknown", str(result))

            return result

        # Create queue of branches to process
        branch_queue = asyncio.Queue()
        for branch in branches:
            await branch_queue.put(branch)

        # Start workers
        async def run_worker(task_id: int):
            """Worker that keeps processing from queue until empty"""
            while not branch_queue.empty():
                try:
                    branch_file = await asyncio.wait_for(branch_queue.get(), timeout=0.1)
                    await worker(branch_file, task_id)
                except asyncio.TimeoutError:
                    break

        # Run all workers in parallel
        workers = [run_worker(task_id) for task_id in branch_task_pool]
        await asyncio.gather(*workers)

    # Final summary
    console.print("\n" + "="*70)
    console.print("[bold green]✓ Profile Generation Complete![/bold green]")
    console.print("="*70 + "\n")

    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_row("Total branches:", str(total))
    summary.add_row("Completed:", f"[green]{stats.completed}[/green]")
    summary.add_row("Filtered:", f"[yellow]{stats.filtered}[/yellow]")
    summary.add_row("Errors:", f"[red]{stats.errors}[/red]")
    summary.add_row("Time elapsed:", stats.elapsed())
    summary.add_row("Avg per branch:",
                   f"{(time.time() - stats.start_time) / max(stats.completed, 1):.1f}s")

    console.print(summary)

    if stats.error_log:
        console.print("\n[red]Errors:[/red]")
        for error in stats.error_log[-5:]:  # Show last 5
            console.print(f"  • {error}")

        if stats.errors > 0:
            console.print(f"\n[yellow]Check debug logs for details:[/yellow]")
            console.print(f"  Phase 1: {debug_dir / 'phase1'}/")
            console.print(f"  Phase 2: {debug_dir / 'phase2'}/")
            console.print(f"  Classifier: {debug_dir / 'classifier'}/")


    console.print("\n[dim]Next: Run hierarchical aggregation[/dim]")
    aggregations_dir = workspace / 'aggregations'
    console.print(f"[dim]  uv run aggregate.py {profiles_dir} {aggregations_dir}[/dim]")


if __name__ == '__main__':
    asyncio.run(main())
