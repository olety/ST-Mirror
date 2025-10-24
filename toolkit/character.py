#!/usr/bin/env python3
"""
Character-Level Processing Orchestrator
Handles full pipeline: Classify → Process → Aggregate → Evolution

Usage:
    from toolkit.character import process_character

    process_character(
        character_name="Ari",
        input_pattern="ST_DATA/chats/Ari*/*.jsonl",
        output_dir="ari_output",
        parallel_branches=10,
        api_key=None
    )
"""

import json
import os
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
from glob import glob

# Rich for beautiful progress displays
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TaskID
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# Import existing modules (relative import since we're in toolkit/)
from .classifier import classify_character_branches

# Global console for Rich output
console = Console()


class ProgressTracker:
    """Progress tracking for parallel branch processing with Rich display (async-safe)."""

    def __init__(self, total_branches: int, progress: Progress):
        self.total = total_branches
        self.completed = 0
        self.failed = 0
        self.active = 0
        # No lock needed - asyncio is single-threaded with cooperative multitasking

        # Rich Progress instance
        self.progress = progress
        self.branch_tasks = {}  # Map branch_file -> task_id

        # Overall progress task
        self.overall_task = progress.add_task(
            f"[cyan]Processing {total_branches} branches",
            total=total_branches
        )

        # Cost tracking
        self.total_cost = 0.0
        self.classification_cost = 0.0

        # Timing
        self.start_time = time.time()
        self.branch_times = []  # List of completion times for ETA

        # Recent completions (for display)
        self.recent_completions = []

        # Failed branches
        self.failed_branches = []

    def create_branch_task(self, branch_file: str, description: str) -> TaskID:
        """Create a progress task for a branch."""
        task_id = self.progress.add_task(
            f"[yellow]{description}",
            total=None  # Indeterminate initially
        )
        self.branch_tasks[branch_file] = task_id
        return task_id

    def update_branch_task(self, branch_file: str, description: str = None, total: int = None, completed: int = None):
        """Update branch task progress."""
        if branch_file not in self.branch_tasks:
            return

        task_id = self.branch_tasks[branch_file]
        kwargs = {}
        if description is not None:
            kwargs['description'] = f"[yellow]{description}"
        if total is not None:
            kwargs['total'] = total
        if completed is not None:
            kwargs['completed'] = completed

        self.progress.update(task_id, **kwargs)

    def complete_branch_task(self, branch_file: str, success: bool = True):
        """Mark branch task as complete and remove it."""
        if branch_file in self.branch_tasks:
            task_id = self.branch_tasks[branch_file]
            self.progress.remove_task(task_id)
            del self.branch_tasks[branch_file]

    def start_branch(self):
        """Mark a branch as active."""
        self.active += 1

    def complete_branch(self, branch_file: str, stats: Dict):
        """Mark branch as completed and record stats."""
        self.active -= 1
        self.completed += 1

        # Update overall progress
        self.progress.update(self.overall_task, advance=1)

        # Record time
        elapsed = stats.get('time', 0)
        self.branch_times.append(elapsed)

        # Record cost
        cost = stats.get('cost', 0.0)
        self.total_cost += cost

        # Add to recent completions (keep last 3)
        self.recent_completions.append({
            'branch_file': Path(branch_file).name,
            'messages': stats.get('messages', 0),
            'chunks': stats.get('chunks', 0),
            'cost': cost,
            'time': elapsed
        })
        if len(self.recent_completions) > 3:
            self.recent_completions.pop(0)

    def fail_branch(self, branch_file: str, error: str):
        """Mark branch as failed."""
        self.active -= 1
        self.failed += 1

        # Update overall progress
        self.progress.update(self.overall_task, advance=1)

        self.failed_branches.append({
            'branch_file': branch_file,
            'error': str(error)
        })

    def get_eta(self) -> Optional[int]:
        """Calculate ETA in seconds based on average branch time."""
        if not self.branch_times:
            return None

        avg_time = sum(self.branch_times) / len(self.branch_times)
        remaining = self.total - self.completed - self.failed
        return int(avg_time * remaining)

    def get_progress(self) -> float:
        """Get progress as 0-1 float."""
        return (self.completed + self.failed) / self.total if self.total > 0 else 0.0

    def get_summary(self) -> Dict:
        """Get current progress summary."""
        return {
            'total': self.total,
            'completed': self.completed,
            'failed': self.failed,
            'active': self.active,
            'queued': self.total - self.completed - self.failed - self.active,
            'progress': self.get_progress(),
            'total_cost': self.total_cost,
            'avg_time': sum(self.branch_times) / len(self.branch_times) if self.branch_times else 0,
            'eta': self.get_eta(),
            'recent_completions': list(self.recent_completions),
            'failed_branches': list(self.failed_branches)
        }


def format_time(seconds: int) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins}m {secs}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours}h {mins}m"


def print_progress_bar(label: str, current: int, total: int, width: int = 40):
    """Print a progress bar."""
    progress = current / total if total > 0 else 0
    filled = int(width * progress)
    bar = '█' * filled + '░' * (width - filled)
    percentage = progress * 100
    print(f"{label} {bar} {current}/{total} ({percentage:.0f}%)")


async def process_single_branch(
    branch_file: str,
    workspace_dir: str,
    model_provider: str,
    model: str,
    api_key: Optional[str] = None,
    progress_tracker: Optional['ProgressTracker'] = None,
    branch_index: int = 0,
    debug: bool = False
) -> Dict:
    """
    Process a single branch: ingest + sequential analysis (async).

    Returns:
        Stats dict with cost, time, messages, chunks
    """
    import asyncio
    start_time = time.time()
    branch_short_name = Path(branch_file).name
    if len(branch_short_name) > 50:
        branch_short_name = branch_short_name[:47] + '...'

    # Create task for this branch
    if progress_tracker:
        progress_tracker.create_branch_task(branch_file, f"[{branch_index+1}] {branch_short_name}")

    try:
        # Import pipeline functions
        import sys
        import io
        import contextlib
        sys.path.insert(0, str(Path(__file__).parent))
        from pipeline import ingest, execute_sequential

        # Create workspace for this branch
        branch_name = Path(branch_file).stem
        # Sanitize branch name to prevent path traversal
        branch_name = branch_name.replace('/', '_').replace('\\', '_').replace('..', '_')
        branch_workspace = Path(workspace_dir) / branch_name
        branch_workspace.mkdir(parents=True, exist_ok=True)

        # Step 1: Ingest + Segment
        session_id = branch_name.replace(' ', '_').replace('-', '_')

        # Import segment function
        from pipeline import segment

        # Update task: Ingesting
        if progress_tracker:
            progress_tracker.update_branch_task(branch_file, f"[{branch_index+1}] {branch_short_name} - Ingesting")

        # Run ingest + segment (no output suppression - let Rich handle the display)
        ingest(
            inputs=[branch_file],
            workspace=str(branch_workspace),
            session_id=session_id
        )
        segment(workspace=str(branch_workspace))

        # Count messages and chunks AFTER segmentation
        clean_file = branch_workspace / 'clean' / f'{session_id}.json'
        with open(clean_file, 'r') as f:
            session_data = json.load(f)
            message_count = len(session_data.get('messages', []))

        chunks_dir = branch_workspace / 'chunks'
        chunk_files = list(chunks_dir.glob('*.json')) if chunks_dir.exists() else []
        total_chunks = len(chunk_files)

        # Update task: Processing chunks with progress bar
        if progress_tracker:
            progress_tracker.update_branch_task(
                branch_file,
                f"[{branch_index+1}] {branch_short_name} - {message_count} msgs → {total_chunks} chunks",
                total=total_chunks,
                completed=0
            )

        # Step 2: Sequential analysis (no output suppression - let Rich handle the display)
        result = await execute_sequential(
            workspace=str(branch_workspace),
            model_provider=model_provider,
            model=model
        )

        # Extract stats
        chunks_processed = result.get('chunks_processed', total_chunks) if result else 0
        cost = result.get('total_cost', 0.0) if result else 0.0

        elapsed = time.time() - start_time

        # Complete branch task
        if progress_tracker:
            progress_tracker.complete_branch_task(branch_file, success=True)

        return {
            'success': True,
            'branch_file': branch_file,
            'messages': message_count,
            'chunks': chunks_processed,
            'cost': cost,
            'time': elapsed
        }

    except Exception as e:
        elapsed = time.time() - start_time

        # Complete branch task with error
        if progress_tracker:
            progress_tracker.complete_branch_task(branch_file, success=False)

        return {
            'success': False,
            'branch_file': branch_file,
            'error': str(e),
            'time': elapsed
        }


async def process_character(
    character_name: str,
    input_pattern: str,
    output_dir: str,
    parallel_branches: int = 10,
    api_key: Optional[str] = None,
    model_provider: str = 'openrouter',
    model: str = 'moonshotai/kimi-k2-0905',
    max_branches: Optional[int] = None,
    debug: bool = False
):
    """
    Process entire character: Classify → Process → Aggregate → Evolution.

    Args:
        character_name: Character name (e.g., "Ari")
        input_pattern: Glob pattern for branch files
        output_dir: Output directory for results
        parallel_branches: Number of branches to process in parallel (default: 10)
        api_key: OpenRouter API key (or from env)
        model_provider: Model provider (default: openrouter)
        model: Model to use (default: moonshotai/kimi-k2-0905)
        debug: Enable verbose debug output (default: False)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    workspaces_dir = output_path / 'workspaces'
    profiles_dir = output_path / 'profiles'
    workspaces_dir.mkdir(exist_ok=True)
    profiles_dir.mkdir(exist_ok=True)

    print(f'\n{"="*80}')
    print(f'PROCESSING CHARACTER: {character_name}')
    print(f'{"="*80}\n')

    # ========== PHASE 1: CLASSIFICATION ==========
    print(f'[1/4] CLASSIFICATION (Flash Lite)')

    classification_file = output_path / 'classification.json'

    classify_start = time.time()
    classification_result = await classify_character_branches(
        character_name=character_name,
        input_pattern=input_pattern,
        output_file=str(classification_file),
        api_key=api_key,
        parallel=min(parallel_branches, 15)  # Cap at 15 for classification
    )
    classify_time = time.time() - classify_start

    # Extract results
    total_branches = classification_result['total_branches']
    kept_branches = classification_result['kept_branches']
    filtered = total_branches - kept_branches
    classification_cost = 0.01  # Approximate

    print(f'\nResults: {kept_branches} kept, {filtered} filtered | Cost: ${classification_cost:.2f} | Time: {format_time(int(classify_time))}\n')

    # Get list of kept branches
    kept_branch_files = [
        b['branch_file']
        for b in classification_result['branches']
        if not b.get('skip', False)
    ]

    if not kept_branch_files:
        print(f'⚠ No valid branches to process! All branches were filtered.')
        return

    # ========== PHASE 2: SEQUENTIAL PROCESSING ==========
    print(f'\n[2/4] SEQUENTIAL PROCESSING (Kimi K2)')
    print(f'Processing {len(kept_branch_files)} branches with {parallel_branches} parallel workers')
    print(f'{"="*80}\n')

    # Create Rich Progress with Live display
    progress = Progress(
        SpinnerColumn(finished_text="✓"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    )

    tracker = ProgressTracker(total_branches=len(kept_branch_files), progress=progress)
    tracker.classification_cost = classification_cost
    tracker.total_cost = classification_cost

    # Estimate total cost
    estimated_cost_per_branch = 0.68  # From validation
    estimated_total = classification_cost + (estimated_cost_per_branch * len(kept_branch_files))

    # Process branches in parallel with asyncio
    # Create semaphore to limit concurrent processing
    semaphore = asyncio.Semaphore(parallel_branches)

    async def process_wrapper(branch_file, branch_index):
        """Async wrapper for process_single_branch with progress tracking and concurrency limiting."""
        # Acquire semaphore to limit concurrent branches
        async with semaphore:
            tracker.start_branch()

            result = await process_single_branch(
                branch_file=branch_file,
                workspace_dir=str(workspaces_dir),
                model_provider=model_provider,
                model=model,
                api_key=api_key,
                progress_tracker=tracker,
                branch_index=branch_index,
                debug=debug
            )

            if result['success']:
                tracker.complete_branch(branch_file, result)
            else:
                tracker.fail_branch(branch_file, result.get('error', 'Unknown error'))

            return result

    # Async execution with Live display
    # Key: asyncio.gather() naturally yields to event loop, allowing Live to refresh spinners
    with Live(progress, console=console, refresh_per_second=10, transient=False):
        # Create tasks for all branches
        tasks = [
            process_wrapper(bf, idx)
            for idx, bf in enumerate(kept_branch_files)
        ]

        # Run concurrently with asyncio.gather
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions that escaped process_wrapper
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Track as failed branch (in case it wasn't already tracked)
                    branch_file = kept_branch_files[i]
                    tracker.fail_branch(branch_file, str(result))
                    print(f'\n[processor] ERROR processing {branch_file}: {result}')
        except Exception as e:
            console.print(f'[red]Fatal error during processing: {e}[/red]')
            raise

    # Final summary
    summary = tracker.get_summary()
    print(f'\n{"="*80}')
    print(f'SEQUENTIAL PROCESSING COMPLETE')
    print(f'{"="*80}')
    print(f'Branches: {summary["completed"]} succeeded, {summary["failed"]} failed')
    print(f'Cost: ${summary["total_cost"]:.2f}')
    print(f'Average: {int(summary["avg_time"])}s/branch' if summary['avg_time'] > 0 else '')
    print(f'{"="*80}\n')

    # Save failed branches if any
    summary = tracker.get_summary()
    if summary['failed_branches']:
        failed_file = output_path / 'failed_branches.json'
        with open(failed_file, 'w') as f:
            json.dump(summary['failed_branches'], f, indent=2)
        print(f'⚠ Saved {len(summary["failed_branches"])} failed branches to {failed_file}\n')

    # ========== PHASE 3: CROSS-BRANCH AGGREGATION ==========
    print(f'\n[3/4] CROSS-BRANCH AGGREGATION')
    print(f'⏳ TO IMPLEMENT - Phase 4a')
    print(f'   Will aggregate {summary["completed"]} branch profiles → character profile\n')

    # Stub: Save placeholder
    character_profile_file = output_path / 'character_profile.json'
    with open(character_profile_file, 'w') as f:
        json.dump({
            'status': 'stub',
            'character': character_name,
            'branches_processed': summary['completed'],
            'note': 'Phase 4a - Cross-branch aggregation not yet implemented'
        }, f, indent=2)

    # ========== PHASE 4: TEMPORAL EVOLUTION ==========
    print(f'\n[4/4] TEMPORAL EVOLUTION')
    print(f'⏳ TO IMPLEMENT - Phase 5')
    print(f'   Will analyze psychological changes over time (June-Oct 2025)\n')

    # Stub: Save placeholder
    evolution_file = output_path / 'evolution.json'
    with open(evolution_file, 'w') as f:
        json.dump({
            'status': 'stub',
            'character': character_name,
            'note': 'Phase 5 - Temporal evolution not yet implemented'
        }, f, indent=2)

    # ========== FINAL SUMMARY ==========
    total_time = time.time() - tracker.start_time

    print(f'\n{"="*80}')
    print(f'COMPLETE: {character_name} Character Profile')
    print(f'{"="*80}')
    print(f'Branches: {summary["completed"]} succeeded, {summary["failed"]} failed')
    print(f'Cost: ${summary["total_cost"]:.2f} total')
    print(f'Time: {format_time(int(total_time))}')
    print(f'\nOutput:')
    print(f'  {classification_file}')
    print(f'  {workspaces_dir}/')
    if summary['failed_branches']:
        print(f'  {output_path / "failed_branches.json"}')
    print(f'  {character_profile_file} (STUB)')
    print(f'  {evolution_file} (STUB)')
    print(f'{"="*80}\n')


if __name__ == '__main__':
    import sys
    import asyncio

    if len(sys.argv) < 4:
        print("Usage: uv run python -m toolkit.character <character> <input_pattern> <output_dir> [parallel_branches]")
        print("Example: uv run python -m toolkit.character Ari 'ST_DATA/chats/Ari*/*.jsonl' ari_output 10")
        sys.exit(1)

    character = sys.argv[1]
    pattern = sys.argv[2]
    output = sys.argv[3]
    parallel = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    asyncio.run(process_character(character, pattern, output, parallel))
