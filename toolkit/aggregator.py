#!/usr/bin/env python3
"""
Adaptive Hierarchical Aggregation System

Automatically aggregates branch profiles with adaptive time splitting:
- Days → Weeks → Months → Years → Life
- Splits periods when they exceed token budget (~30k tokens)
- Maintains consistent schema across all levels

Architecture:
    Branch profiles (individual RPs)
    ↓
    Adaptive splitting (15 profiles max per aggregation)
    ↓
    Weekly summaries (if month > 15 profiles)
    ↓
    Monthly summaries
    ↓
    Yearly summaries
    ↓
    Life profile (overall + companion export)
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import re
from json_repair import repair_json
import asyncio
import httpx

from config import config

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
import time

try:
    from .normalizer import GeminiNormalizer
except ImportError:
    from normalizer import GeminiNormalizer

console = Console()


class AdaptiveAggregator:
    """
    Hierarchical aggregation with adaptive time splitting.

    Automatically splits time periods that exceed token budget.
    """

    def __init__(
        self,
        profiles_dir: Path,
        output_dir: Path,
        api_key: Optional[str] = None,
        skip_existing: bool = True,
        max_parallel: int = 5,
        verbose: bool = False,
    ):
        """
        Initialize aggregator.

        Args:
            profiles_dir: Directory containing branch profile JSONs
            output_dir: Directory to save aggregation results
            api_key: OpenRouter API key (if None, will look for OPENROUTER_API_KEY env var)
            skip_existing: Skip reprocessing existing summaries (default: True)
            max_parallel: Maximum concurrent aggregations (default: 5)
            verbose: Show detailed processing messages (default: False)
        """
        self.profiles_dir = Path(profiles_dir)
        self.output_dir = Path(output_dir)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.skip_existing = skip_existing
        self.max_parallel = max_parallel
        self.verbose = verbose

        # OpenRouter config
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = config.models.aggregation.name

        # JSON normalizer for fixing malformed responses
        self.normalizer = GeminiNormalizer()

        # Create JSON output directory structure
        # weeks/, months/, years/ - Period summary JSONs
        # life/ - Life profile JSONs
        (self.output_dir / "weeks").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "months").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "years").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "life").mkdir(parents=True, exist_ok=True)

        # Create markdown reports directory structure
        # reports/periods/ - Period summary reports (markdown)
        # reports/life/ - Life profile reports (markdown)
        reports_base = self.output_dir.parent / "reports"
        (reports_base / "periods").mkdir(parents=True, exist_ok=True)
        (reports_base / "life").mkdir(parents=True, exist_ok=True)

        # Create debug directory structure
        self.debug_dir = self.output_dir.parent / "debug"
        (self.debug_dir / "aggregator" / "periods").mkdir(parents=True, exist_ok=True)
        (self.debug_dir / "aggregator" / "life").mkdir(parents=True, exist_ok=True)

        # Load prompts and schemas
        prompts_dir = Path(__file__).parent.parent / "prompts"
        with open(prompts_dir / "period_synthesizer.txt") as f:
            self.period_prompt = f.read()
        with open(prompts_dir / "life_synthesizer.txt") as f:
            self.life_prompt = f.read()
        with open(prompts_dir / "period_schema.json") as f:
            self.period_schema = json.load(f)
        with open(prompts_dir / "life_schema.json") as f:
            self.life_schema = json.load(f)

    def load_all_profiles(self) -> List[Dict]:
        """Load all branch profiles from directory."""
        profiles = []
        for profile_file in self.profiles_dir.glob("*_profile.json"):
            with open(profile_file) as f:
                profile = json.load(f)
                profile["_source_file"] = profile_file.name
                profiles.append(profile)

        print(f"Loaded {len(profiles)} branch profiles")
        return profiles

    def extract_date_from_profile(self, profile: Dict) -> Optional[datetime]:
        """
        Extract date from profile's chat_date field (added in pipeline).

        Falls back to parsing profile_id or filename if chat_date is missing.
        """
        # Primary: Use chat_date field from profile (format: "2025-06-07@15h15m23s")
        chat_date = profile.get("chat_date")
        if chat_date:
            # Parse format: "2025-06-07@15h15m23s"
            match = re.match(
                r"(\d{4})-(\d{2})-(\d{2})@(\d{2})h(\d{2})m(\d{2})s", chat_date
            )
            if match:
                year, month, day, hour, minute, second = match.groups()
                try:
                    return datetime(
                        int(year),
                        int(month),
                        int(day),
                        int(hour),
                        int(minute),
                        int(second),
                    )
                except:
                    pass

        # Fallback 1: Extract chat date from profile_id, avoiding generation timestamp
        # Profile ID format: "Branch___2025_10_16@19h31m04s_20251020_222049"
        #                                ^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^
        #                                chat date              generation timestamp
        # Strategy: Remove the generation timestamp suffix, then extract date
        # Generation timestamp format: _YYYYMMDD_HHMMSS at the end
        profile_id = profile.get("profile_id", "")
        profile_id_clean = re.sub(r"_\d{8}_\d{6}$", "", profile_id)

        # Now extract date from cleaned profile_id
        match = re.search(r"(\d{4})[_-](\d{1,2})[_-](\d{1,2})", profile_id_clean)
        if match:
            year, month, day = match.groups()
            try:
                return datetime(int(year), int(month), int(day))
            except:
                pass

        # Fallback 3: Try to extract from source filename
        # Format: "Branch #1738 - 2025-10-06@14h04m39s.jsonl"
        source_file = profile.get("_source_file", "")
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})", source_file)
        if match:
            year, month, day = match.groups()
            try:
                return datetime(int(year), int(month), int(day))
            except:
                pass

        return None

    def estimate_tokens(self, profiles: List[Dict]) -> int:
        """Estimate total tokens for list of profiles."""
        # Estimate ~2000 tokens per profile on average
        TOKENS_PER_PROFILE = 2000
        return len(profiles) * TOKENS_PER_PROFILE

    def group_by_period(
        self, profiles: List[Dict], level: str
    ) -> Dict[str, List[Dict]]:
        """
        Group profiles by time period.

        Args:
            profiles: List of branch profiles
            level: 'day' | 'week' | 'month' | 'year'

        Returns:
            Dict mapping period_id → profiles
        """
        groups = defaultdict(list)

        for profile in profiles:
            date = self.extract_date_from_profile(profile)
            if not date:
                continue

            if level == "day":
                period_id = date.strftime("%Y-%m-%d")
            elif level == "week":
                # ISO week format: 2025-W32
                period_id = f"{date.year}-W{date.isocalendar()[1]:02d}"
            elif level == "month":
                period_id = date.strftime("%Y-%m")
            elif level == "year":
                period_id = str(date.year)
            else:
                raise ValueError(f"Unknown level: {level}")

            groups[period_id].append(profile)

        return dict(groups)

    def needs_splitting(self, profiles: List[Dict]) -> bool:
        """Check if period exceeds token budget and needs splitting."""
        tokens = self.estimate_tokens(profiles)
        return tokens > config.processing.aggregator.token_budget_threshold

    def get_sub_level(self, level: str) -> Optional[str]:
        """Get next level down in hierarchy."""
        hierarchy = ["day", "week", "month", "year", "life"]
        try:
            idx = hierarchy.index(level)
            return hierarchy[idx - 1] if idx > 0 else None
        except ValueError:
            return None

    def get_super_level(self, level: str) -> Optional[str]:
        """Get next level up in hierarchy."""
        hierarchy = ["day", "week", "month", "year", "life"]
        try:
            idx = hierarchy.index(level)
            return hierarchy[idx + 1] if idx < len(hierarchy) - 1 else None
        except ValueError:
            return None

    async def aggregate_period(
        self,
        profiles: List[Dict],
        period_id: str,
        level: str,
        progress=None,
        task_id=None,
        week_slot_tasks=None,
        week_indent=None,
    ) -> Dict:
        """
        Aggregate profiles for a single period with adaptive splitting.

        Args:
            profiles: Branch profiles or summaries to aggregate
            period_id: Period identifier (e.g., '2025-08', '2025-W32')
            level: Time level ('week', 'month', 'year', 'life')
            progress: Optional Rich Progress object for updating parent task
            task_id: Optional parent task ID to update
            week_slot_tasks: Pre-created week slot tasks (if splitting)

        Returns:
            Period summary dict
        """
        # Check if output already exists (resume capability)
        if self.skip_existing:
            if level == "life":
                output_file = self.output_dir / "life" / f"{period_id}_profile.json"
            else:
                output_file = (
                    self.output_dir / f"{level}s" / f"{period_id}_summary.json"
                )

            if output_file.exists():
                with open(output_file) as f:
                    existing = json.load(f)
                if self.verbose:
                    console.print(
                        f"[dim]  {period_id} → Skipped (already exists)[/dim]"
                    )
                return existing

        # Check if splitting needed
        if self.needs_splitting(profiles) and level != "day":
            if self.verbose:
                console.print(
                    f"[cyan]  {period_id} ({len(profiles)} profiles, ~{self.estimate_tokens(profiles)//1000}k tokens) → Splitting to sub-periods[/cyan]"
                )
            return await self._aggregate_with_splitting(
                profiles,
                period_id,
                level,
                progress,
                task_id,
                week_slot_tasks,
                week_indent,
            )
        else:
            if self.verbose:
                console.print(
                    f"[cyan]  {period_id} ({len(profiles)} profiles, ~{self.estimate_tokens(profiles)//1000}k tokens) → Direct aggregation[/cyan]"
                )
            return await self._aggregate_direct(profiles, period_id, level)

    async def _aggregate_with_splitting(
        self,
        profiles: List[Dict],
        period_id: str,
        level: str,
        parent_progress=None,
        parent_task_id=None,
        week_slot_tasks=None,
        week_indent=None,
    ) -> Dict:
        """
        Aggregate by splitting into sub-periods first.

        month → weeks → aggregate weeks
        week → days → aggregate days
        """
        sub_level = self.get_sub_level(level)
        if not sub_level:
            # Can't split further, must aggregate directly
            if self.verbose:
                console.print(
                    f"[yellow]    WARNING: Cannot split {level} further, aggregating {len(profiles)} profiles anyway[/yellow]"
                )
            return await self._aggregate_direct(profiles, period_id, level)

        # Group by sub-periods
        sub_groups = self.group_by_period(profiles, sub_level)

        # Update parent task to show sub-period count + final aggregation
        # Total work = process each sub-period + aggregate them into final result
        # Note: description is already set correctly when task was created (with proper indent)
        if parent_progress and parent_task_id is not None:
            parent_progress.update(
                parent_task_id,
                total=len(sub_groups) + 1,  # +1 for final aggregation
                completed=0,
                # Don't update description - it's already set with correct indent
            )

        # Create queue of sub-periods for parallel processing
        sub_queue = asyncio.Queue()
        for sub_id in sorted(sub_groups.keys()):
            await sub_queue.put((sub_id, sub_groups[sub_id]))

        sub_summaries = []

        # If we have parent progress, use pre-created tasks or create them dynamically
        if parent_progress and parent_task_id is not None:
            # Use pre-created tasks if provided, otherwise create dynamically
            if week_slot_tasks is not None:
                sub_task_slots = week_slot_tasks
                num_sub_workers = len(week_slot_tasks)
            else:
                # Fallback: create tasks dynamically (legacy path)
                # Create task slots for sub-workers
                num_sub_workers = min(self.max_parallel, len(sub_groups))
                sub_task_slots = []
                for i in range(num_sub_workers):
                    task_id = parent_progress.add_task(
                        f"[dim]  Slot {i+1}", total=1, visible=False
                    )
                    sub_task_slots.append(task_id)

            # Worker function for parallel sub-period processing
            async def sub_worker(slot_id: int):
                while True:
                    try:
                        sub_id, sub_profiles = sub_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        parent_progress.update(sub_task_slots[slot_id], visible=False)
                        break

                    # Update slot to show current work
                    # Use provided week_indent or default to 4 spaces
                    indent_str = week_indent if week_indent else "    "
                    parent_progress.update(
                        sub_task_slots[slot_id],
                        description=f"{indent_str}[yellow]↳ {sub_id} ({len(sub_profiles)} profiles)",
                        visible=True,
                        completed=0,
                    )

                    sub_summary = await self.aggregate_period(
                        sub_profiles, sub_id, sub_level
                    )

                    # If synthesis failed, propagate failure upward
                    if sub_summary is None:
                        raise RuntimeError(
                            f"Failed to synthesize {sub_id} after all retries. "
                            f"Cannot continue with {period_id} aggregation."
                        )

                    sub_summaries.append(sub_summary)

                    # Update progress
                    parent_progress.update(sub_task_slots[slot_id], completed=1)

                    # Update parent task (month slot) as each week completes
                    parent_progress.update(parent_task_id, advance=1)

                    sub_queue.task_done()

            # Run workers in parallel (max_parallel concurrent)
            workers = [sub_worker(i) for i in range(num_sub_workers)]
            await asyncio.gather(*workers)

            # Hide sub-tasks when done
            for slot_id in sub_task_slots:
                parent_progress.update(slot_id, visible=False)

        else:
            # No parent progress - create standalone (shouldn't happen in normal flow)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
                refresh_per_second=2,
            ) as sub_progress:

                num_sub_workers = min(self.max_parallel, len(sub_groups))
                sub_task_slots = []
                for i in range(num_sub_workers):
                    task_id = sub_progress.add_task(
                        f"[dim]Slot {i+1}", total=1, visible=False
                    )
                    sub_task_slots.append(task_id)

                async def sub_worker(slot_id: int):
                    while True:
                        try:
                            sub_id, sub_profiles = sub_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            sub_progress.update(sub_task_slots[slot_id], visible=False)
                            break

                        sub_progress.update(
                            sub_task_slots[slot_id],
                            description=f"[yellow]{sub_id} ({len(sub_profiles)} profiles)",
                            visible=True,
                            completed=0,
                        )

                        sub_summary = await self.aggregate_period(
                            sub_profiles, sub_id, sub_level
                        )

                        # If synthesis failed, propagate failure upward
                        if sub_summary is None:
                            raise RuntimeError(
                                f"Failed to synthesize {sub_id} after all retries. "
                                f"Cannot continue with {period_id} aggregation."
                            )

                        sub_summaries.append(sub_summary)

                        sub_progress.update(sub_task_slots[slot_id], completed=1)
                        sub_queue.task_done()

                workers = [sub_worker(i) for i in range(num_sub_workers)]
                await asyncio.gather(*workers)

        # Now aggregate the sub-summaries
        if self.verbose:
            console.print(
                f"[cyan]  {period_id}: Aggregating {len(sub_summaries)} {sub_level} summaries[/cyan]"
            )

        result = await self._aggregate_direct(sub_summaries, period_id, level)

        # Advance parent task for the final aggregation step
        if parent_progress and parent_task_id is not None:
            parent_progress.update(parent_task_id, advance=1)

        return result

    async def _synthesize_with_api(
        self, items: List[Dict], period_id: str, level: str, max_retries: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Call Kimi K2 to synthesize period summary or life profile.

        Args:
            items: Branch profiles or period summaries to aggregate
            period_id: Period identifier
            level: Time level
            max_retries: Maximum retry attempts for network errors (uses config default if None)

        Returns:
            Synthesized profile dict or None if API call fails
        """
        if not self.api_key:
            print(f"    ⚠️  No API key - skipping synthesis for {period_id}")
            return None

        # Use config default if not specified
        if max_retries is None:
            max_retries = config.models.aggregation.retries

        # Prepare input data
        input_data = json.dumps(items, indent=2)

        # Choose prompt and schema
        if level == "life":
            prompt_template = self.life_prompt
            schema = self.life_schema
            schema_name = "life_profile"
        else:
            prompt_template = self.period_prompt
            schema = self.period_schema
            schema_name = "period_summary"

        # Insert data into prompt
        prompt = prompt_template.replace(
            "[INPUT DATA WILL BE INSERTED HERE]", input_data
        )
        prompt = prompt.replace(
            "[INPUT: YEARLY OR MONTHLY SUMMARIES WILL BE INSERTED HERE]", input_data
        )

        # Insert pronoun forms for life-level synthesis
        if level == "life":
            pronouns = config.user.pronouns
            # Map pronoun sets to their forms
            pronoun_forms = {
                "he/him": {"pronouns": "he/him", "subject": "he", "object": "him", "possessive": "his", "possessive_pronoun": "his"},
                "she/her": {"pronouns": "she/her", "subject": "she", "object": "her", "possessive": "her", "possessive_pronoun": "hers"},
                "they/them": {"pronouns": "they/them", "subject": "they", "object": "them", "possessive": "their", "possessive_pronoun": "theirs"},
            }
            forms = pronoun_forms.get(pronouns, pronoun_forms["they/them"])

            # Add name to forms
            forms["name"] = config.user.name

            # Replace pronoun placeholders in prompt
            for key, value in forms.items():
                prompt = prompt.replace(f"{{{key}}}", value)

        # Prepare request with strict schema enforcement
        request = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a psychological profile synthesizer.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": config.models.aggregation.temperature,
            "max_tokens": config.models.aggregation.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
        }

        # Adaptive timeout based on level (larger aggregations need more time)
        timeout_map = {
            "week": config.models.aggregation.timeout_week_seconds,
            "month": config.models.aggregation.timeout_month_seconds,
            "year": config.models.aggregation.timeout_year_seconds,
            "life": config.models.aggregation.timeout_life_seconds
        }
        timeout = timeout_map.get(level, config.models.aggregation.timeout_week_seconds)

        # Setup debug logging
        if level == "life":
            debug_log_file = (
                self.debug_dir
                / "aggregator"
                / "life"
                / f"debug_life_{period_id}_attempt{{attempt}}.log"
            )
        else:
            debug_log_file = (
                self.debug_dir
                / "aggregator"
                / "periods"
                / f"debug_agg_{period_id}_attempt{{attempt}}.log"
            )

        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            current_log_file = Path(str(debug_log_file).format(attempt=attempt + 1))

            # Initialize log file BEFORE API call (so errors are logged too)
            with open(current_log_file, "w") as log:
                log.write(f"=== Aggregator Debug Log ===\n")
                log.write(f"Period: {period_id}\n")
                log.write(f"Level: {level}\n")
                log.write(f"Attempt: {attempt + 1}/{max_retries}\n")
                log.write(f"Model: {self.model}\n\n")

            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": os.environ.get(
                            "OPENROUTER_REFERER",
                            "https://github.com/olety/ST-Mirror",
                        ),
                        "X-Title": os.environ.get(
                            "OPENROUTER_TITLE", "ST-Mirror"
                        ),
                        "Content-Type": "application/json",
                    }

                    response = await client.post(
                        self.openrouter_url, headers=headers, json=request
                    )

                    # Log HTTP response (append to existing log)
                    with open(current_log_file, "a") as log:
                        log.write(f"=== HTTP Response ===\n")
                        log.write(f"Status: {response.status_code}\n")
                        log.write(
                            f"Raw response text (length: {len(response.text)}):\n"
                        )
                        log.write(response.text[:5000])  # First 5000 chars
                        log.write(f"\n\n")

                    # Check for HTTP errors and retry on server errors (5xx)
                    if response.status_code >= 500:
                        # Server error - retry with exponential backoff
                        if attempt < max_retries - 1:
                            wait_time = config.network.backoff_base ** attempt
                            with open(current_log_file, "a") as log:
                                log.write(
                                    f"⚠️  HTTP {response.status_code} (server error) - retrying\n"
                                )
                            console.print(
                                f"[yellow]    ⚠️  HTTP {response.status_code} error (attempt {attempt + 1}/{max_retries})[/yellow]"
                            )
                            console.print(
                                f"[yellow]       Retrying in {wait_time}s...[/yellow]"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            with open(current_log_file, "a") as log:
                                log.write(
                                    f"✗ HTTP {response.status_code} after {max_retries} attempts\n"
                                )
                                log.write(f"Response: {response.text[:500]}\n")
                            console.print(
                                f"[red]    ✗ HTTP {response.status_code} error after {max_retries} attempts[/red]"
                            )
                            console.print(
                                f"[red]       Response: {response.text[:200]}[/red]"
                            )
                            return None

                    # Raise for other HTTP errors (4xx)
                    response.raise_for_status()

                    # Parse API response
                    with open(current_log_file, "a") as log:
                        log.write(f"=== Parsing API Response ===\n")

                    try:
                        result = response.json()
                        with open(current_log_file, "a") as log:
                            log.write(f"✓ API response parsed successfully\n")
                            log.write(f"Generation ID: {result.get('id')}\n")
                            log.write(f"Model used: {result.get('model')}\n")
                            log.write(
                                f"Usage: {json.dumps(result.get('usage', {}), indent=2)}\n\n"
                            )
                    except json.JSONDecodeError as e:
                        # OpenRouter sometimes returns HTTP 200 with garbage (whitespace/empty)
                        # This is a retryable error - likely backend streaming issue
                        with open(current_log_file, "a") as log:
                            log.write(
                                f"✗ Failed to parse OpenRouter API response as JSON!\n"
                            )
                            log.write(f"Error: {e}\n")
                            log.write(f"Full response text:\n{response.text}\n")

                        if attempt < max_retries - 1:
                            wait_time = config.network.backoff_base ** attempt
                            with open(current_log_file, "a") as log:
                                log.write(f"⚠️  Retrying in {wait_time}s...\n")
                            console.print(
                                f"[yellow]    ⚠️  API returned invalid JSON (attempt {attempt + 1}/{max_retries})[/yellow]"
                            )
                            console.print(
                                f"[yellow]       Retrying in {wait_time}s...[/yellow]"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            with open(current_log_file, "a") as log:
                                log.write(f"✗ Failed to get valid JSON after {max_retries} attempts\n")
                            console.print(
                                f"[red]    ✗ API response parse error for {period_id} after {max_retries} attempts[/red]"
                            )
                            console.print(f"[red]       Error: {e}[/red]")
                            console.print(
                                f"[red]       Response preview: {response.text[:200]}...[/red]"
                            )
                            return None

                    # Check for provider streaming errors (e.g., Fireworks 502)
                    choice = result["choices"][0]
                    if "error" in choice:
                        error_info = choice["error"]
                        with open(current_log_file, "a") as log:
                            log.write(f"✗ Provider streaming error detected\n")
                            log.write(f"Message: {error_info.get('message')}\n")
                            log.write(f"Code: {error_info.get('code')}\n")
                            log.write(
                                f"Provider: {error_info.get('metadata', {}).get('provider_name')}\n"
                            )
                            if (
                                error_info.get("metadata", {})
                                .get("raw", {})
                                .get("retryable")
                            ):
                                log.write(f"Error is marked as retryable\n")
                        # Raise retryable error to trigger retry logic
                        raise httpx.RemoteProtocolError(
                            f"Provider streaming error: {error_info.get('message')}"
                        )

                    raw_content = choice["message"]["content"]

                    # Log raw content
                    with open(current_log_file, "a") as log:
                        log.write(f"=== Raw Content (length: {len(raw_content)}) ===\n")
                        log.write(raw_content)
                        log.write(
                            f"\n\n=== Content ends at char {len(raw_content)} ===\n\n"
                        )
                        log.write(f"=== Preprocessing ===\n")

                    # Preprocess: Strip markdown code fences if present
                    # Model sometimes breaks schema by outputting: {"thinking":"..."}\n```json\n{actual json}
                    content_to_parse = raw_content

                    # Remove markdown code fences
                    if "```json" in content_to_parse or "```" in content_to_parse:
                        with open(current_log_file, "a") as log:
                            log.write("Found markdown code fences, extracting JSON...\n")

                        # Extract content between ```json and ``` or after last {
                        # Try to find JSON block in code fence
                        json_match = re.search(r'```json\s*(\{.*\})\s*```', content_to_parse, re.DOTALL)
                        if json_match:
                            content_to_parse = json_match.group(1)
                            with open(current_log_file, "a") as log:
                                log.write(f"Extracted from code fence: {content_to_parse[:200]}...\n")
                        else:
                            # Find last complete JSON object
                            last_brace = content_to_parse.rfind('}')
                            if last_brace > 0:
                                # Find matching opening brace
                                depth = 0
                                for i in range(last_brace, -1, -1):
                                    if content_to_parse[i] == '}':
                                        depth += 1
                                    elif content_to_parse[i] == '{':
                                        depth -= 1
                                        if depth == 0:
                                            content_to_parse = content_to_parse[i:last_brace+1]
                                            with open(current_log_file, "a") as log:
                                                log.write(f"Extracted last complete object: {content_to_parse[:200]}...\n")
                                            break

                    with open(current_log_file, "a") as log:
                        log.write(f"=== Parsing Strategies (cleaned content length: {len(content_to_parse)}) ===\n")

                    # Try to parse model output with multiple strategies
                    parse_error = None
                    for parse_attempt in range(3):
                        try:
                            if parse_attempt == 0:
                                # First attempt: json-repair (handles LLM quirks)
                                # Fixes unescaped newlines, control chars, trailing commas, etc.
                                repaired = repair_json(content_to_parse)
                                synthesized = json.loads(repaired)
                                with open(current_log_file, "a") as log:
                                    log.write(f"✓ Strategy 1 (json-repair) succeeded\n")
                            elif parse_attempt == 1:
                                # Second attempt: direct parse (in case already valid)
                                synthesized = json.loads(content_to_parse)
                                with open(current_log_file, "a") as log:
                                    log.write(
                                        f"✓ Strategy 2 (direct parse) succeeded\n"
                                    )
                            else:
                                # Third attempt: custom normalizer (field name variations)
                                if self.verbose:
                                    console.print(
                                        f"[yellow]       Attempting full normalization...[/yellow]"
                                    )
                                synthesized = self.normalizer.normalize(content_to_parse)
                                with open(current_log_file, "a") as log:
                                    log.write(
                                        f"✓ Strategy {parse_attempt + 1} (custom normalizer) succeeded\n"
                                    )

                            # Validate that we got a dict, not a list or other type
                            if not isinstance(synthesized, dict):
                                with open(current_log_file, "a") as log:
                                    log.write(
                                        f"✗ Parsed result is {type(synthesized).__name__}, not dict\n"
                                    )
                                    log.write(f"Content: {str(synthesized)[:500]}\n")
                                raise ValueError(
                                    f"API returned {type(synthesized).__name__} instead of dict"
                                )

                            if self.verbose:
                                console.print(
                                    f"[green]    ✓ Synthesis complete ({level})[/green]"
                                )
                            return synthesized

                        except (json.JSONDecodeError, ValueError, KeyError) as e:
                            parse_error = e
                            with open(current_log_file, "a") as log:
                                log.write(
                                    f"✗ Strategy {parse_attempt + 1} failed: {e}\n"
                                )
                            if parse_attempt < 2:
                                continue

                    # All parse attempts failed
                    with open(current_log_file, "a") as log:
                        log.write(f"\n✗ All parsing strategies failed\n")
                        log.write(f"Final error: {parse_error}\n")
                    console.print(
                        f"[red]    ✗ Model JSON parse error for {period_id}[/red]"
                    )
                    console.print(f"[red]       Error: {parse_error}[/red]")
                    console.print(
                        f"[red]       Content preview: {content_to_parse[:200]}...[/red]"
                    )
                    return None

            except httpx.HTTPStatusError as e:
                # Client error (4xx) - don't retry
                with open(current_log_file, "a") as log:
                    log.write(f"✗ HTTP {e.response.status_code} (client error)\n")
                    log.write(f"Response: {e.response.text[:500]}\n")
                console.print(
                    f"[red]    ✗ HTTP error for {period_id}: {e.response.status_code}[/red]"
                )
                console.print(f"[red]       Response: {e.response.text[:200]}[/red]")
                return None

            except (
                httpx.RemoteProtocolError,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
            ) as e:
                # Network errors - retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = config.network.backoff_base ** attempt
                    with open(current_log_file, "a") as log:
                        log.write(f"⚠️  Network error - retrying\n")
                    console.print(
                        f"[yellow]    ⚠️  Network error (attempt {attempt + 1}/{max_retries}): {e}[/yellow]"
                    )
                    console.print(
                        f"[yellow]       Retrying in {wait_time}s...[/yellow]"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    with open(current_log_file, "a") as log:
                        log.write(
                            f"✗ Network error after {max_retries} attempts: {e}\n"
                        )
                    console.print(
                        f"[red]    ✗ Network error after {max_retries} attempts: {e}[/red]"
                    )
                    return None

            except Exception as e:
                with open(current_log_file, "a") as log:
                    log.write(f"✗ Unexpected error: {e}\n")
                console.print(f"[red]    ✗ Synthesis error for {period_id}: {e}[/red]")
                return None

        return None

    async def _aggregate_direct(
        self, items: List[Dict], period_id: str, level: str
    ) -> Dict:
        """
        Direct aggregation without splitting.

        This is where we would call Kimi K2 to synthesize.
        For now, returns metadata placeholder.

        Args:
            items: Either branch profiles or period summaries
            period_id: Period identifier
            level: Time level

        Returns:
            Period summary
        """
        # Check if output already exists (resume capability)
        if self.skip_existing:
            if level == "life":
                output_file = self.output_dir / "life" / f"{period_id}_profile.json"
            else:
                output_file = (
                    self.output_dir / f"{level}s" / f"{period_id}_summary.json"
                )

            if output_file.exists():
                with open(output_file) as f:
                    existing = json.load(f)
                if self.verbose:
                    console.print(f"[dim]    → Skipped (already exists)[/dim]")
                return existing

        # Determine date range
        dates = []
        for item in items:
            if "_source_file" in item:
                # Branch profile
                date = self.extract_date_from_profile(item)
                if date:
                    dates.append(date)
            elif "date_range" in item:
                # Period summary
                dr = item["date_range"]
                if dr.get("start"):
                    dates.append(datetime.fromisoformat(dr["start"]))
                if dr.get("end"):
                    dates.append(datetime.fromisoformat(dr["end"]))

        date_range = {
            "start": min(dates).isoformat() if dates else None,
            "end": max(dates).isoformat() if dates else None,
        }

        # Count branches and total decisions
        def count_branches(item):
            if "_source_file" in item:
                return 1
            else:
                return item.get("meta", {}).get("branches_count", 0)

        def count_decisions(item):
            if "_source_file" in item:
                # Branch profile - get from data_summary
                return item.get("data_summary", {}).get("total_decisions", 0)
            else:
                # Period summary - get from meta
                return item.get("meta", {}).get("total_decisions", 0)

        branches_count = sum(count_branches(item) for item in items)
        total_decisions = sum(count_decisions(item) for item in items)

        # Call API to synthesize
        synthesized = await self._synthesize_with_api(items, period_id, level)

        if synthesized:
            # Validate that synthesized is a dict, not a list or other type
            if not isinstance(synthesized, dict):
                console.print(
                    f"[red]    ✗ API returned invalid type for {period_id}: {type(synthesized).__name__} (expected dict)[/red]"
                )
                console.print(
                    f"[red]       Content preview: {str(synthesized)[:200]}[/red]"
                )
                return None

            # Use synthesized profile
            summary = synthesized
            # Add metadata
            summary["period"] = period_id
            summary["level"] = level
            summary["date_range"] = date_range
            if "meta" not in summary:
                summary["meta"] = {}
            summary["meta"].update(
                {
                    "branches_count": branches_count,
                    "total_decisions": total_decisions,
                    "items_aggregated": len(items),
                    "aggregation_method": (
                        "from_sub_summaries"
                        if not items[0].get("_source_file")
                        else "from_branches"
                    ),
                    "estimated_tokens": self.estimate_tokens(items),
                }
            )
        elif not self.api_key:
            # No API key provided - create placeholder
            summary = {
                "period": period_id,
                "level": level,
                "date_range": date_range,
                "meta": {
                    "branches_count": branches_count,
                    "total_decisions": total_decisions,
                    "items_aggregated": len(items),
                    "aggregation_method": (
                        "from_sub_summaries"
                        if not items[0].get("_source_file")
                        else "from_branches"
                    ),
                    "estimated_tokens": self.estimate_tokens(items),
                },
                "current_state": {"note": "Synthesis skipped (no API key)"},
                "evolution": {"note": "Synthesis skipped (no API key)"},
            }
        else:
            # Synthesis failed after retries - don't save placeholder
            console.print(
                f"[red]    ✗ Synthesis failed for {period_id} after all retries[/red]"
            )
            return None

        # Save JSON to appropriate directory
        if level == "life":
            output_file = self.output_dir / "life" / f"{period_id}_profile.json"
        else:
            output_file = self.output_dir / f"{level}s" / f"{period_id}_summary.json"

        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

        if self.verbose:
            console.print(f"[dim]    Saved JSON: {output_file.name}[/dim]")

        # Generate markdown report
        self._generate_report(summary, period_id, level)

        return summary

    def _generate_report(self, data: Dict, period_id: str, level: str):
        """
        Generate markdown report from JSON data.

        Args:
            data: Profile or summary data (JSON)
            period_id: Period identifier
            level: Time level ('week', 'month', 'year', 'life')
        """
        # Determine report directory
        if level == "life":
            reports_dir = self.output_dir.parent / "reports" / "life"
            report_file = reports_dir / f"{period_id}_report.md"
        else:
            reports_dir = self.output_dir.parent / "reports" / "periods"
            report_file = reports_dir / f"{period_id}_report.md"

        reports_dir.mkdir(parents=True, exist_ok=True)

        # Build markdown content
        lines = []

        # Header
        lines.append(f"# {period_id.upper()} - {level.title()} Profile")
        lines.append("")

        # Metadata
        if "meta" in data:
            meta = data["meta"]
            lines.append("## Overview")
            lines.append("")
            lines.append(f"- **Period**: {data.get('period', period_id)}")
            if "date_range" in data:
                dr = data["date_range"]
                start = dr.get("start") or "N/A"
                end = dr.get("end") or "N/A"
                start_str = start[:10] if start != "N/A" else "N/A"
                end_str = end[:10] if end != "N/A" else "N/A"
                lines.append(f"- **Date Range**: {start_str} to {end_str}")
            lines.append(f"- **Branches Analyzed**: {meta.get('branches_count', 0)}")
            lines.append(f"- **Total Decisions**: {meta.get('total_decisions', 0):,}")
            lines.append("")

        # Current State
        if "current_state" in data:
            current = data["current_state"]

            # Big Five
            if "big5" in current:
                lines.append("## Big Five Personality Traits")
                lines.append("")

                for trait_name, trait_data in current["big5"].items():
                    score = trait_data.get("score", 0)
                    lines.append(f"### {trait_name.replace('_', ' ').title()}")
                    lines.append(f"**Score**: {score:.2f}")
                    lines.append("")

                    # Aspects
                    if "aspects" in trait_data:
                        lines.append("**Aspects**:")
                        lines.append("")
                        for aspect_name, aspect_data in trait_data["aspects"].items():
                            aspect_score = aspect_data.get("score", 0)
                            lines.append(
                                f"- **{aspect_name.replace('_', ' ').title()}**: {aspect_score:.2f}"
                            )
                            if "evidence" in aspect_data and aspect_data["evidence"]:
                                evidence = aspect_data["evidence"]
                                if isinstance(evidence, list) and evidence:
                                    lines.append(f"  - {evidence[0]}")
                        lines.append("")

            # Attachment Style
            if "attachment" in current:
                attach = current["attachment"]
                lines.append("## Attachment Style")
                lines.append("")
                lines.append(
                    f"- **Primary Style**: {attach.get('primary_style', 'N/A')}"
                )
                lines.append(f"- **Anxiety Dimension**: {attach.get('anxiety', 0):.2f}")
                lines.append(
                    f"- **Avoidance Dimension**: {attach.get('avoidance', 0):.2f}"
                )
                lines.append("")

        # Period Summary (narrative)
        if "period_summary" in data:
            lines.append("## Period Summary")
            lines.append("")
            lines.append(data["period_summary"])
            lines.append("")

        # Companion Export (for life profiles)
        if level == "life" and "companion_export" in data:
            companion = data["companion_export"]

            if "personality_summary" in companion:
                lines.append("## Personality Summary")
                lines.append("")
                lines.append(companion["personality_summary"])
                lines.append("")

            if "growth_context" in companion:
                gc = companion["growth_context"]
                lines.append("## Growth Journey")
                lines.append("")
                if "healing_journey" in gc:
                    lines.append("### Healing Journey")
                    lines.append(gc["healing_journey"])
                    lines.append("")
                if "current_phase" in gc:
                    lines.append("### Current Phase")
                    lines.append(gc["current_phase"])
                    lines.append("")

        # Write report
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        if self.verbose:
            console.print(f"[dim]    Generated report: {report_file.name}[/dim]")

    async def _synthesize_with_progress(
        self, items: List[Dict], period_id: str, level: str, description: str
    ) -> Optional[Dict]:
        """
        Synthesize with live progress display for year/life aggregation.

        Args:
            items: Items to synthesize
            period_id: Period identifier
            level: Level (year/life)
            description: Description for display

        Returns:
            Synthesized summary
        """
        start_time = time.time()

        # Create status display
        def make_status():
            elapsed = int(time.time() - start_time)
            minutes, seconds = divmod(elapsed, 60)
            time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

            status = f"[cyan]⠋ {description}[/cyan]\n"
            status += f"  ├─ Input: {len(items)} {'summaries' if level != 'year' or len(items) > 1 else 'summary'}\n"
            status += f"  ├─ Synthesizing {level} profile...\n"
            status += f"  └─ [dim]API call in progress - {time_str}[/dim]"
            return status

        # Show live progress with periodic updates
        with Live(make_status(), console=console, refresh_per_second=2) as live:
            # Create task for synthesis
            synthesis_task = asyncio.create_task(self._aggregate_direct(items, period_id, level))

            # Update display periodically while synthesis runs
            while not synthesis_task.done():
                live.update(make_status())
                try:
                    await asyncio.wait_for(asyncio.shield(synthesis_task), timeout=0.5)
                except asyncio.TimeoutError:
                    continue  # Continue updating display

            # Get result (may raise exception if synthesis failed)
            try:
                result = synthesis_task.result()
            except Exception:
                result = None  # Treat exceptions as failure

        # No completion message - final summary shows all results
        return result

    async def build_hierarchy(self, profiles: List[Dict]) -> Optional[Dict]:
        """
        Build complete hierarchical aggregation.

        Returns:
            Life profile (top-level summary), or None if aggregation fails
        """
        if not profiles:
            raise ValueError("No profiles to aggregate")

        print("\n" + "=" * 60)
        print("BUILDING HIERARCHICAL AGGREGATION")
        print("=" * 60)

        # Analyze date range
        dates = [self.extract_date_from_profile(p) for p in profiles]
        dates = [d for d in dates if d]

        if not dates:
            print("\n⚠️  DEBUG: No dates extracted from profiles!")
            for i, p in enumerate(profiles[:3]):
                print(
                    f"  Profile {i}: profile_id={p.get('profile_id', 'N/A')[:60]}, chat_date={p.get('chat_date')}"
                )
            raise ValueError("Could not extract dates from any profiles")

        date_range = (min(dates), max(dates))
        print(
            f"\n📊 Date range: {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}"
        )
        print(f"   (Extracted {len(dates)}/{len(profiles)} dates successfully)")
        print(f"📊 Total profiles: {len(profiles)}")
        print(f"📊 Estimated tokens: ~{self.estimate_tokens(profiles)//1000}k")

        # Determine top level (year or month)
        year_span = date_range[1].year - date_range[0].year
        if year_span > 0:
            top_level = "year"
        else:
            top_level = "month"

        print(
            f"\n📊 Hierarchy: {'Life → Years → Months' if year_span > 0 else 'Life → Months'}"
        )
        print("")

        # Pre-scan the FULL hierarchy (years → months → weeks)
        year_groups = self.group_by_period(profiles, "year")

        # Build complete hierarchy metadata
        hierarchy_metadata = {}

        for year_id in sorted(year_groups.keys()):
            year_profiles = year_groups[year_id]
            month_groups = self.group_by_period(year_profiles, "month")

            year_meta = {"profiles": year_profiles, "months": {}}

            for month_id in sorted(month_groups.keys()):
                month_profiles = month_groups[month_id]
                needs_split = self.needs_splitting(month_profiles)

                month_meta = {
                    "profiles": month_profiles,
                    "needs_split": needs_split,
                }

                if needs_split:
                    week_groups = self.group_by_period(month_profiles, "week")
                    month_meta["weeks"] = week_groups
                    month_meta["total_work"] = len(week_groups) + 1
                else:
                    month_meta["total_work"] = 100

                year_meta["months"][month_id] = month_meta

            hierarchy_metadata[year_id] = year_meta

        period_summaries = []

        # Setup Rich progress bars with full hierarchy
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=2,  # Update display twice per second
        ) as progress:

            # Top level: Life Profile
            # Total = number of months (or years) + 1 for final life synthesis
            life_task = progress.add_task(
                f"[bold cyan]Life Profile ({len(profiles)} branches)",
                total=(
                    len(year_groups)
                    if year_span > 0
                    else len(
                        [m for y in hierarchy_metadata.values() for m in y["months"]]
                    )
                )
                + 1,
            )

            # Create hierarchy of tasks
            task_registry = {}  # Maps period_id to task_id

            for year_id in sorted(hierarchy_metadata.keys()):
                year_meta = hierarchy_metadata[year_id]

                if year_span > 0:
                    # Multi-year: show year level
                    year_task = progress.add_task(
                        f"  [cyan]{year_id} ({len(year_meta['profiles'])} branches)",
                        total=len(year_meta["months"]),
                    )
                    task_registry[year_id] = {
                        "task_id": year_task,
                        "level": "year",
                        "parent": life_task,
                    }

                for month_id in sorted(year_meta["months"].keys()):
                    month_meta = year_meta["months"][month_id]
                    indent = "    " if year_span > 0 else "  "
                    week_indent = indent + "  "  # Weeks get 2 extra spaces

                    if month_meta["needs_split"]:
                        # Month with weeks
                        month_task = progress.add_task(
                            f"{indent}[yellow]{month_id} → {len(month_meta['weeks'])} weeks",
                            total=month_meta["total_work"],
                            completed=0,
                        )

                        # Create week slots
                        num_week_workers = min(
                            self.max_parallel, len(month_meta["weeks"])
                        )
                        week_slots = []
                        for i in range(num_week_workers):
                            slot = progress.add_task(
                                f"{week_indent}[dim]Slot {i+1}",
                                total=1,
                                visible=False,
                            )
                            week_slots.append(slot)
                        month_meta["week_slots"] = week_slots
                        month_meta["week_indent"] = week_indent  # Store for worker
                    else:
                        # Month without splitting
                        month_task = progress.add_task(
                            f"{indent}[yellow]{month_id} ({len(month_meta['profiles'])} profiles)",
                            total=1,
                            completed=0,
                        )

                    task_registry[month_id] = {
                        "task_id": month_task,
                        "level": "month",
                        "parent": (
                            task_registry.get(year_id, {}).get("task_id")
                            if year_span > 0
                            else life_task
                        ),
                        "meta": month_meta,
                    }

            # Create queue and process months in parallel
            month_queue = asyncio.Queue()
            for year_id in sorted(hierarchy_metadata.keys()):
                for month_id in sorted(hierarchy_metadata[year_id]["months"].keys()):
                    await month_queue.put((year_id, month_id))

            # Worker function
            async def month_worker():
                while True:
                    try:
                        year_id, month_id = month_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    month_meta = hierarchy_metadata[year_id]["months"][month_id]
                    month_reg = task_registry[month_id]

                    # Process month
                    summary = await self.aggregate_period(
                        month_meta["profiles"],
                        month_id,
                        "month",
                        progress=progress,
                        task_id=month_reg["task_id"],
                        week_slot_tasks=month_meta.get("week_slots"),
                        week_indent=month_meta.get("week_indent"),
                    )
                    period_summaries.append(summary)

                    # Ensure task reaches 100%
                    current = progress.tasks[month_reg["task_id"]].completed
                    total = progress.tasks[month_reg["task_id"]].total
                    if current < total:
                        progress.update(month_reg["task_id"], completed=total)

                    # Update parent (year or life)
                    progress.update(month_reg["parent"], advance=1)

                    month_queue.task_done()

            # Run workers
            num_workers = min(
                self.max_parallel,
                sum(len(y["months"]) for y in hierarchy_metadata.values()),
            )
            workers = [month_worker() for _ in range(num_workers)]
            await asyncio.gather(*workers)

            # Now synthesize year/life within Progress context to complete the bar
            if year_span > 0:
                # Group month summaries by year
                year_summaries = []
                for year_id in sorted(hierarchy_metadata.keys()):
                    year_month_summaries = [
                        s for s in period_summaries if s["period"].startswith(year_id)
                    ]

                    year_summary = await self._aggregate_direct(
                        year_month_summaries, year_id, "year"
                    )
                    if not year_summary:
                        console.print(
                            f"[red]✗ Failed to synthesize {year_id}, aborting life aggregation[/red]"
                        )
                        return None
                    year_summaries.append(year_summary)

                # Aggregate years to life
                life_profile = await self._aggregate_direct(year_summaries, "life", "life")
                if not life_profile:
                    console.print(
                        f"[red]✗ Failed to synthesize life profile, aborting[/red]"
                    )
                    return None
            else:
                # Single year: aggregate months directly to life
                life_profile = await self._aggregate_direct(
                    period_summaries, "life", "life"
                )
                if not life_profile:
                    console.print(
                        f"[red]✗ Failed to synthesize life profile, aborting[/red]"
                    )
                    return None

            # Mark life synthesis as complete
            progress.update(life_task, advance=1)

        # Save life profile
        life_output = self.output_dir / "life" / "life_profile.json"
        with open(life_output, "w") as f:
            json.dump(life_profile, f, indent=2)

        # Save companion export as separate file
        if "companion_export" in life_profile:
            companion_output = self.output_dir / "life" / "companion_export.json"
            with open(companion_output, "w") as f:
                json.dump(life_profile["companion_export"], f, indent=2)
            # Companion export path shown in completion summary below

        # Beautiful completion summary
        console.print()
        console.print("═" * 70, style="cyan")
        console.print("✓ HIERARCHY COMPLETE", style="bold green")
        console.print("═" * 70, style="cyan")
        console.print()

        # Summary table
        summary_table = Table(show_header=False, box=None, padding=(0, 2))
        summary_table.add_row("Life Profile Created", "")
        summary_table.add_row(
            "  • Branches analyzed:",
            f"[cyan]{life_profile['meta']['branches_count']}[/cyan]",
        )
        summary_table.add_row(
            "  • Decisions processed:",
            f"[cyan]{life_profile['meta']['total_decisions']:,}[/cyan]",
        )
        summary_table.add_row(
            "  • Date range:",
            f"[cyan]{life_profile['date_range']['start'][:10]} to {life_profile['date_range']['end'][:10]}[/cyan]",
        )
        if year_span > 0:
            summary_table.add_row(
                "  • Hierarchy:",
                f"[dim]{len(profiles)} branches → {len(hierarchy_metadata)} years → life[/dim]",
            )
        else:
            summary_table.add_row(
                "  • Hierarchy:",
                f"[dim]{len(profiles)} branches → {len(period_summaries)} months → life[/dim]",
            )
        console.print(summary_table)
        console.print()
        console.print(f"[green]✓[/green] Life profile: {life_output}")
        if "companion_export" in life_profile:
            companion_output = self.output_dir / "life" / "companion_export.json"
            console.print(f"[green]✓[/green] Companion export: {companion_output}")
        console.print()

        return life_profile
