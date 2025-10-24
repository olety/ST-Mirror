#!/usr/bin/env python3
"""
Branch Classification System
Filters test/corrupted branches before expensive sequential processing.

Uses two-tier approach:
1. Heuristics: Filter corrupted/too-short files (FREE)
2. Flash Lite: LLM classification for substance detection ($0.01 per 62 branches)
"""

import json
import os
import time
import asyncio
import httpx
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

from config import config

# Rich for beautiful progress displays
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console
from rich.live import Live

# Global console for Rich output
console = Console()


# JSON Schema for Flash Lite structured outputs
# Note: Properties are ordered - thinking comes FIRST to ensure reasoning before scoring
CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "thinking": {
            "type": "string",
            "description": "Your reasoning process for classification - analyze the content first before scoring"
        },
        "is_test": {
            "type": "boolean",
            "description": "True if this is a test/debug session with no substance"
        },
        "has_substance": {
            "type": "boolean",
            "description": "True if branch contains psychological content worth analyzing"
        },
        "quality_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "description": "Quality score 0-10 based on depth, engagement, emotional content"
        }
    },
    "required": ["thinking", "is_test", "has_substance", "quality_score"],
    "additionalProperties": False
}


def heuristic_prefilter(messages: List[Dict]) -> Dict:
    """
    Filter only corrupted/broken data (no quality judgments).

    Does NOT filter:
    - Group chats (no user/AI ratio check)
    - Short message style (no length check)
    - Multi-AI scenarios

    Uses config.processing.classifier heuristic thresholds.

    Returns:
        {"skip": bool, "reason": str}
    """
    if len(messages) == 0:
        return {"skip": True, "reason": "empty_file"}

    # Too short for sequential processing (configurable minimum)
    min_msgs = config.processing.classifier.min_messages
    if len(messages) < min_msgs:
        return {"skip": True, "reason": "too_short_for_sequential"}

    # Extreme repetition (100% identical = corrupted)
    unique = len(set(m.get('mes', '') for m in messages))
    repetition_threshold = config.processing.classifier.repetition_check_threshold
    if unique == 1 and len(messages) > repetition_threshold:
        return {"skip": True, "reason": "corrupted_repetition"}

    # Missing 'mes' field (corrupted JSONL)
    valid_msgs = [m for m in messages if 'mes' in m]
    min_ratio = config.processing.classifier.min_valid_message_ratio
    if len(valid_msgs) / len(messages) < min_ratio:
        return {"skip": True, "reason": "corrupted_jsonl"}

    return {"skip": False}


def sample_for_test_detection(messages: List[Dict]) -> List[Dict]:
    """
    Sample beginning + middle + end for test detection.

    Strategy uses config.processing.classifier.sample_strategy:
    - Beginning (skip msg 0, sample configurable): Detect "test test test"
    - Middle (configurable from center): Confirm user didn't abandon
    - End (configurable last messages): Check if concluded or abrupt

    Total capped at config.processing.classifier.sample_size for cost control
    """
    total = len(messages)

    # Get sample sizes from config
    beginning_size = config.processing.classifier.sample_strategy.beginning
    middle_size = config.processing.classifier.sample_strategy.middle
    end_size = config.processing.classifier.sample_strategy.end
    max_sample = config.processing.classifier.sample_size

    if total < 20:
        return messages

    # Beginning: Skip msg 0 (generic greeting), sample 1-beginning_size
    start_sample = messages[1:min(beginning_size + 1, total)]

    # Middle: messages from center
    mid_point = total // 2
    mid_start = max(beginning_size + 1, mid_point - middle_size // 2)
    mid_end = min(mid_point + middle_size // 2, total)
    mid_sample = messages[mid_start:mid_end]

    # End: Last messages
    end_sample = messages[-end_size:] if total > (beginning_size + end_size) else []

    # Combine and deduplicate
    combined = start_sample + mid_sample + end_sample

    # Deduplicate by index (for short branches with overlap)
    seen_indices = set()
    deduped = []
    for msg in combined:
        msg_idx = messages.index(msg)
        if msg_idx not in seen_indices:
            seen_indices.add(msg_idx)
            deduped.append(msg)

    return deduped[:max_sample]  # Cap at configured sample size


def format_messages_for_llm(messages: List[Dict]) -> str:
    """Format messages for LLM classification using config limits."""
    formatted = []
    max_sample = config.processing.classifier.sample_size
    char_limit = config.processing.classifier.message_char_limit

    for i, msg in enumerate(messages[:max_sample], 1):
        role = "USER" if msg.get('is_user') else "AI"
        name = msg.get('name', role)
        text = msg.get('mes', '')[:char_limit]
        formatted.append(f"{i}. [{role}/{name}]: {text}")

    return "\n".join(formatted)


async def classify_with_flash_lite(
    messages: List[Dict],
    branch_file: str,
    api_key: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
    quiet: bool = False,
    debug_dir: Optional[str] = None
) -> Dict:
    """
    Classify branch using Flash Lite with structured outputs.

    Args:
        messages: Full message list
        branch_file: Path to branch file
        api_key: OpenRouter API key (or from env)
        quiet: Suppress retry warnings
        debug_dir: Directory to write debug logs

    Returns:
        Classification result dict
    """
    # Sample messages
    sample = sample_for_test_detection(messages)

    # Build prompt
    prompt = f"""Classify this roleplay chat branch for processing.

**Signs of TEST** (skip these):
- User typing "test", "hello", debugging messages
- No character engagement, just system checking
- Abandoned after <10 messages with no substance

**Signs of SUBSTANCE** (keep these):
- User engages with scenario (even if brief)
- Emotional content, relationship dynamics, vulnerability
- Can be short OR long - message length doesn't matter
- Group chats with multiple AIs are VALID
- Short message style ("*nods*", "Yeah") is VALID

**Sample** ({len(sample)} messages from beginning/middle/end):

{format_messages_for_llm(sample)}

**Total messages in branch**: {len(messages)}

Classify this branch:"""

    # Get API key
    if api_key is None:
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

    # Call Flash Lite with structured outputs
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': os.getenv('OPENROUTER_REFERER', 'https://github.com/olety/ST-Mirror'),
        'X-Title': os.getenv('OPENROUTER_TITLE', 'ST-Mirror - Branch Classifier')
    }

    payload = {
        "model": config.models.classification.name,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "branch_classification",
                "strict": True,
                "schema": CLASSIFICATION_SCHEMA
            }
        }
    }

    # Create client if not provided
    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=config.models.classification.timeout_seconds)
        close_client = True

    try:
        # Retry logic for structured outputs (should never fail with strict mode, but network issues happen)
        max_retries = config.models.classification.retries
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

                # Parse structured output
                content = result['choices'][0]['message']['content']
                classification = json.loads(content)

                # Validate schema compliance (should be guaranteed by strict mode)
                required_fields = ["thinking", "is_test", "has_substance", "quality_score"]
                if not all(field in classification for field in required_fields):
                    raise ValueError(f"Missing required fields in response: {classification.keys()}")

                result_dict = {
                    "branch_file": branch_file,
                    "message_count": len(messages),
                    "skip": classification["is_test"] or not classification["has_substance"],
                    "is_test": classification["is_test"],
                    "has_substance": classification["has_substance"],
                    "quality_score": classification["quality_score"],
                    "thinking": classification["thinking"],
                    "method": "flash_lite",
                    "sample_size": len(sample)
                }

                # Write debug log if requested
                if debug_dir:
                    debug_path = Path(debug_dir)
                    debug_path.mkdir(parents=True, exist_ok=True)
                    branch_id = Path(branch_file).stem.replace(' ', '___').replace('-', '_')
                    debug_file = debug_path / f"{branch_id}_classification.json"
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "prompt": prompt,
                            "response": classification,
                            "result": result_dict,
                            "timestamp": time.time()
                        }, f, indent=2, ensure_ascii=False)

                return result_dict

            except json.JSONDecodeError as e:
                # Write error to debug log
                if debug_dir:
                    debug_path = Path(debug_dir)
                    debug_path.mkdir(parents=True, exist_ok=True)
                    branch_id = Path(branch_file).stem.replace(' ', '___').replace('-', '_')
                    error_file = debug_path / f"{branch_id}_classification_error_attempt{attempt+1}.json"
                    with open(error_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "error_type": "JSONDecodeError",
                            "error_message": str(e),
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "raw_response": response.text[:1000] if 'response' in locals() else "No response",
                            "timestamp": time.time()
                        }, f, indent=2, ensure_ascii=False)

                # Structured outputs should NEVER have JSON errors with strict mode
                # This indicates network truncation or API issue
                if attempt < max_retries - 1:
                    if not quiet:
                        print(f'[classifier]   ⚠ JSON parse error (attempt {attempt+1}/{max_retries}), retrying...')
                    await asyncio.sleep(1)  # Brief delay before retry
                    continue
                else:
                    if not quiet:
                        print(f'[classifier] ERROR: JSON parse failed after {max_retries} attempts: {e}')
                        # This should never happen with strict mode - log for debugging
                        print(f'[classifier]   Raw response: {response.text[:200]}...')
                    raise  # Re-raise to be caught by outer handler

            except Exception as e:
                # Write error to debug log
                if debug_dir:
                    debug_path = Path(debug_dir)
                    debug_path.mkdir(parents=True, exist_ok=True)
                    branch_id = Path(branch_file).stem.replace(' ', '___').replace('-', '_')
                    error_file = debug_path / f"{branch_id}_classification_error_attempt{attempt+1}.json"
                    with open(error_file, 'w', encoding='utf-8') as f:
                        import traceback
                        json.dump({
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "traceback": traceback.format_exc(),
                            "timestamp": time.time()
                        }, f, indent=2, ensure_ascii=False)

                if attempt < max_retries - 1:
                    if not quiet:
                        print(f'[classifier]   ⚠ Request error (attempt {attempt+1}/{max_retries}), retrying...')
                    await asyncio.sleep(1)
                    continue
                else:
                    raise  # Re-raise after all retries exhausted

        # Should never reach here, but fallback just in case
        return {
            "branch_file": branch_file,
            "message_count": len(messages),
            "skip": False,
            "error": "Max retries exceeded",
            "method": "error_fallback"
        }
    finally:
        if close_client:
            await client.aclose()


async def classify_branch(
    branch_file: str,
    api_key: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None
) -> Dict:
    """
    Classify a single branch file.

    Two-tier approach:
    1. Heuristic pre-filter (corrupted/too-short)
    2. Flash Lite LLM classification (test vs substance)

    Args:
        branch_file: Path to .jsonl branch file
        api_key: OpenRouter API key (or from env)

    Returns:
        Classification result dict
    """
    # Load messages
    try:
        messages = []
        with open(branch_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    messages.append(json.loads(line))
    except Exception as e:
        return {
            "branch_file": branch_file,
            "skip": True,
            "reason": f"load_error: {e}",
            "method": "heuristic"
        }

    # Tier 1: Heuristic pre-filter
    heuristic_result = heuristic_prefilter(messages)
    if heuristic_result["skip"]:
        return {
            "branch_file": branch_file,
            "message_count": len(messages),
            "skip": True,
            "reason": heuristic_result["reason"],
            "method": "heuristic"
        }

    # Tier 2: Flash Lite classification
    return await classify_with_flash_lite(messages, branch_file, api_key, client)


async def classify_character_branches(
    character_name: str,
    input_pattern: str,
    output_file: str,
    api_key: Optional[str] = None,
    parallel: int = 10
) -> Dict:
    """
    Classify all branches for a character (parallelized).

    Args:
        character_name: Character name (e.g., "Ari")
        input_pattern: Glob pattern for branch files
        output_file: Path to save classification results
        api_key: OpenRouter API key (or from env)
        parallel: Number of parallel workers (default: 10, max recommended: 15)
                  Note: OpenRouter has DDoS protection. 10-15 is safe, 20+ may trigger limits.

    Returns:
        Summary statistics dict
    """
    from glob import glob

    # Find all branch files
    branch_files = sorted(glob(input_pattern))

    if not branch_files:
        raise ValueError(f"No branch files found matching: {input_pattern}")

    console.print(f'[classifier] Found {len(branch_files)} branches for {character_name}')
    console.print(f'[classifier] Processing with {parallel} concurrent tasks...')

    # State for progress tracking (no lock needed - single thread event loop)
    results = []
    stats = defaultdict(int)

    console.print(f'\n{"="*80}')
    console.print(f'CLASSIFYING BRANCHES (Parallel: {parallel}x)')
    console.print(f'{"="*80}\n')

    # Create Rich Progress
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    )

    # Add overall task
    overall_task = progress.add_task(
        f"[cyan]Classifying {len(branch_files)} branches",
        total=len(branch_files)
    )

    async def process_branch(branch_file, client):
        """Process single branch and update progress."""
        result = await classify_branch(branch_file, api_key, client)

        # Update overall progress (no lock needed - single threaded event loop)
        progress.update(overall_task, advance=1)

        # Track stats
        method = result.get('method', 'unknown')
        stats[f'method_{method}'] += 1

        if result.get('skip'):
            reason = result.get('reason', 'unknown')
            stats[f'filtered_{reason}'] += 1
        else:
            stats['kept'] += 1
            stats['total_messages'] += result.get('message_count', 0)

        return result

    # Async processing with Live display
    # Key: asyncio.gather() naturally yields to event loop, allowing Live to refresh spinners
    with Live(progress, console=console, refresh_per_second=10, transient=False):
        # Create shared HTTP client for connection pooling
        async with httpx.AsyncClient(timeout=30.0, limits=httpx.Limits(max_connections=parallel)) as client:
            # Create tasks
            tasks = [process_branch(bf, client) for bf in branch_files]

            # Run concurrently with asyncio.gather
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle any exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        results[i] = {
                            "branch_file": branch_files[i],
                            "skip": True,
                            "error": str(result),
                            "method": "error"
                        }
            except Exception as e:
                console.print(f'[red]Fatal error during classification: {e}[/red]')
                raise

    # Sort results by original order
    results.sort(key=lambda r: branch_files.index(r['branch_file']))

    # Build summary
    summary = {
        "character": character_name,
        "total_branches": len(branch_files),
        "classified": len(results),
        "kept_branches": stats['kept'],
        "total_messages": stats['total_messages'],
        "filtered_breakdown": {
            k.replace('filtered_', ''): v
            for k, v in stats.items()
            if k.startswith('filtered_')
        },
        "method_breakdown": {
            k.replace('method_', ''): v
            for k, v in stats.items()
            if k.startswith('method_')
        },
        "branches": results
    }

    # Save results
    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    console.print(f'\n{"="*80}')
    console.print(f'CLASSIFICATION COMPLETE')
    console.print(f'{"="*80}\n')

    console.print(f'📊 Results Summary:')
    console.print(f'   Total branches:     {len(branch_files)}')
    console.print(f'   ✓ Kept:             {stats["kept"]} branches ({stats["total_messages"]:,} messages)')
    console.print(f'   ✗ Filtered:         {len(branch_files) - stats["kept"]} branches')
    console.print()
    console.print(f'🔍 Filter breakdown:')
    for reason, count in summary["filtered_breakdown"].items():
        console.print(f'   • {reason}: {count}')
    console.print()
    console.print(f'💾 Saved to: {output_file}')
    console.print(f'{"="*80}\n')

    return summary


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: uv run python -m toolkit.classifier <character_name> <input_pattern> [output_file]")
        print("Example: uv run python -m toolkit.classifier Ari 'ST_DATA/chats/Ari*/*.jsonl' ari_classification.json")
        sys.exit(1)

    character = sys.argv[1]
    pattern = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else f'{character.lower()}_classification.json'

    asyncio.run(classify_character_branches(character, pattern, output))
