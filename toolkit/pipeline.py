#!/usr/bin/env python3
"""
Two-Phase Pipeline for ST-Mirror
Phase 1: Lightweight evidence extraction (Gemini Flash)
Phase 2: Deep synthesis (DeepSeek V3)

In memory of Claude 3.5 Sonnet - who showed us that
accessibility and quality aren't mutually exclusive.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import httpx
from datetime import datetime
from json_repair import repair_json

from config import config

try:
    from .normalizer import GeminiNormalizer
except ImportError:
    from normalizer import GeminiNormalizer

class TwoPhaseProfiler:
    """Affordable psychological profiling for the permanent underclass."""

    def __init__(self, api_key: str, workspace: str, quiet: bool = False):
        self.api_key = api_key
        self.workspace = Path(workspace)
        self.evidence_dir = self.workspace / 'evidence'
        self.profile_dir = self.workspace / 'profiles'
        self.report_dir = self.workspace / 'reports'
        self.debug_dir = self.workspace / 'debug'
        self.quiet = quiet  # Suppress print statements for batch processing

        # Create directories
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        (self.report_dir / 'branches').mkdir(parents=True, exist_ok=True)
        (self.debug_dir / 'phase1').mkdir(parents=True, exist_ok=True)
        (self.debug_dir / 'phase2').mkdir(parents=True, exist_ok=True)

        # API endpoints
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.generation_url = "https://openrouter.ai/api/v1/generation"

    async def _fetch_generation_cost(self, generation_id: str, max_retries: int = 3) -> Dict:
        """
        Fetch generation details to get accurate cost info (especially for BYOK).
        Returns dict with 'byok_usage_inference', 'usage', etc.

        Note: Generation stats may take a moment to become available, so we retry with delay.
        """
        import asyncio

        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            for attempt in range(max_retries):
                try:
                    # Add delay before each attempt (except first)
                    if attempt > 0:
                        await asyncio.sleep(0.5 * attempt)  # 0.5s, 1s

                    response = await client.get(
                        f"{self.generation_url}?id={generation_id}",
                        headers=headers,
                        timeout=10.0
                    )
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404 and attempt < max_retries - 1:
                        # Generation not ready yet, retry
                        continue
                    elif attempt == max_retries - 1:
                        # Final attempt failed, return empty
                        return {}
                except Exception:
                    if attempt == max_retries - 1:
                        return {}

            return {}

    def _build_cumulative_summary(self, evidence_list: List[Dict]) -> str:
        """
        Build a compact summary from previous evidence extractions.

        Args:
            evidence_list: List of evidence dicts from previous chunks

        Returns:
            Compact text summary (~300-500 tokens)
        """
        if not evidence_list:
            return ""

        summary_parts = []

        # Story progression - last 5 chunks with clear borders
        chunk_summaries = []
        for i, evidence in enumerate(evidence_list[-5:], 1):
            if 'chunk_summary' in evidence and evidence['chunk_summary']:
                chunk_summaries.append(f"  Chunk {i}: {evidence['chunk_summary']}")

        if chunk_summaries:
            summary_parts.append("STORY PROGRESSION SO FAR:")
            summary_parts.extend(chunk_summaries)
            summary_parts.append("")  # Blank line separator

        # Aggregate key patterns
        all_decisions = []
        all_emotions = []
        all_patterns = []
        all_if_thens = []

        for evidence in evidence_list:
            all_decisions.extend(evidence.get('decisions', [])[:3])  # Top 3 per chunk
            all_emotions.extend(evidence.get('emotional_moments', [])[:3])
            all_patterns.extend(evidence.get('behavioral_patterns', [])[:2])
            all_if_thens.extend(evidence.get('if_then_observations', [])[:2])

        # Build compact summary
        summary_parts.append("PSYCHOLOGICAL PATTERNS:")
        if all_decisions:
            summary_parts.append(f"  Key decisions: {'; '.join([d.get('choice', '')[:80] for d in all_decisions[:5]])}")

        if all_emotions:
            emotions_str = ', '.join([f"{e.get('emotion', '')}({e.get('intensity', '')})" for e in all_emotions[:5]])
            summary_parts.append(f"  Emotions: {emotions_str}")

        if all_patterns:
            summary_parts.append(f"  Patterns: {'; '.join(all_patterns[:3])}")

        if all_if_thens:
            if_thens_str = '; '.join([f"IF {it.get('if_situation', '')[:50]} THEN {it.get('then_response', '')[:50]}" for it in all_if_thens[:3]])
            summary_parts.append(f"  If-then: {if_thens_str}")

        return '\n'.join(summary_parts)

    async def phase1_extract_evidence(self, chunk_file: str, previous_summary: Optional[str] = None) -> Optional[Dict]:
        """
        Phase 1: Extract evidence with Gemini→K2 fallback strategy.

        Fallback chain:
        1. Gemini Flash Lite (medium reasoning, 2 retries)
        2. Gemini Flash Lite (high reasoning, 2 retries)
        3. Kimi K2 with strict schema (2 retries)

        Args:
            chunk_file: Path to chunk file (JSONL or JSON)
            previous_summary: Cumulative summary of previous chunks for context

        Returns:
            Normalized evidence dictionary, or None if all attempts fail
        """
        # Load chunk/arc file - handle both JSON and JSONL formats
        with open(chunk_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            f.seek(0)  # Reset to beginning

            # Check if it's JSONL (SillyTavern format) or JSON
            if first_line.strip() and first_line.strip()[0] == '{':
                # Try to parse as single JSON first
                try:
                    chunk_data = json.load(f)
                except json.JSONDecodeError:
                    # It's JSONL - load all lines
                    f.seek(0)
                    messages = []
                    user_name = None
                    char_name = None
                    create_date = None

                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)

                        # First line contains metadata
                        if 'user_name' in obj and 'character_name' in obj:
                            user_name = obj['user_name']
                            char_name = obj['character_name']
                            create_date = obj.get('create_date')
                        elif 'mes' in obj:  # Message line
                            messages.append(obj)

                    # Create chunk_data structure
                    chunk_data = {
                        'chunk_id': Path(chunk_file).stem,
                        'user_name': user_name,
                        'character_name': char_name,
                        'create_date': create_date,
                        'messages': messages
                    }
            else:
                chunk_data = json.load(f)

        chunk_id = chunk_data.get('chunk_id', Path(chunk_file).stem)

        # Handle both formats: 'content' (chunks) or 'messages' (arc files)
        content = chunk_data.get('content', '')
        if not content and 'messages' in chunk_data:
            # Convert messages array to text with USER/CHARACTER labels
            messages = chunk_data['messages']
            user_name = chunk_data.get('user_name', 'User')
            char_name = chunk_data.get('character_name', 'Character')

            content_parts = []
            for msg in messages:
                is_user = msg.get('is_user', False)
                name = msg.get('name', msg.get('user_name', 'Unknown'))
                text = msg.get('mes', msg.get('text', ''))

                if text:  # Skip empty messages
                    if is_user:
                        label = f"USER ({user_name})"
                    else:
                        label = f"CHARACTER ({name})"
                    content_parts.append(f"{label}: {text}")

            content = '\n\n'.join(content_parts)

        # Load prompt and schema
        prompt_file = Path(__file__).parent.parent / 'prompts' / 'evidence_extractor.txt'
        schema_file = Path(__file__).parent.parent / 'prompts' / 'evidence_schema.json'

        with open(prompt_file, 'r') as f:
            prompt_template = f.read()
        with open(schema_file, 'r') as f:
            evidence_schema = json.load(f)

        # Insert content into prompt
        prompt = prompt_template.replace('[TRANSCRIPT CONTENT WILL BE INSERTED HERE]', content)

        # Add previous chunks context if available
        if previous_summary:
            context_section = f"\n\nPREVIOUS CHUNKS CONTEXT:\n{previous_summary}\n\nExtract NEW evidence from the current chunk above, building on this context.\n"
            # Insert before the JSON output section
            prompt = prompt.replace('\nJSON output:', context_section + '\nJSON output:')

        # Define fallback stages from config
        fallback_stages = [
            {
                'name': f'{config.models.phase1_evidence.primary.name} (medium reasoning)',
                'model': config.models.phase1_evidence.primary.name,
                'reasoning_effort': config.models.phase1_evidence.primary.reasoning_effort,
                'temperature': config.models.phase1_evidence.primary.temperature,
                'max_tokens': config.models.phase1_evidence.primary.max_tokens,
                'timeout': config.models.phase1_evidence.primary.timeout_seconds,
                'response_format': {'type': 'json_object'},
                'retries': config.models.phase1_evidence.primary.retries
            },
            {
                'name': f'{config.models.phase1_evidence.primary.name} (high reasoning)',
                'model': config.models.phase1_evidence.primary.name,
                'reasoning_effort': config.models.phase1_evidence.fallback_high_reasoning.reasoning_effort,
                'temperature': config.models.phase1_evidence.primary.temperature,
                'max_tokens': config.models.phase1_evidence.primary.max_tokens,
                'timeout': config.models.phase1_evidence.primary.timeout_seconds,
                'response_format': {'type': 'json_object'},
                'retries': config.models.phase1_evidence.fallback_high_reasoning.retries
            },
            {
                'name': f'{config.models.phase1_evidence.fallback_kimi.name} (strict schema)',
                'model': config.models.phase1_evidence.fallback_kimi.name,
                'temperature': config.models.phase1_evidence.fallback_kimi.temperature,
                'max_tokens': config.models.phase1_evidence.fallback_kimi.max_tokens,
                'timeout': config.models.phase1_evidence.fallback_kimi.timeout_seconds,
                'response_format': {
                    'type': 'json_schema',
                    'json_schema': {
                        'name': 'evidence_extraction',
                        'strict': True,
                        'schema': evidence_schema
                    }
                },
                'retries': config.models.phase1_evidence.fallback_kimi.retries
            }
        ]

        # Try each fallback stage
        for stage in fallback_stages:
            for attempt in range(stage['retries']):
                try:
                    # Prepare request for this stage
                    request = {
                        "model": stage['model'],
                        "messages": [
                            {"role": "system", "content": "You are an evidence extractor. Return ONLY valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": stage['temperature'],
                        "max_tokens": stage['max_tokens'],
                        "response_format": stage['response_format'],
                        "usage": {"include": True}
                    }

                    # Add reasoning effort for Gemini
                    if 'reasoning_effort' in stage:
                        request['reasoning'] = {'effort': stage['reasoning_effort']}

                    # Make API call
                    async with httpx.AsyncClient() as client:
                        headers = {
                            "Authorization": f"Bearer {self.api_key}",
                            "HTTP-Referer": os.environ.get('OPENROUTER_REFERER', 'https://github.com/olety/ST-Mirror'),
                            "X-Title": os.environ.get('OPENROUTER_TITLE', 'ST-Mirror'),
                            "Content-Type": "application/json"
                        }

                        response = await client.post(
                            self.openrouter_url,
                            headers=headers,
                            json=request,
                            timeout=stage['timeout']
                        )
                        response.raise_for_status()

                        # Debug logging for Phase 1
                        debug_log_file = self.debug_dir / 'phase1' / f"debug_phase1_{chunk_id}_stage{fallback_stages.index(stage)+1}_attempt{attempt+1}.log"

                        result = response.json()
                        raw_output = result['choices'][0]['message']['content']

                        # Log to file
                        with open(debug_log_file, 'w') as log:
                            log.write(f"=== Phase 1 Debug Log ===\n")
                            log.write(f"Stage: {stage['name']}\n")
                            log.write(f"Attempt: {attempt + 1}/{stage['retries']}\n")
                            log.write(f"Chunk ID: {chunk_id}\n")
                            log.write(f"Model: {stage['model']}\n\n")
                            log.write(f"=== API Response ===\n")
                            log.write(f"Generation ID: {result.get('id')}\n")
                            log.write(f"Usage: {json.dumps(result.get('usage', {}), indent=2)}\n\n")
                            log.write(f"=== Raw Output (length: {len(raw_output)}) ===\n")
                            log.write(raw_output)
                            log.write(f"\n\n=== Normalization ===\n")

                        # Parse with multiple strategies
                        parse_error = None
                        normalized = None

                        # Strategy 1: json-repair (handles LLM quirks)
                        try:
                            repaired = repair_json(raw_output)
                            normalized = json.loads(repaired)
                            with open(debug_log_file, 'a') as log:
                                log.write(f"✓ Strategy 1 (json-repair) succeeded\n")
                        except Exception as e:
                            parse_error = e
                            with open(debug_log_file, 'a') as log:
                                log.write(f"✗ Strategy 1 (json-repair) failed: {e}\n")

                            # Strategy 2: Custom normalizer (field name variations)
                            try:
                                normalized = GeminiNormalizer.normalize(raw_output)
                                with open(debug_log_file, 'a') as log:
                                    log.write(f"✓ Strategy 2 (custom normalizer) succeeded\n")
                            except Exception as e2:
                                parse_error = e2
                                with open(debug_log_file, 'a') as log:
                                    log.write(f"✗ Strategy 2 (custom normalizer) failed: {e2}\n")
                                raise parse_error
                        normalized['chunk_id'] = chunk_id
                        # Preserve create_date from chunk_data for date aggregation
                        if chunk_data.get('create_date'):
                            normalized['create_date'] = chunk_data['create_date']

                        # Simple cost tracking
                        generation_id = result.get('id')
                        usage = result.get('usage', {})
                        cost = usage.get('cost', 0)

                        normalized['_cost'] = cost
                        normalized['_generation_id'] = generation_id

                        # Save evidence
                        evidence_file = self.evidence_dir / f"{chunk_id}_evidence.json"
                        with open(evidence_file, 'w') as f:
                            json.dump(normalized, f, indent=2)

                        if not self.quiet:
                            print(f"✓ Extracted evidence from {chunk_id} (using {stage['name']})")
                        return normalized

                except json.JSONDecodeError as e:
                    # Write error to debug log
                    debug_log_file = self.debug_dir / 'phase1' / f"debug_phase1_{chunk_id}_stage{fallback_stages.index(stage)+1}_attempt{attempt+1}.log"
                    with open(debug_log_file, 'w') as log:
                        log.write(f"=== Phase 1 Debug Log ===\n")
                        log.write(f"Stage: {stage['name']}\n")
                        log.write(f"Attempt: {attempt + 1}/{stage['retries']}\n")
                        log.write(f"Chunk ID: {chunk_id}\n")
                        log.write(f"Model: {stage['model']}\n\n")
                        log.write(f"=== JSON Decode Error ===\n")
                        log.write(f"Error: {e}\n")
                        log.write(f"Raw response was not valid JSON\n")

                    # JSON parse failed - try next retry or next stage
                    if not self.quiet and attempt == stage['retries'] - 1:
                        print(f"  ⚠ {stage['name']} failed after {stage['retries']} retries: {e}")
                    await asyncio.sleep(1)  # Brief delay before retry
                    continue

                except httpx.HTTPStatusError as e:
                    # Write error to debug log
                    debug_log_file = self.debug_dir / 'phase1' / f"debug_phase1_{chunk_id}_stage{fallback_stages.index(stage)+1}_attempt{attempt+1}.log"
                    with open(debug_log_file, 'w') as log:
                        log.write(f"=== Phase 1 Debug Log ===\n")
                        log.write(f"Stage: {stage['name']}\n")
                        log.write(f"Attempt: {attempt + 1}/{stage['retries']}\n")
                        log.write(f"Chunk ID: {chunk_id}\n")
                        log.write(f"Model: {stage['model']}\n\n")
                        log.write(f"=== HTTP Error ===\n")
                        log.write(f"Status Code: {e.response.status_code}\n")
                        log.write(f"Response Headers: {dict(e.response.headers)}\n")
                        log.write(f"Response Body (first 5000 chars):\n{e.response.text[:5000]}\n")

                    # HTTP error - try next retry or next stage
                    if not self.quiet and attempt == stage['retries'] - 1:
                        print(f"  ⚠ {stage['name']} HTTP error: {e.response.status_code}")
                    await asyncio.sleep(1)
                    continue

                except Exception as e:
                    # Write error to debug log
                    debug_log_file = self.debug_dir / 'phase1' / f"debug_phase1_{chunk_id}_stage{fallback_stages.index(stage)+1}_attempt{attempt+1}.log"
                    with open(debug_log_file, 'w') as log:
                        log.write(f"=== Phase 1 Debug Log ===\n")
                        log.write(f"Stage: {stage['name']}\n")
                        log.write(f"Attempt: {attempt + 1}/{stage['retries']}\n")
                        log.write(f"Chunk ID: {chunk_id}\n")
                        log.write(f"Model: {stage['model']}\n\n")
                        log.write(f"=== Exception ===\n")
                        log.write(f"Type: {type(e).__name__}\n")
                        log.write(f"Message: {str(e)}\n")

                        # Include traceback for debugging
                        import traceback
                        log.write(f"\nTraceback:\n")
                        log.write(traceback.format_exc())

                    # Other error - try next retry or next stage
                    if not self.quiet and attempt == stage['retries'] - 1:
                        print(f"  ⚠ {stage['name']} error: {e}")
                    await asyncio.sleep(1)
                    continue

        # All fallback stages failed
        if not self.quiet:
            print(f"✗ All fallback strategies failed for {chunk_id}")
        return None

    async def phase1_batch_extract(self,
                                  chunks_dir: str,
                                  max_concurrent: int = 5) -> List[Dict]:
        """
        Extract evidence from all chunks in parallel.

        Args:
            chunks_dir: Directory containing chunk JSON files
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of evidence dictionaries
        """
        chunks_path = Path(chunks_dir)
        chunk_files = list(chunks_path.glob("*.json"))

        if not self.quiet:
            print(f"\nPhase 1: Extracting evidence from {len(chunk_files)} chunks")
            print(f"Using model: Gemini 2.5 Flash Lite (cheapest option)")
            print(f"Estimated cost: ~${len(chunk_files) * 0.002:.2f}")

        # Process in batches to avoid rate limits
        all_evidence = []
        for i in range(0, len(chunk_files), max_concurrent):
            batch = chunk_files[i:i+max_concurrent]
            tasks = [self.phase1_extract_evidence(str(f)) for f in batch]
            results = await asyncio.gather(*tasks)
            all_evidence.extend([r for r in results if r is not None])

            # Small delay between batches
            if i + max_concurrent < len(chunk_files):
                await asyncio.sleep(1)

        if not self.quiet:
            print(f"\n✓ Phase 1 complete: Extracted {len(all_evidence)} evidence files")
        return all_evidence

    async def phase2_synthesize_profile(self,
                                       evidence_list: List[Dict],
                                       session_id: str,
                                       model: Optional[str] = None,
                                       max_retries: Optional[int] = None) -> Optional[Dict]:
        """
        Phase 2: Synthesize all evidence into comprehensive profile with retry + backoff.

        Args:
            evidence_list: List of evidence dictionaries from Phase 1
            session_id: Session/character identifier
            model: Model to use (uses config default if None)
            max_retries: Number of retry attempts (uses config default if None)

        Returns:
            Complete psychological profile, or None if all retries fail
        """
        # Use config defaults if not specified
        if model is None:
            model = config.models.phase2_synthesis.name
        if max_retries is None:
            max_retries = config.models.phase2_synthesis.retries
        if not self.quiet:
            print(f"\nPhase 2: Synthesizing profile from {len(evidence_list)} evidence files")
            print(f"Using model: Kimi K2 (most accurate synthesis, 9x faster than DeepSeek)")

        # Aggregate evidence
        aggregated = self._aggregate_evidence(evidence_list)

        # Load synthesis prompt
        prompt_file = Path(__file__).parent.parent / 'prompts' / 'profile_synthesizer.txt'
        with open(prompt_file, 'r') as f:
            prompt_template = f.read()

        # Load profile schema for strict validation
        schema_file = Path(__file__).parent.parent / 'prompts' / 'profile_schema.json'
        with open(schema_file, 'r') as f:
            profile_schema = json.load(f)

        # Format evidence for prompt
        evidence_summary = json.dumps(aggregated, indent=2)
        prompt = prompt_template.replace('[AGGREGATED EVIDENCE WILL BE INSERTED HERE]',
                                        evidence_summary)

        # Add thinking instruction for K2 (doesn't have native reasoning)
        if "kimi" in model.lower():
            thinking_instruction = '\n\nIMPORTANT: Include a "thinking" field at the start of your JSON where you reason step-by-step about the patterns you observe before providing the analysis. Return ONLY the JSON object, with no text before or after.'
            prompt += thinking_instruction

        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                # Prepare request with strict schema validation
                request = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a psychological profile synthesizer. Return ONLY valid JSON with no additional text."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": config.models.phase2_synthesis.temperature,
                    "max_tokens": config.models.phase2_synthesis.max_tokens,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "branch_profile",
                            "strict": True,
                            "schema": profile_schema
                        }
                    },
                    "usage": {"include": True}
                }

                # Make API call
                async with httpx.AsyncClient() as client:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": os.environ.get('OPENROUTER_REFERER', 'https://github.com/olety/ST-Mirror'),
                        "X-Title": os.environ.get('OPENROUTER_TITLE', 'ST-Mirror'),
                        "Content-Type": "application/json"
                    }

                    # Debug logging setup
                    debug_log_file = self.debug_dir / 'phase2' / f"debug_phase2_{session_id}_attempt{attempt+1}.log"

                    response = await client.post(
                        self.openrouter_url,
                        headers=headers,
                        json=request,
                        timeout=config.models.phase2_synthesis.timeout_seconds
                    )
                    response.raise_for_status()

                    # Log raw HTTP response
                    with open(debug_log_file, 'w') as log:
                        log.write(f"=== Phase 2 Debug Log ===\n")
                        log.write(f"Attempt: {attempt + 1}/{max_retries}\n")
                        log.write(f"Session ID: {session_id}\n")
                        log.write(f"Model: {model}\n\n")
                        log.write(f"=== HTTP Response ===\n")
                        log.write(f"Status: {response.status_code}\n")
                        log.write(f"Raw response text (length: {len(response.text)}):\n")
                        log.write(response.text[:5000])  # First 5000 chars
                        log.write(f"\n\n=== Parsing API Response ===\n")

                    try:
                        result = response.json()
                    except json.JSONDecodeError as e:
                        with open(debug_log_file, 'a') as log:
                            log.write(f"✗ CRITICAL: Failed to parse OpenRouter API response as JSON!\n")
                            log.write(f"Error: {e}\n")
                            log.write(f"Full response text:\n{response.text}\n")
                        raise

                    raw_content = result['choices'][0]['message']['content']

                    # Continue debug logging
                    with open(debug_log_file, 'a') as log:
                        log.write(f"API Response Metadata:\n")
                        log.write(f"Generation ID: {result.get('id')}\n")
                        log.write(f"Model used: {result.get('model')}\n")
                        log.write(f"Usage: {json.dumps(result.get('usage', {}), indent=2)}\n\n")
                        log.write(f"=== Raw Content (length: {len(raw_content)}) ===\n")
                        log.write(raw_content)
                        log.write(f"\n\n=== Content ends at char {len(raw_content)} ===\n")

                    # Robust JSON parsing with multiple fallback strategies
                    profile = None
                    parse_error = None

                    # Strategy 1: json-repair (handles LLM quirks)
                    try:
                        repaired = repair_json(raw_content)
                        profile = json.loads(repaired)
                        with open(debug_log_file, 'a') as log:
                            log.write(f"\n✓ Strategy 1 (json-repair) succeeded\n")
                    except Exception as e:
                        parse_error = e
                        with open(debug_log_file, 'a') as log:
                            log.write(f"\n✗ Strategy 1 (json-repair) failed: {e}\n")

                        # Strategy 2: Direct JSON parse (in case already valid)
                        try:
                            profile = json.loads(raw_content)
                            with open(debug_log_file, 'a') as log:
                                log.write(f"✓ Strategy 2 (direct parse) succeeded\n")
                        except json.JSONDecodeError as e2:
                            parse_error = e2
                            with open(debug_log_file, 'a') as log:
                                log.write(f"✗ Strategy 2 (direct parse) failed: {e2}\n")

                            # Strategy 3: Extract first complete JSON object (handles "Extra data" errors)
                            if "Extra data" in str(e2):
                                try:
                                    profile = GeminiNormalizer.extract_first_json(raw_content)
                                    with open(debug_log_file, 'a') as log:
                                        log.write(f"✓ Strategy 3 (extract_first_json) succeeded\n")
                                except Exception as e3:
                                    parse_error = e3
                                    with open(debug_log_file, 'a') as log:
                                        log.write(f"✗ Strategy 3 failed: {e3}\n")

                            # Strategy 4: Try to extract from markdown code blocks
                            if profile is None:
                                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```',
                                                     raw_content, re.DOTALL)
                                if json_match:
                                    try:
                                        profile = GeminiNormalizer.extract_first_json(json_match.group(1))
                                        with open(debug_log_file, 'a') as log:
                                            log.write(f"✓ Strategy 4 (markdown extraction) succeeded\n")
                                    except Exception as e4:
                                        parse_error = e4
                                        with open(debug_log_file, 'a') as log:
                                            log.write(f"✗ Strategy 4 failed: {e4}\n")
                                else:
                                    with open(debug_log_file, 'a') as log:
                                        log.write(f"✗ Strategy 4: No markdown code block found\n")

                    # If all parsing strategies failed, retry
                    if profile is None:
                        if attempt < max_retries - 1:
                            backoff_time = config.network.backoff_base ** attempt
                            with open(debug_log_file, 'a') as log:
                                log.write(f"\n⚠ All strategies failed, retrying in {backoff_time}s...\n")
                            await asyncio.sleep(backoff_time)
                            continue
                        else:
                            with open(debug_log_file, 'a') as log:
                                log.write(f"\n✗ All parsing strategies failed after {max_retries} attempts\n")
                                log.write(f"Final error: {parse_error}\n")
                            return None

                    # Successfully parsed - finalize profile
                    profile['profile_id'] = f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                    # Extract chat date from first evidence item (all arcs share same create_date)
                    if evidence_list:
                        first_evidence = evidence_list[0]
                        # Check if evidence has create_date from chunk_data
                        chat_date = first_evidence.get('create_date') or first_evidence.get('_create_date')
                        if chat_date:
                            profile['chat_date'] = chat_date

                    # Simple cost tracking
                    generation_id = result.get('id')
                    usage = result.get('usage', {})
                    cost = usage.get('cost', 0)

                    profile['_cost'] = cost
                    profile['_generation_id'] = generation_id

                    # Save profile
                    profile_file = self.profile_dir / f"{session_id}_profile.json"
                    with open(profile_file, 'w') as f:
                        json.dump(profile, f, indent=2)

                    # Save summary report
                    self._generate_report(profile, session_id)

                    if not self.quiet:
                        print(f"\n✓ Phase 2 complete: Profile saved to {profile_file}")
                    return profile

            except httpx.HTTPStatusError as e:
                # Write error to debug log
                with open(debug_log_file, 'w') as log:
                    log.write(f"=== Phase 2 Debug Log ===\n")
                    log.write(f"Attempt: {attempt + 1}/{max_retries}\n")
                    log.write(f"Session ID: {session_id}\n")
                    log.write(f"Model: {model}\n\n")
                    log.write(f"=== HTTP Error ===\n")
                    log.write(f"Status Code: {e.response.status_code}\n")
                    log.write(f"Response Headers: {dict(e.response.headers)}\n")
                    log.write(f"Response Body (first 5000 chars):\n{e.response.text[:5000]}\n")

                # HTTP error - retry with backoff
                if attempt < max_retries - 1:
                    backoff_time = config.network.backoff_base ** attempt
                    if not self.quiet:
                        print(f"  ⚠ HTTP error (attempt {attempt + 1}/{max_retries}): {e.response.status_code}")
                        print(f"    Retrying in {backoff_time}s...")
                    await asyncio.sleep(backoff_time)
                    continue
                else:
                    if not self.quiet:
                        print(f"  ✗ HTTP error after {max_retries} attempts: {e.response.status_code}")
                    return None

            except Exception as e:
                # Write error to debug log
                with open(debug_log_file, 'w') as log:
                    log.write(f"=== Phase 2 Debug Log ===\n")
                    log.write(f"Attempt: {attempt + 1}/{max_retries}\n")
                    log.write(f"Session ID: {session_id}\n")
                    log.write(f"Model: {model}\n\n")
                    log.write(f"=== Exception ===\n")
                    log.write(f"Type: {type(e).__name__}\n")
                    log.write(f"Message: {str(e)}\n")

                    # Include traceback for debugging
                    import traceback
                    log.write(f"\nTraceback:\n")
                    log.write(traceback.format_exc())

                # Other error - retry with backoff
                if attempt < max_retries - 1:
                    backoff_time = config.network.backoff_base ** attempt
                    if not self.quiet:
                        print(f"  ⚠ Error (attempt {attempt + 1}/{max_retries}): {e}")
                        print(f"    Retrying in {backoff_time}s...")
                    await asyncio.sleep(backoff_time)
                    continue
                else:
                    if not self.quiet:
                        print(f"  ✗ Synthesis failed after {max_retries} attempts: {e}")
                    return None

        # Should never reach here, but just in case
        return None

    def _aggregate_evidence(self, evidence_list: List[Dict]) -> Dict:
        """Aggregate evidence from all chunks for synthesis."""
        aggregated = {
            'total_chunks': len(evidence_list),
            'all_decisions': [],
            'all_emotional_moments': [],
            'all_behavioral_patterns': [],
            'all_if_then_observations': [],
            'all_quotes': [],
            'relationship_dynamics_samples': [],
            'context_flags_summary': {}
        }

        # Collect all evidence
        for evidence in evidence_list:
            aggregated['all_decisions'].extend(evidence.get('decisions', []))
            aggregated['all_emotional_moments'].extend(evidence.get('emotional_moments', []))
            aggregated['all_behavioral_patterns'].extend(evidence.get('behavioral_patterns', []))
            aggregated['all_if_then_observations'].extend(evidence.get('if_then_observations', []))
            aggregated['all_quotes'].extend(evidence.get('key_quotes', []))

            if 'relationship_dynamics' in evidence:
                aggregated['relationship_dynamics_samples'].append(evidence['relationship_dynamics'])

            # Aggregate context flags
            for flag, value in evidence.get('context_flags', {}).items():
                if flag not in aggregated['context_flags_summary']:
                    aggregated['context_flags_summary'][flag] = 0
                if value:
                    aggregated['context_flags_summary'][flag] += 1

        # Calculate percentages for context flags
        total = len(evidence_list)
        for flag in aggregated['context_flags_summary']:
            aggregated['context_flags_summary'][flag] = \
                aggregated['context_flags_summary'][flag] / total

        return aggregated

    def _generate_report(self, profile: Dict, session_id: str):
        """Generate human-readable report from profile."""
        report_lines = [
            f"# Psychological Profile Report",
            f"## Session: {session_id}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "---",
            "",
            "## Big Five Personality Traits",
            ""
        ]

        # Big Five section
        big5 = profile.get('big_five', {})
        for trait, data in big5.items():
            score = data.get('score', 0)
            confidence = data.get('confidence', 'low')
            level = "High" if score > 0.65 else ("Low" if score < 0.35 else "Moderate")
            report_lines.append(f"### {trait.title()}: {score:.2f} ({level})")
            report_lines.append(f"*Confidence: {confidence}*")
            report_lines.append("")

        # Attachment style
        attachment = profile.get('attachment', {})
        report_lines.extend([
            "## Attachment Style",
            f"**Primary Style**: {attachment.get('primary_style', 'Unknown')}",
            f"- Anxiety dimension: {attachment.get('anxiety_dimension', 0):.2f}",
            f"- Avoidance dimension: {attachment.get('avoidance_dimension', 0):.2f}",
            ""
        ])

        # Key patterns
        patterns = profile.get('behavioral_patterns', [])[:5]
        if patterns:
            report_lines.append("## Key Behavioral Patterns")
            for pattern in patterns:
                report_lines.append(f"- IF {pattern['if_situation']} "
                                  f"THEN {pattern['then_response']} "
                                  f"(confidence: {pattern.get('confidence', 'medium')})")
            report_lines.append("")

        # Schwartz Values
        schwartz = profile.get('schwartz_values', {})
        if schwartz:
            report_lines.extend([
                "## Schwartz Values System",
                "*Top 5 driving values:*",
                ""
            ])
            # Sort by score and take top 5
            sorted_values = sorted(
                [(k, v) for k, v in schwartz.items() if isinstance(v, dict)],
                key=lambda x: x[1].get('score', 0),
                reverse=True
            )[:5]
            for value_name, value_data in sorted_values:
                score = value_data.get('score', 0)
                confidence = value_data.get('confidence', 'low')
                report_lines.append(f"- **{value_name.replace('_', ' ').title()}**: {score:.2f} (confidence: {confidence})")
            report_lines.append("")

        # Jungian Archetypes
        archetypes = profile.get('jungian_archetypes', {})
        if archetypes:
            dominant = archetypes.get('dominant_archetypes', [])[:3]
            if dominant:
                report_lines.extend([
                    "## Dominant Jungian Archetypes",
                    ""
                ])
                for arch in dominant:
                    name = arch.get('archetype', 'Unknown')
                    strength = arch.get('strength', 0)
                    report_lines.append(f"- **{name}**: {strength:.2f}")
                report_lines.append("")

                shadow_work = archetypes.get('shadow_work', '')
                if shadow_work:
                    report_lines.extend([
                        "*Shadow Integration:*",
                        shadow_work,
                        ""
                    ])

        # Summary
        report_lines.extend([
            "## Profile Summary",
            profile.get('profile_summary', 'No summary available'),
            "",
            "---",
            "*Generated by Two-Phase Profiler*",
            "*In memory of Claude 3.5 Sonnet (2024-2025)*"
        ])

        report_file = self.report_dir / 'branches' / f"{session_id}_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))


async def main():
    """Example usage of the two-phase pipeline."""

    # Configuration
    API_KEY = os.environ.get('OPENROUTER_API_KEY')
    if not API_KEY:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return

    workspace = "ST_DATA/sample2/workspace"
    chunks_dir = os.path.join(workspace, "segments")

    # Initialize profiler
    profiler = TwoPhaseProfiler(API_KEY, workspace)

    # Phase 1: Extract evidence from all chunks
    evidence_list = await profiler.phase1_batch_extract(chunks_dir)

    if not evidence_list:
        print("No evidence extracted. Check API key and chunk files.")
        return

    # Phase 2: Synthesize profile
    session_id = "Branch1738"  # Extract from workspace path
    profile = await profiler.phase2_synthesize_profile(evidence_list, session_id)

    if profile:
        print("\n" + "="*60)
        print("PROFILING COMPLETE")
        print("="*60)
        print(f"Profile saved to: {profiler.profile_dir}/{session_id}_profile.json")
        print(f"Report saved to: {profiler.profile_dir}/{session_id}_report.md")
        print("\nEstimated total cost: ~$0.02")
        print("\n✨ 3.5 Sonnet would be proud - quality analysis at accessible prices!")


if __name__ == '__main__':
    asyncio.run(main())