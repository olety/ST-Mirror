# ST-Mirror Toolkit

Core library for psychological profile extraction from SillyTavern chat logs.

**Pipeline:** Classifier → Segmentation → Evidence Extraction → Profile Synthesis → Temporal Aggregation

## Modules

### classifier.py
Two-tier filtering to prevent wasting API credits on test/corrupted branches.

**Tier 1: Heuristic (FREE)** - Filters corrupted data: min messages (100), valid ratio (50%), repetition check<br/>
**Tier 2: LLM (~$0.0001/branch)** - Gemini Flash Lite samples beginning/middle/end for substance detection<br/>
**Key Functions:**
- `heuristic_prefilter(messages) -> {"skip": bool, "reason": str}`
- `classify_with_flash_lite(messages, branch_file, ...) -> {"skip": bool, "is_test": bool, "has_substance": bool}`
- `classify_character_branches(character_name, pattern, output_file, parallel=10)` - Parallel processing (max 15)

**Used by:** `profile.py:26`

### segmentation.py
Arc-aware segmentation for long RPs. Detects narrative transitions based on OOC/sexual/personal content shifts.

**Key Functions:**
- `segment_by_arcs(jsonl_path, output_dir, window_size=50, min_arc_length=100) -> List[str]`
- `analyze_window(messages, window_size=50)` - Tracks ooc/ic/sexual/personal/emotional ratios
- `detect_arc_transitions(windows, ooc_threshold=0.3, content_threshold=0.25)` - Finds boundaries
- `classify_arc(window_stats)` - Types: meta_conversation, erotic_fantasy, genuine_intimacy, romantic_connection, mixed_intimate, casual_roleplay

**Output:** `{branch_id}_arc{N}_{type}.json` + `{branch_id}_arc_summary.json`<br/>
**Used by:** `profile.py:25`

### pipeline.py
Two-phase profiling: lightweight evidence extraction (Gemini Flash Lite ~$0.002) → deep synthesis (Kimi K2 ~$0.005).

**Phase 1:** Extract evidence with 3-stage fallback (Gemini medium → Gemini high → Kimi K2 strict schema)<br/>
**Phase 2:** Synthesize complete profile (Big Five, attachment, Schwartz values, Jungian archetypes)<br/>
**Key Class:**
```python
class TwoPhaseProfiler:
    async def phase1_extract_evidence(chunk_file, previous_summary=None) -> Dict
        # Cumulative context prevents re-analysis (~300-500 token summary)

    async def phase2_synthesize_profile(evidence_list, session_id) -> Dict
        # Strict JSON schema + 3 retries with exponential backoff
```

**Prompts:** `../prompts/evidence_extractor.txt`, `evidence_schema.json`, `profile_synthesizer.txt`, `profile_schema.json`<br/>
**Outputs:** Evidence (`workspace/evidence/`), Profiles (`workspace/profiles/`), Reports (`workspace/reports/branches/`)<br/>
**Used by:** `profile.py:24`

### aggregator.py
Hierarchical temporal aggregation with adaptive splitting (Branches → Weeks → Months → Years → Life).

**Adaptive Splitting:** Auto-splits periods >15 profiles or >30k tokens. Example: 50-profile month → 4 weeks → month summary.<br/>
**Key Class:**
```python
class AdaptiveAggregator:
    async def build_hierarchy(profiles) -> Optional[Dict]
        # Returns life profile with current_state (recency-weighted), evolution (early vs late),
        # stable_traits (variance <0.15), companion_export

    async def aggregate_period(profiles, period_id, level) -> Dict
        # Checks token budget, splits if needed, synthesizes with Kimi K2
```

**Date Extraction:** Primary: `chat_date` field → Fallback: `profile_id` → Fallback: `_source_file`<br/>
**Timeouts:** Week (2m), Month (3m), Year (4m), Life (5m)<br/>
**Outputs:** Weekly/monthly/yearly summaries, life profile, companion export<br/>
**Used by:** `aggregate.py:14`

### normalizer.py
Deterministic JSON normalizer for Gemini Flash outputs. Handles field name variations, enum mappings, type coercion.

**Key Class:**
```python
class GeminiNormalizer:
    @classmethod
    def normalize(raw_output) -> Dict
        # Workflow: Parse JSON → Normalize fields → Fix types → Ensure required → Enforce limits

    @classmethod
    def extract_first_json(text) -> Dict
        # Brace-counting to handle "Extra data" errors
```

**Mappings:** decision_points→decisions, emotions→emotional_moments, angry→anger, sub→submissive, etc.<br/>
**Array Limits:** Decisions (5), emotions (5), quotes (5), patterns (3), if-then (3)<br/>
**Used by:** `pipeline.py:23`, `aggregator.py:52`

## Entry Points

| File | Imports | Purpose |
|------|---------|---------|
| `profile.py` | `TwoPhaseProfiler`, `segment_by_arcs`, `heuristic_prefilter`, `classify_with_flash_lite` | CLI for branch → profile |
| `aggregate.py` | `AdaptiveAggregator` | CLI for profile → life hierarchy |

**Standalone:**
- `uv run python -m toolkit.classifier <character> <pattern> [output]`
- `uv run python -m toolkit.normalizer` (test suite)

## Configuration

Uses `../config.yaml`:

**Models:** classification (Flash Lite), phase1_evidence (Flash Lite + fallbacks), phase2_synthesis (Kimi K2), aggregation (Kimi K2)<br/>
**Processing:** classifier (max_parallel, sampling strategy), profiler (arc detection), aggregator (splitting thresholds)<br/>
**Network:** backoff_base (2), max_connections (10)<br/>
**User:** name, pronouns (for companion export)

## Debug Logs

`ST_DATA/workspace/debug/`:
- `classifier/{branch_id}_classification.json`
- `phase1/debug_phase1_{chunk_id}_stage{N}_attempt{N}.log`
- `phase2/debug_phase2_{session_id}_attempt{N}.log`
- `aggregator/periods/debug_agg_{period_id}_attempt{N}.log`

## Retry Strategies

**Network errors (5xx, timeouts):** Exponential backoff (1s, 2s, 4s), always retryable<br/>
**JSON parse errors:** json-repair → direct parse → normalizer → markdown extraction<br/>
**Client errors (4xx):** Not retryable
