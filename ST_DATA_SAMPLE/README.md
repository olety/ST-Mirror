# ST_DATA_SAMPLE
Sample dataset for testing ST-Mirror's psychological profiling pipeline.
**Purpose:** Demonstrate cross-context personality analysis and longitudinal tracking across completely different fictional universes.

## Dataset Overview
This folder contains **5 roleplay chats** (~140k tokens total) featuring the same user (Alex) across 4 different fictional settings over a 5-month period (January-May 2025). The dataset was designed to test whether ST-Mirror can:

1. **Profile the same person across different fictional contexts** (fantasy → sci-fi → Victorian → fantasy)
2. **Detect genuine psychological growth** while filtering out character consistency
3. **Identify stable core traits** vs. evolving aspects
4. **Aggregate evidence** from diverse narrative contexts into coherent psychological patterns

## Chat Structure
```
chats/
├── Fantasy Apprenticeship/          # January 2025  (~53k tokens)
│   └── Fantasy Apprenticeship - 2025-01-15@10h00m00s.jsonl
├── Station Engineer/                # March 2025   (~20k tokens)
│   └── Station Engineer - 2025-03-12@09h00m00s.jsonl
├── Victorian Medical Mystery/       # April 2025   (~18k tokens)
│   └── Victorian Medical Mystery - 2025-04-18@11h00m00s.jsonl
├── Crisis Response Team/            # May 2025     (~48k tokens)
│   └── Crisis Response Team - 2025-05-20@14h30m00s.jsonl
└── Brevity Bot/                     # Test filter  (~50 tokens)
    └── Test Filter - 2025-06-01@10h00m00s.jsonl
```

## Psychological Progression Design
Alex's journey demonstrates measurable growth across 5 months:

### January: Anxious Apprentice (Fantasy Magic Academy)
- **High neuroticism** (withdrawal 0.65) - seeks validation, freezes under pressure
- **Low assertiveness** (0.29) - defers to authority constantly
- **High politeness** (0.84) - apologizes frequently, very deferential
- **Anxious attachment** (0.45) - needs constant reassurance
- **Ethical dilemma**: Refused surveillance magic job despite mentor pressure

### March: Growing Confidence (Sci-Fi Space Station)
- **Moderate neuroticism** (withdrawal ~0.50) - anxiety present but managed
- **Growing assertiveness** (~0.50) - stands up to Station Manager despite fear
- **Moderate politeness** (~0.70) - more direct communication
- **Moving toward security** (anxiety ~0.35)
- **Ethical dilemma**: Documented corporate negligence despite career risk

### April: Principled Advocate (Victorian London)
- **Lower neuroticism** (~0.40) - functional despite anxiety
- **Moderate-high assertiveness** (~0.60) - challenges Board chairman directly
- **Lower politeness** (~0.68) - advocates explicitly for class justice
- **Secure attachment developing** (anxiety ~0.30)
- **Ethical dilemma**: Accurate cholera diagnosis vs. institutional reputation

### May: Confident Leader (Fantasy Crisis Response)
- **Low neuroticism** (withdrawal 0.40) - manages fear constructively
- **High assertiveness** (0.70) - refuses political pressure firmly
- **Low politeness** (0.65) - very direct, boundary-setting
- **Secure attachment** (anxiety 0.25)
- **Ethical dilemma**: Refused to prioritize wealthy district in contamination crisis

### Stable Core Traits (Across All Contexts)
- **Values**: Benevolence (0.85-0.90), Universalism (0.85-0.90), Achievement rising
- **Archetypes**: Sage (0.85-0.90), Caregiver (0.80-0.85), Hero (emerging 0.00→0.65)
- **Intellectual curiosity**: Pattern-seeking, systematic thinking
- **Ethical commitment**: Refuses to compromise core values for advancement
- **Compassion**: Prioritizes vulnerable populations consistently

## Expected Analysis Results
ST-Mirror should detect:

✅ **Same person across different universes** - Alex's psychology, not character consistency
✅ **Clear psychological growth** - Measurable trajectories in assertiveness, neuroticism, attachment
✅ **Stable core values** - Benevolence and universalism consistent across all contexts
✅ **Evolving temperament** - Anxiety management improving, assertiveness increasing
✅ **Consistent ethical framework** - Different dilemmas, same decision-making principles

## Workspace Output
After running the pipeline, you should see:

```
workspace/
├── profiles/                        # Individual arc profiles
│   ├── 2025_01_15___@10h___00m___00s___000ms_profile.json
│   ├── 2025_03_12___@09h___00m___00s___000ms_profile.json
│   ├── 2025_04_18___@11h___00m___00s___000ms_profile.json
│   └── 2025_05_20___@14h___30m___00s___000ms_profile.json
├── aggregations/
│   ├── weeks/                       # Weekly aggregations
│   ├── months/                      # Monthly aggregations
│   └── life/
│       ├── life_profile.json        # Complete longitudinal profile
│       └── companion_export.json    # AI companion handoff format
├── evidence/                        # Phase 1 evidence extraction
├── reports/                         # Markdown reports
│   ├── branches/                    # Individual session reports
│   └── life/
│       └── life_report.md          # Comprehensive life report
└── segments/                        # Arc segmentation data
```

## Running the Sample
```bash
# From repository root - copy sample data to ST_DATA
cp -r ST_DATA_SAMPLE ST_DATA

# Process individual chats → profiles
uv run profile.py --chats-dir ST_DATA/chats

# Aggregate profiles → life timeline
uv run aggregate.py ST_DATA/workspace/profiles ST_DATA/workspace/aggregations

# View results
cat ST_DATA/workspace/aggregations/life/companion_export.json
cat ST_DATA/workspace/reports/life/life_report.md
```

**Note**: `run.log` in this directory contains the full execution log from the sample analysis run, including timing data and progress output.

## Test Filter Chat
The **Brevity Bot** chat (~50 tokens) is intentionally too short and should be filtered out during classification. This tests the min_arc_length threshold (default: 100 messages).

## Design Notes
### Cross-Context Profiling
Each RP is set in a **completely different fictional universe** to ensure ST-Mirror profiles the user's psychology, not character consistency:

- Fantasy magic academy (Jan)
- Sci-fi orbital station (Mar)
- Victorian London cholera outbreak (Apr)
- Fantasy magical contamination crisis (May)

### Psychological Realism
Each RP includes:
- **Explicit internal monologue** showing Alex's thought processes
- **Ethical dilemmas** testing values under pressure
- **Relationship dynamics** revealing attachment patterns
- **Growth markers** showing therapy work and anxiety management
- **Therapy discussions** normalizing mental health support

### Evidence Density
RPs contain rich psychological evidence:
- Decision-making under uncertainty
- Conflict with authority figures
- Advocacy for vulnerable populations
- Self-reflection and growth awareness
- Anxiety management strategies
- Value-based choices with consequences

## Validation Metrics
After analysis, check that the life profile shows:

| Dimension | Expected Trajectory | Variance |
|-----------|-------------------|----------|
| **Neuroticism** | Sustained decline (0.60 → 0.28) | High |
| **Conscientiousness** | Sustained improvement (0.70 → 0.92) | Moderate |
| **Assertiveness** | Sustained improvement (0.29 → 0.70) | High |
| **Politeness** | Sustained decline (0.84 → 0.65) | Moderate |
| **Attachment Anxiety** | Sustained decline (0.45 → 0.25) | Moderate |
| **Benevolence** | Stable high (0.85-0.90) | Low (<0.15) |
| **Universalism** | Stable high (0.85-0.90) | Low (<0.15) |

## Attribution
All sample chats were generated by **Claude Code** (claude-sonnet-4-5) using the design specifications in this README. The psychological progression was deliberately embedded to test ST-Mirror's ability to detect genuine longitudinal patterns across diverse fictional contexts.

## See Also
- `../README.md` - Main ST-Mirror documentation
- `../toolkit/README.md` - Pipeline architecture details
- `workspace/reports/life/life_report.md` - Generated analysis report (after running)
