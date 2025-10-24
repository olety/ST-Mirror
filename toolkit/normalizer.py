#!/usr/bin/env python3
"""
Deterministic normalizer for Gemini Flash JSON outputs.
Handles common quirks without needing LLM repair.

In memory of Claude 3.5 Sonnet - who taught us that elegance
comes from handling edge cases gracefully.
"""

import json
import re
from typing import Dict, Any, List, Union

class GeminiNormalizer:
    """Normalize Gemini's creative JSON interpretations."""

    # Common field name variations Gemini might produce
    FIELD_MAPPINGS = {
        # Decisions variations
        'decision_points': 'decisions',
        'choices': 'decisions',
        'decision': 'decisions',

        # Emotional variations
        'emotions': 'emotional_moments',
        'emotional_states': 'emotional_moments',
        'emotion_moments': 'emotional_moments',

        # Quote variations
        'quotes': 'key_quotes',
        'important_quotes': 'key_quotes',
        'revealing_quotes': 'key_quotes',

        # Pattern variations
        'patterns': 'behavioral_patterns',
        'behaviors': 'behavioral_patterns',
        'repeated_behaviors': 'behavioral_patterns',

        # If-then variations
        'if_then': 'if_then_observations',
        'if_then_patterns': 'if_then_observations',
        'conditionals': 'if_then_observations',

        # Context variations
        'flags': 'context_flags',
        'context': 'context_flags',
        'scene_flags': 'context_flags',

        # Relationship variations
        'dynamics': 'relationship_dynamics',
        'relationship': 'relationship_dynamics',
        'relational_dynamics': 'relationship_dynamics'
    }

    # Emotion enum normalizations
    EMOTION_MAPPINGS = {
        'angry': 'anger',
        'mad': 'anger',
        'frustrated': 'anger',
        'afraid': 'fear',
        'scared': 'fear',
        'worried': 'anxiety',
        'anxious': 'anxiety',
        'sad': 'sadness',
        'depressed': 'sadness',
        'happy': 'joy',
        'excited': 'joy',
        'love': 'affection',
        'caring': 'affection',
        'vulnerable': 'vulnerability',
        'exposed': 'vulnerability'
    }

    # Power balance normalizations
    POWER_MAPPINGS = {
        'subordinate': 'submissive',
        'sub': 'submissive',
        'dom': 'dominant',
        'dominating': 'dominant',
        'balanced': 'equal',
        'even': 'equal',
        'changing': 'shifting',
        'variable': 'shifting'
    }

    @classmethod
    def extract_first_json(cls, text: str) -> Dict[str, Any]:
        """
        Extract the first valid JSON object from text, handling extra data after JSON.

        This handles the "Extra data: line 2 column 1" error when LLMs add text after JSON.
        """
        # Try to find and parse just the first complete JSON object
        # Count braces to find where the JSON object ends
        depth = 0
        in_string = False
        escape_next = False
        start_idx = text.find('{')

        if start_idx == -1:
            raise ValueError("No JSON object found in text")

        for i in range(start_idx, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        # Found complete JSON object
                        json_str = text[start_idx:i+1]
                        return json.loads(json_str)

        # Fallback to original text if we couldn't parse
        raise ValueError("Could not find complete JSON object")

    @classmethod
    def normalize(cls, raw_output: Union[str, Dict]) -> Dict[str, Any]:
        """
        Normalize Gemini output to match expected schema.

        Args:
            raw_output: Either JSON string or already parsed dict

        Returns:
            Normalized dictionary matching evidence_schema.json
        """
        # Parse if string
        if isinstance(raw_output, str):
            # Clean common JSON issues
            raw_output = cls._clean_json_string(raw_output)
            try:
                data = json.loads(raw_output)
            except json.JSONDecodeError as e:
                # Handle "Extra data" error - extract first JSON object
                if "Extra data" in str(e):
                    try:
                        data = cls.extract_first_json(raw_output)
                    except:
                        data = None
                else:
                    data = None

                # If extraction failed, try other methods
                if data is None:
                    # Try to extract JSON from markdown code blocks
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```',
                                         raw_output, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(1))
                    else:
                        # Last resort - try to find JSON-like structure
                        json_match = re.search(r'(\{.*\})', raw_output, re.DOTALL)
                        if json_match:
                            try:
                                data = cls.extract_first_json(json_match.group(1))
                            except:
                                data = json.loads(json_match.group(1))
                        else:
                            raise ValueError("Could not extract JSON from output")
        else:
            data = raw_output

        # Normalize field names
        data = cls._normalize_field_names(data)

        # Normalize specific fields
        data = cls._normalize_decisions(data)
        data = cls._normalize_emotions(data)
        data = cls._normalize_relationships(data)
        data = cls._normalize_booleans(data)
        data = cls._normalize_numbers(data)

        # Ensure required fields exist
        data = cls._ensure_required_fields(data)

        # Enforce limits
        data = cls._enforce_limits(data)

        return data

    @classmethod
    def _clean_json_string(cls, s: str) -> str:
        """Clean common JSON formatting issues."""
        # Remove trailing commas
        s = re.sub(r',\s*}', '}', s)
        s = re.sub(r',\s*]', ']', s)

        # Fix single quotes (Gemini sometimes uses them)
        # But be careful not to break strings with apostrophes
        s = re.sub(r"(?<=[{\[:,])\s*'([^']*)'(?=\s*[,\]}:])", r'"\1"', s)

        # Remove comments (Gemini sometimes adds them)
        s = re.sub(r'//.*$', '', s, flags=re.MULTILINE)
        s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)

        return s.strip()

    @classmethod
    def _normalize_field_names(cls, data: Dict) -> Dict:
        """Normalize field names to expected schema."""
        normalized = {}

        for key, value in data.items():
            # Check if this is a known variation
            normalized_key = cls.FIELD_MAPPINGS.get(key.lower(), key)
            normalized[normalized_key] = value

        return normalized

    @classmethod
    def _normalize_decisions(cls, data: Dict) -> Dict:
        """Normalize decision format."""
        if 'decisions' in data and isinstance(data['decisions'], list):
            normalized_decisions = []
            for decision in data['decisions'][:5]:  # Max 5
                if isinstance(decision, dict):
                    # Normalize field names within decision
                    norm_decision = {}
                    norm_decision['choice'] = str(decision.get('choice',
                                                   decision.get('chosen',
                                                   decision.get('action', ''))))[:100]
                    norm_decision['alternative'] = str(decision.get('alternative',
                                                      decision.get('other_option',
                                                      decision.get('rejected', ''))))[:100]
                    if 'context' in decision:
                        norm_decision['context'] = str(decision['context'])[:200]
                    normalized_decisions.append(norm_decision)
                elif isinstance(decision, str):
                    # Try to parse string format "chose X over Y"
                    match = re.match(r'chose?\s+(.+?)\s+over\s+(.+)', decision, re.I)
                    if match:
                        normalized_decisions.append({
                            'choice': match.group(1)[:100],
                            'alternative': match.group(2)[:100]
                        })
            data['decisions'] = normalized_decisions
        return data

    @classmethod
    def _normalize_emotions(cls, data: Dict) -> Dict:
        """Normalize emotion values to valid enums."""
        if 'emotional_moments' in data and isinstance(data['emotional_moments'], list):
            normalized_emotions = []
            for emotion in data['emotional_moments'][:5]:  # Max 5
                if isinstance(emotion, dict):
                    norm_emotion = {}

                    # Normalize emotion enum
                    raw_emotion = str(emotion.get('emotion', '')).lower()
                    norm_emotion['emotion'] = cls.EMOTION_MAPPINGS.get(
                        raw_emotion, raw_emotion)

                    # Ensure valid emotion
                    valid_emotions = ['anger', 'fear', 'sadness', 'joy',
                                    'surprise', 'vulnerability', 'affection', 'anxiety']
                    if norm_emotion['emotion'] not in valid_emotions:
                        norm_emotion['emotion'] = 'surprise'  # Default

                    # Normalize intensity
                    raw_intensity = str(emotion.get('intensity', 'medium')).lower()
                    if raw_intensity in ['low', 'medium', 'high']:
                        norm_emotion['intensity'] = raw_intensity
                    else:
                        # Try to map numeric
                        try:
                            intensity_val = float(raw_intensity)
                            if intensity_val < 0.33:
                                norm_emotion['intensity'] = 'low'
                            elif intensity_val < 0.67:
                                norm_emotion['intensity'] = 'medium'
                            else:
                                norm_emotion['intensity'] = 'high'
                        except:
                            norm_emotion['intensity'] = 'medium'

                    if 'trigger' in emotion:
                        norm_emotion['trigger'] = str(emotion['trigger'])[:100]

                    normalized_emotions.append(norm_emotion)

            data['emotional_moments'] = normalized_emotions
        return data

    @classmethod
    def _normalize_relationships(cls, data: Dict) -> Dict:
        """Normalize relationship dynamics."""
        if 'relationship_dynamics' in data:
            rd = data['relationship_dynamics']
            if isinstance(rd, dict):
                # Normalize power balance
                if 'power_balance' in rd:
                    power = str(rd['power_balance']).lower()
                    rd['power_balance'] = cls.POWER_MAPPINGS.get(power, power)
                    if rd['power_balance'] not in ['submissive', 'equal', 'dominant', 'shifting']:
                        rd['power_balance'] = 'equal'

                # Normalize intimacy level
                if 'intimacy_level' in rd:
                    intimacy = str(rd['intimacy_level']).lower()
                    if intimacy not in ['distant', 'casual', 'close', 'intimate']:
                        rd['intimacy_level'] = 'casual'

                data['relationship_dynamics'] = rd
        return data

    @classmethod
    def _normalize_booleans(cls, data: Dict) -> Dict:
        """Convert various boolean representations."""
        def to_bool(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ['true', 'yes', '1', 't', 'y']
            if isinstance(val, (int, float)):
                return val > 0
            return False

        # Context flags
        if 'context_flags' in data and isinstance(data['context_flags'], dict):
            for key in data['context_flags']:
                data['context_flags'][key] = to_bool(data['context_flags'][key])

        # Relationship dynamics booleans
        if 'relationship_dynamics' in data and isinstance(data['relationship_dynamics'], dict):
            for key in ['conflict_present', 'boundary_setting']:
                if key in data['relationship_dynamics']:
                    data['relationship_dynamics'][key] = to_bool(
                        data['relationship_dynamics'][key])

        return data

    @classmethod
    def _normalize_numbers(cls, data: Dict) -> Dict:
        """Convert string numbers to actual numbers."""
        # IC/OOC ratio
        if 'ic_ooc_ratio' in data:
            try:
                ratio = float(str(data['ic_ooc_ratio']).rstrip('%') ) / 100 \
                        if '%' in str(data['ic_ooc_ratio']) else float(data['ic_ooc_ratio'])
                data['ic_ooc_ratio'] = max(0.0, min(1.0, ratio))
            except:
                data['ic_ooc_ratio'] = 0.5  # Default

        # Word count
        if 'word_count' in data:
            try:
                data['word_count'] = int(data['word_count'])
            except:
                data['word_count'] = 0

        return data

    @classmethod
    def _ensure_required_fields(cls, data: Dict) -> Dict:
        """Ensure all required fields exist with defaults."""
        defaults = {
            'chunk_id': 'unknown',
            'decisions': [],
            'emotional_moments': [],
            'key_quotes': [],
            'context_flags': {
                'has_vulnerability': False,
                'has_conflict': False,
                'has_intimacy': False,
                'has_humor': False,
                'has_creativity': False,
                'has_boundary_setting': False,
                'has_emotional_support': False
            }
        }

        for key, default_value in defaults.items():
            if key not in data:
                data[key] = default_value

        return data

    @classmethod
    def _enforce_limits(cls, data: Dict) -> Dict:
        """Enforce array length limits."""
        limits = {
            'decisions': 5,
            'emotional_moments': 5,
            'key_quotes': 5,
            'behavioral_patterns': 3,
            'if_then_observations': 3
        }

        for field, limit in limits.items():
            if field in data and isinstance(data[field], list):
                data[field] = data[field][:limit]

        return data


def test_normalizer():
    """Test the normalizer with various Gemini quirks."""

    # Test case 1: Field name variations
    test1 = {
        'decision_points': [
            {'chosen': 'help', 'rejected': 'ignore'}
        ],
        'emotions': [
            {'emotion': 'angry', 'intensity': '0.8'}
        ],
        'quotes': ['test quote'],
        'flags': {'has_conflict': 'yes'}
    }

    result1 = GeminiNormalizer.normalize(test1)
    assert 'decisions' in result1
    assert result1['decisions'][0]['choice'] == 'help'
    assert result1['emotional_moments'][0]['emotion'] == 'anger'
    assert result1['emotional_moments'][0]['intensity'] == 'high'
    assert result1['context_flags']['has_conflict'] == True

    print("✓ Test 1 passed: Field name normalization")

    # Test case 2: JSON in markdown
    test2 = """
    Here's the analysis:
    ```json
    {
        "chunk_id": "test123",
        "decisions": [
            {"choice": "defend", "alternative": "flee"}
        ],
        "emotional_moments": [],
        "key_quotes": ["I won't back down"],
        "context_flags": {}
    }
    ```
    """

    result2 = GeminiNormalizer.normalize(test2)
    assert result2['chunk_id'] == 'test123'
    assert result2['decisions'][0]['choice'] == 'defend'

    print("✓ Test 2 passed: JSON extraction from markdown")

    print("\nAll tests passed! Normalizer ready for Gemini's creative outputs.")


if __name__ == '__main__':
    test_normalizer()