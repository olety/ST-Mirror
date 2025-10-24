#!/usr/bin/env python3
"""
Arc-aware segmentation for ST-Mirror

Detects narrative arc transitions and splits long RPs accordingly,
then processes each arc separately to handle context shifts.
"""

import os, json, glob
from typing import List, Dict, Any, Tuple
from collections import defaultdict

def detect_ooc_markers(text: str) -> bool:
    """Detect OOC (out-of-character) markers"""
    text_lower = text.lower()
    return any([
        'ooc' in text_lower,
        '((' in text and '))' in text,
        text.strip().startswith('>'),
        text.strip().startswith('[OOC'),
        '(((' in text,
    ])

def analyze_window(messages: List[Dict], window_size: int = 50) -> List[Dict[str, Any]]:
    """Analyze RP in windows to detect arc characteristics"""
    windows = []

    for i in range(0, len(messages), window_size):
        window = messages[i:i+window_size]
        if not window:
            continue

        # Calculate metrics
        ooc_count = 0
        ic_markers = 0
        sexual_content = 0
        personal_markers = 0
        emotional_markers = 0
        avg_length = 0

        for msg in window:
            text = msg.get('text', '') or msg.get('mes', '') or msg.get('content', '')
            text_lower = text.lower()

            # OOC detection
            if detect_ooc_markers(text):
                ooc_count += 1

            # IC markers (asterisks for actions)
            if text.count('*') > 2:
                ic_markers += 1

            # Sexual content markers
            sexual_words = ['fuck', 'cock', 'pussy', 'cum', 'sex', 'orgasm', 'moan',
                           'thrust', 'penetrat', 'nipple', 'breast', 'ass', 'dick']
            if any(w in text_lower for w in sexual_words):
                sexual_content += 1

            # Personal/real markers (genuine conversation)
            personal_words = ['actually', 'real life', 'honestly', 'i think', 'i feel',
                            'personally', 'to be honest', 'in reality', 'my actual']
            if any(w in text_lower for w in personal_words):
                personal_markers += 1

            # Emotional vulnerability markers
            emotional_words = ['love', 'care', 'scared', 'afraid', 'worried', 'trust',
                             'hurt', 'cry', 'tears', 'heart', 'feel']
            if any(w in text_lower for w in emotional_words):
                emotional_markers += 1

            avg_length += len(text)

        n = len(window)
        windows.append({
            'start_idx': i,
            'end_idx': min(i + window_size, len(messages)),
            'ooc_ratio': ooc_count / n,
            'ic_ratio': ic_markers / n,
            'sexual_ratio': sexual_content / n,
            'personal_ratio': personal_markers / n,
            'emotional_ratio': emotional_markers / n,
            'avg_length': avg_length / n
        })

    return windows

def detect_arc_transitions(windows: List[Dict],
                           ooc_threshold: float = 0.3,
                           content_threshold: float = 0.25) -> List[int]:
    """Detect where narrative arcs transition based on metric shifts"""
    transitions = [0]  # Always start with 0

    for i in range(1, len(windows)):
        prev = windows[i-1]
        curr = windows[i]

        # Calculate shifts
        ooc_shift = abs(curr['ooc_ratio'] - prev['ooc_ratio'])
        sexual_shift = abs(curr['sexual_ratio'] - prev['sexual_ratio'])
        personal_shift = abs(curr['personal_ratio'] - prev['personal_ratio'])

        # Detect significant transitions
        is_transition = (
            ooc_shift > ooc_threshold or
            sexual_shift > content_threshold or
            personal_shift > content_threshold
        )

        if is_transition:
            transitions.append(curr['start_idx'])

    return transitions

def classify_arc(window_stats: Dict) -> str:
    """Classify the type of narrative arc based on metrics"""
    ooc = window_stats['ooc_ratio']
    sexual = window_stats['sexual_ratio']
    personal = window_stats['personal_ratio']
    emotional = window_stats['emotional_ratio']

    # High OOC = meta-conversation
    if ooc > 0.4:
        return 'meta_conversation'

    # High sexual, low personal = fantasy/ERP
    elif sexual > 0.3 and personal < 0.2:
        return 'erotic_fantasy'

    # Low sexual, high personal = genuine intimacy
    elif sexual < 0.2 and personal > 0.3:
        return 'genuine_intimacy'

    # High personal + emotional = real romantic
    elif personal > 0.25 and emotional > 0.25:
        return 'romantic_connection'

    # Mixed
    elif sexual > 0.2 and personal > 0.2:
        return 'mixed_intimate'

    # Default
    else:
        return 'casual_roleplay'

def segment_by_arcs(jsonl_path: str,
                    output_dir: str,
                    window_size: int = 50,
                    min_arc_length: int = 100) -> List[str]:
    """
    Segment a long RP by narrative arcs and save each arc separately

    Returns list of arc file paths
    """

    # Load messages
    messages = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    messages.append(json.loads(line))
                except:
                    continue

    if len(messages) < min_arc_length:
        print(f'[arc-segment] File too short for arc detection ({len(messages)} messages)')
        return []

    # Analyze windows
    windows = analyze_window(messages, window_size)

    # Detect transitions
    transitions = detect_arc_transitions(windows)

    # Add end boundary
    if transitions[-1] != len(messages):
        transitions.append(len(messages))

    print(f'[arc-segment] Detected {len(transitions)-1} arcs in {len(messages)} messages')

    # Create arcs
    arcs = []
    arc_files = []

    # Extract base name once (needed for summary even if no arcs created)
    base_name = os.path.splitext(os.path.basename(jsonl_path))[0]

    for i in range(len(transitions) - 1):
        start_idx = transitions[i]
        end_idx = transitions[i + 1]

        # Skip tiny arcs
        if end_idx - start_idx < min_arc_length:
            continue

        arc_messages = messages[start_idx:end_idx]

        # Calculate arc statistics from windows
        arc_windows = [w for w in windows if w['start_idx'] >= start_idx and w['end_idx'] <= end_idx]
        if arc_windows:
            avg_stats = {
                'ooc_ratio': sum(w['ooc_ratio'] for w in arc_windows) / len(arc_windows),
                'sexual_ratio': sum(w['sexual_ratio'] for w in arc_windows) / len(arc_windows),
                'personal_ratio': sum(w['personal_ratio'] for w in arc_windows) / len(arc_windows),
                'emotional_ratio': sum(w['emotional_ratio'] for w in arc_windows) / len(arc_windows),
            }
            arc_type = classify_arc(avg_stats)
        else:
            arc_type = 'unknown'

        arc_data = {
            'arc_id': i + 1,
            'arc_type': arc_type,
            'start_message': start_idx,
            'end_message': end_idx,
            'message_count': len(arc_messages),
            'statistics': avg_stats if arc_windows else {},
            'messages': arc_messages
        }

        arcs.append(arc_data)

        # Save arc to file
        arc_file = os.path.join(output_dir, f'{base_name}_arc{i+1:02d}_{arc_type}.json')

        os.makedirs(output_dir, exist_ok=True)
        with open(arc_file, 'w', encoding='utf-8') as f:
            json.dump(arc_data, f, ensure_ascii=False, indent=2)

        arc_files.append(arc_file)

        print(f'  Arc {i+1}: messages {start_idx}-{end_idx} ({len(arc_messages)} msgs) - {arc_type}')
        if arc_windows:
            print(f'    Stats: OOC={avg_stats["ooc_ratio"]*100:.0f}% Sexual={avg_stats["sexual_ratio"]*100:.0f}% Personal={avg_stats["personal_ratio"]*100:.0f}%')

    # Save summary
    summary = {
        'source_file': jsonl_path,
        'total_messages': len(messages),
        'num_arcs': len(arcs),
        'arcs': [
            {
                'arc_id': a['arc_id'],
                'arc_type': a['arc_type'],
                'start_message': a['start_message'],
                'end_message': a['end_message'],
                'message_count': a['message_count'],
                'statistics': a['statistics']
            }
            for a in arcs
        ]
    }

    summary_file = os.path.join(output_dir, f'{base_name}_arc_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f'[arc-segment] Saved summary to {summary_file}')

    return arc_files

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Arc-aware RP segmentation')
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output-dir', required=True, help='Output directory for arc files')
    parser.add_argument('--window-size', type=int, default=50, help='Window size for analysis')
    parser.add_argument('--min-arc-length', type=int, default=100, help='Minimum messages per arc')

    args = parser.parse_args()

    segment_by_arcs(args.input, args.output_dir, args.window_size, args.min_arc_length)
