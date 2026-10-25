"""Offline evaluation of reviewed call transcripts; no model inference or uploads."""
import json
import math
import unicodedata
from collections import defaultdict


def normalize(text):
    if not isinstance(text, str):
        raise ValueError('Transcript must be a string; missing is not silence')
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def validate(rows):
    if not rows:
        raise ValueError('No observations')
    ids, calls = set(), {}
    for row in rows:
        for key in ('id', 'call_id', 'split', 'language'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError('Missing identifier: ' + key)
        if row['id'] in ids:
            raise ValueError('Duplicate segment id')
        ids.add(row['id'])
        if row['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Unknown split')
        if calls.setdefault(row['call_id'], row['split']) != row['split']:
            raise ValueError('Call leaked across splits')
        for key in ('reference', 'hypothesis'):
            normalize(row.get(key))
        if not isinstance(row.get('is_silence'), bool):
            raise ValueError('A reviewed silence label is required')
        if row['is_silence'] != (normalize(row['reference']) == ''):
            raise ValueError('Empty reference and reviewed silence label disagree')
        for key in ('audio_seconds', 'inference_seconds'):
            v = row.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError('Invalid timing: ' + key)


def score(rows):
    validate(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['split'], row['language'])].append(row)
    results = []
    for (split, language), group in sorted(groups.items()):
        words = word_errors = chars = char_errors = silent = false_speech = 0
        for row in group:
            ref, hyp = normalize(row['reference']), normalize(row['hypothesis'])
            if row['is_silence']:
                silent += 1
                false_speech += bool(hyp)
                continue
            words += len(ref.split())
            word_errors += distance(ref.split(), hyp.split())
            # Unicode code points, with spaces removed; not grapheme clusters.
            ref_chars, hyp_chars = ref.replace(' ', ''), hyp.replace(' ', '')
            chars += len(ref_chars)
            char_errors += distance(ref_chars, hyp_chars)
        results.append(dict(split=split, language=language, segments=len(group),
                            reference_words=words, word_errors=word_errors,
                            wer=word_errors / words if words else None,
                            codepoint_cer=char_errors / chars if chars else None,
                            reviewed_silence_segments=silent,
                            false_speech_segments=false_speech,
                            false_speech_rate=false_speech / silent if silent else None,
                            aggregate_rtf=sum(r['inference_seconds'] for r in group) /
                            sum(r['audio_seconds'] for r in group)))
    return {'normalization': 'NFC, casefold, collapse whitespace; preserve punctuation and script',
            'limitations': 'WER is token-weighted within split/language. No transliteration equivalence. '
            'CER uses Unicode code points, not grapheme clusters. Silence scored separately. '
            'RTF requires comparable measured hardware and timing boundaries. No model runs in this code.',
            'groups': results}


def load(path):
    with open(path, encoding='utf-8') as source:
        return [json.loads(line) for line in source if line.strip()]
