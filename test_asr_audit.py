import copy
import unittest
from asr_audit import distance, normalize, score

BASE = dict(id='1', call_id='a', split='test', language='English', reference='one two',
            hypothesis='one two', is_silence=False, audio_seconds=2, inference_seconds=1)

class EvaluationTests(unittest.TestCase):
    def row(self, **changes):
        return dict(BASE, **changes)
    def test_exact(self):
        self.assertEqual(score([BASE])['groups'][0]['wer'], 0)
    def test_edits(self):
        self.assertEqual(distance('a b c'.split(), 'a d'.split()), 2)
    def test_missing_hypothesis_is_deletion(self):
        self.assertEqual(score([self.row(hypothesis='')])['groups'][0]['wer'], 1)
    def test_silence_separate(self):
        d=score([self.row(reference='', hypothesis='hello', is_silence=True)])['groups'][0]
        self.assertIsNone(d['wer']); self.assertEqual(d['false_speech_rate'], 1)
    def test_silent_correct(self):
        d=score([self.row(reference='', hypothesis='', is_silence=True)])['groups'][0]
        self.assertEqual(d['false_speech_rate'], 0)
    def test_empty_not_assumed_silence(self):
        with self.assertRaises(ValueError): score([self.row(reference='')])
    def test_null_rejected(self):
        with self.assertRaises(ValueError): score([self.row(hypothesis=None)])
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError): score([BASE, BASE])
    def test_call_leakage_rejected(self):
        with self.assertRaises(ValueError): score([BASE, self.row(id='2',split='train')])
    def test_same_call_same_split_allowed(self):
        self.assertEqual(score([BASE,self.row(id='2')])['groups'][0]['segments'],2)
    def test_token_weighting(self):
        rows=[self.row(reference='one',hypothesis='wrong'),self.row(id='2',reference='a b c',hypothesis='a b c')]
        self.assertEqual(score(rows)['groups'][0]['wer'],.25)
    def test_insertions_not_clipped(self):
        self.assertEqual(score([self.row(reference='a',hypothesis='b c d')])['groups'][0]['wer'],3)
    def test_unicode_and_script(self):
        self.assertEqual(normalize('Cafe\u0301'), normalize('CAFÉ'))
        self.assertNotEqual(normalize('कल'),normalize('kal'))
    def test_invalid_duration(self):
        for value in [0,-1,float('nan'),float('inf'),True]:
            with self.assertRaises(ValueError): score([self.row(audio_seconds=value)])
    def test_rtf_aggregate(self):
        d=score([BASE,self.row(id='2',audio_seconds=8,inference_seconds=1)])['groups'][0]
        self.assertEqual(d['aggregate_rtf'],.2)
    def test_split_separation(self):
        self.assertEqual(len(score([BASE,self.row(id='2',call_id='b',split='validation')])['groups']),2)

if __name__ == '__main__': unittest.main()
