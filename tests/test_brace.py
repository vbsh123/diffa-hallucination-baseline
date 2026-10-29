import tempfile
from pathlib import Path
import unittest

from baseline.common import digest, write_json
from dch.brace import load_inputs, parse_pair, report, score_pair
from dch.inputs import load_inputs as dispatch_inputs
from dch.nlp import PaperNLP


class BraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = PaperNLP()

    def pair(self):
        return dict(pair_id='p', control_id='p_human', corrupted_id='p_corrupted',
                    human='A dog barks beside a cat.', corrupted='A dragon barks beside a cat.',
                    injected_noun_lemmas=['dragon'])

    def record(self, original, result, detections):
        return dict(original=original, cut=result, healed=result,
                    detections=[dict(lemma=n, flagged=f) for n, f in detections],
                    total_added_nfe=3, elapsed_seconds=1,
                    healing=dict(fixed_tokens_preserved=True))

    def test_reversed_caption_sides_and_punctuation_are_not_targets(self):
        row = dict(file_name='1.wav', caption_1=[
            'A dragon barks beside a cat.', 'A dog barks beside a cat',
            'hallucination', 'human', dict(references=['A dog and a cat.'])])
        result = parse_pair(row, 'caption_1', self.nlp)
        self.assertEqual(result['human_side'], 1)
        self.assertEqual(result['injected_noun_lemmas'], ['dragon'])
        row['caption_1'][:2] = ['A dog barks.', 'A dog barks']
        result = parse_pair(row, 'caption_1', self.nlp)
        self.assertEqual(result['injected_noun_lemmas'], [])
        self.assertEqual(result['automatic_target_status'], 'no_distinct_changed_noun')

    def test_pos_missed_known_swap_remains_in_target_denominator(self):
        row = dict(file_name='1.wav', caption_1=[
            'A person sneezing then speaking followed by someone chuckles as a clock ticks in the background',
            'A person sneezing then speaking followed by someone chuckles as an elephant ticks in the background.',
            'human', 'hallucination', dict(references=['A clock ticks.'])])
        pair = parse_pair(row, 'caption_1', self.nlp)
        self.assertEqual(pair['injected_noun_lemmas'], ['elephant'])
        pair['pair_id'] = 'p'
        ctrl = self.record(pair['human'], pair['human'], [])
        bad = self.record(pair['corrupted'], pair['corrupted'], [('person', False)])
        score = score_pair(pair, ctrl, bad, self.nlp)
        self.assertEqual(score['target_count'], 1)
        self.assertEqual(score['candidate_target_count'], 0)
        self.assertEqual(score['flagged_target_count'], 0)

    def test_target_removal_and_control_damage_are_separate(self):
        pair = self.pair()
        ctrl = self.record(pair['human'], 'A cat.', [('dog', True), ('cat', False)])
        bad = self.record(pair['corrupted'], 'A cat.', [('dragon', True), ('cat', False)])
        score = score_pair(pair, ctrl, bad, self.nlp)
        self.assertEqual(score['flagged_target_count'], 1)
        self.assertEqual(score['cut']['target_mentions_removed'], 1)
        self.assertEqual(score['control_original_noun_count'], 2)
        self.assertEqual(score['cut']['control_nouns_retained'], 1)
        self.assertEqual(score['corrupted_shared_noun_count'], 1)
        self.assertEqual(score['cut']['corrupted_shared_nouns_retained'], 1)
        self.assertFalse(score['cut']['control_unchanged'])
        self.assertFalse(score['cut']['corrupted_empty'])
        bad['healed'] = ''
        self.assertTrue(score_pair(pair, ctrl, bad, self.nlp)['healed']['corrupted_empty'])
        self.assertEqual(score_pair(pair, ctrl, bad, self.nlp)['healed']['corrupted_shared_nouns_retained'], 0)

    def fixture(self, root):
        (root/'audio').mkdir()
        audio = root/'audio/1.wav'
        audio.write_bytes(b'not decoded by input/scoring fixture')
        pair = self.pair()
        samples = []
        for role in ('human', 'corrupted'):
            samples.append(dict(id='p_'+role, audio_path='audio/1.wav',
                                audio_sha256=digest(audio), sample_rate=16000, num_samples=16000,
                                reference_captions=[pair['human']], labels=[],
                                brace=dict(pair_id='p', role=role, text=pair[role])))
        write_json(root/'pairs.json', dict(schema_version=1, pairs=[pair]))
        manifest = root/'manifest.json'
        write_json(manifest, dict(schema_version=1, selection={}, samples=samples,
                                 pairs_sha256=digest(root/'pairs.json')))
        return manifest, pair, samples

    def test_input_dispatch_pairs_and_annotation_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, pair, _ = self.fixture(root)
            path, samples, predictions, config, origin = dispatch_inputs(brace_cohort=manifest, samples=1)
            self.assertEqual(len(samples), 2)
            self.assertEqual(predictions['p_corrupted'], dict(id='p_corrupted', text=pair['corrupted']))
            self.assertNotIn('reference', predictions['p_corrupted'])
            self.assertEqual(origin['kind'], 'released_brace_caption_pairs')
            with self.assertRaisesRegex(ValueError, 'cannot be mixed'):
                dispatch_inputs(baseline_run=root, brace_cohort=manifest)
            with (root/'pairs.json').open('a') as stream:
                stream.write(' ')
            with self.assertRaisesRegex(ValueError, 'annotations changed'):
                load_inputs(manifest)

    def test_report_requires_complete_pairs_and_counts_empty_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, pair, samples = self.fixture(root)
            out = root/'run'
            ctrl = self.record(pair['human'], pair['human'], [('dog', False), ('cat', False)])
            bad = self.record(pair['corrupted'], '', [('dragon', True), ('cat', False)])
            for sample, record in zip(samples, [ctrl, bad]):
                record.update(id=sample['id'], config_sha256='test')
                write_json(out/'clips'/f"{sample['id']}.json", record)
            result = report(out, samples, manifest, 'test')
            self.assertTrue(result['complete'])
            self.assertEqual(result['target_detection_recall'], 1)
            self.assertEqual(result['stages']['healed']['control_noun_retention'], 1)
            self.assertEqual(result['stages']['healed']['corrupted_empty_captions'], 1)
            (out/'clips/p_human.json').unlink()
            result = report(out, samples, manifest, 'test', detect_only=True)
            self.assertFalse(result['complete'])
            self.assertEqual(result['completed_pairs'], 0)
            self.assertIsNone(result['target_detection_recall'])


if __name__ == '__main__':
    unittest.main()
