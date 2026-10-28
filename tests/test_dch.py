import ast
import csv
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from baseline.common import digest,write_json
from dch.backend import chat_strings
from dch.cut import cut_caption
from dch.heal import heal_caption
from dch.inputs import load_inputs
from dch.nlp import PaperNLP
from dch.report import report


class DCHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = PaperNLP()

    def cut(self,text,flags):
        return cut_caption(text,self.nlp.candidates(text),flags,self.nlp)

    def test_physical_word_lemmas_and_paper_stoplists(self):
        groups = self.nlp.candidates("Dogs and people stand beside a machine. The atmosphere and overall scene have color and size.")
        lemmas = {g['lemma'] for g in groups}
        self.assertTrue({'dog','people','machine'} <= lemmas)
        self.assertFalse({'atmosphere','scene','color','size'} & lemmas)
        self.assertTrue(self.nlp.physical('people'))
        self.assertFalse(self.nlp.physical('second'))
        self.assertIn('machine',{g['lemma'] for g in self.nlp.candidates('A sewing machine runs.')})
        self.assertNotIn('sewing machine',{g['lemma'] for g in self.nlp.candidates('A sewing machine runs.')})
        adapted = PaperNLP(audio_meta=True)
        self.assertFalse({'audio','recording','clip'} & {g['lemma'] for g in adapted.candidates('The audio is a recording. The clip ends.')})

    def test_negation_is_not_silently_changed_from_paper(self):
        self.assertIn('dog',{g['lemma'] for g in self.nlp.candidates('There is no dog here.')})

    def test_list_and_first_conjunct_preserve_accepted_objects(self):
        result = self.cut('A cat, a dog, and a horse are nearby.',{'dog'})
        self.assertEqual(result['text'],'A cat and a horse are nearby.')
        self.assertEqual(result['events'][0]['rule'],'released_reference_grammar')
        result = self.cut('A cat and a dog sit on a bench.',{'cat'})
        self.assertIn('a dog',result['text'])
        self.assertIn('bench',result['text'])
        self.assertNotIn('cat',result['text'])

    def test_preposition_clause_and_sentence_rules(self):
        self.assertEqual(self.cut('A cat sits beside a dog.',{'dog'})['text'],'A cat sits.')
        self.assertEqual(self.cut('A cat sleeps, while a dog runs, under a tree.',{'dog'})['text'],
                         'A cat sleeps under a tree.')
        self.assertEqual(self.cut('A cat sleeps. A dog runs.',{'dog'})['text'],'A cat sleeps.')

    def test_protected_clause_and_compound_are_not_blindly_deleted(self):
        result = self.cut('The cat chases the dog.',{'dog'})
        self.assertEqual(result['text'],'The cat chases the dog.')
        self.assertEqual(result['remaining_flagged_lemmas'],['dog'])
        self.assertEqual(result['events'][0]['status'],'residual')
        result = self.cut('A dog holds a train horn.',{'train'})
        self.assertIn('horn',result['text'])
        self.assertIn('dog',result['text'])

    def test_repeated_rejected_mentions_and_empty_cut(self):
        result = self.cut('A dog is sleeping. The dog is running.',{'dog'})
        self.assertEqual(result['text'],'')
        self.assertEqual(result['remaining_flagged_lemmas'],[])
        class NoBackend:
            tokenizer = None
            def __getattr__(self,key):
                raise AssertionError('Empty cut must not trigger new caption generation')
        repaired = heal_caption('A dog barks.','',NoBackend(),self.nlp,{'dog'},set())
        self.assertEqual(repaired['status'],'empty_cut_caption')
        self.assertEqual(repaired['nfe'],0)

    def test_supplied_source_files_are_byte_identical(self):
        root=Path('dch/reference')
        metadata=json.loads((root/'SOURCE.json').read_text())
        for name,expected in metadata['files'].items():
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),expected,name)

    def test_candidates_and_grammar_match_supplied_functions(self):
        from dch.reference.candidates import candidates
        from dch.reference.cut import cut
        text='A cat, a dog, and a horse are nearby.'
        expected=candidates(list(text),text)
        self.assertEqual([g['lemma'] for g in self.nlp.candidates(text)],list(expected))
        self.assertEqual(self.cut(text,{'dog'})['text'],cut(text,['dog'],['cat','horse']))

    def test_prompt_matches_official_chat_template(self):
        source = Path('external/DIFFA/DIFFA-2/src/dataloader.py')
        if not source.exists():
            self.skipTest('Pinned upstream checkout is not present')
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_chat_prompt')
        namespace = {}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
        build = namespace['build_chat_prompt']
        messages = [{'role':'system','content':'You are a helpful voice assistant.'},
                    {'role':'user','content':'This is the audio: '}]
        prefix = build(messages,add_generation_prompt=False)[:-10]
        messages[-1]['content'] = 'This is the audio: <audio>. Describe the audio.'
        self.assertEqual(chat_strings('Describe the audio.'),(prefix,build(messages,add_generation_prompt=True)))

    def test_saved_inputs_and_blinded_report_without_regeneration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root/'data'
            data.mkdir()
            audio = data/'clip.wav'
            audio.write_bytes(b'not decoded in this input/report fixture')
            sample = dict(id='clip',audio_path='clip.wav',audio_sha256=digest(audio),
                          sample_rate=16000,num_samples=16000,duration_seconds=1,
                          reference_captions=['A dog barks.'],labels=['dog'])
            manifest = data/'manifest.json'
            write_json(manifest,dict(schema_version=1,samples=[sample]))
            source = root/'baseline'
            write_json(source/'config.json',dict(manifest=str(manifest),manifest_sha256=digest(manifest),
                                                instruction='Describe the audio.',decode=dict(gen_length=128)))
            write_json(source/'predictions/clip.json',dict(id='clip',text='A dog barks.',
                                                          run_config_sha256=digest(source/'config.json')))
            path,samples,predictions,config,origin = load_inputs(source)
            self.assertEqual(predictions['clip']['text'],'A dog barks.')
            out = root/'dch'
            write_json(out/'clips/clip.json',dict(id='clip',config_sha256='test',original='A dog barks.',
                       cut='A dog barks.',healed='A dog barks.',detections=[],detector_nfe=0,total_added_nfe=0,
                       cutting=dict(remaining_flagged_lemmas=[]),healing=dict(status='no_cut',fixed_tokens_preserved=True)))
            result = report(out,samples,path,'test')
            self.assertTrue(result['complete'])
            key=json.loads((out/'private_judge_key.json').read_text())
            self.assertEqual({v['version'] for v in key.values()},{'original','cut','healed'})
            for file in (out/'judge_inputs').glob('*.jsonl'):
                items = [json.loads(line) for line in file.read_text().splitlines()]
                self.assertEqual(len(items),1)
                self.assertNotIn('version',items[0])
            saved=json.loads((source/'predictions/clip.json').read_text())
            saved['run_config_sha256']='wrong'
            write_json(source/'predictions/clip.json',saved)
            with self.assertRaisesRegex(ValueError,'Mixed or invalid'):
                load_inputs(source)


if __name__=='__main__':
    unittest.main()
