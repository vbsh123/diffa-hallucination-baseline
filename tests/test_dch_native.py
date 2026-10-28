"""Opt-in pinned-tokenizer/tiny-upstream CPU checks; never downloads full weights."""
import os
from pathlib import Path
import sys
import unittest


@unittest.skipUnless(os.environ.get('DCH_NATIVE_TEST')=='1','Set DCH_NATIVE_TEST=1 with torch and cached pinned tokenizer')
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import AutoTokenizer
        from dch.backend import DiffaBackend
        sys.dont_write_bytecode=True
        sys.path.insert(0,str(Path('external/DIFFA/DIFFA-2').resolve()))
        from src.fast_dllm.model.configuration_llada import LLaDAConfig
        from src.fast_dllm.model.modeling_llada import LLaDAModelLM
        torch.set_num_threads(2)
        torch.manual_seed(7)
        config=LLaDAConfig(d_model=32,n_layers=1,n_heads=4,n_kv_heads=4,max_sequence_length=512,
                           vocab_size=126464,embedding_size=126464,mask_token_id=126336,
                           block_type='llama',activation_type='silu',rope=True,flash_attention=False)
        llm=LLaDAModelLM(config,init_params=True).to(torch.bfloat16).eval()
        class Wrapper(torch.nn.Module):
            def __init__(self,llm):
                super().__init__()
                self.llm_model=llm
        cls.torch=torch
        cls.model=Wrapper(llm)
        cls.tokenizer=AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct',
                     revision='08b83a6feb34df1a6011b80c3c00c7563e963b07',local_files_only=True)
        cls.backend=DiffaBackend(cls.model,cls.tokenizer,torch.zeros(1,3,32,dtype=torch.bfloat16),'Describe the audio.')

    def test_raw_forward_matches_upstream_audio_embedding_insertion(self):
        from src.fast_dllm.diffa_generate import insert_audio_embedding
        b,t=self.backend,self.torch
        prompt,insertion=b.prompt('Describe the audio.')
        canvas=t.tensor([[b.mask_id,b.mask_id]])
        with t.inference_mode():
            embedding=self.model.llm_model.get_input_embeddings()(t.cat([prompt,canvas],dim=1))
            inserted=insert_audio_embedding(embedding,b.speech,t.tensor([insertion]))
            expected=self.model.llm_model(inputs_embeds=inserted,use_cache=False).logits[0,-2:]
            actual=b.answer_logits(canvas)
        self.assertTrue(t.equal(expected,actual))

    def test_detector_is_one_forward_and_uses_pooled_answer_variants(self):
        from dch.backend import logodds
        b,t=self.backend,self.torch
        b.cache.clear()
        before=b.nfe
        result=b.verify('dog')
        self.assertEqual(b.nfe-before,1)
        self.assertEqual(result['answer_positions'],2)
        self.assertEqual(result['flagged'],result['score']<0)
        self.assertEqual(b.verify('dog'),result)
        self.assertEqual(b.nfe-before,1)
        scores=t.tensor([0.,1.,2.,3.,4.])
        result=logodds(scores,[0,1],[2,3])
        probs=t.softmax(scores.double(),dim=0)
        expected=t.log(probs[[0,1]].sum())-t.log(probs[[2,3]].sum())
        self.assertAlmostEqual(result['score'],float(expected),places=12)

    def test_local_infill_never_changes_frozen_tokens_or_emits_banned_tokens(self):
        b=self.backend
        ids=self.tokenizer.encode('A man speaks near a bird.',add_special_tokens=False)+[b.eos_id]*3
        before=b.nfe
        positions=[1,2]
        output,trace=b.infill(ids,positions,{17,18,19})
        self.assertEqual(b.nfe-before,len(positions))
        self.assertEqual(len(trace),len(positions))
        self.assertTrue(all(output[i]==ids[i] for i in range(len(ids)) if i not in positions))
        self.assertTrue(all(output[i] not in b.special_ids|{17,18,19} for i in positions))

    def test_complete_heal_matches_unmodified_supplied_helper(self):
        from dch.backend import DiffaBackend
        from dch.heal import heal_caption
        from dch.nlp import PaperNLP
        from dch.reference.heal import heal,seam_windows
        b=DiffaBackend(self.model,self.tokenizer,self.backend.speech,'Describe the audio.')
        original,cut='A cat, a dog, is sleeping.','A cat is sleeping.'
        actual=heal_caption(original,cut,b,PaperNLP(tokenizer=self.tokenizer),{'dog'},{'cat'},16)
        other=DiffaBackend(self.model,self.tokenizer,self.backend.speech,'Describe the audio.')
        expected,info=heal(other,other.prompt_embeds(None,other.instruction),original,cut,
                           ['dog'],['cat'],lambda x:other.verify(x,use_cache=False)['flagged'],16,2)
        self.assertEqual(actual['text'],expected)
        self.assertEqual(actual['reference_info'],info)
        self.assertEqual(b.nfe,other.nfe)
        self.assertTrue(actual['fixed_tokens_preserved'])
        self.assertTrue(set(actual['changed_positions'])<=set(actual['mutable_positions']))

    def test_actual_tokenizer_candidates_match_supplied_helper(self):
        from dch.nlp import PaperNLP
        from dch.reference.candidates import candidates
        text='A man speaks beside a sewing machine and two dogs.'
        ids=self.tokenizer.encode(text,add_special_tokens=False)
        strings=[self.tokenizer.decode([i]) for i in ids]
        expected=candidates(strings,text)
        actual=PaperNLP(tokenizer=self.tokenizer).candidates(text)
        self.assertEqual([r['lemma'] for r in actual],list(expected))
        for row in actual:
            self.assertEqual([m['token_indices'] for m in row['mentions']],expected[row['lemma']])


if __name__=='__main__':
    unittest.main()
