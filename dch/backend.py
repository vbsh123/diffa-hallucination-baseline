"""DIFFA's model interface for the unmodified released DCH helpers."""
import math

SYSTEM='You are a helpful voice assistant.'
QUESTION='Is there a sound of {object} in the audio?\nAnswer the question using a single word or phrase.'


def chat_strings(instruction):
    system='<|startoftext|><|start_header_id|>system<|end_header_id|>\n\n'+SYSTEM+'<|eot_id|>'
    prefix=system+'<|start_header_id|>user<|end_header_id|>\n\nThis is the audio: '
    full=prefix+'<audio>. '+instruction+'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    return prefix,full


def answer_variants(tokenizer):
    result={}
    for label in ('Yes','No'):
        ids=[]
        for form in (label,label.lower(),' '+label,' '+label.lower()):
            encoded=tokenizer.encode(form,add_special_tokens=False)
            if len(encoded)!=1:
                raise ValueError(f'Released detector assumes single-token Yes/No variants: {form!r} -> {encoded}')
            ids.append(encoded[0])
        if len(set(ids))!=len(ids):
            raise ValueError('Released detector would double-count duplicate answer variants on this tokenizer')
        result[label.lower()]=ids
    if set(result['yes']) & set(result['no']):
        raise ValueError('Overlapping Yes/No variants')
    return result


def logodds(logits,yes_ids,no_ids):
    import torch
    values=logits.to(torch.float64)
    yes=torch.logsumexp(values[yes_ids],0)
    no=torch.logsumexp(values[no_ids],0)
    normalizer=torch.logsumexp(values,0)
    score=float(yes-no)
    if not math.isfinite(score):
        raise ValueError('Nonfinite detector score')
    py,pn=float((yes-normalizer).exp()),float((no-normalizer).exp())
    return dict(score=score,p_yes=py,p_no=pn,yes_no_mass=py+pn)


class DiffaBackend:
    def __init__(self,model,tokenizer,speech,instruction,threshold=0.):
        import torch
        self.model,self.tokenizer,self.tok,self.speech=model,tokenizer,tokenizer,speech
        self.instruction,self.threshold=instruction,threshold
        self.device=model.llm_model.device
        self.mask_id=model.llm_model.config.mask_token_id
        if self.mask_id!=126336:
            raise ValueError('Released infill assumes the LLaDA MASK id 126336')
        self.eos_id=self.eot=tokenizer.eos_token_id
        if self.eot is None:
            raise ValueError('Missing native DIFFA EOS id')
        self.embed=model.llm_model.get_input_embeddings()
        with torch.inference_mode():
            self.mask_embed=self.embed(torch.tensor([self.mask_id],device=self.device))
        self.answer_ids=answer_variants(tokenizer)
        self.special_ids=set(tokenizer.all_special_ids)|{self.mask_id,self.eot}
        self.nfe=0
        self.cache={}
        self.prompts={}
        self.last_decoded_canvas=None
        self.last_logits=None

    @staticmethod
    def audio_question(question):
        return question.replace('Is there a ','Is there a sound of ',1).replace(
            'Are there any ','Are there any sounds of ',1).replace('in the image?','in the audio?')

    def prompt(self,instruction):
        if instruction not in self.prompts:
            prefix,text=chat_strings(instruction)
            prefix_ids=self.tokenizer(prefix,return_tensors='pt')
            ids=self.tokenizer(text,return_tensors='pt').input_ids.to(self.device)
            insertion=int(prefix_ids.attention_mask.sum())-1
            if (ids==self.mask_id).any() or not 0<=insertion<=ids.shape[1]:
                raise ValueError('Invalid DIFFA prompt')
            self.prompts[instruction]=(ids,insertion)
        return self.prompts[instruction]

    def prompt_embeds(self,unused_input,question):
        import torch
        # The author detector supplies an image-worded question. Only that text
        # and the model-specific multimodal embedding preparation are adapted.
        question=self.audio_question(question) if 'in the image?' in question else question
        ids,insertion=self.prompt(question)
        with torch.inference_mode():
            embeddings=self.embed(ids)
            return torch.cat([embeddings[:,:insertion],self.speech.to(embeddings.dtype),
                              embeddings[:,insertion:]],dim=1)

    def logits(self,embeds,start=0):
        import torch
        if embeds.shape[1]>self.model.llm_model.config.max_sequence_length:
            raise ValueError('Audio, prompt and answer exceed native context length')
        with torch.inference_mode():
            result=self.model.llm_model(inputs_embeds=embeds,use_cache=False).logits[:,start:].float()
        self.nfe+=embeds.shape[0]
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite native answer logits')
        self.last_logits=result[0,0].clone()
        return result

    def answer_logits(self,canvas,instruction=None):
        import torch
        context=self.prompt_embeds(None,self.instruction if instruction is None else instruction)
        ids=torch.as_tensor(canvas,dtype=torch.long,device=self.device).reshape(1,-1)
        return self.logits(torch.cat([context,self.embed(ids).to(context.dtype)],1),context.shape[1])[0]

    def verify(self,lemma,use_cache=True):
        from .reference.detect import yes_no,question_for,POST
        if use_cache and lemma in self.cache:
            return self.cache[lemma]
        before=self.nfe
        score=yes_no(self,None,lemma)
        diagnostics=logodds(self.last_logits,self.answer_ids['yes'],self.answer_ids['no'])
        diagnostics['score']=score  # Use the released float32 scoring result itself.
        diagnostics.update(lemma=lemma,question=self.audio_question(question_for(lemma).format(o=lemma)+POST),
                           threshold=self.threshold,flagged=score<self.threshold,answer_positions=2,
                           scored_position=0,nfe=self.nfe-before)
        if self.nfe-before!=1:
            raise AssertionError('Detector must use exactly one native forward')
        if use_cache:
            self.cache[lemma]=diagnostics
        return diagnostics

    def canvas(self,text,gen_length):
        from .reference.model import Model
        return Model.canvas(self,text,gen_length)

    def decode_answer(self,ids):
        from .reference.model import Model
        self.last_decoded_canvas=list(ids)
        return Model.decode_answer(self,ids)

    def forbidden_words(self,lemmas,nlp=None):
        from .reference.heal import ban_ids
        result=set()
        for word in lemmas:
            result.update(ban_ids(self.tok,word))
        return result

    def infill(self,original,positions,global_bans,position_bans=None):
        from .reference.model import infill
        positions=sorted(set(positions))
        bans={p:set(global_bans)|set((position_bans or {}).get(p,())) for p in positions}
        context=self.prompt_embeds(None,self.instruction)
        result=infill(self,context,original,positions,ban=bans)
        if any(result[i]!=original[i] for i in range(len(original)) if i not in positions):
            raise AssertionError('Released infill changed a frozen token')
        return result,[dict(position=p,token_id=result[p]) for p in positions]
