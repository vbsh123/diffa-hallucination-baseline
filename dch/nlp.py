"""Adapter around the unmodified candidate extractor supplied in DCH-main.zip."""
import hashlib
import json
from pathlib import Path

from .reference.candidates import STOP,VAGUE,candidates,is_physical

AUDIO_META_STOP = frozenset({"audio","recording","clip"})
GROUP_STOP,VAGUE_STOP = frozenset(STOP),frozenset(VAGUE)


class PaperNLP:
    def __init__(self,audio_meta=False,tokenizer=None):
        import nltk
        from nltk.corpus import wordnet
        from nltk.stem import WordNetLemmatizer
        self.nltk,self.wn,self.tokenizer = nltk,wordnet,tokenizer
        self.audio_meta = audio_meta
        self.lemmatizer = WordNetLemmatizer()
        try:
            wordnet.ensure_loaded()
            nltk.pos_tag(nltk.word_tokenize('A test.'))
        except LookupError as error:
            raise RuntimeError('Missing DCH NLP resources; run bash setup_dch.sh') from error

    @staticmethod
    def physical(lemma):
        return is_physical(lemma)

    def candidates(self,text):
        # Source helper uses concatenated per-token decodes. Character tokens are
        # equivalent for CPU-only inventory; real runs use the actual tokenizer.
        if self.tokenizer is None:
            strings = list(text)
        else:
            ids = self.tokenizer.encode(text,add_special_tokens=False)
            strings = [self.tokenizer.decode([i]) for i in ids]
        if ''.join(strings) != text:
            raise ValueError('Released DCH token-span helper cannot exactly align this caption; no guessed spans used')
        starts = [0]
        for value in strings:
            starts.append(starts[-1]+len(value))
        result = []
        for lemma,mentions in candidates(strings,text).items():
            if self.audio_meta and lemma in AUDIO_META_STOP:
                continue
            result.append(dict(lemma=lemma,mentions=[dict(surface=text[starts[min(m)]:starts[max(m)+1]],
                               start=starts[min(m)],end=starts[max(m)+1],token_indices=m)
                               for m in mentions]))
        return result

    def provenance(self):
        source = Path(__file__).parent/'reference'
        meta = json.loads((source/'SOURCE.json').read_text())
        resources = {}
        names = ['taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.weights.json',
                 'taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.classes.json',
                 'taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.tagdict.json',
                 'tokenizers/punkt_tab/english/abbrev_types.txt','tokenizers/punkt_tab/english/collocations.tab',
                 'tokenizers/punkt_tab/english/ortho_context.tab','tokenizers/punkt_tab/english/sent_starters.txt']
        for name in names:
            with self.nltk.data.find(name).open() as stream:
                resources[name] = hashlib.sha256(stream.read()).hexdigest()
        for name in ['data.noun','data.adj','data.verb','data.adv','index.noun','index.adj','index.verb',
                     'index.adv','noun.exc','adj.exc','verb.exc','adv.exc','lexnames']:
            with self.wn.root.join(name).open() as stream:
                resources['wordnet/'+name] = hashlib.sha256(stream.read()).hexdigest()
        for name,expected in meta['files'].items():
            if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Released DCH file changed: {name}')
        return dict(nltk=self.nltk.__version__,wordnet=self.wn.get_version(),resources=resources,
                    reference_source=meta,group_stop=sorted(STOP),vague_stop=sorted(VAGUE),
                    audio_meta_stoplist_added=self.audio_meta,
                    audio_meta_stoplist=sorted(AUDIO_META_STOP) if self.audio_meta else [])
