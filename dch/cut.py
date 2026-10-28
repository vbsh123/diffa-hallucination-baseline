"""Call the supplied author's deletion grammar without changing its regexes."""
from difflib import SequenceMatcher
from .reference.cut import cut
from .reference.candidates import surface_forms


def cut_caption(text,candidates,rejected,nlp):
    rejected = sorted(set(rejected))
    accepted = sorted(g['lemma'] for g in candidates if g['lemma'] not in rejected)
    result = cut(text,rejected,accepted)
    remaining = [n for n in rejected if surface_forms(result,n)]
    # Audit alignment only. These spans do not change the author's cut decisions.
    spans = [(i1,i2) for tag,i1,i2,j1,j2 in SequenceMatcher(a=text,b=result,autojunk=False).get_opcodes()
             if tag in ('delete','replace')]
    events = [dict(lemma=n,status='residual' if n in remaining else 'removed',
                   rule='released_reference_grammar',surface_forms=surface_forms(text,n)) for n in rejected]
    return dict(text=result,spans=spans,events=events,remaining_flagged_lemmas=remaining)
