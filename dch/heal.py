"""Call the ZIP's seam/repair implementation unchanged, then audit token locality."""


def heal_caption(original,cut,backend,nlp,flagged,accepted,capacity=128,window=2):
    if original==cut or not cut:
        return dict(text=cut,status='no_cut' if original==cut else 'empty_cut_caption',
                    mutable_positions=[],changed_positions=[],fixed_tokens_preserved=True,nfe=0,
                    new_object_checks=[],nonword_retries=0)
    from .reference.heal import heal,seam_windows,invalid_words
    canvas,positions,seams=seam_windows(backend,original,cut,capacity,window)
    # Reject ambiguous carrier text rather than silently fixing span coordinates.
    if backend.tok.decode(canvas[:len(backend.tok.encode(cut,add_special_tokens=False))]).strip()!=cut:
        raise ValueError('Cut caption does not round-trip through the released canvas helper')
    before=backend.nfe
    checks=[]
    def is_bad(lemma):
        record=dict(backend.verify(lemma,use_cache=False))
        checks.append(record)
        return record['flagged']
    backend.last_decoded_canvas=list(canvas)
    text,info=heal(backend,backend.prompt_embeds(None,backend.instruction),original,cut,
                   sorted(flagged),sorted(accepted),is_bad,capacity,window)
    final=backend.last_decoded_canvas
    changed=[i for i,(a,b) in enumerate(zip(canvas,final)) if a!=b]
    if len(final)!=len(canvas) or not set(changed).issubset(positions):
        raise AssertionError('Repair changed a fixed token outside the released seam windows')
    return dict(text=text,status='healed' if positions else 'no_mutable_seam_tokens',
                cut_canvas_ids=canvas,healed_canvas_ids=final,mutable_positions=positions,
                seam_count=seams,changed_positions=changed,fixed_tokens_preserved=True,
                nfe=backend.nfe-before,new_object_checks=checks,reference_info=info,
                nonword_retries=info.get('nonword_retries',0),
                residual_nonword_spans=invalid_words(backend,final,positions))
