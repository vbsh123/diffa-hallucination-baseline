"""Detect-Cut-Heal on one caption.

    out = correct(model, image, question, caption, tau=0.0, window=2)

1. Candidates: the physical-object nouns of the caption (candidates.py).
2. Detect: one yes/no question per candidate; flag the objects with s(o) < tau.
3. Cut: delete each flagged object with the smallest phrase that holds it; sentences that
   name a verified (unflagged) object are protected.
4. Heal: re-mask `window` tokens on each side of every seam and let the model infill them,
   flagged words forbidden, non-words rejected, new objects re-verified.

The returned dict holds the original and edited captions, the detector scores, the flagged
objects, the cut text (the Cut-only variant), the flagged objects that survive (mentions
the grammar could not detach), and the number of forward passes spent on detection and
on the edit. A caption without flagged objects is returned unchanged.
"""
import time

import torch

from .candidates import candidates, surface_forms
from .cut import cut
from .detect import detect, yes_no
from .heal import heal


def correct(m, image, question, caption, tau=0.0, window=2, gen_length=128):
    """Run DCH on `caption` (written for `image` in answer to `question`, the same prompt
    that produced it, so the repair sees the context the caption was written in)."""
    image = image.convert("RGB")
    out = {"orig": caption}
    nfe0 = m.nfe
    t0 = time.time()
    ids, n_text = m.canvas(caption, gen_length)
    strs = [m.tok.decode([i]) for i in ids]
    cands = candidates(strs[:n_text], caption)
    scores = detect(m, image, sorted(cands))
    out["scores"] = scores
    flagged = sorted(o for o, s in scores.items() if s < tau)
    protected = sorted(o for o, s in scores.items() if s >= tau)
    out["flagged"] = flagged
    out["nfe_detect"] = m.nfe - nfe0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    out["sec_detect"] = round(t1 - t0, 3)
    nfe1 = m.nfe
    if not flagged:
        out.update(cut_text=caption, edited=caption, residual=[], nfe_edit=0, sec_edit=0.0)
        return out
    cut_text = cut(caption, flagged, protected)
    out["cut_text"] = cut_text
    if cut_text == caption:                      # nothing could be detached
        out.update(edited=caption, residual=flagged, nfe_edit=0, sec_edit=0.0, cuts=0)
        return out
    pemb = m.prompt_embeds(image, question)
    is_bad = lambda o: yes_no(m, image, o) < tau
    edited, info = heal(m, pemb, caption, cut_text, flagged, protected, is_bad, gen_length, window)
    out["edited"] = edited
    out.update(info)
    out["residual"] = [o for o in flagged if surface_forms(edited, o)]
    out["nfe_edit"] = m.nfe - nfe1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    out["sec_edit"] = round(time.time() - t1, 3)
    return out
