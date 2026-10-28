"""Cut: delete each flagged object together with the smallest phrase that holds it.

Re-filling a flagged word in place keeps a frame that still calls for an object, and the
model often fills it with another absent one. Deleting the phrase removes the obligation.
For every mention of a flagged object the grammar removes, in order of preference,

    item      a list item with its connector      "..., a bowl, ..."  / "... and a bowl"
    first     the first conjunct of a coordination "a horse and a shield"
    prep      a prepositional phrase               "a laptop sits on a desk"
    clause    a comma-delimited clause
    sentence  the whole sentence

A sentence (or clause) that also names a protected object, one the detector accepted,
is never deleted; a flagged mention with no detachable phrase inside such a sentence is
left in place. Regular expressions only, no model call.
"""
import re

DET = r"(?:a|an|the|some|two|three|four|several|another|one|other|more|few|many|various)"
CONN = r"(?:,\s*and|,\s*or|,|\s+and|\s+or|\s+as well as)"
# modifiers between determiner and noun: any word that is not a verb/preposition/clause marker
MOD = (r"(?:(?!(?:is|are|was|were|be|there|has|have|had|with|of|in|on|at|to|which|that|"
       r"who|while|where|and|or|its|their|including|such|as|like|for|from|by)\b)[\w-]+\s+)")
PREPS = (r"on top of|in front of|next to|on|in|at|with|near|beside|by|under|behind|inside|"
         r"atop|against|beneath|above|holding|carrying|wearing")


def sentences(text):
    return list(re.finditer(r"[^.!?]*[.!?]|[^.!?]+$", text))


def _adjective_comma(text, it):
    """A bare ', X NOUN' item (no determiner) right after an adjective is an adjective
    series ('a lush, green field'), not a list item."""
    g = it.group(0)
    if not g.lstrip().startswith(",") or re.match(rf",\s*(?:and\s+|or\s+)?{DET}\b", g.lstrip(), re.I):
        return False
    from nltk.corpus import wordnet as wn
    prev = re.findall(r"[\w-]+", text[:it.start()])
    if not prev:
        return False
    w = prev[-1].lower()
    n_adj = len(wn.synsets(w, pos=wn.ADJ)) + len(wn.synsets(w, pos=wn.ADJ_SAT))
    return n_adj > len(wn.synsets(w, pos=wn.NOUN))


def _cut(text, a0, a1):
    """Delete text[a0:a1]; an appositive between two commas takes both commas."""
    left = text[:a0].rstrip()
    right = text[a1:].lstrip()
    if text[a0:a1].lstrip().startswith(",") and right.startswith(","):
        return left + " " + right[1:].lstrip()
    if left.endswith(",") and right.startswith(","):
        return left[:-1] + " " + right[1:].lstrip()
    if left.endswith(",") and right[:1] in (".", "!", "?"):
        return left[:-1] + right
    return text[:a0] + text[a1:]


def retract_one(text, word, protect=()):
    """Delete every mention of the surface form `word` with its phrase. A sentence is
    deleted whole only if it mentions none of the `protect` words."""
    w = re.escape(word)
    item = re.compile(rf"{CONN}\s+(?:{DET}\s+)?{MOD}{{0,3}}?{w}\b(?:\s+of\s+[\w-]+)?", re.I)
    prep = re.compile(rf"\s+(?:{PREPS})\s+(?:{DET}\s+)?{MOD}{{0,3}}?{w}\b(?:\s+of\s+[\w-]+)?", re.I)
    first = re.compile(rf"\b(?:{DET}\s+)?{MOD}{{0,3}}?{w}\b(?:\s+of\s+[\w-]+)?(?:\s+and|\s+or)\s+(?={DET}\b)", re.I)
    for _ in range(50):
        m = re.search(rf"\b{w}\b", text, re.I)
        if not m:
            break
        best = None
        for pat in (item, first, prep):
            for it in pat.finditer(text):
                if it.start() <= m.start() < it.end() and not _adjective_comma(text, it):
                    best = it
                    break
            if best is not None:
                break
        if best is not None:
            text = _cut(text, best.start(), best.end())
            continue
        for s in sentences(text):
            if s.start() <= m.start() < s.end():
                if any(re.search(rf"\b{re.escape(p)}\b", s.group(0), re.I) for p in protect):
                    # the sentence holds a protected object: try the comma clause instead
                    cl = None
                    for c in re.finditer(r",[^,.!?]*(?=[,.!?])", s.group(0)):
                        if c.start() + s.start() <= m.start() < c.end() + s.start():
                            cl = c
                            break
                    if cl is None or any(re.search(rf"\b{re.escape(p)}\b", cl.group(0), re.I) for p in protect):
                        return re.sub(r"[ \t]{2,}", " ", text).strip()     # leave it in place
                    a0, a1 = s.start() + cl.start(), s.start() + cl.end()
                    text = _cut(text, a0, a1 + (1 if a1 < len(text) and text[a1] == "," else 0))
                    break
                text = text[:s.start()] + text[s.end():]
                break
        else:
            break
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def cut(caption, flagged, protected):
    """Delete every flagged object (lemmas) from the caption, protecting the sentences
    that name a verified object. Returns the cut text (equal to the caption if nothing
    could be detached)."""
    from .candidates import surface_forms
    keep = [s for n in protected for s in surface_forms(caption, n)]
    text = caption
    for n in flagged:
        for s in surface_forms(caption, n):
            text = retract_one(text, s, protect=keep)
    return text
