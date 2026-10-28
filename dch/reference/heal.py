"""Heal: repair the seams that deletion leaves, with the model.

Each cut is located by a word-level alignment between the original and the cut text;
w = 2 tokens on each side of every seam (sentence-final punctuation excluded) are re-masked
and infilled by the model with the image, the prompt and the text on both sides visible,
one token per forward pass in the sampler's own low-confidence order. Three safeguards:

  * the tokens of the flagged words are forbidden inside the windows, so a deleted object
    cannot be written back;
  * a fill that spells a non-word (not a WordNet lemma, not a function word) is rejected
    and the window refilled with that token also forbidden, up to twice;
  * any object the repair writes into a window is re-verified with the detector; a
    rejected one is forbidden and the windows are refilled once more.
"""
import difflib

import numpy as np

from .model import infill

FUNCTION_WORDS = set("""a an the and or but nor so yet of in on at to for with by from into onto upon over under
above below between among through during before after about against without within along across behind beyond near
off out up down as than that this these those there here which who whom whose what where when while is are was were
be been being has have had do does did its it they them their his her he she we our you your i my me one two three
four five six seven eight nine ten several some many few each every other another such no not also very just too
can could would should will may might must""".split())


def ban_ids(tok, word):
    """First token of the word, its plural forms and its capitalized form, with and
    without a leading space: the ids forbidden inside a repair window."""
    out = set()
    for v in {word, word + "s", word + "es", word.capitalize()}:
        for pre in ("", " "):
            enc = tok.encode(pre + v, add_special_tokens=False)
            if enc:
                out.add(enc[0])
    return out


def invalid_words(m, ids, positions):
    """Alphabetic words touching `positions` that are neither WordNet lemmas nor function
    words, as lists of token indices (the repair spelled a non-word)."""
    from nltk.corpus import wordnet as wn
    pos = set(positions)
    out = []
    strs = [m.tok.decode([i]) for i in ids]
    j = 0
    while j < len(strs):
        if not strs[j].strip() or not strs[j].strip()[0].isalpha():
            j += 1
            continue
        k = j + 1
        while k < len(strs) and strs[k][:1].isalpha():     # continuation pieces of the word
            k += 1
        span = list(range(j, k))
        if pos & set(span):
            w = "".join(strs[t] for t in span).strip().lower()
            if (w.isalpha() and len(w) >= 3 and w not in FUNCTION_WORDS and not wn.synsets(w)
                    and not wn.synsets(w.rstrip("s")) and not wn.morphy(w)):
                out.append(span)
        j = k
    return out


def seam_windows(m, orig, cut_text, gen_length, w=2):
    """Canvas ids of the cut text and the token positions to re-mask: w tokens on each
    side of every cut, found by aligning the words of the original and the cut text."""
    ow, nw = orig.split(" "), cut_text.split(" ")
    offs = np.cumsum([0] + [len(x) + 1 for x in nw])
    cuts = [int(offs[j1]) for tag, i1, i2, j1, j2 in
            difflib.SequenceMatcher(a=ow, b=nw, autojunk=False).get_opcodes() if tag in ("delete", "replace")]
    ids, n_text = m.canvas(cut_text, gen_length)
    starts = np.cumsum([0] + [len(m.tok.decode([i])) for i in ids])
    win = set()
    for c in cuts:
        k = int(np.searchsorted(starts, c, side="right")) - 1
        for j in range(max(0, k - w), min(n_text, k + w)):
            win.add(j)
    win = sorted(j for j in win if m.tok.decode([ids[j]]).strip() not in (".", "!", "?"))
    return ids, win, len(cuts)


def heal(m, pemb, orig, cut_text, flagged, protected, is_bad, gen_length=128, w=2):
    """Repair the seams of `cut_text` (the original `orig` with the `flagged` objects
    deleted). `is_bad(object)` is the detector's verdict, used to re-verify objects the
    repair writes; `protected` objects are never re-verified. Returns (text, info)."""
    from .candidates import candidates
    ids, win, n_cuts = seam_windows(m, orig, cut_text, gen_length, w)
    info = {"cuts": n_cuts, "repaired": len(win), "nonword_retries": 0, "reverified": 0}
    if not win:
        return cut_text, info
    bans = set()
    for n in flagged:
        bans |= ban_ids(m.tok, n)
    ban_pos = {j: set(bans) for j in win}
    new = infill(m, pemb, ids, win, ban=ban_pos)
    for _ in range(2):                              # reject non-words spelled around the ban
        bad = invalid_words(m, new, win)
        if not bad:
            break
        for span in bad:
            for t in span:
                if t in ban_pos:
                    ban_pos[t].add(new[t])
        new = infill(m, pemb, new, win, ban=ban_pos)
        info["nonword_retries"] += 1
    # re-verify the objects the repair wrote inside the windows
    text = m.decode_answer(new)
    n_new = len(m.tok.encode(text, add_special_tokens=False))
    strs = [m.tok.decode([i]) for i in new]
    written = {n for n, mens in candidates(strs[:n_new], text).items()
               if any(men[0] in win for men in mens)}
    info["heal_objects"] = sorted(written)
    rejected = sorted(n for n in written if n not in protected and is_bad(n))
    if rejected:
        for n in rejected:
            bans |= ban_ids(m.tok, n)
        for j in win:
            ban_pos[j] |= bans
        new = infill(m, pemb, new, win, ban=ban_pos)
        info["reverified"] = len(rejected)
        info["heal_rejected"] = rejected
    return m.decode_answer(new), info
