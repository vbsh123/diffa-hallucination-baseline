"""Object candidates O(c): the nouns of a caption that denote physical objects.

Candidates are found with the standard NLTK tagger (nouns, alphabetic, at least three
letters), grouped by WordNet lemma, filtered by a short stoplist of scene and meta nouns
("image", "background", "side", ...), and kept only if one of the word's first three
WordNet senses is a physical object, a living thing or food (`is_physical`). No benchmark
vocabulary is used, so the same candidates serve COCO, AMBER and any other caption.

Each candidate maps to its mentions as spans of answer-canvas token indices, which Heal
uses to find the objects written inside a repair window.
"""
import re

# scene / meta nouns that are never object candidates
STOP = set("image picture photo scene view side left right top bottom front back middle "
           "center centre foreground background corner edge area part piece set lot number "
           "kind type sort way color colour shade hue atmosphere setting moment detail "
           "details thing things something feature features appearance shape size style "
           "design time day overall sense presence".split())

# physical in WordNet but not checkable as objects: deleting them never removes a
# hallucination, only information
VAGUE = set("""backdrop background foreground surface light lighting pattern element item object
material addition connection direction position point spot mix pop snapshot representation
frame scene view setting atmosphere observer viewer sentinel band border platform stand
tool upright piece part section area space structure form shape design texture tone shade
hue color colour detail feature touch sense presence moment contrast composition
arrangement display collection variety array assortment group number""".split())

_PHYS = {}


def ensure_nltk():
    """Download the NLTK resources the candidate extractor and the scorers need."""
    import nltk
    for path, name in [("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab"),
                       ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
                       ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
                       ("corpora/wordnet", "wordnet")]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass


def is_physical(lemma):
    """True if the noun's first WordNet sense is not a color/attribute/measure/time and
    one of its first three senses is a physical object, living thing or food."""
    if lemma in VAGUE:
        return False
    if lemma not in _PHYS:
        from nltk.corpus import wordnet as wn
        ss = wn.synsets("person" if lemma == "people" else lemma.replace(" ", "_"), pos=wn.NOUN)[:3]
        ok = False
        if ss:
            first = {h.name() for path in ss[0].hypernym_paths() for h in path}
            if first & {"color.n.01", "attribute.n.02", "measure.n.02", "time_period.n.01"}:
                ss = []
        for s in ss:
            hyp = {h.name() for path in s.hypernym_paths() for h in path}
            if hyp & {"object.n.01", "whole.n.02", "living_thing.n.01", "organism.n.01",
                      "food.n.01", "food.n.02"}:
                ok = True
                break
        _PHYS[lemma] = ok
    return _PHYS[lemma]


def word_rows(tokens, word, occurrence):
    """Canvas token indices spanning the occurrence-th match of `word` (plural forms too)
    in the concatenation of the per-token strings."""
    starts, pos = [], 0
    text = "".join(tokens).lower()
    for t in tokens:
        starts.append(pos)
        pos += len(t)
    hits = [(m.start(), m.end()) for m in re.finditer(rf"\b{re.escape(word)}(?:s|es)?\b", text)]
    if occurrence >= len(hits):
        return []
    c0, c1 = hits[occurrence]
    return [i for i, s in enumerate(starts) if s < c1 and s + len(tokens[i]) > c0]


def noun_groups(tok_strings, caption):
    """{lemma: [mention, ...]} for every noun of the caption that passes the stoplist;
    a mention is the list of canvas token indices of one occurrence. Callers filter with
    `is_physical` to get the object candidates."""
    import nltk
    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()
    seen, groups = {}, {}
    for word, pos in nltk.pos_tag(nltk.word_tokenize(caption)):
        if not pos.startswith("NN") or not word.isalpha():
            continue
        w = word.lower()
        n = lem.lemmatize(w)
        occ = seen.get(w, 0)
        seen[w] = occ + 1
        if n in STOP or len(n) < 3:
            continue
        rows = word_rows(tok_strings, w, occ)
        if rows:
            groups.setdefault(n, []).append(rows)
    return groups


def candidates(tok_strings, caption):
    """Object candidates of a caption: {lemma: mentions}, physical nouns only."""
    return {n: mens for n, mens in noun_groups(tok_strings, caption).items() if is_physical(n)}


def surface_forms(caption, lemma):
    """The words of the caption whose lemma is `lemma` ("cars", "car"), longest first."""
    import nltk
    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()
    return sorted({w for w in nltk.word_tokenize(caption)
                   if w.isalpha() and (lem.lemmatize(w.lower()) == lemma or w.lower() == lemma)},
                  key=len, reverse=True)
