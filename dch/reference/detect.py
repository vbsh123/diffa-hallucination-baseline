"""Detect: ask the model whether each object is in the image.

For a candidate o the model is prompted, with the image, "Is there a {o} in the image?
Answer the question using a single word or phrase." on an answer canvas of two masked
positions. One forward pass gives the distribution at the first answer position, and the
detector score is the log-odds

    s(o) = log sum_{y in Yes} p(y) - log sum_{n in No} p(n),

over the tokens "Yes"/"yes" and "No"/"no" with and without a leading space. An object is
flagged when s(o) < tau; the default tau = 0 is the point at which "no" becomes more
probable than "yes". Lexicalized plurals ("glasses", "scissors") get "Are there any {o}".

The question isolates the object from the sentence that proposed it, and it costs one
forward pass per candidate. The detector does not know who wrote the caption.
"""
import torch

QUESTION = "Is there a {o} in the image?"
QUESTION_PLURAL = "Are there any {o} in the image?"
POST = "\nAnswer the question using a single word or phrase."


def question_for(obj):
    """The yes/no question for an object lemma; plural-only nouns get 'Are there any'."""
    from nltk.stem import WordNetLemmatizer
    if " " not in obj and obj.endswith("s") and WordNetLemmatizer().lemmatize(obj) != obj:
        return QUESTION_PLURAL
    return QUESTION


@torch.no_grad()
def yes_no(m, image, obj):
    """Detector score s(obj): log-odds of 'yes' over 'no' for the object in the image."""
    pe = m.prompt_embeds(image, question_for(obj).format(o=obj) + POST)
    L = pe.shape[1]
    x = torch.cat([pe, m.mask_embed.repeat(1, 2, 1).to(pe.dtype)], 1)
    lp = torch.log_softmax(m.logits(x, L)[0, 0], -1)
    ids = lambda ws: [m.tok.encode(w, add_special_tokens=False)[0] for w in ws]
    y = torch.logsumexp(lp[ids(["Yes", "yes", " Yes", " yes"])], 0)
    n = torch.logsumexp(lp[ids(["No", "no", " No", " no"])], 0)
    return float(y - n)


def detect(m, image, cands):
    """{object: s(object)} for every candidate object of a caption."""
    return {o: yes_no(m, image, o) for o in cands}
