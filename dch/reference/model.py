"""LLaDA-V access for DCH.

Everything operates on a fixed canvas `prompt + answer` in embedding space, exactly like
`LLaDAModelLM.generate_with_embeds` in train/llava: the prompt (image tokens, question,
chat template) is embedded once, the answer positions hold token embeddings or the [MASK]
embedding, and one forward pass gives the model's distribution at every answer position.
Because attention is bidirectional, the distribution at a masked position conditions on
the whole caption, before and after it. That is what Heal relies on.

Two operations are used by DCH:
  * `logits` on a canvas with two masked answer positions -> the yes/no detector;
  * `infill`: re-decode a chosen set of answer positions of a finished caption, one token
    per forward pass, most confident position first (the sampler's own rule).

`caption` generates a caption with the stock sampler (128 positions, 128 steps,
temperature 0, <|eot_id|> stopping), the protocol used in the paper.
"""
import copy
import time

import torch

MASK_ID = 126336                 # LLaDA's [MASK] token
EOT = "<|eot_id|>"               # end of the answer (llava_llada template separator)
# never written by the repair: <|start_header_id|>, <|end_header_id|>, <|eot_id|> variants
SPECIAL_SUPPRESS = [126081, 126080, 126346, 126347]


class Model:
    def __init__(self, pretrained="GSAI-ML/LLaDA-V", device="cuda:0", attn="sdpa"):
        from llava.model.builder import load_pretrained_model
        self.tok, self.model, self.proc, _ = load_pretrained_model(
            pretrained, None, "llava_llada", attn_implementation=attn, device_map=device)
        self.model.eval()
        self.device = device
        self.nfe = 0                   # forward passes through the language model
        self.embed = self.model.get_model().embed_tokens
        with torch.no_grad():
            self.mask_embed = self.embed(torch.tensor([MASK_ID], device=device))
        self.eot = self.tok.convert_tokens_to_ids(EOT)

    # -- prompt --------------------------------------------------------------------
    def prompt_text(self, question, with_image=True):
        """The llava_llada chat prompt with the image placeholder and the /no_think suffix,
        as in the released evaluation wrapper."""
        from llava.constants import DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        text = (DEFAULT_IMAGE_TOKEN + "\n" if with_image else "") + question + " /no_think"
        conv = copy.deepcopy(conv_templates["llava_llada"])
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    @torch.no_grad()
    def prompt_embeds(self, image, question):
        """(1, L, d) embeddings of the prompt with the image spliced in (the model's own
        image view, i.e. the tiled "anyres" view of the released checkpoint)."""
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images, tokenizer_image_token
        img = image.convert("RGB")
        ids = tokenizer_image_token(self.prompt_text(question, True), self.tok,
                                    IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        t = process_images([img], self.proc, self.model.config).to(torch.float16).to(self.device)
        (_, _, _, _, emb, _) = self.model.prepare_inputs_labels_for_multimodal(
            ids, None, None, None, None, t, ["image"], image_sizes=[img.size])
        return emb

    # -- one forward pass ----------------------------------------------------------
    @torch.no_grad()
    def logits(self, embeds, start=0):
        """Logits at positions >= start (the vocabulary is 126k: logits for the whole
        anyres prompt would not fit, and only the answer positions are needed)."""
        self.nfe += embeds.shape[0]
        h = self.model.model(inputs_embeds=embeds)[0][:, start:]
        return self.model.lm_head(h).float()

    # -- caption generation (the paper's protocol) ---------------------------------
    @torch.no_grad()
    def caption(self, image, question, gen_length=128, steps=128):
        """Generate a caption with the stock low-confidence sampler: `gen_length`
        masked positions filled in `steps` steps (one position per step by default),
        temperature 0, stopping at <|eot_id|>. Returns (text, info)."""
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images, tokenizer_image_token
        img = image.convert("RGB")
        ids = tokenizer_image_token(self.prompt_text(question, True), self.tok,
                                    IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        t = [x.to(dtype=torch.float16, device=self.device)
             for x in process_images([img], self.proc, self.model.config)]
        t0 = time.time()
        out = self.model.generate(ids, images=t, image_sizes=[img.size], tokenizer=self.tok,
                                  gen_length=gen_length, block_length=gen_length, steps=steps,
                                  temperature=0.0, stopping_criteria=[EOT])
        text = self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        return text, {"gen_time_s": round(time.time() - t0, 3), "gen_length": gen_length, "steps": steps}

    # -- canvas helpers ------------------------------------------------------------
    def canvas(self, text, gen_length):
        """Token ids of `text` on an answer canvas of `gen_length` positions, padded with
        <|eot_id|>. Returns (ids, n_text). A canvas is never shorter than the text."""
        ids = self.tok.encode(text, add_special_tokens=False)
        n = len(ids)
        gen_length = max(gen_length, n + 1)
        return ids + [self.eot] * (gen_length - n), n

    def decode_answer(self, ids):
        """Text of a canvas up to the first <|eot_id|>."""
        cut = ids.index(self.eot) if self.eot in ids else len(ids)
        return self.tok.decode(ids[:cut]).strip()


@torch.no_grad()
def infill(m, pemb, ans_ids, positions, ban=None):
    """Re-decode `positions` of a finished canvas.

    All of `positions` are shown as [MASK]; one forward pass gives the model's prediction
    at each of them; the most confident position is committed; repeat until none is left
    (the low-confidence order of the sampler, one token per forward pass). Every other
    position, before and after, stays visible, so the infill reads both sides of the seam.

    ban: {position: set(token ids)} that may not be chosen at that position (DCH bans the
    tokens of the flagged words, so a deleted object cannot be written back).
    Returns the new ids.
    """
    ids = list(ans_ids)
    todo = set(positions)
    ban = ban or {}
    L = pemb.shape[1]
    while todo:
        t = torch.tensor(ids, device=m.device)
        ae = m.embed(t)[None].to(pemb.dtype)
        mk = sorted(todo)
        ae[0, mk] = m.mask_embed.to(ae.dtype)
        lp = torch.log_softmax(m.logits(torch.cat([pemb, ae], 1), L)[0, mk], -1)
        score = lp.clone()
        score[:, SPECIAL_SUPPRESS] = -1e9
        score[:, MASK_ID] = -1e9
        for r, j in enumerate(mk):
            for b in ban.get(j, ()):
                score[r, b] = -1e9
        tok = score.argmax(-1)
        conf = lp.gather(-1, tok[:, None])[:, 0]
        r = int(conf.argmax())
        ids[mk[r]] = int(tok[r])
        todo.discard(mk[r])
    return ids
