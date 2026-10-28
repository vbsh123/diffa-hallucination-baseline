# Source attribution

The six Python method files in this directory are copied **byte for byte** from
the user-supplied `Downloads/DCH-main.zip`, code for *Detect, Cut, and Heal:
Correcting Object Hallucinations in Masked-Diffusion Vision-Language Models*.

The archive identifies source commit `db099487f853ba2947fad6ffd032ba55690b9774`.
`SOURCE.json` records the archive and per-file SHA-256 hashes. Its README states
that DCH is released under the same terms as LLaDA-V; the supplied
`train/LICENSE` is preserved here as `LICENSE` (Apache 2.0).

The surrounding `dch/` package adapts the model interface and question to DIFFA
audio. It does not instantiate the image-model class or call the image pipeline.
Candidate extraction, deletion, seam selection, ban rules, invalid-word checks,
repair and confidence ranking use these supplied helpers. Runtime checks and
output records are added by the surrounding adapter, not by modifying these files.
