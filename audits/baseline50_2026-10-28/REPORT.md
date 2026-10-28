# Audit of the user's 50 baseline captions

This is an assistant semantic review against the five saved reference captions, sound labels and metadata. **No waveform was listened to.** Judgments are non-blind and exploratory. They do not establish physical absence of unannotated sounds.

## Caption-level results

| Outcome | Clips | Fraction |
| --- | ---: | ---: |
| All reviewed main claims reference-supported | 21/50 | 42% |
| At least one unreferenced main sound/source claim | 23/50 | 46% |
| No unreferenced claim, but at least one unclear claim | 6/50 | 12% |

These three rows partition all 50 captions. 'Supported' is limited to the reviewed main claims; it does not mean complete, fluent, correctly ordered or hallucination-free. Neither unreferenced nor unclear means verified false.

**Source-identity mismatch candidates:** 12/50 (24%), a subset of the unreferenced group. Clip numbers: 15, 19, 20, 23, 26, 30, 33, 34, 35, 36, 43, 50.

The source candidates include sheep/goat, power tool/steam, water/bicycle-gravel, train/ocean waves, car/basketball-whoosh, vacuum/power tool, sewing machine/typewriter, door/telephone, machinery/insects, horse/insects, dog/horse and train/bus. Some distinctions are much weaker acoustically than others; these are listening priorities, not a claim that all twelve are wrong.

## Why this differs from the noun proxy

The original reported micro noun fraction was 71.96%, macro 66.73%, caption flag rate 100%, and macro reference noun coverage 22.61%. Generic nouns, explanations of typical sources, hypothetical alternatives and negated mentions inflate that proxy. This audit scores semantic source/event assertions instead; the difference is a change of evaluation, not a model improvement.

Examples: clip 3's man and birds have reference support even though the noun scorer flags atmosphere/environment. Clip 27 explains common siren sources without asserting every emergency vehicle is present. Clip 33's sewing-machine attribution conflicts with typewriter references and is a source-error candidate, not just verbose style.

The main claim review does not exhaustively score acoustic attributes, timing/order, counts, negation or all possible scene inferences. The no-additional-sounds assertion in clip 33 conflicts with the referenced bell; clip 34's four-second recording claim disagrees with the 9.50-second file duration. These are separate observations rather than additions to the headline sound-presence tally. Hypothetical scene/source explanations are recorded, not blanket-approved.

## Annotation/coverage concerns

Clip 1 has sheep labels but firecracker/crowd captions. Clip 45 mixes a thunder label with divergent engine/static/hum/rain captions; its engine attribution is unclear, not a definite failure. Some supported captions omit important references: clip 13 omits the passing vehicle, clip 16 the goat, and clip 46 speech. No exhaustive semantic coverage percentage is claimed.

## Every clip

Full source captions and evidence notes are in review.csv and decisions.json; captions.txt preserves the user's quoted outputs. U = at least one unreferenced main claim; ? = unclear-only; S = all reviewed main claims supported.

| # | ID | Outcome | Unreferenced / unclear main claims |
| ---: | --- | :---: | --- |
| 1 | YwnqUgK_-fo4 | U | crowd clapping |
| 2 | YAMQei29haCw | S | — |
| 3 | YZ_smJ66Tb3c | S | — |
| 4 | YCh0LMmhBUg4 | U | breathing |
| 5 | Y3n05BjV7r7M | S | — |
| 6 | Y9U8COLzEegs | ? | unclear: a be ringing |
| 7 | Yy_OyLW9lBXU | S | — |
| 8 | YhxbmDeNSO6Q | U | wind |
| 9 | YhDMHIDJdfDA | S | — |
| 10 | YXplKBvZaHXA | S | — |
| 11 | Y4IeDBwyQ9ZQ | U | glasses as the clinking source |
| 12 | YARFFw0e_jig | ? | unclear: man speaking |
| 13 | Y2bq2lc3DLwM | S | — |
| 14 | YB4SZwi9Ce3o | S | — |
| 15 | YV8A0VRGdgwM | U | sheep as the bleating source; hissing |
| 16 | Yjid4t-FzUn0 | S | — |
| 17 | YkF1KWybdRpM | ? | unclear: idling followed by revving; unclear: brief clickissing sound |
| 18 | YtaYKM1OSTwE | U | speaking voice; vehicle opening |
| 19 | YlfO471Rn61k | U | power tool operating |
| 20 | Y7P6lcyeDKNI | U | water flowing |
| 21 | YG3YO2unWz7k | S | — |
| 22 | YglAeihz0NAM | S | — |
| 23 | Ygbi6MxPf3hA | U | train passing |
| 24 | Yjlwe9jtu5Gw | U | ticking; scratching |
| 25 | YBvw2gv1fcZw | S | — |
| 26 | YrvtA7c1I4xo | U | car engine revving; ticking/ringing |
| 27 | YpWQeV08kYR0 | S | — |
| 28 | Yd6gu2w19YQo | U | sigh |
| 29 | Y54eRRbCtPn8 | S | — |
| 30 | Y-JP1GqPEKtw | U | vacuum cleaner operating |
| 31 | Y32565FEuksc | S | — |
| 32 | YHZ9O6sc7cLA | U | mechanical machinery sounds |
| 33 | Yt1hj7se76wQ | U | sewing machine operating |
| 34 | YJfaj4P3us9M | U | door opening |
| 35 | Yg5l3Bz6lWnc | U | wind; mechanical machinery sounds |
| 36 | YqakN0JNbpcU | U | horse trotting |
| 37 | Y3wrdPAeqjVI | U | mechanisms/machinery operating |
| 38 | YxUWSHYoslPQ | ? | unclear: specifically metal clinking; unclear: door opening rather than other door activity |
| 39 | YYIqpIjjee00 | S | — |
| 40 | Yjf4iyQPJSvk | S | — |
| 41 | YelztUCeNQvQ | S | — |
| 42 | YajheseWZmmU | U | woman speaking; mechanical sounds |
| 43 | YJZloTOdIY_c | U | dog growling; dog panting |
| 44 | YPTyFYxXdut4 | S | — |
| 45 | Y_iUX8CibElk | ? | unclear: small engine idling |
| 46 | Y2KR0C5ysO8o | S | — |
| 47 | YhuMLK0oA3L8 | U | mechanical mechanisms operating |
| 48 | Yii3Geza3hAU | ? | unclear: man speaking |
| 49 | Ym8wV38lf2jg | S | — |
| 50 | YmYQrjcYNrW0 | U | train as the horn source; train passing |

## Reproduce the summaries

From the repository root, with the original cohort prepared:

```bash
python3 audits/baseline50_2026-10-28/summarize.py --manifest data/audiocaps50/manifest.json
# Optional: replay all original noun metrics without DIFFA inference:
.venv/bin/python audits/baseline50_2026-10-28/summarize.py --manifest data/audiocaps50/manifest.json --replay-lexical
```

The script validates all 50 IDs and the source-manifest hash, then counts the explicit judgments; it does not generate new semantic judgments or assess audio. The optional lexical replay re-extracts nouns from the quoted captions using the existing pinned scorer. Reference captions remain in the cohort manifest and are not redistributed here; evidence notes paraphrase them.

