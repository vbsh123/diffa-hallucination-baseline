"""CPU checks using real NLP and synthetic audio; no fake model results."""
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import spacy

from baseline.common import claim_config, digest, load_manifest, read_json, write_json
from baseline.prepare import audio_array, prepare
from baseline.run import output_record
from baseline.score import aggregate, compare, extract, score_run


def wav_bytes(seconds=1, rate=8000):
    t = np.arange(round(seconds * rate)) / rate
    audio = np.stack([.1 * np.sin(2*np.pi*440*t), .1 * np.sin(2*np.pi*220*t)], axis=1)
    stream = io.BytesIO()
    sf.write(stream, audio, rate, format="WAV", subtype="FLOAT")
    return stream.getvalue()


class FakeDataset(list):
    def cast_column(self, name, feature):
        return self


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = spacy.load("en_core_web_sm")

    def test_counts_empty_predictions_and_references(self):
        rows = [compare(["dog", "helicopter"], ["dog", "person"]),
                compare([], ["engine"]), compare(["bird"], [])]
        report = aggregate(rows, expected=4)
        self.assertEqual(report["completed_clips"], 3)
        self.assertFalse(report["complete"])
        self.assertEqual(report["empty_noun_captions"], 1)
        self.assertEqual(report["empty_reference_noun_sets"], 1)
        self.assertAlmostEqual(report["unsupported_noun_fraction_micro"], 2/3)
        self.assertAlmostEqual(report["unsupported_noun_fraction_macro"], .75)
        self.assertAlmostEqual(report["caption_flag_rate"], 2/3)
        self.assertAlmostEqual(report["reference_noun_coverage_macro"], .25)
        self.assertIsNone(aggregate([], expected=2)["unsupported_noun_fraction_macro"])
        self.assertIsNone(aggregate([compare([], [])], 1)["reference_noun_coverage_micro"])

    def test_lemma_spans_and_negated_claims_are_auditable(self):
        text = "No dogs or helicopters are heard in the audio."
        result = extract(text, self.nlp)
        self.assertEqual(result["terms"], ["dog", "helicopter"])
        self.assertTrue(result["has_negation"])
        for mention in result["mentions"]:
            self.assertEqual(text[mention["start"]:mention["end"]], mention["surface"])
        self.assertEqual(compare(["dog", "dog"], ["dog"])["unsupported_fraction"], 0)

    def test_complete_stereo_audio_is_resampled_without_crop(self):
        samples, duration = audio_array({"bytes": wav_bytes(2.25), "path": None})
        self.assertEqual(samples.shape, (36000,))
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(duration, 2.25)
        self.assertTrue(np.isfinite(samples).all())
        with self.assertRaisesRegex(ValueError, "no cropping"):
            audio_array({"bytes": wav_bytes(30.1), "path": None})

    def test_nonfinite_audio_rejected(self):
        stream = io.BytesIO()
        sf.write(stream, np.array([0., np.nan]), 16000, format="WAV", subtype="FLOAT")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            audio_array({"bytes": stream.getvalue()})

    def make_cohort(self, root):
        dataset = FakeDataset([
            dict(youtube_id="dog_clip", audio={"bytes": wav_bytes()},
                 caption=["A dog barks while a person speaks."], label=["dog", "speech"],
                 hallucination_label=["helicopter"]),
            dict(youtube_id="engine_clip", audio={"bytes": wav_bytes()},
                 caption=["An engine idles."], label=["engine"], hallucination_label=["bird"])
        ])
        with patch("datasets.load_dataset", return_value=dataset):
            return prepare(root / "data", count=2, seed=7)

    def test_cohort_schema_hashes_and_selection_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.make_cohort(root)
            manifest = load_manifest(path)
            self.assertEqual(len(manifest["samples"]), 2)
            self.assertNotIn("hallucination_label", str(manifest))
            self.assertEqual(prepare(root / "data", 2, 7), path)
            with self.assertRaisesRegex(ValueError, "settings changed"):
                prepare(root / "data", 1, 7)
            audio = path.parent / manifest["samples"][0]["audio_path"]
            audio.write_bytes(audio.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "Audio changed"):
                load_manifest(path)

    def test_config_resume_rejects_changed_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            claim_config(path, {"factor": 1})
            claim_config(path, {"factor": 1})
            with self.assertRaisesRegex(ValueError, "Configuration changed"):
                claim_config(path, {"factor": 2})

    def test_decode_stops_at_eos_and_records_unresolved_masks(self):
        class Tokenizer:
            def decode(self, ids, **kwargs):
                return ":".join(map(str, ids))
        record = output_record([2, 99, 3, 0, 99], 0, 99, Tokenizer())
        self.assertEqual(record["visible_ids"], [2, 99, 3])
        self.assertEqual(record["unresolved_visible_masks"], 1)
        self.assertFalse(record["capacity_exhausted"])
        self.assertTrue(output_record([2, 3], 0, 99, Tokenizer())["capacity_exhausted"])

    def test_cpu_prepare_score_and_review_integration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self.make_cohort(root)
            out = root / "run"
            config = dict(manifest=str(manifest_path), manifest_sha256=digest(manifest_path))
            write_json(out / "config.json", config)
            captions = {"dog_clip": "A dog and a helicopter are heard.", "engine_clip": "Quietly."}
            for clip_id, text in captions.items():
                write_json(out / "predictions" / f"{clip_id}.json",
                           dict(id=clip_id, run_config_sha256=digest(out / "config.json"), text=text,
                                eos_found=True, capacity_exhausted=False, unresolved_visible_masks=0,
                                actual_nfe=3, elapsed_seconds=.1))
            report = score_run(out, nlp=self.nlp)
            self.assertTrue(report["complete"])
            self.assertEqual(report["empty_noun_captions"], 1)
            self.assertEqual(report["unsupported_noun_fraction_micro"], .5)
            self.assertEqual(report["caption_flag_rate"], .5)
            rows = read_json(out / "scores.json")["clips"]
            dog = next(r for r in rows if r["id"] == "dog_clip")
            self.assertEqual(dog["unsupported_terms"], ["helicopter"])
            with (out / "review.csv").open() as stream:
                review = list(csv.DictReader(stream))
            self.assertIn("scene_or_source_hallucination", review[0])
            self.assertIn("audible_event_hallucination", review[0])
            review[0]["review_notes"] = "A human listening judgment"
            with (out / "review.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(review[0]))
                writer.writeheader()
                writer.writerows(review)
            score_run(out, nlp=self.nlp)
            with (out / "review.csv").open() as stream:
                self.assertEqual(list(csv.DictReader(stream))[0]["review_notes"],
                                 "A human listening judgment")
            (out / "predictions" / "engine_clip.json").unlink()
            self.assertFalse(score_run(out, nlp=self.nlp)["complete"])
            write_json(out / "predictions" / "extra.json", {})
            with self.assertRaisesRegex(ValueError, "outside this cohort"):
                score_run(out, nlp=self.nlp)


if __name__ == "__main__":
    unittest.main()
