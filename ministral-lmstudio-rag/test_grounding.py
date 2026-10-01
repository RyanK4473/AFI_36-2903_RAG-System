"""Regression checks for missing evidence and cross-reference hallucinations."""
import json
import unittest
from unittest.mock import patch

import chatbot


def passage(text):
    return {"source": "test.pdf", "page": 1, "text": text, "score": 0.36}


def decision(quote, source_id=1):
    return json.dumps({"answerable": True, "evidence": [{"id": source_id, "quote": quote}]})


class GroundingTests(unittest.TestCase):
    def test_cropped_referral_cannot_authorize_saluting_rules(self):
        hits = [passage("Reference AFI 34-1201, Protocol regarding customs and courtesies, "
                        "when a salute is or is not required. Uniforms may be worn during travel.")]
        quote = "Protocol regarding customs and courtesies, when a salute is or is not required"
        with patch.object(chatbot, "generate", return_value=(decision(quote), "stop")) as model:
            answer, _, sources = chatbot.grounded_answer(None, "test", "What are the customs of saluting?", hits)
        self.assertIn(chatbot.NO_EVIDENCE, answer)
        self.assertIn("Reference AFI 34-1201", answer)
        self.assertEqual(model.call_count, 1)
        self.assertTrue(sources[0]["text"].startswith("Reference AFI"))

    def test_invented_quote_cannot_authorize_an_answer(self):
        hits = [passage("Uniforms may be worn during travel.")]
        with patch.object(chatbot, "generate", return_value=(decision("Salutes are not required during travel."), "stop")) as model:
            answer, _, _ = chatbot.grounded_answer(None, "test", "Must I salute during travel?", hits)
        self.assertEqual(answer, chatbot.NO_EVIDENCE)
        self.assertEqual(model.call_count, 1)

    def test_pdf_word_spacing_can_be_repaired_without_inventing_text(self):
        hits = [passage("The five elemen ts are neatness and cleanliness.")]
        quoted = chatbot.checked_quote({"id": 1, "quote": "five elements are **neatness** and cleanliness"}, hits)
        self.assertEqual(quoted["text"], hits[0]["text"])

    def test_wrong_citation_is_rejected(self):
        with self.assertRaises(ValueError):
            chatbot.checked_quote({"id": 2, "quote": "Not required."}, [passage("Not required.")])

    def test_missing_or_malformed_evidence_fails_closed(self):
        for reply, reason in [("not JSON", "stop"), (decision("Not required."), "length")]:
            with self.subTest(reply=reply), patch.object(chatbot, "generate", return_value=(reply, reason)):
                answer, _, _ = chatbot.grounded_answer(None, "test", "Must I salute?", [passage("Not required.")])
            self.assertEqual(answer, chatbot.NO_EVIDENCE)

    def test_supported_specific_rule_is_answered(self):
        source = "Saluting is not required when wearing the mess dress and semi-formal dress uniforms."
        expected = "Saluting is not required in those uniforms [1]."
        replies = [(decision(source), "stop"), (expected, "stop"),
                   ('{"relevant":true,"supported":true}', "stop")]
        with patch.object(chatbot, "generate", side_effect=replies):
            answer, _, _ = chatbot.grounded_answer(None, "test", "Is saluting required in mess dress?", [passage(source)])
        self.assertEqual(answer, expected)

    def test_unsupported_summary_falls_back_to_source_text(self):
        source = "Saluting is not required when wearing the mess dress uniform."
        replies = [(decision(source), "stop"), ("Saluting is required in mess dress [1].", "stop"),
                   ('{"relevant":true,"supported":false}', "stop")]
        with patch.object(chatbot, "generate", side_effect=replies):
            answer, _, _ = chatbot.grounded_answer(None, "test", "Is saluting required in mess dress?", [passage(source)])
        self.assertIn(source, answer)
        self.assertNotIn("Saluting is required", answer)

    def test_unrelated_fact_is_not_returned_as_an_answer(self):
        source = "Foreign decoration wear criteria depend on the type of device."
        replies = [(decision(source), "stop"), (source + " [1]", "stop"),
                   ('{"relevant":false,"supported":true}', "stop")]
        with patch.object(chatbot, "generate", side_effect=replies):
            answer, _, _ = chatbot.grounded_answer(None, "test", "What are saluting customs?", [passage(source)])
        self.assertEqual(answer, chatbot.NO_EVIDENCE)

    def test_overlapping_chunks_restore_a_complete_rule(self):
        overlap = "responsibilities and standards for dress and personal appearance"
        left = passage("This instruction provides " + overlap + " of personnel and refers to DoDI 1300")
        right = passage(overlap + " of personnel and refers to DoDI 1300.17 for religious accommodation.")
        merged = chatbot.merge_overlapping_passages([left, right])
        self.assertEqual(len(merged), 1)
        self.assertIn("DoDI 1300.17 for religious accommodation.", merged[0]["text"])


if __name__ == "__main__":
    unittest.main()
