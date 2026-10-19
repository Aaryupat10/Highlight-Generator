import unittest

from nlp_analysis import analyze_event, normalize_commentary


class NormalizeCommentaryTests(unittest.TestCase):
    def test_collapses_whitespace_but_preserves_original_punctuation(self):
        text = normalize_commentary("  ＧＯＡＬ!!!\u00a0Haaland’s   shot …  ")
        self.assertEqual(text, "ＧＯＡＬ!!! Haaland’s shot …")


class EventCueTests(unittest.TestCase):
    def test_affirmative_goal_is_asserted_and_emphasis_is_kept(self):
        result = analyze_event("GOAL!!! Haaland scores!", "goal")
        self.assertEqual(result["event_context"], "asserted_or_live")
        self.assertTrue(result["has_excitement_cue"])

    def test_negated_goal_is_not_asserted(self):
        result = analyze_event("Haaland didn't score the goal.", "goal")
        self.assertEqual(result["event_context"], "negated")

    def test_disallowed_goal_is_not_asserted(self):
        result = analyze_event("That goal was ruled out.", "goal")
        self.assertEqual(result["event_context"], "negated")

    def test_hypothetical_goal_is_not_asserted(self):
        result = analyze_event("He could have scored a goal.", "goal")
        self.assertEqual(result["event_context"], "hypothetical")

    def test_past_season_goal_is_retrospective(self):
        result = analyze_event("He scored last season.", "goal")
        self.assertEqual(result["event_context"], "retrospective")

    def test_positive_event_in_later_sentence_is_preserved(self):
        result = analyze_event("No goal was given. Now, goal for Arsenal!", "goal")
        self.assertEqual(result["event_context"], "asserted_or_live")

    def test_not_only_is_not_treated_as_negation(self):
        result = analyze_event("Not only did Haaland score, he assisted too.", "goal")
        self.assertEqual(result["event_context"], "asserted_or_live")
        self.assertFalse(result["has_negation"])

    def test_positive_clause_after_but_is_not_inherited_as_negated(self):
        result = analyze_event("He didn't score, but now there is a goal for Arsenal.", "goal")
        self.assertEqual(result["event_context"], "asserted_or_live")

    def test_no_event_phrase_is_reported(self):
        result = analyze_event("Patient possession in midfield.", "goal")
        self.assertEqual(result["event_context"], "no_event_phrase")


if __name__ == "__main__":
    unittest.main()
