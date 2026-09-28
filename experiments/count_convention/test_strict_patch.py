"""Tests for the strict, thrombosis-scoped COUNT rule.

  python -m experiments.count_convention.test_strict_patch
"""
import json
import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.count_convention import strict_patch as S

THROMBOSIS = {"Patient": {"table_readable_name": "p", "columns": [{"name": "ID"}, {"name": "SEX"}]},
              "Laboratory": {"table_readable_name": "l", "columns": [{"name": "ID"}, {"name": "WBC"}]},
              "Examination": {"table_readable_name": "e", "columns": [{"name": "ID"}]}}
OTHER = {"schools": {"table_readable_name": "s", "columns": [{"name": "CDSCode"}]},
         "frpm": {"table_readable_name": "f", "columns": [{"name": "CDSCode"}]}}


class ScopeTest(unittest.TestCase):
    def test_detects_thrombosis(self):
        self.assertTrue(S.is_thrombosis(THROMBOSIS))

    def test_rejects_other_databases(self):
        self.assertFalse(S.is_thrombosis(OTHER))
        self.assertFalse(S.is_thrombosis({}))

    def test_detects_when_extra_tables_present(self):
        schema = dict(THROMBOSIS, Extra={"columns": []})
        self.assertTrue(S.is_thrombosis(schema))


class RewriteTest(unittest.TestCase):
    def test_strict_section_added_and_idempotent(self):
        out = S.rewrite("BASE")
        self.assertIn(S.MARKER, out)
        self.assertEqual(out, S.rewrite(out))

    def test_rule_e_is_rescoped_to_projection_only(self):
        out = S.rewrite("x\n" + S.RULE_E_OLD + "\ny")
        self.assertNotIn(S.RULE_E_OLD, out)
        rule_e = re.sub(r"\s+", " ", S.RULE_E_NEW)
        self.assertIn("`SELECT DISTINCT` in the PROJECTION only", rule_e)
        self.assertIn("does not apply to COUNT()", rule_e)
        self.assertNotIn("COUNT(DISTINCT element)", rule_e)   # no COUNT example left in RULE E

    def test_rule_o_claims_only_count(self):
        text = re.sub(r"\s+", " ", S.STRICT_SECTION)
        self.assertIn("governs the argument of COUNT() and nothing else", text)
        self.assertIn("`SELECT DISTINCT` in the projection is RULE E's business", text)

    def test_states_must_not_rather_than_prefer(self):
        text = re.sub(r"\s+", " ", S.STRICT_SECTION)
        self.assertIn("you MUST write plain", text)
        self.assertNotIn("PREFER plain", text)

    def test_unknown_rule_e_wording_is_not_mangled(self):
        out = S.rewrite("no rule E here")
        self.assertIn("no rule E here", out)


class PatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not S.install():
            raise unittest.SkipTest("src not importable")

    def _system(self, schema):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        return builder.build("how many patients?", schema, list(builder.strategy_prompts)[0],
                             evidence="")["system"]

    def test_applies_to_thrombosis(self):
        system = self._system(THROMBOSIS)
        self.assertIn(S.MARKER, system)
        self.assertNotIn(S.RULE_E_OLD, system)

    def test_does_not_apply_to_other_databases(self):
        system = self._system(OTHER)
        self.assertNotIn(S.MARKER, system)
        self.assertIn(S.RULE_E_OLD, system)          # RULE E intact elsewhere

    def test_rules_a_to_k_survive_for_thrombosis(self):
        system = self._system(THROMBOSIS)
        for kept in ("RULE A", "RULE C", "RULE K"):
            self.assertIn(kept, system)

    def test_real_thrombosis_schema_is_detected(self):
        path = ROOT / "data/bird_data/schemas/thrombosis_prediction_schema.json"
        if not path.exists():
            self.skipTest("schema file missing")
            return
        schema = json.loads(path.read_text(encoding="utf-8"))["tables"]
        self.assertTrue(S.is_thrombosis(schema))
        self.assertIn(S.MARKER, self._system(schema))




class JudgeTest(unittest.TestCase):
    """The judge must stop preferring COUNT(DISTINCT), or it reverses the generation rule."""

    def test_exact_judge_wording_is_matched(self):
        from src.prompt import JUDGE_PROMPT
        self.assertTrue(S.JUDGE_RULE_E_OLD in JUDGE_PROMPT["system"]
                        or "PROJECTION ONLY" in JUDGE_PROMPT["system"])

    def test_patch_applies_and_is_idempotent(self):
        from src.prompt import JUDGE_PROMPT
        self.assertTrue(S.patch_judge_prompt())
        first = JUDGE_PROMPT["system"]
        self.assertTrue(S.patch_judge_prompt())
        self.assertEqual(first, JUDGE_PROMPT["system"])

    def test_old_preference_is_gone_and_new_one_is_stated(self):
        from src.prompt import JUDGE_PROMPT
        S.patch_judge_prompt()
        system = re.sub(r"\s+", " ", JUDGE_PROMPT["system"])
        self.assertNotIn("PREFER candidates that use DISTINCT when JOIN multiplies rows", system)
        self.assertIn("you MUST select a plain `COUNT(<column>)` candidate whenever one is offered",
                      system)
        # the judge overrode a mere preference with this argument, so it must be named and refused
        self.assertIn("is exactly the one you must NOT make", system)
        self.assertIn("RULE C chooses WHICH column goes inside COUNT()", system)

    def test_judge_rules_other_than_e_survive(self):
        from src.prompt import JUDGE_PROMPT
        S.patch_judge_prompt()
        for kept in ("RULE A", "RULE C", "RULE K"):
            self.assertIn(kept, JUDGE_PROMPT["system"])

    def test_unknown_wording_reports_failure_rather_than_silently_passing(self):
        import src.prompt as sp
        original = sp.JUDGE_PROMPT["system"]
        try:
            sp.JUDGE_PROMPT["system"] = "a judge prompt with no rule E"
            self.assertFalse(S.patch_judge_prompt())
        finally:
            sp.JUDGE_PROMPT["system"] = original


class ChecklistTest(unittest.TestCase):
    """The numbered checklist routed DISTINCT through schema reasoning - the opposite of RULE O."""

    @classmethod
    def setUpClass(cls):
        if not S.install():
            raise unittest.SkipTest("src not importable")

    def test_checklist_clause_is_rescoped(self):
        out = S.rewrite("6. Follow Rule A, " + S.CHECKLIST_OLD + ", Rule F")
        self.assertNotIn(S.CHECKLIST_OLD, out)
        self.assertIn("`SELECT DISTINCT` in the projection only", out)
        self.assertIn("Rule O (the argument of COUNT()", out)

    def test_checklist_is_rescoped_in_the_real_prompt(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        for strategy in builder.strategy_prompts:
            system = builder.build("how many patients?", THROMBOSIS, strategy,
                                   focused_schema=THROMBOSIS, evidence="")["system"]
            self.assertNotIn("DISTINCT via schema reasoning", system, strategy.value)

    def test_other_databases_keep_the_original_checklist(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        system = builder.build("how many schools?", OTHER, list(builder.strategy_prompts)[0],
                               evidence="")["system"]
        self.assertIn("DISTINCT via schema reasoning", system)


class FixerTest(unittest.TestCase):
    """The fixer's 'KEEP DISTINCT otherwise' branch fires on exactly the thrombosis shape."""

    @classmethod
    def setUpClass(cls):
        if not S.install():
            raise unittest.SkipTest("src not importable")

    def test_rewrite_points_the_rule_at_the_evidence(self):
        from src.prompt import FIXER_PROMPT
        out = S.fixer_rewrite(FIXER_PROMPT["system"])
        self.assertIsNotNone(out)
        text = re.sub(r"\s+", " ", out)
        self.assertNotIn("KEEP DISTINCT otherwise", text)
        self.assertIn("decided by the evidence, not by fan-out", text)
        self.assertIn("never ADD DISTINCT to a COUNT() that does not have it", text)

    def test_rewrite_is_idempotent_and_reports_unknown_wording(self):
        from src.prompt import FIXER_PROMPT
        once = S.fixer_rewrite(FIXER_PROMPT["system"])
        self.assertEqual(once, S.fixer_rewrite(once))
        self.assertIsNone(S.fixer_rewrite("a fixer prompt with no COUNT rule"))

    def test_scope_is_taken_from_the_database_path(self):
        self.assertTrue(S._is_thrombosis_db(
            Path("data/bird_data/dev_databases/thrombosis_prediction/thrombosis_prediction.sqlite")))
        self.assertFalse(S._is_thrombosis_db(Path("data/x/california_schools.sqlite")))
        self.assertFalse(S._is_thrombosis_db(None))

    def test_a_patient_and_laboratory_only_schema_is_still_in_scope(self):
        """The focused schema of a Patient+Laboratory question never mentions Examination."""
        from src.selection.fixer import SQLFixer
        self.assertTrue(getattr(SQLFixer, "_qasql_count_strict", False))
        self.assertTrue(S._is_thrombosis_db(
            Path("out/dev_databases/thrombosis_prediction/thrombosis_prediction.sqlite")))

    def test_shared_prompt_dict_is_left_untouched_outside_the_call(self):
        from src.prompt import FIXER_PROMPT
        S.install()
        self.assertIn("KEEP DISTINCT otherwise", FIXER_PROMPT["system"])


class PlacementTest(unittest.TestCase):
    """Position is the whole finding: buried at the bottom the rule was obeyed 12/29 and the judge
    0/11; moved above RULE A / above EVALUATION CRITERIA the judge went 11/11 on the same inputs."""

    @classmethod
    def setUpClass(cls):
        if not S.install():
            raise unittest.SkipTest("src not importable")

    def test_rule_zero_precedes_rule_a_in_every_strategy(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        for strategy in builder.strategy_prompts:
            system = builder.build("how many patients?", THROMBOSIS, strategy,
                                   focused_schema=THROMBOSIS, evidence="")["system"]
            zero, rule_a = system.find("RULE 0"), system.find("**RULE A")
            self.assertGreater(zero, -1, strategy.value)
            self.assertLess(zero, rule_a, strategy.value)
            self.assertEqual(system.count("RULE 0 (READ THIS"), 1, strategy.value)

    def test_rule_zero_is_not_appended_at_the_end_any_more(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        system = builder.build("q", THROMBOSIS, list(builder.strategy_prompts)[0],
                               focused_schema=THROMBOSIS, evidence="")["system"]
        tail = system[-400:]
        self.assertNotIn("RULE 0", tail)

    def test_judge_note_precedes_the_criteria_list(self):
        from src.prompt import JUDGE_PROMPT
        system = JUDGE_PROMPT["system"]
        note, criteria = system.find("RULE 0, BEFORE ANY OTHER CRITERION"), system.find("EVALUATION CRITERIA")
        self.assertGreater(note, -1)
        self.assertLess(note, criteria)
        self.assertEqual(system.count("RULE 0, BEFORE ANY OTHER CRITERION"), 1)

    def test_judge_note_is_idempotent_and_reports_a_missing_anchor(self):
        from src.prompt import JUDGE_PROMPT
        once = JUDGE_PROMPT["system"]
        self.assertEqual(S.patch_judge_top_note(once), once)
        self.assertIsNone(S.patch_judge_top_note("a judge prompt with no criteria heading"))


class AllDatabasesTest(unittest.TestCase):
    """QASQL_COUNT_ALL=1 lifts the scope; without it nothing outside thrombosis is touched."""

    def tearDown(self):
        os.environ.pop("QASQL_COUNT_ALL", None)

    def test_off_by_default(self):
        self.assertFalse(S.all_databases())
        self.assertFalse(S.is_thrombosis(OTHER))
        self.assertFalse(S._is_thrombosis_db(Path("x/california_schools.sqlite")))

    def test_on_covers_every_database(self):
        os.environ["QASQL_COUNT_ALL"] = "1"
        self.assertTrue(S.all_databases())
        self.assertTrue(S.is_thrombosis(OTHER))
        self.assertTrue(S._is_thrombosis_db(Path("x/california_schools.sqlite")))

    def test_on_still_rejects_an_empty_schema(self):
        os.environ["QASQL_COUNT_ALL"] = "1"
        self.assertFalse(S.is_thrombosis({}))
        self.assertFalse(S._is_thrombosis_db(None))

    def test_rule_reaches_another_database_when_lifted(self):
        os.environ["QASQL_COUNT_ALL"] = "1"
        S.install()
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        system = builder.build("how many schools?", OTHER, list(builder.strategy_prompts)[0],
                               evidence="")["system"]
        self.assertIn("RULE 0", system)
        self.assertLess(system.find("RULE 0"), system.find("**RULE A"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
