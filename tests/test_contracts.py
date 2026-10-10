import json
from pathlib import Path
import unittest
from infrx_bench import contracts

ROOT=Path(__file__).resolve().parents[1]


class Contracts(unittest.TestCase):
    def test_examples_use_structural_and_semantic_validation(self):
        for name in ("cpu-mock.json","b200-screen.proposed.json","context-thinking-sweep.proposed.json"):
            contracts.validate("config",json.loads((ROOT/"configs"/name).read_text()))

    def test_required_fields_fail(self):
        with self.assertRaises(ValueError):contracts.validate("final",{"schema":"infrx-final/2"})
        with self.assertRaises(ValueError):contracts.validate("attempts",{"attempt_id":"x","scheduled_at":1,"status":"invented"})

    def test_shed_and_failed_rows_remain_valid_evidence(self):
        for status in ("shed","failed"):
            contracts.validate("attempts",{"attempt_id":"x","scheduled_at":1.0,"status":status,"usage":None})

    def test_unknown_schema_keywords_fail(self):
        with self.assertRaises(ValueError):contracts.check({}, {"made_up":True})

    def test_quality_requires_actual_coverage(self):
        with self.assertRaises(ValueError):contracts.validate("quality",{"quality_diagnostic":{"n":0,"passed":0},"canary_case_ids":[],"canary_identity":"x","configuration_contract":{},"cell_gate_passed":True})
