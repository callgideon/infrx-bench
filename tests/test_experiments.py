import copy
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest

from infrx_bench import analysis, cli, config, export, fixture, sessions

ROOT=Path(__file__).resolve().parents[1]


def example():
    return json.loads((ROOT/"configs/cpu-mock.json").read_text())


class Experiments(unittest.TestCase):
    def test_configuration_and_budget_are_explicit(self):
        cfg=example(); config.validate(cfg)
        cfg["budget_seconds"]=.01
        with self.assertRaises(ValueError): config.validate(cfg)

    def test_identity_override_and_isolation_are_refused(self):
        cfg=example(); cfg["cells"][0]["body"]={"model":"different"}
        with self.assertRaises(ValueError): config.validate(cfg)
        cfg=example(); cfg["capabilities"]["cache_salt_verified"]=False
        with self.assertRaises(ValueError): config.validate(cfg)

    def test_real_runtime_placeholders_are_execution_blockers(self):
        cfg=json.loads((ROOT/"configs/thor-rehearsal.template.json").read_text())
        config.validate(cfg)
        with self.assertRaises(ValueError): config.validate(cfg,execution=True)

    def test_session_continuity_salt_and_bounded_completion(self):
        cfg=example(); cell=cfg["cells"][-1]
        with tempfile.TemporaryDirectory() as tmp, fixture.server() as (srv,base):
            cfg["endpoints"]=[base]
            result=sessions.run(config.driver_config(cfg,cell),cell,Path(tmp)/"isolated")
            self.assertEqual(result["sessions"]["completed"],2)
            self.assertEqual({x["cache_salt"] for x in srv.bodies},{"infrx-tenant:a","infrx-tenant:b"})
        cfg=example(); cell=cfg["cells"][-2]
        with tempfile.TemporaryDirectory() as tmp, fixture.server() as (srv,base):
            cfg["endpoints"]=[base]
            result=sessions.run(config.driver_config(cfg,cell),cell,Path(tmp)/"sessions")
            self.assertEqual(result["sessions"]["completed"],2)
            second=[b for b in srv.bodies if len(b["messages"])==4]
            self.assertEqual(len(second),2)
            self.assertEqual(second[0]["messages"][2]["role"],"assistant")
            self.assertGreater(result["batch_window_seconds"],0)

    def test_session_deadline_sheds_and_blocks_followup(self):
        cfg=example(); cell=copy.deepcopy(cfg["cells"][-2])
        cell["duration_seconds"]=.01; cell["batch_window_seconds"]=.1
        with tempfile.TemporaryDirectory() as tmp, fixture.server() as (srv,base):
            cfg["endpoints"]=[base]
            result=sessions.run(config.driver_config(cfg,cell),cell,Path(tmp)/"limited")
        self.assertEqual(result["counts"]["shed"],2)
        self.assertEqual(result["sessions"]["completed"],0)
        self.assertFalse(cli.gate(result,cell))

    def test_quality_gate_stops_before_capacity(self):
        cfg=example()
        with tempfile.TemporaryDirectory() as tmp, fixture.server(fail_quality=True) as (srv,base):
            cfg["endpoints"]=[base]
            with self.assertRaises(ValueError): cli.execute(cfg,Path(tmp)/"run",mock_token=srv.token)
            self.assertFalse((Path(tmp)/"run/closed-c1").exists())
            summary=json.loads((Path(tmp)/"run/closed-c1-canary-before/summary.json").read_text())
            self.assertFalse(summary["cell_gate_passed"])

    def test_incomplete_canary_coverage_stops_capacity(self):
        cfg=example(); cfg["canary"]["duration_seconds"]=.001
        with tempfile.TemporaryDirectory() as tmp, fixture.server(delay=.01) as (srv,base):
            cfg["endpoints"]=[base]
            with self.assertRaises(ValueError): cli.execute(cfg,Path(tmp)/"run",mock_token=srv.token)
            summary=json.loads((Path(tmp)/"run/closed-c1-canary-before/summary.json").read_text())
            self.assertLess(len(summary["canary_case_ids"]),10)
            self.assertFalse(summary["cell_gate_passed"])

    def test_b200_proposal_is_not_execution_ready(self):
        cfg=json.loads((ROOT/"configs/b200-screen.proposed.json").read_text())
        config.validate(cfg)
        with self.assertRaises(ValueError): config.validate(cfg,execution=True)

    def test_complete_resume_requires_immutable_artifacts(self):
        cfg=example(); cfg["cells"]=cfg["cells"][:1]
        with tempfile.TemporaryDirectory() as tmp, fixture.server() as (srv,base):
            cfg["endpoints"]=[base]; root=Path(tmp)/"run"
            cli.execute(cfg,root,mock_token=srv.token); count=len(srv.bodies)
            cli.execute(cfg,root,resume=True,mock_token=srv.token)
            self.assertEqual(len(srv.bodies),count)
            (root/"closed-c1/attempts.jsonl").write_text("changed")
            with self.assertRaises(ValueError): cli.execute(cfg,root,resume=True,mock_token=srv.token)

    def test_completed_retry_is_retained_on_later_resume(self):
        cfg=example(); cfg["cells"]=cfg["cells"][:1]
        with tempfile.TemporaryDirectory() as tmp, fixture.server() as (srv,base):
            cfg["endpoints"]=[base]; root=Path(tmp)/"run"
            cli.execute(cfg,root,mock_token=srv.token)
            (root/"experiment-complete.json").unlink()
            (root/"closed-c1").rename(root/"closed-c1-retry-1")
            (root/"closed-c1").mkdir()
            count=len(srv.bodies); cli.execute(cfg,root,resume=True,mock_token=srv.token)
            self.assertEqual(len(srv.bodies),count)

    def test_deadline_preserves_no_false_completion(self):
        cfg=example(); cfg["end_utc"]=(dt.datetime.now(dt.timezone.utc)-dt.timedelta(seconds=1)).isoformat()
        with tempfile.TemporaryDirectory() as tmp, fixture.server() as (srv,base):
            cfg["endpoints"]=[base]
            with self.assertRaises(TimeoutError): cli.execute(cfg,Path(tmp)/"run",mock_token=srv.token)
            self.assertFalse((Path(tmp)/"run/experiment-complete.json").exists())

    def test_verified_export_fails_on_missing_or_changed_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/"a"; b=Path(tmp)/"b"; a.write_bytes(b"artifact")
            self.assertTrue(export.copy(a,b)["destination_verified"])
            b.write_bytes(b"changed")
            with self.assertRaises(ValueError): export.verify(a,b)

    def test_user_capacity_and_cost_are_not_concurrency(self):
        summary={"elapsed_s_including_drain":60,"goodput_requests_per_s":2,"counts":{"success":120},
            "successful_output_tokens":1200,"valid_capacity_evidence":True,"cell_gate_passed":True}
        cost=analysis.capacity_cost(summary,80,2,users=30,allocated_gpus=4)
        self.assertEqual(cost["capacity_users_scenario"],60)
        self.assertIsNone(cost["sustainable_users_qualified"])
        self.assertEqual(cost["hypothetical_fractional_hourly_usd"],40)
        self.assertAlmostEqual(cost["actual_full_node_cost_for_observed_cell_usd"],80/60)
        summary["queue_stability"]={"status":"stable"}
        self.assertIsNone(analysis.capacity_cost(summary,80,2)["sustainable_users_qualified"])
        summary["experiment_purpose"]="b200_research"
        summary["deployment_qualification"]="SMOKE_PASSED"
        summary["tail_latency_qualified"]=True
        self.assertIsNone(analysis.capacity_cost(summary,80,2)["sustainable_users_qualified"])

    def test_canary_comparison_rejects_changed_thinking(self):
        a={"canary_identity":"same","quality_diagnostic":{"n":10,"passed":10}}
        b={"canary_identity":"same","quality_diagnostic":{"n":10,"passed":9}}
        self.assertFalse(analysis.quality_comparison(a,b,0)["pass"])
        b["canary_identity"]="different"
        with self.assertRaises(ValueError): analysis.quality_comparison(a,b)

    def test_reservation_charges_idle_and_failed_time(self):
        cost=analysis.reservation_cost(48,80,1000,240,other_hourly=2)
        self.assertEqual(cost["whole_node_cost_usd"],3936)
        self.assertEqual(cost["usd_per_accepted_task"],3.936)
        with self.assertRaises(ValueError): analysis.reservation_cost(48,80,-1,240)


if __name__=="__main__": unittest.main()
