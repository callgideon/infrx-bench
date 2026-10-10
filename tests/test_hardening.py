import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from infrx_bench import analysis,budget,cli,config,fixture,harness,identity,policy,sweeps

ROOT=Path(__file__).resolve().parents[1]


def example():
    cfg=json.loads((ROOT/"configs/cpu-mock.json").read_text())
    cfg["cells"]=cfg["cells"][:1]
    return cfg


class Hardening(unittest.TestCase):
    def test_external_corpus_bytes_are_resume_identity(self):
        with tempfile.TemporaryDirectory() as tmp,fixture.server() as (srv,base):
            cfg=example();cfg["endpoints"]=[base];corpus=Path(tmp)/"corpus.json"
            data={"entries":[{"id":"one","messages":[{"role":"user","content":"immutable prompt"}]}]}
            corpus.write_text(json.dumps(data));cfg["cells"][0]["corpus_file"]=str(corpus)
            out=Path(tmp)/"run";cli.prepare(cfg,out,mock_token=srv.token)
            corpus.write_text(json.dumps(data,indent=2))
            with self.assertRaisesRegex(ValueError,"Frozen input bytes"):cli.prepare(cfg,out,resume=True,mock_token=srv.token)
            self.assertEqual(len(srv.bodies),0)

    def test_external_baseline_bytes_are_resume_identity(self):
        with tempfile.TemporaryDirectory() as tmp,fixture.server() as (srv,base):
            cfg=example();cfg["endpoints"]=[base];baseline=Path(tmp)/"baseline.json"
            baseline.write_text('{"score":1}');cfg["baseline_canary_summary"]=str(baseline)
            out=Path(tmp)/"run";cli.prepare(cfg,out,mock_token=srv.token)
            baseline.write_text('{"score":0}')
            with self.assertRaisesRegex(ValueError,"Frozen input bytes"):cli.prepare(cfg,out,resume=True,mock_token=srv.token)

    def test_external_canary_bytes_are_resume_identity(self):
        with tempfile.TemporaryDirectory() as tmp,fixture.server() as (srv,base):
            cfg=example();cfg["endpoints"]=[base];cases=Path(tmp)/"cases.json"
            data=json.loads((ROOT/"infrx_bench/cases.json").read_text());cases.write_text(json.dumps(data))
            cfg["canary_cases_file"]=str(cases);out=Path(tmp)/"run";cli.prepare(cfg,out,mock_token=srv.token)
            data[0]["expected"]=[999];cases.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"Frozen input bytes"):cli.prepare(cfg,out,resume=True,mock_token=srv.token)

    def test_deadline_survives_resume_and_expires(self):
        with tempfile.TemporaryDirectory() as tmp,fixture.server() as (srv,base):
            cfg=example();cfg["endpoints"]=[base];out=Path(tmp)/"run"
            with mock.patch("time.time",return_value=1000):_,first=cli.prepare(cfg,out,mock_token=srv.token)
            with mock.patch("time.time",return_value=1100):_,second=cli.prepare(cfg,out,resume=True,mock_token=srv.token)
            self.assertEqual(first,second);self.assertEqual(first["expires_at"],1240)
            with mock.patch("time.time",return_value=1241):
                with self.assertRaises(TimeoutError):cli.prepare(cfg,out,resume=True,mock_token=srv.token)

    def test_deadline_window_tampering_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp,fixture.server() as (srv,base):
            cfg=example();cfg["endpoints"]=[base];out=Path(tmp)/"run";cli.prepare(cfg,out,mock_token=srv.token)
            path=out/"run-window.json";window=json.loads(path.read_text());window["expires_at"]+=1000;path.write_text(json.dumps(window))
            with self.assertRaisesRegex(ValueError,"window changed"):cli.prepare(cfg,out,resume=True,mock_token=srv.token)

    def test_request_drain_is_clamped_to_absolute_deadline(self):
        with fixture.server(delay=.3) as (srv,base):
            start=time.monotonic()
            row=harness.request(base,"fixture/model",[{"role":"user","content":"test"}],32,5,time.time(),"bounded",deadline_epoch=time.time()+.04)
            self.assertLess(time.monotonic()-start,.25);self.assertEqual(row["status"],"failed")
            self.assertEqual(row["error"],"request_timeout")

    def test_expired_export_does_not_copy(self):
        from infrx_bench import export
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/"source";dst=Path(tmp)/"dest";src.write_bytes(b"fixture")
            with self.assertRaises(TimeoutError):export.copy(src,dst,time.time()-1)
            self.assertFalse(Path(str(dst)+".receipt.json").exists())

    def test_mock_label_cannot_target_real_arm(self):
        cfg=example();cfg["arm"]="native-sglang-baseline"
        with self.assertRaises(ValueError):config.validate(cfg)
        cfg=example()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,"controlled"):cli.prepare(cfg,Path(tmp)/"run")

    def test_blocked_arm_and_precision_relabel_are_refused(self):
        cfg=json.loads((ROOT/"configs/b200-screen.proposed.json").read_text());cfg["arm"]="nvfp8-candidate"
        with self.assertRaises(ValueError):config.validate(cfg)
        cfg=json.loads((ROOT/"configs/b200-screen.proposed.json").read_text());cfg["identity"]["precision"]="nvfp8"
        with self.assertRaises(ValueError):config.validate(cfg)

    def test_explicit_role_and_positive_slo_are_required(self):
        cfg=example();cfg.pop("arm_role")
        with self.assertRaises(ValueError):config.validate(cfg)
        cfg=example();cfg["cells"][0].pop("minimum_slo_fraction")
        with self.assertRaises(ValueError):config.validate(cfg)
        cfg=example();cfg["canary"]["minimum_quality_fraction"]=0
        with self.assertRaises(ValueError):config.validate(cfg)

    def test_canaries_derive_each_configuration_controls(self):
        cfg=example();cell=cfg["cells"][0];cell["thinking_mode"]="fixture-mode";cell["sampling"]={"temperature":.5,"top_p":.9}
        cell["body"]={"fixture_thinking":"enabled","ignore_eos":True};cell["output_tokens"]=512
        before,scored,after=config.expand(cfg)
        self.assertEqual(config.contract(before),config.contract(scored));self.assertEqual(config.contract(after),config.contract(scored))
        self.assertNotIn("ignore_eos",before["body"]);self.assertEqual(before["output_tokens"],512)

    def test_final_canary_failure_prevents_capacity_receipt(self):
        with tempfile.TemporaryDirectory() as tmp,fixture.server() as (srv,base):
            cfg=example();cfg["endpoints"]=[base];out=Path(tmp)/"run";original=harness.run_cell
            def run(driver,root,**kwargs):
                if str(root).endswith("canary-after"):srv.fail_quality=True
                return original(driver,root,**kwargs)
            with mock.patch.object(harness,"run_cell",side_effect=run):
                with self.assertRaises(ValueError):cli.execute(cfg,out,mock_token=srv.token)
            self.assertTrue((out/"closed-c1/summary.json").exists());self.assertFalse((out/"experiment-complete.json").exists())
            with self.assertRaises((ValueError,OSError)):analysis.qualify_summary(out,out/"closed-c1/summary.json")

    def test_summary_flags_alone_never_qualify_capacity(self):
        summary={"elapsed_s_including_drain":60,"goodput_requests_per_s":2,"counts":{"success":120},"successful_output_tokens":1200,
                 "experiment_purpose":"b200_research","cell_gate_passed":True,"valid_capacity_evidence":True,"tail_latency_qualified":True,
                 "deployment_qualification":"SMOKE_PASSED","queue_stability":{"status":"stable"}}
        self.assertIsNone(analysis.capacity_cost(summary,80,2)["sustainable_users_qualified"])

    def test_queue_drain_cannot_hide_admission_growth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"completion.json").write_text(json.dumps({"start_at":0,"end_at":10,"admission_started_at":0,"admission_ended_at":5}))
            points=[(i,i if i<=5 else 10-i) for i in range(11)]
            (root/"telemetry.jsonl").write_text(''.join(json.dumps({"at":x,"metrics":{"fixture":f"waiting {y}\n"}})+'\n' for x,y in points))
            result=cli.queue_stability(root,{"arrival_rate":2,"queue_metric":"waiting","minimum_stability_seconds":5})
            self.assertEqual(result["status"],"growing_or_not_drained");self.assertEqual(result["final_queue"],5)

    def test_stable_queue_with_rejections_fails_joint_gate(self):
        summary={"counts":{"success":5,"shed":5,"failed":0,"unfinished":0},"attempts":10,"complete":True,"missing_usage_attempts":0,
                 "invalid_protocol_attempts":[],"malformed_journal_lines":0,"goodput_requests_per_s":5,"elapsed_s_including_drain":1,
                 "queue_stability":{"status":"stable"},"tail_latency_qualified":True,"valid_capacity_evidence":True,"quality_diagnostic":{"n":5,"passed":5}}
        cell={"kind":"load","evidence_role":"capacity","minimum_slo_fraction":.95,"minimum_task_quality_fraction":1,"max_error_fraction":.01}
        self.assertFalse(cli.gate(summary,cell))

    def test_real_qualification_does_not_accept_a_status_label(self):
        cfg=json.loads((ROOT/"configs/b200-screen.proposed.json").read_text())
        frozen={"config":cfg,"inputs":{"qualification":{"data":{"schema":"infrx-qualification/2","status":"QUALIFIED_WITH_EVIDENCE","identity":cfg["identity"],"runtime":cfg["runtime"]}}}}
        with self.assertRaisesRegex(ValueError,"Missing inventory"):policy.qualification(frozen)

    def test_sweep_reserves_output_and_preserves_source(self):
        cfg=example();original=copy.deepcopy(cfg)
        cfg["sweep"]={"context_pairs":[{"actual_input_tokens":2048,"configured_max_tokens":4096,"corpus_file":"proposed.json"}],
                      "thinking_modes":[{"name":"none","body":{},"output_tokens":512}]}
        with tempfile.TemporaryDirectory() as tmp:
            result=sweeps.write(cfg,Path(tmp)/"sweep")
            self.assertFalse(result["execution"]);self.assertEqual(len(result["configs"]),1)
            generated=json.loads(Path(result["configs"][0]).read_text());config.validate(generated)
            self.assertEqual(generated["cells"][0]["configured_context_max_tokens"],4096)
        self.assertEqual(cfg["cells"],original["cells"])
        cfg["sweep"]["context_pairs"][0]["configured_max_tokens"]=2200
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,"reserve"):sweeps.write(cfg,Path(tmp)/"bad")
