import copy
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest import mock

from infrx_bench import config, deployment, policy

ROOT=Path(__file__).resolve().parents[1]


def pair(engine):
    spec=json.loads((ROOT/f"configs/deployments/native-{engine}-b200.proposed.json").read_text())
    benchmark=json.loads((ROOT/f"configs/b200-{engine}-native.proposed.json").read_text())
    return spec,benchmark


def evidence(spec,benchmark):
    identity=benchmark["identity"];runtime=benchmark["runtime"]
    documents={"inventory":{"status":"VERIFIED","hardware":{"gpu_model":"NVIDIA B200","node_gpu_count":8,"allocated_gpu_count":spec["topology"]["allocated_gpu_count"]}},
               "weights":{"verified":True,"repo":identity["repo"],"revision":identity["revision"],"files":[{"path":"synthetic-test-only.safetensors","sha256":"0"*64}]},
               "runtime":{"runtime":runtime,"inspection_status":"VERIFIED","configured_context_max_tokens":16384,
                          "request_modes":spec["request_modes"],"deployment_contract":deployment.contract(spec)},
               "smoke":{"identity":identity,"runtime":runtime,"generation_status":"PASSED","failed_attempts":0,"successful_output_tokens":1}}
    return {"config":benchmark,"inputs":{"qualification":{"data":{"schema":"infrx-qualification/2","status":"QUALIFIED_WITH_EVIDENCE","identity":identity,"runtime":runtime}},
            **{"qualification:"+name:{"data":document} for name,document in documents.items()}}}


class Deployment(unittest.TestCase):
    def test_both_templates_match_without_execution(self):
        with mock.patch("subprocess.run") as run,mock.patch("subprocess.Popen") as popen:
            for engine,gpus,port in (("sglang","0,1,2,3",18300),("vllm","0,1",18400)):
                spec,benchmark=pair(engine)
                result=deployment.preview(spec,benchmark,"/srv/infrx/native",gpus,port,"infrx-test",f"configs/b200-{engine}-native.proposed.json")
                self.assertFalse(result["executed"]);self.assertEqual(result["qualification"],"NOT_RUN")
                self.assertEqual(shlex.split(result["review_shell_command"]),result["docker_argv"])
                self.assertEqual(result["smoke_request"]["chat_template_kwargs"],{"thinking":False})
                self.assertEqual(result["smoke_request"]["reasoning_effort"],"none")
            run.assert_not_called();popen.assert_not_called()

    def test_image_offline_mount_and_network_are_explicit(self):
        for engine,gpus in (("sglang","0,1,2,3"),("vllm","0,1")):
            spec,_=pair(engine);argv=deployment.command(spec,"/srv/weights",gpus,18300,"infrx-test")
            self.assertIn("--pull=never",argv);self.assertIn("/srv/weights:/model:ro",argv)
            self.assertIn("HF_HUB_OFFLINE=1",argv);self.assertIn("TRANSFORMERS_OFFLINE=1",argv)
            self.assertIn("127.0.0.1:18300:30000",argv);self.assertNotIn("--privileged",argv)
            self.assertNotIn("--language-model-only",argv);self.assertFalse(any("speculative" in token for token in argv))
            self.assertIn(spec["runtime"]["image_digest"],argv)

    def test_exact_engine_topology_and_parsers(self):
        spec,_=pair("sglang");argv=deployment.command(spec,"/srv/weights","0,1,2,3",18300,"infrx-test")
        self.assertEqual(argv[argv.index("--tp")+1],"4");self.assertEqual(argv[argv.index("--ep-size")+1],"4")
        self.assertIn("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0",argv);self.assertNotIn("--attention-backend",argv)
        self.assertEqual(argv[argv.index("--context-length")+1],"16384")
        spec,_=pair("vllm");argv=deployment.command(spec,"/srv/weights","0,1",18400,"infrx-test")
        self.assertEqual(argv[argv.index("--tensor-parallel-size")+1],"2")
        self.assertEqual(json.loads(argv[argv.index("--engram-config")+1]),{"cpu_offload":True})
        self.assertEqual(argv[argv.index("--reasoning-parser")+1],"deepseek_v41")
        self.assertEqual(argv[argv.index("--max-model-len")+1],"16384")

    def test_bad_gpu_selection_mount_and_name_are_rejected(self):
        spec,_=pair("sglang")
        for gpus in ("0,1","0,1,1,3","0,1,2,8","all","0,1,2,3;echo nope"):
            with self.assertRaises(ValueError):deployment.command(spec,"/srv/weights",gpus,18300,"infrx-test")
        for path in ("relative","/srv/../weights","/srv/weights:rw","/srv/weights\n"):
            with self.assertRaises(ValueError):deployment.command(spec,path,"0,1,2,3",18300,"infrx-test")
        with self.assertRaises(ValueError):deployment.command(spec,"/srv/weights","0,1,2,3",18300,"other-service")

    def test_shell_rendering_preserves_literal_weight_path(self):
        spec,benchmark=pair("vllm")
        path="/srv/weights $(never-execute)"
        result=deployment.preview(spec,benchmark,path,"0,1",18400,"infrx-test","config.json")
        self.assertIn(path+":/model:ro",shlex.split(result["review_shell_command"]))

    def test_mutable_or_mismatched_image_is_refused(self):
        spec,_=pair("sglang")
        for image in ("lmsysorg/sglang:dev-dsv41","lmsysorg/sglang@sha256:"+"0"*64):
            bad=copy.deepcopy(spec);bad["runtime"]["image_digest"]=image
            with self.assertRaisesRegex(ValueError,"Image"):deployment.validate(bad)

    def test_wrong_benchmark_contract_context_thinking_and_port_are_refused(self):
        spec,benchmark=pair("vllm")
        for change in (lambda cfg:cfg["deployment_contract"].update(engram_cpu_offload=False),
                       lambda cfg:cfg["cells"][0].update(configured_context_max_tokens=32768),
                       lambda cfg:cfg["cells"][0].update(body={"reasoning_effort":"high"})):
            bad=copy.deepcopy(benchmark);change(bad)
            with self.assertRaises(ValueError):deployment.validate_benchmark(spec,bad)
        with self.assertRaisesRegex(ValueError,"port"):deployment.preview(spec,benchmark,"/srv/weights","0,1",18300,"infrx-test","config.json")

    def test_real_template_execution_stays_blocked(self):
        for engine in ("sglang","vllm"):
            _,benchmark=pair(engine)
            with self.assertRaisesRegex(ValueError,"Reconciled"):config.validate(benchmark,execution=True)

    def test_on_node_contract_must_match_even_with_verified_labels(self):
        spec,benchmark=pair("vllm");frozen=evidence(spec,benchmark)
        self.assertEqual(policy.qualification(frozen)["hardware"]["allocated_gpu_count"],2)
        frozen["inputs"]["qualification:runtime"]["data"]["deployment_contract"]["engram_cpu_offload"]=False
        with self.assertRaisesRegex(ValueError,"topology/offload/scheduler"):policy.qualification(frozen)

    def test_observed_gpu_allocation_must_match(self):
        spec,benchmark=pair("sglang");frozen=evidence(spec,benchmark)
        frozen["inputs"]["qualification:inventory"]["data"]["hardware"]["allocated_gpu_count"]=2
        with self.assertRaisesRegex(ValueError,"GPU allocation"):policy.qualification(frozen)

    def test_cli_has_no_execute_switch(self):
        with mock.patch("sys.argv",["deployment","--execute"]),mock.patch("subprocess.run") as run:
            with self.assertRaises(SystemExit):deployment.main()
            run.assert_not_called()
