"""Write separately qualified context/thinking configs; never start engines or collect."""
import copy
from pathlib import Path
import re
from . import config, harness


def write(cfg,out):
    spec=cfg.get("sweep")
    if not isinstance(spec,dict):raise ValueError("Explicit sweep specification required")
    contexts=spec.get("context_pairs",[]);modes=spec.get("thinking_modes",[])
    if not 1<=len(contexts)<=12 or not 1<=len(modes)<=4 or len(contexts)*len(modes)>24:
        raise ValueError("Choose a bounded sparse sweep, <=24 configs")
    destination=Path(out)
    if destination.exists():raise ValueError("Sweep output must be new; preserve existing files")
    documents=[]
    for pair in contexts:
        actual,maximum=pair["actual_input_tokens"],pair["configured_max_tokens"]
        if not isinstance(actual,int) or not isinstance(maximum,int) or not 128<=actual<maximum<=1048576:
            raise ValueError("Valid actual/configured context pair required")
        for mode in modes:
            if not re.fullmatch(r"[a-z0-9_-]{1,20}",mode["name"]):raise ValueError("Safe thinking mode name required")
            generated=copy.deepcopy(cfg);generated.pop("sweep",None)
            generated["qualification_file"]=pair.get("qualification_file","")
            generated["label"]="PROPOSED separate deployment/configuration qualification required"
            for cell in generated["cells"]:
                output=mode["output_tokens"]
                if actual+output>maximum:raise ValueError("Context pair must reserve the full output budget")
                cell["corpus_file"]=pair["corpus_file"]
                cell["actual_input_token_target"]=actual;cell["configured_context_max_tokens"]=maximum
                cell["thinking_mode"]=mode["name"];cell["body"]=mode["body"]
                cell["output_tokens"]=output
                if pair.get("baseline_canary_file"):cell["baseline_canary_file"]=pair["baseline_canary_file"]
            config.validate(generated)
            name=f"input{actual}-max{maximum}-{mode['name']}.json"
            documents.append((name,generated))
    destination.mkdir(parents=True)
    for name,document in documents:harness.write_json(destination/name,document)
    return {"execution":False,"configs":[str(destination/name) for name,document in documents],
            "gate":"Each real config requires its own exact context/runtime/mode evidence. No server mutation or collection."}
