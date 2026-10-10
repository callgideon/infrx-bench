"""Validate the documented JSON contract subset without third-party dependencies."""
import argparse
import json
from pathlib import Path
import re
import math

from . import identity

SCHEMAS = Path(__file__).resolve().parents[1]/"schemas/contracts.json"


def _type(value, kind):
    return {"object": isinstance(value, dict), "array": isinstance(value, list),
            "string": isinstance(value, str), "boolean": isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int,float)) and not isinstance(value,bool) and math.isfinite(value),
            "null": value is None}.get(kind, False)


def check(value, schema, location="$ "):
    supported={"type","required","properties","const","enum","minItems","minimum","pattern"}
    if set(schema)-supported:
        raise ValueError("Unsupported schema keyword; do not silently skip validation")
    types = schema.get("type")
    if types and not any(_type(value, kind) for kind in ([types] if isinstance(types,str) else types)):
        raise ValueError(location+" has wrong type")
    if "const" in schema and value != schema["const"]:
        raise ValueError(location+" differs from required constant")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(location+" differs from allowed values")
    if isinstance(value, dict):
        if set(schema.get("required", []))-value.keys():
            raise ValueError(location+" missing required fields")
        for key, subschema in schema.get("properties", {}).items():
            if key in value:
                check(value[key],subschema,location+"."+key)
    if "pattern" in schema and (not isinstance(value,str) or not re.search(schema["pattern"],value)):
        raise ValueError(location+" failed pattern")
    if "minimum" in schema and (not _type(value,"number") or value < schema["minimum"]):
        raise ValueError(location+" below minimum")
    if "minItems" in schema and (not isinstance(value,list) or len(value)<schema["minItems"]):
        raise ValueError(location+" has too few items")


def validate(kind, document):
    schemas = identity.read_json(SCHEMAS)["data"]
    if kind not in schemas:
        raise ValueError("Unknown artifact contract")
    check(document,schemas[kind])
    if kind=="config":
        from . import config
        config.validate(document)
    if kind=="assets":
        from . import assets
        assets.validate(document)
    return document


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind",required=True);parser.add_argument("--input",required=True)
    parser.add_argument("--jsonl",action="store_true")
    args=parser.parse_args()
    try:
        count=0
        path=Path(args.input)
        if args.jsonl:
            if path.is_symlink() or not path.is_file() or path.stat().st_size>identity.MAX_INPUT_BYTES:
                raise ValueError("Bounded regular JSONL file required")
            with path.open() as stream:
                for line in stream:
                    if len(line)>4*1024*1024:
                        raise ValueError("JSONL line cap exceeded")
                    validate(args.kind,json.loads(line));count+=1
            if not count:
                raise ValueError("Empty evidence journal")
        else:
            validate(args.kind,identity.read_json(path)["data"]);count=1
        print(json.dumps({"contract":args.kind,"records":count,"structural_validation":"PASSED"}));return 0
    except (ValueError,OSError,KeyError) as error:
        print("BLOCKED: "+str(error));return 2


if __name__=="__main__":
    raise SystemExit(main())
