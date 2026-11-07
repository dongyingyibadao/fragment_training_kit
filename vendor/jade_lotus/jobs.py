"""Explicit, bounded job submission and monitoring for the investigated workspace."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from jade_lotus.platform import DEFAULT_WORKSPACE, GPU_PATTERN, InspireClient, OAuthCredentials, PlatformError, write_json
from jade_lotus.runtime import writer_lock


def job_id(record):
    return record.get("job_id") or record.get("response", {}).get("data", {}).get("job_id")


def validate(payload):
    if payload.get("workspace_id") != DEFAULT_WORKSPACE or payload.get("task_priority") != 1:
        raise PlatformError("Only this workspace's explicitly requested priority 1 is supported")
    if payload.get("project_id") != "project-0a63ad2d-c102-4f9f-bdf4-af1a164bc0b0":
        raise PlatformError("Project must be the discovered 具身智能-具身决策 project")
    configs = payload.get("framework_config", [])
    if len(configs) != 1 or configs[0].get("instance_count") != 1 or configs[0].get("gpu_count") not in (1, 2):
        raise PlatformError("Only a single host with one or two GPUs is supported")
    declared_gpu = configs[0].get("resource_spec_price", {}).get("gpu_count")
    if declared_gpu is not None and declared_gpu != configs[0]["gpu_count"]:
        raise PlatformError("GPU count differs between the framework and resource specification")
    if configs[0].get("image") != "docker.sii.shaipower.online/inspire-studio/ljx:261009":
        raise PlatformError("Image must be the requested ljx:261009")
    gpu_type = configs[0].get("resource_spec_price", {}).get("gpu_type", "")
    # NVIDIA_H200_SXM_141G uses underscores, which are word characters in regex.
    if not GPU_PATTERN.search(gpu_type.replace("_", " ")):
        raise PlatformError("Only discovered H100/H200 specifications are supported")
    if not payload.get("command") or not payload.get("name"):
        raise PlatformError("A command and a stable job name are required")
    if not 0 < int(payload.get("max_running_time_ms", 0)) <= 24 * 3600 * 1000:
        raise PlatformError("An explicit runtime limit of at most 24 hours is required")


def submit(client, payload, record_path):
    """Record the request before sending; never blindly repeat an ambiguous POST."""
    validate(payload)
    record_path = Path(record_path)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    with writer_lock(record_path.parent / (record_path.stem + "-submit-lock")):
        if record_path.exists():
            previous = json.loads(record_path.read_text())
            if previous.get("request") != payload:
                raise PlatformError("The submission record belongs to a different request")
            if job_id(previous):
                return previous
            raise PlatformError("An earlier submission is unresolved; inspect the platform job list before retrying")
        record = {"request": payload, "request_sha256": digest, "status": "submitting"}
        write_json(record_path, record)
        try:
            response = client.request("POST", "/api/v1/train_job/create", payload)
        except Exception as error:
            # An explicit server rejection did not create a task. A timeout,
            # malformed response or 5xx may have lost a successful response.
            rejected = isinstance(error, PlatformError) and (
                str(error).startswith("Platform API rejected ")
                or bool(re.match(r"HTTP 4(?!08)\d\d:", str(error)))
            )
            record["status"] = "submission_rejected" if rejected else "submission_unknown"
            write_json(record_path, record)
            raise
        record.update(response=response, status="submitted")
        identifier = job_id(record)
        if not identifier:
            record["status"] = "submission_unknown"
            write_json(record_path, record)
            raise PlatformError("Create returned no job_id; inspect the task list before retrying")
        record["job_id"] = identifier
        write_json(record_path, record)
        return record


def inspect_job(client, record):
    identifier = job_id(record)
    if not identifier:
        raise PlatformError("Submission record has no job_id")
    detail = client.request("POST", "/api/v1/train_job/detail", {"job_id": identifier})
    data = detail["data"]
    if data.get("priority_level") != "LOW" or str(data.get("priority_name")) != "1":
        client.request("POST", "/api/v1/train_job/stop", {"job_id": identifier})
        raise PlatformError("Created job was not LOW/1; stopped it to preserve the requested priority")
    instances = client.request("POST", "/api/v1/train_job/instance_list", {"jobId": identifier, "page_num": 1, "page_size": 20})
    return {"job_id": identifier, "status": data.get("status"), "priority_level": data.get("priority_level"),
            "priority_name": data.get("priority_name"), "detail": detail, "instances": instances}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", default="~/.local/share/jade-lotus/credentials.json")
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("submit")
    create.add_argument("--payload", type=Path, required=True)
    create.add_argument("--record", type=Path, required=True)
    for action in ("status", "stop"):
        child = sub.add_parser(action)
        child.add_argument("--record", type=Path, required=True)
        if action == "status":
            child.add_argument("--output", type=Path)
    args = parser.parse_args()
    client = InspireClient(OAuthCredentials(args.credentials))
    try:
        if args.action == "submit":
            record = submit(client, json.loads(args.payload.read_text()), args.record)
            status = inspect_job(client, record)
            print(json.dumps({k: status[k] for k in ("job_id", "status", "priority_level", "priority_name")}))
        else:
            record = json.loads(args.record.read_text())
            if args.action == "stop":
                response = client.request("POST", "/api/v1/train_job/stop", {"job_id": job_id(record)})
                print(json.dumps(response))
            else:
                status = inspect_job(client, record)
                if args.output:
                    write_json(args.output, status)
                print(json.dumps({k: status[k] for k in ("job_id", "status", "priority_level", "priority_name")}))
        return 0
    except (PlatformError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
