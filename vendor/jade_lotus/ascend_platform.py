"""QZ Ascend discovery and explicit low-priority job requests.

Discovery, listing and inspection never stop a job. Resource/project/image and
priority identities come from a workspace inventory, not NVIDIA defaults.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import http.cookiejar
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from jade_lotus.platform import (DEFAULT_BACKEND, NoRedirect,
                                OAuthCredentials, PlatformError, compact_name,
                                identifier, items, priority_restriction,
                                resource_summary, sanitized, write_json)
from jade_lotus.runtime import writer_lock


WORKSPACES = (
    "ws-e5a87523-5ec2-4afa-a14a-56c3f4088d14",
    "ws-8f82eaee-1717-4b74-a6fa-afbc6f231845",
)
TERMINAL = {"job_succeeded", "job_failed", "job_stopped", "job_completed",
            "job_finished", "job_deleted", "job_cancelled", "job_canceled",
            "submission_rejected", "cancelled_before_submission"}
RUNNING = {"job_running", "running"}
API = {
    "user": "/api/v2/user?Action=GetUserDetail",
    "permissions": "/api/v2/user?Action=GetPermissions",
    "projects": "/api/v2/project?Action=GetProjectListV2",
    "images": "/api/v2/image?Action=ListImages",
    "groups": "/api/v2/workspace?Action=ListLogicComputeGroups",
    "resource": "/api/v2/workspace?Action=GetLogicComputeGroupResource",
    "prices": "/api/v2/resource-price?Action=GetLogicComputeGroupResourceSpecPrices",
    "schedule": "/api/v2/train?Action=GetTrainScheduleConfig",
    "create": "/api/v2/train?Action=CreateJobConsole",
    "list": "/api/v2/train?Action=ListJobs",
    "detail": "/api/v2/train?Action=GetJob",
    "instances": "/api/v2/train?Action=ListJobInstances",
    "logs": "/api/v2/train?Action=GetJobLog",
    "events": "/api/v2/train?Action=ListJobEvents",
    "node_events": "/api/v2/cluster?Action=ListNodeEvents",
    "stop": "/api/v2/train?Action=StopJob",
}
READ_ONLY_API = frozenset(API[key] for key in (
    "user", "permissions", "projects", "images", "groups", "resource", "prices", "schedule",
    "list", "detail", "instances", "logs", "events", "node_events"))
SESSION_HOSTS = frozenset(("qz.sii.edu.cn", "keycloak-inspire-prod.sii.edu.cn", "cas.sii.edu.cn"))
SESSION_MAX_REDIRECTS = 8
SESSION_TIMEOUT_SECONDS = 45


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def public(value):
    """Keep API evidence useful without serializing session/capability fields."""
    if isinstance(value, dict):
        return {key: ("[redacted]" if any(part in key.casefold() for part in
                ("token", "cookie", "password", "secret", "authorization", "viewer_url", "webshell"))
                else public(child)) for key, child in sanitized(value).items()}
    if isinstance(value, list):
        return [public(child) for child in value]
    return value


def positive_int(value, name, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlatformError(f"{name} must be a positive integer")
    if not math.isfinite(value) or int(value) != value or value < 1 or (maximum and value > maximum):
        raise PlatformError(f"{name} must be a positive integer" + (f" <= {maximum}" if maximum else ""))
    return int(value)


def npu_type(spec, group=None):
    """Only explicit Ascend device types count; never resolve by a group name."""
    info = spec.get("gpu_info") or spec.get("npu_info") or {}
    raw = spec.get("gpu_type") or spec.get("npu_type") or info.get("gpu_type") or info.get("npu_type")
    brand = str(info.get("brand", spec.get("brand", ""))).upper()
    if isinstance(raw, str) and (raw.upper().startswith("ASCEND_") or
                               (brand == "ASCEND" and raw.strip())):
        return raw
    if raw and group:
        for entry in group.get("gpu_type_stats", []):
            candidate = entry.get("gpu_info") or {}
            if candidate.get("gpu_type") == raw:
                return npu_type({"gpu_info": candidate})
    return None


class CookieClient:
    """Use existing SSO cookies; renew the QZ session without login forms."""
    cookie_session = True

    def __init__(self, cookie_file, backend_url=DEFAULT_BACKEND):
        self.path = Path(cookie_file).expanduser().resolve()
        if stat.S_IMODE(self.path.stat().st_mode) & 0o077:
            raise PlatformError("QZ cookie jar must be private (chmod 600)")
        if urllib.parse.urlparse(backend_url).scheme != "https":
            raise PlatformError("Platform requests require HTTPS")
        self.backend_url = backend_url.rstrip("/")
        self.jar = http.cookiejar.MozillaCookieJar(str(self.path))
        self.jar.load(ignore_discard=True, ignore_expires=True)
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar), NoRedirect())
        self.evidence = []
        self._last_request = 0.0

    def headers(self):
        return {"Content-Type": "application/json", "Accept": "application/json"}

    @staticmethod
    def _session_url_allowed(url):
        try:
            parsed = urllib.parse.urlsplit(url)
            return (parsed.scheme == "https" and parsed.hostname in SESSION_HOSTS
                    and parsed.port in (None, 443) and not parsed.username and not parsed.password)
        except ValueError:
            return False

    def _save_cookie_jar(self):
        descriptor, temporary = tempfile.mkstemp(prefix="." + self.path.name + ".", dir=self.path.parent)
        try:
            os.fchmod(descriptor, 0o600)
            os.close(descriptor)
            descriptor = None
            self.jar.save(temporary, ignore_discard=True, ignore_expires=True)
            os.chmod(temporary, 0o600)
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _renew_cookie_session(self):
        """Follow only the official HTTPS SSO chain; never log its query/cookies."""
        succeeded, redirects = False, 0
        deadline = time.monotonic() + SESSION_TIMEOUT_SECONDS
        url = self.backend_url + "/?t=" + str(int(time.time() * 1000))
        backend = urllib.parse.urlsplit(self.backend_url)
        try:
            if backend.hostname != urllib.parse.urlsplit(DEFAULT_BACKEND).hostname:
                return False
            for redirects in range(SESSION_MAX_REDIRECTS + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._session_url_allowed(url):
                    return False
                request = urllib.request.Request(url, method="GET", headers={"Accept": "text/html"})
                try:
                    with self.opener.open(request, timeout=min(10, remaining)) as response:
                        final_url = response.geturl()
                        if (response.getcode() != 200 or not self._session_url_allowed(final_url)
                                or urllib.parse.urlsplit(final_url).netloc != backend.netloc):
                            return False
                        self._save_cookie_jar()
                        succeeded = True
                        return True
                except urllib.error.HTTPError as error:
                    location = error.headers.get("Location") if error.headers else None
                    code = error.code
                    error.close()
                    if code not in (301, 302, 303, 307, 308) or not location:
                        return False
                    url = urllib.parse.urljoin(url, location)
            return False
        except (OSError, ValueError):
            return False
        finally:
            self.evidence.append({"at": now_iso(), "event": "cookie_session_renewal",
                                  "succeeded": succeeded, "redirects": redirects})

    def request(self, method, path, body=None):
        if method != "POST" or path not in API.values():
            raise PlatformError("Only the explicit official QZ V2 API actions are allowed")
        for attempt in range(2):
            delay = 1 / 3 - (time.monotonic() - self._last_request)
            if delay > 0:
                time.sleep(delay)
            self._last_request = time.monotonic()
            request = urllib.request.Request(self.backend_url + path, method=method,
                data=json.dumps(body if body is not None else {}).encode(), headers=self.headers())
            try:
                with self.opener.open(request, timeout=30) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as error:
                code = error.code
                error.close()
                if code == 401 and attempt == 0 and self.cookie_session:
                    renewed = self._renew_cookie_session()
                    if renewed and path in READ_ONLY_API:
                        continue
                raise PlatformError(f"HTTP {code}: {method} {path}") from None
            except (urllib.error.URLError, TimeoutError, ValueError):
                raise PlatformError(f"Invalid or unavailable API response: {method} {path}") from None
        if not isinstance(result, dict):
            raise PlatformError("Platform returned a non-object JSON response")
        self.evidence.append({"at": now_iso(), "method": method, "path": path,
                              "request": public(body), "response": public(result)})
        metadata = result.get("ResponseMetadata")
        if not isinstance(metadata, dict):
            raise PlatformError("Platform returned an invalid V2 response envelope")
        error = metadata.get("Error")
        if error:
            code = error.get("Code", "unknown") if isinstance(error, dict) else "unknown"
            raise PlatformError(f"Platform API rejected {method} {path} (code={code})")
        return {"code": 0, "message": "", "data": result.get("Result"), "metadata": public(metadata)}


class OAuthClient(CookieClient):
    """The same explicit V2 contract with a private official OAuth credential."""
    cookie_session = False

    def __init__(self, credentials, backend_url=DEFAULT_BACKEND):
        if urllib.parse.urlparse(backend_url).scheme != "https":
            raise PlatformError("Platform requests require HTTPS")
        self.credentials = OAuthCredentials(credentials)
        self.backend_url = backend_url.rstrip("/")
        self.opener = urllib.request.build_opener(NoRedirect())
        self.evidence = []
        self._last_request = 0.0

    def headers(self):
        return dict(super().headers(), Authorization="Bearer " + self.credentials.token(),
                    **{"inspire-api": "true", "X-Inspire-Client-Source": "jade-lotus"})


def make_client(credentials=None, cookie_file=None, backend_url=DEFAULT_BACKEND):
    if bool(credentials) == bool(cookie_file):
        raise PlatformError("Provide exactly one private OAuth credential file or QZ cookie jar")
    return (CookieClient(cookie_file, backend_url) if cookie_file else
            OAuthClient(credentials, backend_url))


def _match_projects(projects, config):
    if config.get("project_id"):
        return [p for p in projects if identifier(p, "project_id", "id") == config["project_id"]]
    name = config.get("project_name", "具身智能 具身决策")
    return [p for p in projects if compact_name(p.get("name", p.get("project_name", ""))) == compact_name(name)]


def _match_images(images, config):
    available = [im for im in images if im.get("status") == "SUCCESS"]
    if config.get("image"):
        return [im for im in available if identifier(im, "address", "url") == config["image"]]
    if config.get("image_id"):
        return [im for im in available if identifier(im, "image_id", "id") == config["image_id"]]
    if config.get("image_query"):
        query = config["image_query"]
        return [im for im in available if query in (str(im.get("version", "")), str(im.get("name", "")))
                or str(im.get("name", "")).endswith(":" + query)]
    # Return all choices; an ambiguous image never silently chooses the first.
    return available


def discover(client, workspace_configs=None, workspace_ids=WORKSPACES):
    """Read both spaces; configured selectors never hide other raw candidates."""
    configs = {c["workspace_id"]: c for c in (workspace_configs or [])}
    if not set(workspace_ids).issubset(WORKSPACES) or not set(configs).issubset(WORKSPACES):
        raise PlatformError("Ascend discovery is limited to the two authorized workspaces")
    user_response = client.request("POST", API["user"], {})
    user = user_response.get("data") or {}
    result = {"schema_version": 1, "backend": "ascend", "at": now_iso(),
              "user": {key: user.get(key) for key in ("id", "name")},
              "workspaces": [], "groups": [], "selections": [], "errors": []}
    for workspace_id in workspace_ids:
        config = configs.get(workspace_id, {})
        try:
            permission = client.request("POST", API["permissions"], {"workspace_id": workspace_id})
            projects = items(client.request("POST", API["projects"],
                {"page": 1, "page_size": -1, "filter": {"workspace_id": workspace_id, "check_admin": True}}))
            images = items(client.request("POST", API["images"],
                {"page": 0, "page_size": -1, "filter": {"visibility": "VISIBILITY_PRIVATE",
                 "source_list": ["SOURCE_PRIVATE", "SOURCE_PUBLIC"],
                 "registry_hint": {"workspace_id": workspace_id}}}), "images", "list", "items")
            group_response = client.request("POST", API["groups"],
                {"pageNum": 1, "pageSize": 10000, "filter": {"workspace_id": workspace_id,
                 "support_job_type": "distributed_training", "include_gpu_type_stats": True}})
            schedule = client.request("POST", API["schedule"], {"workspace_id": workspace_id}).get("data") or {}
        except (PlatformError, OSError, TimeoutError) as error:
            result["errors"].append({"workspace_id": workspace_id, "error": str(error)})
            continue
        project_matches, image_matches = _match_projects(projects, config), _match_images(images, config)
        result["workspaces"].append({"workspace_id": workspace_id, "permissions": public(permission.get("data")),
            "projects": public(projects), "images": public(images), "project_matches": public(project_matches),
            "image_matches": public(image_matches), "schedule": public(schedule)})
        for group in items(group_response, "logic_compute_groups", "list", "items"):
            group_id = identifier(group, "logic_compute_group_id", "id")
            if not group_id:
                continue
            try:
                resource = client.request("POST", API["resource"],
                    {"workspace_id": workspace_id, "logic_compute_group_id": group_id})
                price_query = {"workspace_id": workspace_id, "logic_compute_group_id": group_id,
                               "schedule_config_type": "SCHEDULE_CONFIG_TYPE_TRAIN"}
                if len(project_matches) == 1:
                    price_query["project_id"] = identifier(project_matches[0], "project_id", "id")
                if (config.get("priority") or {}).get("task_priority"):
                    price_query["task_priority"] = config["priority"]["task_priority"]
                specs = items(client.request("POST", API["prices"], price_query),
                    "lcg_resource_spec_prices", "resource_spec_prices", "lcgResourceSpecPrices", "items", "list")
            except (PlatformError, OSError, TimeoutError) as error:
                result["errors"].append({"workspace_id": workspace_id, "logic_compute_group_id": group_id,
                                         "error": str(error)})
                continue
            summary = {"workspace_id": workspace_id, "id": group_id, "name": group.get("name"),
                       "resources": resource_summary(resource), "specs": public(specs),
                       "gpu_type_stats": public(group.get("gpu_type_stats", [])),
                       "storage_disks": public(group.get("storage_disks", [])),
                       "price_query": public(price_query),
                       "resource_query": {"status": "fresh", "observed_at": now_iso()}}
            result["groups"].append(summary)
            for spec in specs:
                device = npu_type(spec, group)
                if device is None or len(project_matches) != 1 or len(image_matches) != 1:
                    continue
                try:
                    positive_int(spec.get("gpu_count"), "gpu_count", 32)
                except PlatformError:
                    continue
                if config.get("group_ids") and group_id not in config["group_ids"]:
                    continue
                if config.get("quota_ids") and spec.get("quota_id") not in config["quota_ids"]:
                    continue
                project, image = project_matches[0], image_matches[0]
                result["selections"].append({"workspace_id": workspace_id,
                    "project_id": identifier(project, "project_id", "id"), "project_name": project.get("name"),
                    "logic_compute_group_id": group_id, "logic_compute_group_name": group.get("name"),
                    "npu_type": device, "image_id": identifier(image, "image_id", "id"),
                    "image": identifier(image, "address", "url"), "image_type": image.get("source"),
                    "image_visibility": image.get("visibility"), "resource_spec_price": copy.deepcopy(spec),
                    "priority": copy.deepcopy(config.get("priority")),
                    "allowed_priority_levels": priority_restriction(schedule, str(spec.get("quota_id", "")), group_id),
                    "discovered_at": result["at"]})
    result["evidence"] = public(client.evidence)
    return result


def validate_selection(selection):
    if selection.get("workspace_id") not in WORKSPACES:
        raise PlatformError("Selection is outside the two authorized Ascend workspaces")
    for key in ("project_id", "logic_compute_group_id", "image", "image_type", "resource_spec_price"):
        if not selection.get(key):
            raise PlatformError(f"Selection is missing {key}")
    if selection["image_type"] not in {"SOURCE_PRIVATE", "SOURCE_PUBLIC", "SOURCE_OFFICIAL"}:
        raise PlatformError("Unknown platform image source")
    spec = selection["resource_spec_price"]
    if not npu_type(spec) or not spec.get("quota_id"):
        raise PlatformError("Selection requires a discovered Ascend type and quota")
    if spec.get("logic_compute_group_id") not in (None, "", selection["logic_compute_group_id"]):
        raise PlatformError("Resource specification belongs to another group")
    for key in ("cpu_count", "gpu_count", "memory_size_gib"):
        positive_int(spec.get(key), key, 32 if key == "gpu_count" else None)
    priority = selection.get("priority") or {}
    if priority.get("verified") is not True or priority.get("level") != "LOW" or not priority.get("name"):
        raise PlatformError("An explicitly verified LOW priority mapping is required")
    positive_int(priority.get("task_priority"), "task_priority", 10)
    if priority["task_priority"] not in (1, 2, 3) or str(priority["name"]) != str(priority["task_priority"]):
        raise PlatformError("LOW priority requires an official task priority in 1..3 with the same name")
    excluded = selection.get("exclude_nodes", [])
    if (not isinstance(excluded, list)
            or any(not isinstance(node, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,253}", node)
                   for node in excluded) or len(excluded) != len(set(excluded))):
        raise PlatformError("exclude_nodes must contain unique node names")


def render_payload(selection, command, name, max_running_seconds=86400, retries=2):
    validate_selection(selection)
    if not isinstance(command, str) or not command.strip() or not re.fullmatch(r"[\w.\-\u4e00-\u9fa5]{1,255}", name):
        raise PlatformError("A nonempty command and stable job name are required")
    positive_int(max_running_seconds, "max_running_seconds", 86400)
    if type(retries) is not int or not 0 <= retries <= 10:
        raise PlatformError("fault retries must be an integer in 0..10")
    raw = selection["resource_spec_price"]
    cpu, cards, memory = (int(raw[key]) for key in ("cpu_count", "gpu_count", "memory_size_gib"))
    group_id = selection["logic_compute_group_id"]
    spec = {"cpu_type": raw.get("cpu_type") or (raw.get("cpu_info") or {}).get("cpu_type", ""),
            "cpu_count": cpu, "gpu_type": npu_type(raw), "gpu_count": cards,
            "memory_size_gib": memory, "logic_compute_group_id": group_id, "quota_id": raw["quota_id"]}
    component = {"image_type": selection["image_type"], "image": selection["image"],
                 "instance_count": 1, "resource_spec_price": spec,
                 "cpu": cpu, "gpu_count": cards, "mem_gi": memory}
    if selection.get("shm_gi") is not None:
        component["shm_gi"] = positive_int(selection["shm_gi"], "shm_gi")
    payload = {"workspace_id": selection["workspace_id"], "project_id": selection["project_id"],
            "logic_compute_group_id": group_id, "name": name, "framework": "pytorch", "command": command,
            "task_priority": selection["priority"]["task_priority"],
            "max_running_time_ms": str(max_running_seconds * 1000), "reserve_on_success_ms": "60000",
            "auto_fault_tolerance": bool(retries), "fault_tolerance_max_retry": retries,
            "enable_notification": False, "enable_slow_detect": False, "enable_vccl": False,
            "enable_troubleshoot": False, "framework_config": [component]}
    if selection.get("exclude_nodes"):
        payload["exclude_nodes"] = list(selection["exclude_nodes"])
    return payload


def validate_payload(payload, selection):
    """Compare the full contract to the selected inventory before any create."""
    expected = render_payload(selection, payload.get("command", ""), payload.get("name", ""),
        int(payload.get("max_running_time_ms", 0)) // 1000, payload.get("fault_tolerance_max_retry", 2))
    if payload != expected:
        raise PlatformError("Job request differs from the selected immutable Ascend contract")


def job_id(record):
    return record.get("job_id") or (record.get("response") or {}).get("data", {}).get("job_id")


def submit(client, payload, record_path, selection):
    validate_payload(payload, selection)
    record_path = Path(record_path)
    with writer_lock(record_path.parent / (record_path.stem + "-submit-lock")):
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record.get("request") != payload:
                raise PlatformError("Existing submission record describes another request")
            if job_id(record):
                return record
            raise PlatformError("Submission was already attempted; resolve its identity before posting again")
        record = {"schema_version": 1, "at": now_iso(), "status": "submitting", "request": payload,
                  "selection": copy.deepcopy(selection),
                  "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}
        write_json(record_path, record)
        try:
            response = client.request("POST", API["create"], payload)
        except Exception as error:
            rejected = isinstance(error, PlatformError) and (str(error).startswith("Platform API rejected ")
                or bool(re.match(r"HTTP 4(?!08)\d\d:", str(error))))
            record["status"] = "submission_rejected" if rejected else "submission_unknown"
            write_json(record_path, record)
            raise
        record.update(response=public(response), job_id=(response.get("data") or {}).get("job_id"))
        record["status"] = "submitted" if record["job_id"] else "submission_unknown"
        write_json(record_path, record)
        if not record["job_id"]:
            raise PlatformError("Create response has no job ID; the reservation remains unresolved")
        return record


def list_own_jobs(client, user_id, workspace_ids=WORKSPACES):
    if not user_id:
        raise PlatformError("An authenticated user ID is required for owned task discovery")
    result = []
    for workspace_id in workspace_ids:
        if workspace_id not in WORKSPACES:
            raise PlatformError("Job listing is outside the authorized Ascend workspaces")
        for page in range(1, 51):
            response = client.request("POST", API["list"], {"workspace_id": workspace_id,
                "created_by": user_id, "page_num": page, "page_size": 100, "scope": 1})
            batch = items(response, "jobs", "items", "list")
            if any(row.get("workspace_id", workspace_id) != workspace_id for row in batch):
                raise PlatformError("Job listing returned another workspace")
            result.extend(dict(row, workspace_id=workspace_id) for row in batch)
            data = response.get("data") or {}
            total = data.get("total")
            if len(batch) < 100 or (isinstance(total, (int, float)) and page * 100 >= total):
                break
        else:
            raise PlatformError("Job listing exceeded 5000 records in one workspace")
    return result


def resolve_submission(record, platform_jobs):
    request = record["request"]
    fields = ("name", "command", "workspace_id", "project_id", "logic_compute_group_id")
    matches = [row for row in platform_jobs if all(row.get(key) == request[key] for key in fields)]
    if len(matches) > 1:
        raise PlatformError("More than one official task matches one durable submission")
    return dict(record, job_id=matches[0]["job_id"], status="recovered_from_exact_list_match") if matches else record


def inspect_job(client, identifier):
    """Read detail/instances only. Priority enforcement belongs to the controller."""
    detail = client.request("POST", API["detail"], {"job_id": identifier})
    instances = client.request("POST", API["instances"],
                               {"jobId": identifier, "page_num": 1, "page_size": 100})
    return {"detail": public(detail.get("data") or {}), "instances": public(items(instances, "items", "instances", "list")),
            "instances_response": public(instances)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("discover", "render"))
    parser.add_argument("--credentials")
    parser.add_argument("--cookie-file")
    parser.add_argument("--workspace-config", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--command")
    parser.add_argument("--name")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "discover":
            configs = json.loads(args.workspace_config.read_text()) if args.workspace_config else []
            result = discover(make_client(args.credentials, args.cookie_file), configs)
        else:
            result = render_payload(json.loads(args.selection.read_text()), args.command, args.name)
        write_json(args.output, public(result))
        print(json.dumps({"output": str(args.output), "submitted": False,
                          "selections": len(result.get("selections", [])), "errors": len(result.get("errors", []))}))
        return 0
    except (PlatformError, OSError, ValueError, TypeError) as error:
        print(f"ascend_platform: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
