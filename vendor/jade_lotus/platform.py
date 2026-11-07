"""Read-only Inspire discovery and explicit rendering of preemptible train jobs.

HTTP and OAuth use the Python standard library. Reading an Inspire Code ``ic1:``
credential additionally needs Node's built-in crypto module; no package install
is needed. Tokens are only read from the explicitly supplied private file.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_WORKSPACE = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
DEFAULT_BACKEND = "https://qz.sii.edu.cn"
DEFAULT_KEYCLOAK = "https://keycloak-inspire-prod.sii.edu.cn"
SECRET_KEYS = {"access_token", "refresh_token", "id_token", "device_code", "client_secret", "password", "authorization"}
GPU_PATTERN = re.compile(r"(?<![A-Z0-9])(H100|H200)(?![A-Z0-9])", re.I)


class PlatformError(RuntimeError):
    """A diagnostic that never includes an HTTP response or a credential value."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A login redirect must not turn into a successful HTML API response.
        return None


def sanitized(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("[redacted]" if k.lower() in SECRET_KEYS else sanitized(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitized(v) for v in value]
    return value


def write_json(path: Path | str, value: Any, private: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o600 if private else 0o644)
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def json_http(url: str, method: str, headers: dict, body: bytes | None = None) -> dict:
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise PlatformError("HTTP 401: platform authentication expired or is missing; complete official login again.") from None
        raise PlatformError(f"HTTP {exc.code}: {method} {urllib.parse.urlparse(url).path}") from None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError):
        raise PlatformError(f"Invalid or unavailable API response: {method} {urllib.parse.urlparse(url).path}") from None
    if not isinstance(result, dict):
        raise PlatformError("API returned a non-object JSON response.")
    return result


_CRYPTO_JS = r"""
const crypto = require('node:crypto');
const os = require('node:os');
let input = '';
process.stdin.on('data', b => { input += b; });
process.stdin.on('end', () => {
  try {
    const data = JSON.parse(input);
    const key = crypto.pbkdf2Sync(os.hostname() + os.homedir(), Buffer.from('inspire-opencode-v1'), 10000, 32, 'sha256');
    if (data.mode === 'decrypt') {
      if (!data.value.startsWith('ic1:')) throw new Error('format');
      const raw = Buffer.from(data.value.slice(4), 'base64');
      const decipher = crypto.createDecipheriv('aes-256-gcm', key, raw.subarray(0, 12));
      decipher.setAuthTag(raw.subarray(12, 28));
      process.stdout.write(Buffer.concat([decipher.update(raw.subarray(28)), decipher.final()]).toString('utf8'));
    } else {
      const iv = crypto.randomBytes(12);
      const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
      const ct = Buffer.concat([cipher.update(data.value, 'utf8'), cipher.final()]);
      process.stdout.write('ic1:' + Buffer.concat([iv, cipher.getAuthTag(), ct]).toString('base64'));
    }
  } catch { process.exitCode = 1; }
});
"""


def official_crypto(mode: str, value: str) -> str:
    try:
        process = subprocess.run(["node", "-e", _CRYPTO_JS], input=json.dumps({"mode": mode, "value": value}), text=True, capture_output=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        raise PlatformError("Official encrypted credentials need Node with built-in crypto on the original host/user.") from None
    if process.returncode or not process.stdout:
        raise PlatformError("Cannot decrypt official credentials on this host/user; complete official login again.")
    return process.stdout


def timestamp(value: Any) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        try:
            return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0


class OAuthCredentials:
    def __init__(self, path: str | Path, keycloak_url: str = DEFAULT_KEYCLOAK, client_id: str = "inspire-code"):
        self.path = Path(path).expanduser()
        try:
            if stat.S_IMODE(self.path.stat().st_mode) & 0o077:
                raise PlatformError("Credential JSON must be private (chmod 600).")
            self.data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            raise PlatformError("Cannot read the specified private credential JSON.") from None
        if not isinstance(self.data, dict):
            raise PlatformError("Credential JSON must be an object.")
        self.keycloak_url = keycloak_url.rstrip("/")
        self.client_id = self.data.get("client_id") or client_id
        self.official = self.data.get("version") == 2 and self.data.get("type") == "refresh_token"
        self._access_token = self.data.get("access_token", "")
        self._expires_at = timestamp(self.data.get("expires_at"))
        if not self._expires_at:
            self._expires_at = timestamp(self.data.get("obtained_at")) + timestamp(self.data.get("expires_in"))
        self._lock = threading.Lock()

    def token(self, force_refresh: bool = False) -> str:
        with self._lock:
            if not force_refresh and self._access_token and self._expires_at > time.time() + 30:
                return self._access_token
            refresh = self.data.get("refresh_token")
            if not refresh:
                raise PlatformError("Credential JSON has no usable access/refresh token; complete official login again.")
            if self.official:
                refresh = official_crypto("decrypt", refresh)
            form = {"grant_type": "refresh_token", "client_id": self.client_id, "refresh_token": refresh}
            if self.data.get("client_secret"):
                form["client_secret"] = self.data["client_secret"]
            tokens = json_http(self.keycloak_url + "/realms/inf-internal/protocol/openid-connect/token", "POST", {"Content-Type": "application/x-www-form-urlencoded", "inspire-api": "true"}, urllib.parse.urlencode(form).encode())
            if not tokens.get("access_token"):
                raise PlatformError("OAuth refresh failed; complete official login again.")
            self._access_token = tokens["access_token"]
            self._expires_at = time.time() + float(tokens.get("expires_in", 0))
            if self.official:
                if tokens.get("refresh_token"):
                    self.data["refresh_token"] = official_crypto("encrypt", tokens["refresh_token"])
            else:
                self.data.update(tokens)
                self.data["obtained_at"] = time.time()
                self.data["expires_at"] = self._expires_at
            write_json(self.path, self.data, private=True)
            return self._access_token


class InspireClient:
    def __init__(self, credentials: OAuthCredentials, backend_url: str = DEFAULT_BACKEND):
        if urllib.parse.urlparse(backend_url).scheme != "https":
            raise PlatformError("The backend URL must use HTTPS.")
        self.credentials = credentials
        self.backend_url = backend_url.rstrip("/")
        self.evidence: list[dict] = []
        self._last_request = 0.0

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        if not path.startswith("/api/v1/"):
            raise PlatformError("Only the documented platform API prefix is allowed.")
        # Match the official client's 3 QPS budget. Discovery is bounded and sequential.
        delay = 1 / 3 - (time.monotonic() - self._last_request)
        if delay > 0:
            time.sleep(delay)
        self._last_request = time.monotonic()
        headers = {"Authorization": "Bearer " + self.credentials.token(), "Content-Type": "application/json", "inspire-api": "true", "X-Inspire-Client-Source": "jade-lotus"}
        result = json_http(self.backend_url + path, method, headers, json.dumps(body).encode() if body is not None else None)
        # Do not silently treat a 401 as a fresh empty inventory or repeat a write.
        self.evidence.append({"at": dt.datetime.now(dt.timezone.utc).isoformat(), "method": method, "path": path, "request": sanitized(body), "response": sanitized(result)})
        if result.get("code") not in (0, "0"):
            raise PlatformError(f"Platform API rejected {method} {path} (code={result.get('code')}).")
        return result


def items(response: dict, *keys: str) -> list[dict]:
    data = response.get("data", {})
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in keys or ("items", "list"):
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, dict)]
    return []


def identifier(item: dict, *keys: str) -> str:
    return str(next((item[k] for k in keys if item.get(k)), ""))


def compact_name(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", value).casefold()


def gpu_model(spec: dict, group: dict | None = None) -> str | None:
    descriptors = " ".join(str(spec.get(k, "")) for k in ("gpu_type", "gpu_type_display", "gpu_product_simple"))
    if isinstance(spec.get("gpu_info"), dict):
        descriptors += " " + json.dumps(spec["gpu_info"])
    matches = set(x.upper() for x in GPU_PATTERN.findall(descriptors))
    if len(matches) == 1:
        return matches.pop()
    if matches or not group:
        return None
    # An opaque gpu_type may be resolved only by an exact dictionary match.
    raw_type = spec.get("gpu_type")
    for entry in group.get("gpu_type_stats", []):
        info = entry.get("gpu_info", {})
        if raw_type and info.get("gpu_type") == raw_type:
            return gpu_model(info)
    return None


def prices(spec: dict) -> dict:
    # Empty/absent price is unknown. A zero price alone does not prove a free tier.
    return {k: v for k, v in spec.items() if ("price" in k.lower() or "coupon" in k.lower()) and v not in (None, "")}


def resource_summary(response: dict) -> dict:
    data = response.get("data", {})
    raw = data.get("logic_resouces", data.get("logic_resources", {})) if isinstance(data, dict) else {}
    out = copy.deepcopy(raw)
    for resource, total, used in [("gpu", "gpu_total", "gpu_used"), ("cpu", "cpu_total", "cpu_used"), ("memory_gi", "memory_gi_total", "memory_gi_used")]:
        try:
            out[resource + "_available"] = float(raw[total]) - float(raw[used])
        except (KeyError, ValueError, TypeError):
            out[resource + "_available"] = None
    return out


def priority_restriction(schedule: dict, quota_id: str, group_id: str) -> list[str] | None:
    """Expose the server restriction without guessing which priority it refers to."""
    configured = schedule.get("predef_train_spec", [])
    if isinstance(configured, str):
        try:
            configured = json.loads(configured)
        except json.JSONDecodeError:
            return None
    for entry in configured if isinstance(configured, list) else []:
        if str(entry.get("id", "")) == quota_id and group_id in entry.get("logic_compute_group_ids", []):
            return entry.get("allowed_priority_levels")
    return None


def discover(client: InspireClient, workspace_id: str = DEFAULT_WORKSPACE, project_name: str = "具身智能 具身决策", image_query: str = "261009") -> dict:
    permissions = client.request("GET", f"/api/v1/user/permissions/{workspace_id}")
    user = client.request("GET", "/api/v1/user/detail")
    projects_response = client.request("POST", "/api/v1/project/list", {"page": 1, "page_size": -1, "filter": {"workspace_id": workspace_id, "check_admin": True}})
    projects = items(projects_response)
    project_matches = [p for p in projects if compact_name(p.get("name", p.get("project_name", ""))) == compact_name(project_name)]
    # The server's name filter does not match versions reliably (verified for 261009).
    images_response = client.request("POST", "/api/v1/image/list", {"page": 0, "page_size": -1, "filter": {"visibility": "VISIBILITY_PRIVATE", "source_list": ["SOURCE_PRIVATE", "SOURCE_PUBLIC"], "registry_hint": {"workspace_id": workspace_id}}})
    image_matches = [im for im in items(images_response, "images", "list", "items") if (str(im.get("version", "")) == image_query or str(im.get("name", "")) == image_query or str(im.get("name", "")).endswith(":" + image_query)) and im.get("status") == "SUCCESS" and im.get("visibility") == "VISIBILITY_PRIVATE"]
    group_response = client.request("POST", "/api/v1/logic_compute_groups/list", {"pageNum": 1, "pageSize": 10000, "filter": {"workspace_id": workspace_id, "support_job_type": "distributed_training", "include_gpu_type_stats": True}})
    groups = items(group_response, "logic_compute_groups", "list", "items")
    schedule = client.request("GET", f"/api/v1/notebook/schedule/{workspace_id}")
    project_detail = None
    if len(project_matches) == 1:
        project_detail = client.request("GET", "/api/v1/project/" + identifier(project_matches[0], "project_id", "id"))
    summaries, selections, errors = [], [], []
    for group in groups:
        group_id = identifier(group, "logic_compute_group_id", "id")
        if not group_id:
            continue
        try:
            resource = client.request("GET", "/api/v1/compute_resources/logic_compute_groups/" + group_id)
            specs_response = client.request("POST", "/api/v1/resource_prices/logic_compute_groups", {"workspace_id": workspace_id, "logic_compute_group_id": group_id, "schedule_config_type": "SCHEDULE_CONFIG_TYPE_TRAIN"})
            specs = items(specs_response, "lcg_resource_spec_prices", "resource_spec_prices", "lcgResourceSpecPrices", "items", "list")
        except PlatformError as exc:
            errors.append({"logic_compute_group_id": group_id, "error": str(exc)})
            continue
        summary = {"id": group_id, "name": group.get("name"), "gpu_type_stats": group.get("gpu_type_stats", []), "storage_disks": group.get("storage_disks", []), "resources": resource_summary(resource), "specs": specs}
        summaries.append(summary)
        for spec in specs:
            try:
                small_gpu_spec = int(spec.get("gpu_count", 0)) in (1, 2)
            except (TypeError, ValueError):
                small_gpu_spec = False
            model = gpu_model(spec, group)
            if not small_gpu_spec or model not in {"H100", "H200"} or len(project_matches) != 1 or len(image_matches) != 1:
                continue
            image = image_matches[0]
            selections.append({"workspace_id": workspace_id, "project_id": identifier(project_matches[0], "project_id", "id"), "project_name": project_matches[0].get("name"), "project_priority_level": project_matches[0].get("priority_level"), "project_priority_name": project_matches[0].get("priority_name"), "logic_compute_group_id": group_id, "logic_compute_group_name": group.get("name"), "gpu_model": model, "image_id": identifier(image, "image_id", "id"), "image": identifier(image, "address", "url"), "image_type": image.get("source"), "image_visibility": image.get("visibility"), "resource_spec_price": copy.deepcopy(spec), "allowed_priority_levels": priority_restriction(schedule.get("data", {}), str(spec.get("quota_id", "")), group_id), "priority_restriction_semantics_verified": False, "observed_prices": prices(spec), "free_priority_verified": False})
    return {"schema_version": 1, "at": dt.datetime.now(dt.timezone.utc).isoformat(), "workspace_id": workspace_id, "user": {k: user.get("data", {}).get(k) for k in ("id", "name")}, "permissions": permissions.get("data"), "project_matches": project_matches, "project_detail": project_detail, "image_matches": image_matches, "is_fair_workspace": group_response.get("data", {}).get("is_fair_workspace"), "schedule": schedule.get("data"), "groups": summaries, "selections": selections, "errors": errors, "policy": {"minimum_priority": 1, "allowed_gpu_models": ["H100", "H200"], "free_priority_verified": False, "same_priority_fifo_verified": False, "preemption_grace_period_verified": False}, "evidence": client.evidence}


def render_payload(selection: dict, command: str, name: str, max_running_seconds: int = 1800, retries: int = 3) -> dict:
    if not command.strip() or not name or not re.fullmatch(r"[\w.\-\u4e00-\u9fa5]{1,255}", name):
        raise PlatformError("A non-empty command and a valid job name are required.")
    if max_running_seconds <= 0 or not 1 <= retries <= 2000:
        raise PlatformError("Runtime must be positive; retry count must be 1..2000.")
    required = ("workspace_id", "project_id", "logic_compute_group_id", "image", "image_type", "resource_spec_price")
    if any(not selection.get(k) for k in required):
        raise PlatformError("Selection lacks explicit project, image, group or specification identifiers.")
    spec = copy.deepcopy(selection["resource_spec_price"])
    if not spec.get("quota_id") or gpu_model(spec) not in {"H100", "H200"}:
        raise PlatformError("Selection must contain a discovered H100/H200 GPU type and quota_id.")
    try:
        cpu, gpu, memory = float(spec["cpu_count"]), float(spec["gpu_count"]), float(spec["memory_size_gib"])
    except (KeyError, ValueError, TypeError):
        raise PlatformError("The selected resource specification is incomplete.") from None
    if not all(math.isfinite(x) for x in (cpu, gpu, memory)) or cpu <= 0 or gpu not in (1, 2) or memory <= 0:
        raise PlatformError("Only positive CPU/memory specifications with one or two H100/H200 GPUs are supported.")
    # Preserve integral resource counts as JSON integers for Go/protobuf fields.
    cpu = int(cpu) if cpu.is_integer() else cpu
    memory = int(memory) if memory.is_integer() else memory
    gpu = int(gpu)
    if selection["image_type"] not in {"SOURCE_PRIVATE", "SOURCE_PUBLIC", "SOURCE_OFFICIAL"}:
        raise PlatformError("Invalid image source type.")
    if selection.get("image_visibility") != "VISIBILITY_PRIVATE":
        raise PlatformError("The requested image must be personally visible.")
    group_id = selection["logic_compute_group_id"]
    if spec.get("logic_compute_group_id") not in (None, "", group_id):
        raise PlatformError("Specification belongs to a different compute group.")
    gpu_type = spec.get("gpu_type") or spec.get("gpu_info", {}).get("gpu_type", "")
    cpu_type = spec.get("cpu_type") or spec.get("cpu_info", {}).get("cpu_type", "")
    # The pricing API returns richer objects than the strict create protobuf
    # accepts (including cpu_info/gpu_info and price/version fields).
    spec = {
        "cpu_type": cpu_type,
        "cpu_count": cpu,
        "gpu_type": gpu_type,
        "gpu_count": gpu,
        "memory_size_gib": memory,
        "logic_compute_group_id": group_id,
        "quota_id": spec["quota_id"],
    }
    return {"workspace_id": selection["workspace_id"], "project_id": selection["project_id"], "logic_compute_group_id": group_id, "name": name, "framework": "pytorch", "command": command, "task_priority": 1, "max_running_time_ms": str(max_running_seconds * 1000), "reserve_on_success_ms": "60000", "auto_fault_tolerance": True, "fault_tolerance_max_retry": retries, "enable_notification": False, "enable_slow_detect": False, "enable_vccl": False, "enable_troubleshoot": False, "framework_config": [{"image_type": selection["image_type"], "image": selection["image"], "instance_count": 1, "resource_spec_price": spec, "cpu": cpu, "gpu_count": gpu, "mem_gi": memory}]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    discovery = sub.add_parser("discover", help="Read API inventory; never creates a job")
    discovery.add_argument("--credentials", required=True, help="Private OAuth or official Inspire Code JSON")
    discovery.add_argument("--backend-url", default=DEFAULT_BACKEND)
    discovery.add_argument("--keycloak-url", default=DEFAULT_KEYCLOAK)
    discovery.add_argument("--workspace-id", default=DEFAULT_WORKSPACE)
    discovery.add_argument("--project-name", default="具身智能 具身决策")
    discovery.add_argument("--image", default="261009")
    discovery.add_argument("--output", required=True)
    render = sub.add_parser("render", help="Render a selected job request locally; never submits")
    render.add_argument("--selection", required=True, help="A single object from discovery.selections")
    render.add_argument("--command", required=True)
    render.add_argument("--name", required=True)
    render.add_argument("--max-running-seconds", type=int, default=1800)
    render.add_argument("--retries", type=int, default=3)
    render.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "discover":
            client = InspireClient(OAuthCredentials(args.credentials, args.keycloak_url), args.backend_url)
            result = discover(client, args.workspace_id, args.project_name, args.image)
            write_json(args.output, result)
            print(json.dumps({"output": args.output, "groups": len(result["groups"]), "selections": len(result["selections"]), "errors": len(result["errors"]), "project_matches": len(result["project_matches"]), "image_matches": len(result["image_matches"])}, ensure_ascii=False))
        else:
            selection = json.loads(Path(args.selection).read_text())
            result = render_payload(selection, args.command, args.name, args.max_running_seconds, args.retries)
            write_json(args.output, result)
            print(json.dumps({"output": args.output, "task_priority": 1, "instance_count": 1, "submitted": False}))
        return 0
    except (PlatformError, OSError, json.JSONDecodeError) as exc:
        print(f"platform: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
