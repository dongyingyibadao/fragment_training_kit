"""Read-only evidence audit for completed native AdamW dynamic-refill runs.

Use --run-dir RUN_ROOT --output NEW_JSON. No model, Torch, device framework,
platform client or numerical oracle is loaded. Hash agreement verifies saved
bytes and recorded protocol evidence, not mathematical gradient correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fragment_training_kit  # Select only this checkout's bundled runtime.
from jade_lotus.refill_protocol import read_allocation, mapping_digest, validate_cancel, validate_permit


SCHEMA = "fragment-refill-evidence-audit-v1"
PRECISION = "native_model_and_native_adamw_moments_no_fp32_master"
WIDTHS = {"bool": 1, "uint8": 1, "int8": 1, "int16": 2, "int32": 4, "int64": 8,
          "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}
TERMINAL = {"job_succeeded", "job_failed", "job_stopped", "job_completed", "job_finished",
            "job_deleted", "job_cancelled", "job_canceled"}
BLOCK_BYTES = 4 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"{label}: expected integer >= {minimum}")
    return value


def finite(value, label, *, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0), f"{label}: expected finite number")
    return value


def component(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", value)
            and value not in (".", ".."), "Unsafe evidence identity/path component")
    return value


def sha_text(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Malformed SHA256")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON evidence key")
        result[key] = value
    return result


def parse(raw):
    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON evidence constant: {value}")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=nonfinite)


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}

    def path(self, value):
        path = Path(value)
        require(".." not in path.parts, "Evidence path contains traversal")
        path = path if path.is_absolute() else self.root / path
        require(not path.is_symlink(), "Evidence must be an owned file, not a symbolic link")
        resolved = path.resolve()
        require(resolved.is_relative_to(self.root), "Evidence escapes the supplied run root")
        return resolved

    def remember(self, path, sha):
        key = str(self.path(path).relative_to(self.root))
        require(key not in self.hashes or self.hashes[key] == sha, f"Evidence changed during audit: {key}")
        self.hashes[key] = sha

    def json(self, path):
        path = self.path(path)
        raw = path.read_bytes()
        self.remember(path, hashlib.sha256(raw).hexdigest())
        value = parse(raw)
        require(isinstance(value, dict), f"Expected JSON object: {path.name}")
        return value

    def hash_file(self, path):
        path = self.path(path)
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
                sha.update(block)
        self.remember(path, sha.hexdigest())
        return sha.hexdigest()


def identity(member):
    require(isinstance(member, dict), "Missing member identity")
    worker = component(member.get("worker_id"))
    require(re.fullmatch(r"w[0-9]{6}", worker) and worker != "w000000", "Invalid worker ID")
    return worker, component(member.get("session_id")), integer(member.get("pid"), "member PID", 1)


def identities(members):
    require(isinstance(members, list) and 1 <= len(members) <= 64, "Expected 1..64 exact members")
    values = {identity(member) for member in members}
    require(len({item[0] for item in values}) == len(members), "Duplicate member identity")
    return values


def _logical(value, tensors):
    if isinstance(value, dict):
        if set(value) == {"__native_tensor__"}:
            name = value["__native_tensor__"]
            entry = tensors[name]
            return {"tensor": name, **{key: entry[key] for key in ("dtype", "shape", "nbytes", "sha256")}}
        return {key: _logical(child, tensors) for key, child in value.items()}
    return [_logical(child, tensors) for child in value] if isinstance(value, list) else value


def _options(value):
    if isinstance(value, dict):
        require("__native_tensor__" not in value, "Tensor-valued optimizer options are outside this audit contract")
        if set(value) == {"__native_tuple__"}:
            return [_options(child) for child in value["__native_tuple__"]]
        return {key: _options(child) for key, child in value.items()}
    return [_options(child) for child in value] if isinstance(value, list) else value


def verify_checkpoint(evidence, checkpoint, spec, step):
    """Independently rehash manifest, every chunk/tensor, and the whole payload."""
    directory = evidence.path(checkpoint["path"])
    manifest = evidence.json(directory / "manifest.json")
    require(manifest.get("format") == "jade-native-resident-state-v1"
            and manifest.get("precision_contract") == PRECISION, "Unsupported native checkpoint format")
    require(manifest.get("manifest_sha256") == digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}),
            "Checkpoint manifest SHA256 mismatch")
    require(integer(manifest.get("step"), "checkpoint step") == step, "Checkpoint committed step differs")
    metadata = manifest.get("metadata", {})
    expected = {"run_id": spec["run_id"], "config_sha256": spec["config_sha256"], "boundary_step": step,
                "global_batch": 256, "activation_recomputation": False, "seed": spec["seed"],
                "rng_scheme": spec["rng_scheme"], "precision_contract": spec["precision_contract"]}
    require(all(metadata.get(k) == v and type(metadata.get(k)) is type(v) for k, v in expected.items()),
            "Checkpoint metadata/input identity differs")
    tensors = manifest.get("tensors")
    require(isinstance(tensors, dict) and tensors, "Checkpoint tensor catalog is absent")
    payload = evidence.path(directory / "state.bin")
    size = integer(manifest.get("payload_bytes"), "checkpoint bytes")
    require(payload.stat().st_size == size, "Checkpoint payload length differs")
    chunk_limit = integer(manifest.get("chunk_bytes"), "checkpoint chunk size", 8)
    cursor, whole = 0, hashlib.sha256()
    with payload.open("rb") as stream:
        for name, entry in sorted(tensors.items(), key=lambda item: (item[1]["offset"], item[1]["nbytes"] != 0)):
            shape, dtype = entry["shape"], entry["dtype"]
            require(isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape)
                    and dtype in WIDTHS and entry["device_role"] in ("cpu", "parameter_device"),
                    "Invalid checkpoint tensor shape/dtype/device")
            count = integer(entry["nbytes"], "tensor bytes")
            require(integer(entry["offset"], "tensor offset") == cursor and count == math.prod(shape) * WIDTHS[dtype],
                    "Checkpoint tensor offsets/lengths are not contiguous")
            leaf, consumed = hashlib.sha256(), 0
            require(isinstance(entry["chunks"], list), "Missing checkpoint chunks")
            for chunk in entry["chunks"]:
                length = integer(chunk["nbytes"], "chunk bytes", 1)
                require(length <= chunk_limit and length % WIDTHS[dtype] == 0
                        and integer(chunk["offset"], "chunk offset") == cursor and consumed + length <= count,
                        "Checkpoint chunk offsets/lengths differ")
                piece, remaining = hashlib.sha256(), length
                while remaining:
                    block = stream.read(min(BLOCK_BYTES, remaining))
                    require(bool(block), "Checkpoint payload is truncated")
                    piece.update(block); leaf.update(block); whole.update(block)
                    remaining -= len(block)
                require(piece.hexdigest() == sha_text(chunk["sha256"]), "Checkpoint chunk SHA256 mismatch")
                consumed += length
                cursor += length
            require(consumed == count and leaf.hexdigest() == sha_text(entry["sha256"]), "Checkpoint tensor SHA256 mismatch")
        require(not stream.read(1), "Checkpoint payload has trailing bytes")
    require(cursor == size and whole.hexdigest() == sha_text(manifest["payload_sha256"]), "Checkpoint payload SHA256 mismatch")
    evidence.remember(payload, whole.hexdigest())
    model_sha = digest({k: _logical(manifest[k], tensors) for k in ("model_parameters", "model_buffers")})
    optimizer_sha = digest({**{k: _logical(manifest[k], tensors) for k in
        ("optimizer_defaults", "optimizer_groups", "optimizer_states")}, "step": step, "precision": PRECISION})
    require((model_sha, optimizer_sha) == (manifest["model_sha256"], manifest["optimizer_sha256"]),
            "Checkpoint logical model/Adam SHA256 differs")
    options = {"groups": [_options(group["options"]) for group in manifest["optimizer_groups"]],
               "defaults": _options(manifest["optimizer_defaults"])}
    config_sha = hashlib.sha256(json.dumps(options, sort_keys=True, allow_nan=False).encode()).hexdigest()
    receipt = checkpoint["receipt"]
    require(receipt.get("payload_bytes_verified") is True and receipt.get("step") == step
            and all(receipt.get(k) == manifest[k] for k in ("model_sha256", "optimizer_sha256", "manifest_sha256", "payload_bytes")),
            "Checkpoint receipt disagrees with independently verified bytes")
    return {"path": str(directory), "step": step, "payload_bytes": size, "payload_sha256": whole.hexdigest(),
            "manifest_sha256": manifest["manifest_sha256"], "model_sha256": model_sha,
            "optimizer_sha256": optimizer_sha, "optimizer_config_sha256": config_sha,
            "payload_bytes_independently_verified": True, "max_read_block_bytes": BLOCK_BYTES}


def _report(evidence, spec, member, token, phase):
    worker, session, _ = identity(member)
    token = component(token)
    record = evidence.json(f"probe/reports/{worker}/{session}/{token}/{phase}.json")
    require(identity(record) == identity(member) and record.get("run_id") == spec["run_id"]
            and record.get("token") == token and record.get("phase") == phase, "Stale report identity/run/token/phase")
    return record


def _aligned_candidate(evidence, spec, member, event):
    worker, session, _ = identity(member)
    reports = evidence.path(f"probe/reports/{worker}/{session}")
    boundary = event["step"]
    matches = {}
    for phase in ("loaded", "inspected"):
        matches[phase] = []
        for path in reports.glob(f"*/{phase}.json"):
            value = evidence.json(path)
            if value.get("completed_step") == boundary:
                matches[phase].append(_report(evidence, spec, member, path.parent.name, phase))
    require(len(matches["loaded"]) == 1 and bool(matches["inspected"]),
            "Admission lacks exact candidate load/inspection reports at its boundary")
    loaded = matches["loaded"][0]["receipt"]
    authoritative = event["checkpoint"]["receipt"]
    require(all(loaded.get(k) is True for k in ("verified_loaded_state", "parameter_pointers_preserved", "input_identity_verified", "payload_bytes_verified"))
            and loaded.get("step") == boundary
            and evidence.path(loaded["checkpoint"]) == evidence.path(event["checkpoint"]["path"])
            and all(loaded.get(k) == authoritative[k] for k in ("manifest_sha256", "model_sha256", "optimizer_sha256", "payload_bytes")),
            "Candidate did not load its complete authoritative checkpoint")
    for inspected in matches["inspected"]:
        require(type(inspected.get("checkpoint_calls")) is int and inspected["checkpoint_calls"] == 0
                and [inspected.get(k) for k in ("model_sha256", "optimizer_sha256", "optimizer_config_sha256")] == event["state_hashes"],
                "Candidate inspection disagrees with admitted native model/Adam/configuration")


def _stats(values):
    return {"count": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None, "min": min(values) if values else None,
            "max": max(values) if values else None}


def audit_run(run_dir):
    evidence = Evidence(run_dir)
    root = evidence.root
    source = evidence.json("source_manifest.json")
    require(source.get("schema") == "fragment-training-kit-prepared-v1" and isinstance(source.get("files"), dict),
            "Missing frozen source manifest")
    files = source["files"]
    require({"training.json", "site.json", "runtime/scripts/probe_resident_openpi_fault.py"}.issubset(files),
            "Frozen source manifest omits required configuration/controller")
    actual = {str(p.relative_to(root)) for p in evidence.path("runtime").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    require(actual == {name for name in files if name.startswith("runtime/")}, "Frozen runtime file inventory differs")
    for name, expected in files.items():
        require(evidence.hash_file(name) == sha_text(expected), f"Frozen source/config SHA256 differs: {name}")
    config, site = evidence.json("training.json"), evidence.json("site.json")
    spec, summary = evidence.json("probe/spec.json"), evidence.json("probe/summary.json")
    require(spec.get("config_sha256") == evidence.hashes["training.json"]
            and spec.get("source_sha256") == files["runtime/scripts/probe_resident_openpi_fault.py"],
            "Controller configuration/source identity differs")
    require(evidence.path(spec["config"]) == root / "training.json" and evidence.path(spec["run_dir"]) == root / "probe",
            "Controller names a different run/config")
    # Native OpenPI and the generic adapter read different physical batch keys.
    # Resolve the same key/default as the selected runtime; an irrelevant alias
    # must never make a two-sample physical batch pass the four-sample check.
    custom_adapter = bool(config.get("resident_adapter"))
    microbatch = config.get("worker_microbatch_size" if custom_adapter else "microbatch_size",
                           4 if custom_adapter else 2)
    require(spec.get("pod_refill") is True and spec.get("pod_recovery") is True
            and config.get("resident_pod_refill") is True and config.get("disable_activation_checkpointing") is True
            and config.get("batch_size") == 256 and type(microbatch) is int and microbatch == 4,
            "Expected explicit refill, global256/atomic4 and disabled activation recomputation")
    steps, start = integer(spec.get("steps"), "target steps", 1), integer(spec.get("start_step", 0), "start step")
    require(start < steps and summary.get("status") == "passed" and summary.get("run_id") == spec["run_id"]
            and summary.get("final_states_identical") is True and summary.get("committed_steps") == steps
            and summary.get("executed_steps") == steps - start and summary.get("healthy_rollback_count") == 0,
            "Run lacks a complete passed/no-rollback summary")
    require(site.get("steps") == steps and 1 <= integer(site.get("cards"), "physical pool", 1) <= 64,
            "Site target/physical budget differs")
    commits = summary.get("commits")
    if summary.get("commit_journal"):
        journal = evidence.path(summary["commit_journal"])
        require(journal == root / "probe/commits.jsonl", "Commit journal belongs to another run")
        evidence.hash_file(journal)
        ledger = [parse(line) for line in journal.read_text().splitlines() if line.strip()]
        require(isinstance(commits, list) and commits and ledger[-len(commits):] == commits, "Summary/journal suffix differs")
        commits = ledger
    require(isinstance(commits, list) and len(commits) == steps - start, "A full commit ledger is required")
    require([c.get("step") for c in commits] == list(range(start + 1, steps + 1)), "Commit sequence is incomplete or duplicated")

    allocations, owners = {}, {}
    for path in sorted(evidence.path("refill/allocations").glob("*.json")):
        raw = evidence.json(path)
        mapping = read_allocation(root, path.stem, len(raw["members"]))
        require(mapping == raw, "Allocation changed during audit")
        allocations[path.stem] = mapping
        for member in mapping["members"]:
            require(member["worker_id"] not in owners, "Historical member identity was reused")
            owners[member["worker_id"]] = mapping
    initial_maps = [a for a in allocations.values() if a["role"] == "initial"]
    initial_workers = {m["worker_id"] for a in initial_maps for m in a["members"]}
    require(initial_workers == set(spec["initial_workers"]) and len(spec["initial_workers"]) == len(initial_workers)
            and len(initial_workers) == spec["cards"] == site["active_cards"]
            and [len(a["members"]) for a in initial_maps] == site["job_layout"], "Initial allocation/layout differs")
    initial_ids = identities(summary["initial_members"])
    require({m[0] for m in initial_ids} == initial_workers, "Initial exact identities differ from allocations")
    ledger = evidence.json("refill/state.json")
    require(ledger.get("run_dir") == str(root) and ledger.get("config_sha256") == spec["config_sha256"], "Refill ledger identity differs")
    batches = ledger.get("history", []) + ([ledger["current"]] if ledger.get("current") else [])
    require(len({b["batch_id"] for b in batches}) == len(batches), "Duplicate refill batch ledger")
    batches = {b["batch_id"]: b for b in batches}
    admitted, admissions, boundary_steps = {}, [], set()
    for event in summary.get("refill_events", []):
        boundary = integer(event.get("step"), "refill boundary")
        require(start <= boundary < steps, "Refill event is outside executed COMMIT boundaries")
        pause = finite(event.get("boundary_pause_seconds"), "admission pause", nonnegative=True)
        require(event.get("optimizer_updates_during_admission") == 0 and event.get("healthy_rollback_count") == 0,
                "Refill performed an optimizer update or healthy rollback")
        boundary_steps.add(boundary + 1)
        if event.get("status") == "rejected":
            require(event["batch_id"] in batches and batches[event["batch_id"]]["phase"] == "cancelled",
                    "Rejected refill lacks controller cancellation")
            continue
        require(event.get("status") == "admitted" and event.get("candidate_count") == 16
                and event.get("full_native_state_alignment_verified") is True, "Admission lacks exact16/full-state alignment")
        candidates = identities(event["candidates"])
        require(len(candidates) == 16, "Admission must contain exactly16 distinct candidates")
        component(event["batch_id"])
        require(evidence.json(f"refill/admissions/{event['batch_id']}.json") == event, "Summary and immutable admission differ")
        batch = batches.get(event["batch_id"], {})
        require(batch.get("phase") == "admitted" and batch.get("admission") == event, "Admission differs from controller ledger")
        batch_workers = {m["worker_id"] for a in allocations.values() if a["batch_id"] == event["batch_id"] for m in a["members"]}
        require(batch_workers == {m[0] for m in candidates}, "Admission differs from complete allocated batch")
        union = identities(event["members"])
        prior = initial_ids if boundary == start else identities(commits[boundary - start - 1]["members"])
        require(candidates.issubset(union) and union - candidates <= prior, "Admission includes an unknown healthy member")
        hashes = event.get("state_hashes", [])
        require(isinstance(hashes, list) and len(hashes) == 3 and all(sha_text(v) for v in hashes), "Missing aligned state hashes")
        receipt = event["checkpoint"]["receipt"]
        require(receipt.get("step") == boundary and receipt.get("payload_bytes_verified") is True
                and [receipt.get("model_sha256"), receipt.get("optimizer_sha256")] == hashes[:2], "Admission checkpoint differs from aligned state")
        for member in event["candidates"]:
            worker = member["worker_id"]
            require(worker not in admitted and worker not in initial_workers and owners.get(worker, {}).get("role") == "candidate",
                    "Admission repeats an existing member")
            _aligned_candidate(evidence, spec, member, event)
            admitted[worker] = (identity(member), boundary)
        admissions.append({"batch_id": event["batch_id"], "step": boundary, "cards": 16,
            "member_ids": sorted(batch_workers), "layout": [len(a["members"]) for a in allocations.values() if a["batch_id"] == event["batch_id"]],
            "boundary_pause_seconds": pause})

    ready_cache = {}
    def authorized(member, update):
        key = identity(member)
        worker = key[0]
        require(worker in owners, "Trained member has no immutable allocation")
        if worker in initial_workers:
            require(key in initial_ids, "Initial worker session changed")
        else:
            require(worker in admitted and admitted[worker][0] == key and update > admitted[worker][1],
                    "Candidate trained before admission or with another session")
        if worker not in ready_cache:
            ready = evidence.json(f"probe/ready/{worker}.json")
            mapping = owners[worker]
            slot = next(m for m in mapping["members"] if m["worker_id"] == worker)
            require(identity(ready) == key and ready.get("run_id") == spec["run_id"]
                    and ready.get("allocation_group") == mapping["worker_prefix"]
                    and ready.get("device_index") == slot["device_index"]
                    and ready.get("allocated_devices") == len(mapping["members"]), "READY differs from exact allocated device/session")
            if mapping["role"] == "candidate":
                prefix = mapping["worker_prefix"]
                reservation = evidence.json(f"refill/reservations/{prefix}.json")
                permit = evidence.json(f"refill/permits/{prefix}.json")
                require(validate_permit(mapping, reservation, permit) and ready.get("replacement") is True
                        and ready.get("refill_wrapper_session") == permit["wrapper_session"]
                        and ready.get("refill_batch_id") == mapping["batch_id"], "Candidate READY lacks its exact load permit")
            proof = ready.get("adapter_device")
            if proof is not None:
                require(proof.get("device") == f"npu:{slot['device_index']}" and proof.get("visible_devices") == len(mapping["members"])
                        and proof.get("resident_adapter") == config.get("resident_adapter")
                        and "Ascend" in proof.get("device_name", ""), "Adapter READY does not prove its assigned Ascend device")
            ready_cache[worker] = ready

    observed, used, authorizations = [], set(), {}
    previous = initial_ids
    recovery_steps = set()
    lost = set()
    for event in summary.get("pod_recovery_events", []):
        recovery_steps.add(integer(event["step"], "recovery base step") + 1)
        lost.update(identity(m) for m in event.get("lost", []))
    for commit in commits:
        step = integer(commit["step"], "commit step", 1)
        survivors, prepared = identities(commit["members"]), identities(commit["prepared_members"])
        require(survivors <= prepared, "Commit survivors differ from prepared identities")
        require(commit.get("samples") == 256 and commit.get("atomic_units") == 64 and commit.get("pod_recovery") is True,
                "Commit does not describe global256/64 atomic units")
        assignments = commit["assignments"]
        require(set(assignments) == {m[0] for m in prepared}, "Assignment owners differ from prepared members")
        units = [u for values in assignments.values() for u in values]
        require(all(type(u) is int for u in units) and sorted(units) == list(range(64)), "Atomic units are missing or duplicated")
        token, reduction = component(commit["token"]), component(commit["reduction_token"])
        require(token not in authorizations, "Optimizer update token was reused")
        authorization = evidence.json(f"probe/pod/authorizations/{token}.json")
        require(authorization.get("run_id") == spec["run_id"] and authorization.get("config_sha256") == spec["config_sha256"]
                and authorization.get("base_step") == step - 1 and authorization.get("update_token") == token
                and authorization.get("reduction_token") == reduction and authorization.get("root_token") == commit["root_token"]
                and authorization.get("assignments") == assignments and identities(authorization["prepared_members"]) == prepared
                and authorization.get("optimizer_replay_forbidden") is True, "APPLY authorization differs from committed update")
        generation = integer(authorization["group"]["generation"], "reduction generation")
        require(identities(authorization["group"]["members"]) == prepared
                and integer(commit["generation"], "commit generation") >= generation, "Prepared group generation differs")
        authorizations[token] = (step, prepared, generation, commit)
        losses, norms, timings = [], [], {}
        for member in commit["prepared_members"]:
            authorized(member, step)
            used.add(member["worker_id"])
            reduced = _report(evidence, spec, member, reduction, "pod_reduced")
            local = reduced["local"]
            require(reduced.get("base_step") == step - 1 and reduced.get("generation") == generation
                    and reduced.get("status") == "reduced" and sorted(local["units"]) == sorted(assignments[member["worker_id"]])
                    and local.get("samples") == 4 * len(local["units"]), "Reduced contribution identity/coverage differs")
            losses.append(finite(local["loss_contribution"], "local loss"))
        for member in commit["members"]:
            record = _report(evidence, spec, member, token, "updated")
            require(record.get("completed_step") == step and type(record.get("checkpoint_calls")) is int
                    and record["checkpoint_calls"] == 0 and type(record.get("optimizer_applications")) is int
                    and record["optimizer_applications"] == 1 and record.get("root_token") == commit["root_token"]
                    and record.get("reduction_token") == reduction and record.get("generation") == generation
                    and sorted(record["local"]["units"]) == sorted(assignments[member["worker_id"]]),
                    "Updated report violates exact once-Adam/reduction/no-recompute contract")
            norms.append(finite(record["grad_norm"], "gradient norm", nonnegative=True))
            for name, value in record.get("timings", {}).items():
                timings.setdefault(name, []).append(finite(value, f"worker timing {name}", nonnegative=True))
        require(all(math.isclose(n, norms[0], rel_tol=2e-4, abs_tol=2e-5) for n in norms), "Survivor reduced gradient norms disagree")
        if survivors != prepared or prepared != previous or commit.get("forward_recovery"):
            recovery_steps.add(step)
        previous = survivors
        observed.append({"step": step, "loss": finite(math.fsum(losses), "global loss"), "gradient_norm": norms[0],
            "members": len(survivors), "prepared_members": len(prepared), "member_ids": sorted(m[0] for m in survivors),
            "generation": commit["generation"],
            "commit_seconds": finite(commit["commit_seconds"], "commit duration", nonnegative=True),
            "commit_interval_seconds": finite(commit["commit_interval_seconds"], "commit interval", nonnegative=True),
            "worker_timings_max_seconds": {name: max(values) for name, values in timings.items()}})

    # A failed/hidden candidate computation must not be laundered as an unused
    # cancelled loader merely because its reports were omitted from summary.
    seen_updates = set()
    for phase in ("pod_local", "pod_reduced", "updated"):
        for path in evidence.path("probe/reports").glob(f"*/*/*/{phase}.json"):
            record = evidence.json(path)
            require(record.get("phase") == phase and record.get("run_id") == spec["run_id"]
                    and identity(record)[:2] == (path.parents[2].name, path.parents[1].name)
                    and record.get("token") == path.parent.name, "Uncommitted training report has stale identity")
            base = record.get("base_step") if phase == "pod_reduced" else record.get("completed_step")
            update = integer(base, "report step") + (0 if phase == "updated" else 1)
            require(start < update <= steps, "Training report lies outside audited updates")
            authorized(record, update)
            used.add(record["worker_id"])
            if phase == "updated":
                entry = authorizations.get(record["token"])
                require(entry is not None and entry[0] == update and identity(record) in entry[1], "Unauthorized optimizer update report")
                key = identity(record), update
                require(key not in seen_updates, "A replica applied the same optimizer update twice")
                seen_updates.add(key)
                require(type(record.get("optimizer_applications")) is int and record["optimizer_applications"] == 1
                        and type(record.get("checkpoint_calls")) is int and record["checkpoint_calls"] == 0
                        and record.get("reduction_token") == entry[3]["reduction_token"]
                        and record.get("root_token") == entry[3]["root_token"] and record.get("generation") == entry[2],
                        "Additional updated report violates its APPLY authorization")
                finite(record.get("grad_norm"), "additional gradient norm", nonnegative=True)

    ignored = []
    for directory in ("errors", "exits"):
        for path in evidence.path(f"probe/{directory}").glob("*.json"):
            record = evidence.json(path)
            key = identity(record)
            require(record.get("run_id") == spec["run_id"], "Error/exit belongs to another run")
            if directory == "exits" and record.get("exit_code") == 0:
                continue
            if directory == "exits" and key[0] in used and key in lost and record.get("phase_timed_out") is not True:
                continue  # A fenced active preemption is distinct from a device execution error.
            require(key[0] not in used and owners.get(key[0], {}).get("role") == "candidate",
                    "A trained/initial worker reported an execution error or unfenced exit")
            mapping = owners[key[0]]
            cancellation = evidence.json(f"refill/cancelled/{mapping['worker_prefix']}.json")
            batch = batches.get(mapping["batch_id"], {})
            require(validate_cancel(mapping, cancellation) and batch.get("phase") == "cancelled"
                    and mapping["worker_prefix"] in batch.get("allocations", []), "Unused candidate failure lacks matching controller cancellation")
            ignored.append(str(path.relative_to(root)))

    final_members = summary["final_member_identities"]
    final_ids = identities(final_members)
    require(final_ids <= identities(commits[-1]["members"]) and summary.get("final_members") == len(final_ids), "Final members differ from last COMMIT")
    checkpoint = verify_checkpoint(evidence, summary["final_checkpoint"], spec, steps)
    for member in final_members:
        worker, session, _ = identity(member)
        finals = list(evidence.path(f"probe/reports/{worker}/{session}").glob("*/finished.json"))
        require(len(finals) == 1, "Expected one finished report per final exact member")
        record = _report(evidence, spec, member, finals[0].parent.name, "finished")
        require(record.get("completed_step") == steps and type(record.get("checkpoint_calls")) is int and record["checkpoint_calls"] == 0
                and all(record.get(k) == checkpoint[k] for k in ("model_sha256", "optimizer_sha256", "optimizer_config_sha256")),
                "Final worker model/Adam/config differs from verified checkpoint")

    fleet, cleanup = evidence.json("ascend_fleet/state.json"), evidence.json("cleanup_receipt.json")
    require(fleet.get("run_dir") == str(root) and isinstance(fleet.get("jobs"), list), "Platform fleet belongs to another run")
    require(len(fleet["jobs"]) == len(allocations) and {job["worker_id"] for job in fleet["jobs"]} == set(allocations),
            "Platform history and allocation maps differ")
    for job in fleet["jobs"]:
        mapping = allocations[job["worker_id"]]
        require(job.get("member_ids") == [m["worker_id"] for m in mapping["members"]]
                and job.get("intent_id") == mapping["intent_id"], "Platform intent differs from immutable members")
        require(job.get("official_terminal") is True and job.get("status") in TERMINAL and job.get("job_id"),
                "Historical platform job lacks an official terminal state")
    require(cleanup.get("resources_released") is True and type(cleanup.get("active_jobs")) is int
            and cleanup["active_jobs"] == 0 and type(cleanup.get("local_requested_reserved_cards")) is int
            and cleanup["local_requested_reserved_cards"] == 0, "Cleanup has not confirmed zero owned resources")
    excluded = {start + 1, start + 2} | recovery_steps | boundary_steps
    steady = {}
    for count in sorted({48, 64} | {row["members"] for row in observed}):
        rows = [r for r in observed if r["members"] == count and r["step"] not in excluded]
        steady[str(count)] = {"steps": [r["step"] for r in rows],
            "commit_seconds": _stats([r["commit_seconds"] for r in rows]),
            "commit_interval_seconds": _stats([r["commit_interval_seconds"] for r in rows]),
            "communication_seconds_rank_max": _stats([r["worker_timings_max_seconds"]["communication_seconds"]
                for r in rows if "communication_seconds" in r["worker_timings_max_seconds"]])}
    index_sha = digest(evidence.hashes)
    final_layout = []
    for prefix, allocation in allocations.items():
        active = [m[0] for m in final_ids if owners[m[0]]["worker_prefix"] == prefix]
        if active:
            final_layout.append({"worker_prefix": prefix, "allocated_cards": len(allocation["members"]),
                                 "final_active_members": len(active), "member_ids": sorted(active)})
    elapsed = None
    if "training_started_at" in summary and "completed_at" in summary:
        elapsed = finite(finite(summary["completed_at"], "completion time")
                         - finite(summary["training_started_at"], "training start time"), "training elapsed time", nonnegative=True)
    return {"schema": SCHEMA, "status": "passed", "run_dir": str(root), "run_id": spec["run_id"],
        "auditor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Read-only saved native dynamic-refill evidence and checkpoint-byte audit; no numerical oracle",
        "model_or_device_framework_imported": False, "numerical_gradient_oracle_performed": False,
        "start_step": start, "completed_step": steps, "updates_audited": len(observed), "global_batch": 256, "atomic_units": 64,
        "initial_cards": len(initial_workers), "final_cards": len(final_ids), "physical_pool_limit": site["cards"],
        "initial_job_layout": site["job_layout"], "final_active_allocation_layout": final_layout,
        "historical_allocated_member_count": len(owners), "training_elapsed_seconds": elapsed,
        "first_update_loss": observed[0]["loss"], "last_update_loss": observed[-1]["loss"],
        "admissions": admissions, "admission_pause_seconds_total": sum(a["boundary_pause_seconds"] for a in admissions),
        "steps": observed, "steady_by_cards": steady, "steady_excluded_steps": sorted(excluded & {r["step"] for r in observed}),
        "steady_excluded_recovery_steps": sorted(recovery_steps), "steady_excluded_after_boundary_steps": sorted(boundary_steps),
        "timing_scope": "Exclude first two updates, membership/recovery updates and the first interval after every admission/rejection. Worker communication includes peer arrival and stream dependencies, not pure network time.",
        "checkpoint": checkpoint, "all_final_state_hashes_identical": True,
        "unused_cancelled_candidate_failures": ignored, "platform_jobs_officially_terminal": len(fleet["jobs"]),
        "resources_released": True, "evidence_file_count": len(evidence.hashes),
        "evidence_index_sha256": index_sha, "evidence_sha256": evidence.hashes,
        "limitations": ["Recorded hashes/protocol receipts are not a signed attestation or an independent gradient oracle.",
            "No model is loaded and no task-quality evaluation or checkpoint inference is performed.",
            "Payload bytes are rehashed; tensor numerical values and Adam mathematics are not recomputed.",
            "Timing describes this saved run and may include shared-filesystem/control overhead; it is not pure HCCL bandwidth.",
            "Unused candidate READY is optional; malformed evidence and nonterminal historical platform jobs are rejected."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Audit output already exists: {args.output}")
    result = audit_run(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "updates_audited": result["updates_audited"],
        "initial_cards": result["initial_cards"], "final_cards": result["final_cards"],
        "evidence_index_sha256": result["evidence_index_sha256"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
