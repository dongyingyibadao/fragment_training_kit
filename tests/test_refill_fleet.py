"""Local official-API doubles exercise refill identity and physical accounting.

The fake implements only explicit QZ endpoints and never uses credentials,
network clients, devices, subprocesses or the original experiment repository.
"""
import copy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from fragment_training_kit.tests.test_topology_placement import inventory, node, selection
from jade_lotus import ascend_fleet
from jade_lotus.ascend_platform import API, RUNNING, render_payload
from jade_lotus.refill_fleet import RefillFleet
from jade_lotus.refill_controller import RefillController
from jade_lotus.refill_protocol import (ALLOCATION_SCHEMA, RESERVATION_SCHEMA, PERMIT_SCHEMA,
    IDENTITY_FIELDS, atomic_write, mapping_digest, read_allocation, validate_permit)
from jade_lotus.refill_state import cancel, load_state, save_state
from jade_lotus.runtime import writer_lock


COMMAND = "python fixture.py --run-dir {run_dir} --worker-prefix {worker_id} --devices {npu_count}"


def read(path):
    return json.loads(Path(path).read_text())


class OfficialClient:
    """Keep remote state separate from the local intent and submission ledgers."""
    def __init__(self, run):
        self.run = run
        self.jobs, self.calls, self.creates, self.stops = {}, [], [], []
        self.hidden_jobs, self.allocated_overrides = set(), {}
        self.unknown_next = False
        self.new_priority = "LOW"
        self.evidence = []

    def add(self, payload, identifier, *, status="job_running", priority="LOW"):
        self.jobs[identifier] = {**copy.deepcopy(payload), "job_id": identifier, "status": status,
            "priority_level": priority, "priority_name": "1", "current_running_round": 1}

    def request(self, method, path, body=None):
        assert method == "POST", "Only the recorded official POST endpoints are supported"
        self.calls.append((method, path, copy.deepcopy(body)))
        if path == API["list"]:
            rows = [row for jid, row in self.jobs.items()
                    if jid not in self.hidden_jobs and row["workspace_id"] == body["workspace_id"]]
            start = (body["page_num"] - 1) * body["page_size"]
            return {"data": {"jobs": copy.deepcopy(rows[start:start + body["page_size"]]), "total": len(rows)}}
        if path == API["detail"]:
            return {"data": copy.deepcopy(self.jobs[body["job_id"]])}
        if path == API["instances"]:
            jid = body["jobId"]
            detail = self.jobs[jid]
            count = self.allocated_overrides.get(jid, ascend_fleet.cards(detail))
            rows = [] if detail["status"] not in RUNNING else [{"instance_id": f"instance-{jid}",
                "node_name": f"node-{jid}", "instance_status": "job_running", "running_round": 1,
                "allocated_device_ids": list(range(count)), "allocation_started_at": time.time() - 10}]
            return {"data": {"instances": rows}}
        if path == API["resource"]:
            return {"data": {"logic_resources": {"gpu_total": 128, "gpu_used": 64,
                "cpu_total": 1024, "cpu_used": 256, "memory_gi_total": 8192, "memory_gi_used": 2048}}}
        if path == API["nodes"]:
            return {"data": {"node_dimensions": [node("new-candidate-host", 16)], "total": 1}}
        if path == API["events"]:
            return {"data": {"items": [{"message": "Traceback: ModuleNotFoundError: candidate_fixture"}]}}
        if path == API["stop"]:
            assert body["job_id"] in self.jobs
            self.stops.append(body["job_id"])
            # A requested stop does not constitute official resource release.
            return {"data": {"accepted": True}}
        if path == API["create"]:
            prefix = body["name"].rsplit("-", 1)[1]
            mapping = read_allocation(self.run, prefix, ascend_fleet.cards(body))
            fleet = read(self.run / "ascend_fleet/state.json")
            entry = next(row for row in fleet["jobs"] if row["worker_id"] == prefix)
            assert entry["intent_id"] == mapping["intent_id"]
            assert entry["member_ids"] == [m["worker_id"] for m in mapping["members"]]
            assert read(entry["record_path"])["request"] == body
            assert sum(ascend_fleet.cards(row["payload"]) for row in fleet["jobs"]
                       if ascend_fleet.active(row)) <= 64
            self.creates.append(copy.deepcopy(body))
            jid = f"job-created-{len(self.creates)}"
            self.add(body, jid, priority=self.new_priority)
            if self.unknown_next:
                self.unknown_next = False
                self.hidden_jobs.add(jid)
                raise TimeoutError("Server accepted create; response and listing visibility are temporarily lost")
            return {"data": {"job_id": jid}}
        raise AssertionError(f"Unexpected official endpoint: {path}")


class Harness:
    def __init__(self, tmp_path, layout=(16, 16, 16)):
        self.run = tmp_path / "run"
        self.run.mkdir()
        atomic_write(self.run / "training.json", {"resident_pod_refill": True})
        self.sha = hashlib.sha256((self.run / "training.json").read_bytes()).hexdigest()
        self.path = self.run / "ascend_fleet/state.json"
        self.registry = tmp_path / "registry.json"
        self.inventory = inventory()
        self.site = {"cards": 64, "active_cards": sum(layout), "job_layout": list(layout),
                     "training_config": {"resident_refill_max_intents": 512}}
        self.client = OfficialClient(self.run)
        atomic_write(self.path, {"schema_version": 1, "backend": "ascend", "run_dir": str(self.run),
            "user_id": self.inventory["user"]["id"], "command_template": COMMAND,
            "next_worker": 1, "next_member": 1, "jobs": [], "failures": [], "group_backoff": {}})
        for size in layout:
            self.seed(size, role="initial")
        self.coordinator = RefillFleet(self.client, self.run, self.site,
            initial_planner=lambda *_: pytest.fail("This fixture already owns its complete initial layout"))

    def fleet(self):
        return read(self.path)

    def batch(self):
        return load_state(self.run, self.sha)

    def begin_batch(self):
        state = self.batch()
        batch = {"batch_id": f"b{state['next_batch']:06d}", "phase": "reserving", "allocations": [],
                 "created_at": time.time(), "created_step": 10}
        state["next_batch"] += 1
        state["current"] = batch
        save_state(self.run, state)
        return batch

    def seed(self, size, *, role="candidate", status="job_running", priority="LOW"):
        fleet = self.fleet()
        prefix = f"w{fleet['next_worker']:06d}"
        start = fleet["next_member"]
        chosen = selection(size)
        payload = render_payload(chosen, f"python initial_fixture.py --prefix {prefix}", f"seed-{prefix}", 1800, 0)
        batch = self.batch()
        mapping = {"schema": ALLOCATION_SCHEMA, "run_dir": str(self.run), "config_sha256": self.sha,
            "worker_prefix": prefix, "intent_id": f"intent-{prefix}", "role": role,
            "batch_id": batch["current"]["batch_id"] if role == "candidate" else None,
            "placement_selection": chosen, "placement_reason": "fixture_seed",
            "members": [{"worker_id": f"w{start+i:06d}", "device_index": i} for i in range(size)]}
        atomic_write(self.run / "refill/allocations" / f"{prefix}.json", mapping)
        jid = f"job-seeded-{prefix}"
        entry = {"worker_id": prefix, "workspace_id": chosen["workspace_id"],
            "group_id": chosen["logic_compute_group_id"], "selection": chosen, "payload": payload,
            "status": status, "job_id": jid, "created_at": time.time(), "reason": "fixture_seed",
            "record_path": str(self.run / "ascend_fleet/submissions" / f"{prefix}.json"),
            "refill_role": role, "refill_batch_id": mapping["batch_id"], "intent_id": mapping["intent_id"],
            "member_ids": [m["worker_id"] for m in mapping["members"]]}
        fleet["jobs"].append(entry)
        fleet["next_worker"] += 1
        fleet["next_member"] += size
        atomic_write(self.path, fleet)
        if role == "candidate":
            batch["current"]["allocations"].append(prefix)
            save_state(self.run, batch)
        self.client.add(payload, jid, status=status, priority=priority)
        return entry, mapping

    def reconcile(self, *, plan=None, intent=None):
        return ascend_fleet.reconcile(self.client, self.run, COMMAND, self.inventory,
            card_limit=64, global_card_limit=64, fill=True, registry_path=self.registry, retries=0,
            state_planner=plan or self.coordinator.plan, intent_hook=intent or self.coordinator.intent,
            retire_predicate=self.coordinator.retire, state_hook=self.coordinator.restore,
            isolate_candidate_failures=True)


def test_dynamic_member_history_exceeds_64_without_overallocating_physical_cards(tmp_path):
    h = Harness(tmp_path, layout=(16, 16, 16, 16))
    original = h.fleet()
    retired = original["jobs"][0]
    retired.update(status="job_stopped", official_terminal=True)
    h.client.jobs[retired["job_id"]]["status"] = "job_stopped"
    atomic_write(h.path, original)
    h.begin_batch()
    result = h.reconcile()
    fleet = h.fleet()
    candidate = fleet["jobs"][-1]
    assert candidate["refill_role"] == "candidate"
    assert candidate["member_ids"] == [f"w{i:06d}" for i in range(65, 81)]
    assert len({w for entry in fleet["jobs"] for w in entry["member_ids"]}) == 80
    assert sum(ascend_fleet.cards(entry["payload"]) for entry in fleet["jobs"] if ascend_fleet.active(entry)) == 64
    assert result["local_requested_reserved_cards"] == 64 and len(h.client.creates) == 1
    assert not result["blocked_reason"]


def test_unknown_post_reserves_all_cards_and_is_never_retried(tmp_path):
    h = Harness(tmp_path)
    h.begin_batch()
    h.client.unknown_next = True
    first = h.reconcile()
    candidate = h.fleet()["jobs"][-1]
    allocation = h.run / "refill/allocations" / f"{candidate['worker_id']}.json"
    original = allocation.read_bytes()
    assert candidate["status"] == "submission_unknown" and not candidate["job_id"]
    assert read(candidate["record_path"])["status"] == "submission_unknown"
    assert first["local_requested_reserved_cards"] == 64
    unresolved = h.reconcile()
    assert unresolved["unresolved_reservations"] == 1
    assert unresolved["local_requested_reserved_cards"] == 64 and len(h.client.creates) == 1
    h.client.hidden_jobs.clear()
    resolved = h.reconcile()
    recovered = h.fleet()["jobs"][-1]
    assert recovered["job_id"] == "job-created-1"
    assert recovered["status"] == "job_running"
    assert resolved["local_requested_reserved_cards"] == 64 and len(h.client.creates) == 1
    assert allocation.read_bytes() == original


def test_worker_loss_never_releases_one_card_from_a_still_running_16_card_job(tmp_path):
    h = Harness(tmp_path, layout=(16, 16, 16, 16))
    first = h.fleet()["jobs"][0]
    h.client.allocated_overrides[first["job_id"]] = 15
    atomic_write(h.run / "probe/exits/w000001-session.json", {
        "worker_id": "w000001", "session_id": "session", "exit_code": -9})
    atomic_write(h.run / "refill/progress.json", {
        "active_member_ids": [f"w{i:06d}" for i in range(2, 65)], "completed_step": 10})
    h.begin_batch()
    result = h.reconcile()
    assert result["verified_allocated_cards"] == 63
    assert result["local_requested_reserved_cards"] == 64
    assert result["active_jobs"] == 4 and result["planned"] == []
    assert not h.client.creates and not h.client.stops
    assert h.fleet()["jobs"][0]["official_terminal"] is False


def test_sixty_active_plus_four_reserved_blocks_refill_and_cannot_load_a_partial_batch(tmp_path):
    h = Harness(tmp_path, layout=(16, 16, 16, 8, 4))
    h.begin_batch()
    _, mapping = h.seed(4)
    reservation = {**mapping, "schema": RESERVATION_SCHEMA, "mapping_sha256": mapping_digest(mapping),
        "wrapper_session": "wrapper-4", "pid": 999, "hostname": "candidate-host", "heartbeat_seq": 1,
        "updated_at": time.time(), "state": "reserved"}
    atomic_write(h.run / "refill/reservations" / f"{mapping['worker_prefix']}.json", reservation)
    result = h.reconcile()
    assert result["local_requested_reserved_cards"] == 64 and result["planned"] == []
    assert not h.client.creates
    batch = h.batch()["current"]
    assert batch["phase"] == "reserving" and len(batch["allocations"]) == 1
    members = [{"worker_id": f"w{i:06d}"} for i in range(1, 61)]
    group = {"members": members, "generation": 0}
    pod = SimpleNamespace(run=h.run / "probe", api=object(),
        spec={"run_id": "fixture-run", "config_sha256": h.sha, "heartbeat_timeout": 15},
        watch=SimpleNamespace(keys={member["worker_id"]: object() for member in members}))
    controller = RefillController(pod, {"resident_refill_target_cards": 64,
        "resident_refill_max_batches": 32, "resident_refill_candidate_timeout": 1800})
    controller._admit = lambda *_: pytest.fail("Four candidates cannot alter the active training group")
    assert controller.on_commit(members, group, 10) == (members, group)
    assert h.batch()["current"]["phase"] == "reserving"
    partial_permit = {"schema": PERMIT_SCHEMA, **{key: mapping[key] for key in IDENTITY_FIELDS},
        "mapping_sha256": mapping_digest(mapping), "wrapper_session": reservation["wrapper_session"],
        "batch_member_ids": [m["worker_id"] for m in mapping["members"]]}
    assert not validate_permit(mapping, reservation, partial_permit)
    assert not (h.run / "refill/permits").exists() and not (h.run / "refill/admissions").exists()


def test_three_candidate_software_failures_never_block_healthy_initial_training(tmp_path):
    h = Harness(tmp_path)
    h.begin_batch()
    for _ in range(3):
        h.seed(1, status="job_failed")
    result = h.reconcile()
    failures = h.fleet()["failures"]
    assert len(failures) == 3
    assert all(row["optional_candidate"] and row["kind"] == "software_or_capacity" for row in failures)
    assert not result["blocked_reason"] and result["running_jobs"] == 3
    assert result["local_requested_reserved_cards"] == 48
    assert not h.client.stops and not h.client.creates
    assert h.batch()["current"]["failure"] == "Candidate job ended before admission"


@pytest.mark.parametrize("when", ["existing_observation", "just_submitted"])
def test_bad_candidate_priority_stops_only_candidate_and_retains_healthy_initial_jobs(tmp_path, when):
    h = Harness(tmp_path)
    healthy_ids = {entry["job_id"] for entry in h.fleet()["jobs"]}
    h.begin_batch()
    if when == "existing_observation":
        bad, _ = h.seed(16, priority="HIGH")
        identifier = bad["job_id"]
    else:
        h.client.new_priority = "HIGH"
        identifier = "job-created-1"
    result = h.reconcile()
    assert h.client.stops == [identifier]
    assert not healthy_ids.intersection(h.client.stops)
    assert not result["blocked_reason"]
    assert result["local_requested_reserved_cards"] == 64  # Stop request is not a terminal receipt.
    candidate = h.fleet()["jobs"][-1]
    assert candidate["refill_priority_error"] is True and candidate["stop_requested"]
    assert all(entry["status"] == "job_running" for entry in h.fleet()["jobs"][:3])


def test_cancellation_after_intent_publication_blocks_post_without_stopping_initial_jobs(tmp_path):
    h = Harness(tmp_path)
    h.begin_batch()

    def intent_then_cancel(fleet, entry, placement):
        h.coordinator.intent(fleet, entry, placement)
        with writer_lock(h.run / "refill/lock"):
            state = h.batch()
            cancel(h.run, state, state["current"], "fixture cancellation before POST")

    result = h.reconcile(intent=intent_then_cancel)
    candidate = h.fleet()["jobs"][-1]
    assert candidate["status"] == "cancelled_before_submission"
    assert not Path(candidate["record_path"]).exists()
    assert read_allocation(h.run, candidate["worker_id"], 16)["members"]
    assert not h.client.creates and not h.client.stops
    assert result["local_requested_reserved_cards"] == 48 and not result["blocked_reason"]


def test_cancellation_between_plan_and_intent_defers_without_global_failure(tmp_path):
    h = Harness(tmp_path)
    h.begin_batch()

    def plan_then_cancel(*args):
        placements = h.coordinator.plan(*args)
        assert sum(p["cards"] for p in placements) == 16
        with writer_lock(h.run / "refill/lock"):
            state = h.batch()
            cancel(h.run, state, state["current"], "fixture cancellation before immutable intent")
        return placements

    result = h.reconcile(plan=plan_then_cancel)
    assert len(h.fleet()["jobs"]) == 3
    assert h.fleet()["next_member"] == 49 and h.fleet()["next_worker"] == 4
    assert not h.client.creates and not h.client.stops
    assert not result["blocked_reason"] and result["local_requested_reserved_cards"] == 48


def test_recover_orphan_submission_preserves_original_members_and_does_not_repost(tmp_path):
    h = Harness(tmp_path)
    h.begin_batch()
    h.client.unknown_next = True
    h.reconcile()
    fleet = h.fleet()
    orphan = fleet["jobs"].pop()
    mapping_path = h.run / "refill/allocations" / f"{orphan['worker_id']}.json"
    mapping_bytes = mapping_path.read_bytes()
    fleet["next_worker"], fleet["next_member"] = 4, 49
    atomic_write(h.path, fleet)
    h.client.hidden_jobs.clear()
    result = h.reconcile()
    recovered = h.fleet()["jobs"][-1]
    assert recovered["recovered_orphan_record"] is True
    assert recovered["member_ids"] == orphan["member_ids"]
    assert recovered["intent_id"] == orphan["intent_id"]
    assert recovered["job_id"] == "job-created-1"
    assert h.fleet()["next_member"] == 65 and h.fleet()["next_worker"] == 5
    assert mapping_path.read_bytes() == mapping_bytes
    assert len(h.client.creates) == 1 and result["local_requested_reserved_cards"] == 64


def test_restore_mapping_written_before_durable_intent_replays_same_allocation(tmp_path):
    h = Harness(tmp_path)
    batch = h.begin_batch()
    fleet = h.fleet()
    h.coordinator.selected_batch = batch["batch_id"]
    selected = selection(16)
    placement = {"cards": 16, "selection": selected, "reason": "original_idle_node_choice"}
    entry = {"worker_id": "w000004", "selection": selected,
             "payload": render_payload(selected, "python fixture.py", "unfinished-w000004", 1800, 0)}
    h.coordinator.intent(fleet, entry, placement)
    mapping_path = h.run / "refill/allocations/w000004.json"
    original = mapping_path.read_bytes()
    # Simulate death before state.jobs/next_member was durably advanced. The
    # immutable allocation and batch membership survive; there was no POST.
    assert len(h.fleet()["jobs"]) == 3 and h.fleet()["next_member"] == 49
    result = h.reconcile()
    recovered = h.fleet()["jobs"][-1]
    assert recovered["member_ids"] == [f"w{i:06d}" for i in range(49, 65)]
    assert recovered["reason"] == "original_idle_node_choice"
    assert mapping_path.read_bytes() == original
    assert len(h.client.creates) == 1 and result["local_requested_reserved_cards"] == 64
