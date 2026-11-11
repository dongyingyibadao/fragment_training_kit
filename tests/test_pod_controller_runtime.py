"""Stateful controller protocol checks; no model, devices, jobs or HCCL.

The fake actors keep their own step, accumulated units, group and reduction
release. A new command must satisfy that actor's state, rather than merely
returning the report the controller expects. Device/backend cleanup and actual
numerical equivalence require the separate real-worker experiments.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from fragment_training_kit import launch  # Select the standalone vendor.
from jade_lotus import pod_controller_runtime as runtime
from jade_lotus.pod_recovery import PodRecoveryError


@dataclass(frozen=True)
class Session:
    worker_id: str
    session_id: str
    child_pid: int


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        self.now += 0.01
        return self.now

    time = monotonic

    def sleep(self, seconds):
        self.now += max(seconds, 0.01)


class Actors:
    def __init__(self, run, count=6):
        self.run = run
        self.members = [{"worker_id": f"w{i:06d}", "session_id": f"session-{i}", "pid": 100 + i}
                        for i in range(1, count + 1)]
        self.spec = {"run_id": "actor-run", "config_sha256": "c" * 64,
                     "seed": 42, "rng_scheme": "fixture-atomic4", "phase_timeout": 10,
                     "pod_step_timeout": 40, "pod_max_recoveries": 64}
        self.files, self.reports, self.log = {}, {}, []
        self.command_counts, self.applications = Counter(), Counter()
        self.rules, self.mutations, self.omissions = {}, {}, {}
        self.history = {}
        self.sequence = 0
        self.watch = None
        group = self._group(self.members, 0)
        self.actors = {m["worker_id"]: dict(member=m, alive=True, step=0, units=None,
            group=deepcopy(group), root=None, reducing=None, released=None, finished=False) for m in self.members}

    def ControllerLiveness(self, run, spec):
        assert run == self.run and spec == self.spec
        self.watch = SimpleNamespace(keys={}, detector=SimpleNamespace(fenced_sessions=set()))

        def register(members, *, receipt):
            assert receipt
            for member in members:
                self.watch.keys[member["worker_id"]] = Session(
                    member["worker_id"], member["session_id"], member["pid"])

        self.watch.register_ready = register
        self.watch.poll = lambda: []
        return self.watch

    @staticmethod
    def _identity(member):
        return {key: member[key] for key in ("worker_id", "session_id", "pid")}

    def _same_identity(self, left, right):
        return self._identity(left) == self._identity(right)

    @staticmethod
    def _group(members, generation, *, port=None):
        return {"members": deepcopy(members), "generation": generation, "port": port or 29000}

    @staticmethod
    def assignments(members):
        return {member["worker_id"]: list(range(index, 64, len(members)))
                for index, member in enumerate(members)}

    @staticmethod
    def validate_coverage(mapping):
        assert sorted(unit for units in mapping.values() for unit in units) == list(range(64))

    @staticmethod
    def _control_poll_seconds(spec):
        return 0.01

    @staticmethod
    def _boundary_metadata(spec, step):
        return {"run_id": spec["run_id"], "boundary_step": step,
                "config_sha256": spec["config_sha256"], "global_batch": 256,
                "seed": spec["seed"], "rng_scheme": spec["rng_scheme"],
                "activation_recomputation": False}

    @staticmethod
    def native_state():
        return {"model_sha256": "a" * 64, "optimizer_sha256": "b" * 64,
                "optimizer_config_sha256": "d" * 64}

    def native_receipt(self, step):
        return {"step": step, "manifest_sha256": "f" * 64,
                "payload_bytes_verified": True, **self.native_state()}

    def _read(self, path):
        return deepcopy(self.files.get(path))

    def _write(self, path, value):
        self.files[path] = deepcopy(value)
        self.log.append(("write", str(path), deepcopy(value)))
        if path.parent == self.run / "abort":
            for worker, actor in self.actors.items():
                if actor["alive"] and actor["reducing"] == value["token"]:
                    actor.update(reducing=None, released=None, group=None)
                    self.record(worker, value["token"], "pod_aborted", completed_step=actor["step"],
                                state_storage_pointers_unchanged=True)
        elif path.parent == self.run / "pod" / "reduce_release":
            self.command_counts["release"] += 1
            self.inject("release", self.command_counts["release"], "before")
            for worker, actor in self.actors.items():
                if actor["alive"] and actor["reducing"] == value["token"]:
                    actor.update(reducing=None, released=value["token"])
                    self.record(worker, value["token"], "pod_released", completed_step=actor["step"],
                                generation=actor["group"]["generation"])

    def record(self, worker, token, phase, **fields):
        row = {**self.actors[worker]["member"], "run_id": self.spec["run_id"],
               "token": token, "phase": phase, **deepcopy(fields)}
        row.update(self.mutations.get((worker, phase), {}))
        for field in self.omissions.get((worker, phase), ()):
            row.pop(field, None)
        self.reports[worker, token, phase] = row

    def _reports(self, run, spec, members, token, phase):
        rows = {}
        for member in members:
            row = self.reports.get((member["worker_id"], token, phase))
            if row is None:
                return None
            assert self._same_identity(member, row)
            assert row["run_id"] == spec["run_id"] and row["token"] == token and row["phase"] == phase
            rows[member["worker_id"]] = deepcopy(row)
        return rows

    def lose_on(self, action, occurrence, workers, *, when="before"):
        self.rules[action, occurrence, when] = list(workers)

    def inject(self, action, occurrence, when):
        for worker in self.rules.pop((action, occurrence, when), []):
            actor = self.actors[worker]
            assert actor["alive"]
            actor["alive"] = False
            key = self.watch.keys[worker]
            self.watch.detector.fenced_sessions.add(key)
            self.files[self.run / "exits" / f"{worker}-{key.session_id}.json"] = {
                **actor["member"], "run_id": self.spec["run_id"], "exit_code": -9}
            self.log.append(("fence", worker, action))

    def append_units(self, worker, units):
        actor = self.actors[worker]
        history = self.history.setdefault((worker, actor["step"]), [])
        assert not set(history).intersection(units), "A surviving actor recomputed an accumulated unit"
        assert len(units) == len(set(units))
        history.extend(units)
        actor["units"].extend(units)

    def local(self, actor):
        return {"units": actor["units"][:], "samples": 4 * len(actor["units"]),
                "loss_contribution": sum(unit + 1 for unit in actor["units"]) / 64}

    def _send(self, run, spec, members, action, **fields):
        self.sequence += 1
        token = fields.pop("token", f"command-{self.sequence}")
        command = {"token": token, "action": action, **deepcopy(fields)}
        self.command_counts[action] += 1
        occurrence = self.command_counts[action]
        if action == "pod_apply":
            authorization = self._read(run / "pod" / "authorizations" / f"{token}.json")
            assert authorization is not None, "APPLY was sent before durable authorization"
            assert authorization["phase"] == "applying"
            assert authorization["update_token"] == token
            assert authorization["root_token"] == fields["root_token"]
            assert authorization["reduction_token"] == fields["reduction_token"]
            assert authorization["base_step"] == fields["expected_step"]
            assert authorization["optimizer_replay_forbidden"] is True
            self.validate_coverage(authorization["assignments"])
        self.log.append(("send", action, deepcopy(command), [m["worker_id"] for m in members]))
        self.inject(action, occurrence, "before")
        for member in members:
            worker, actor = member["worker_id"], self.actors[member["worker_id"]]
            if not actor["alive"]:
                continue
            assert not actor["finished"], "A finished actor received another command"
            assert actor["step"] == fields["expected_step"], "Stale or repeated optimizer step"
            assert actor["reducing"] is None, "New command overtook a live reduction"
            if action == "pod_compute":
                assert actor["units"] is None and actor["group"] == fields["group"]
                actor.update(units=[], root=fields["root_token"], released=None)
                self.append_units(worker, fields["assignments"][worker])
                self.record(worker, token, "pod_local", completed_step=actor["step"],
                            complete_local_backward=True, local=self.local(actor))
            elif action == "pod_replay":
                assert actor["units"] is not None and actor["root"] == fields["root_token"]
                self.append_units(worker, fields["replay"][worker])
                actor["released"] = None
                self.record(worker, token, "pod_local", completed_step=actor["step"],
                            complete_local_backward=True, local=self.local(actor))
            elif action == "pod_reset":
                actor.update(group=None, released=None)
                self.record(worker, token, "pod_reset_done", completed_step=actor["step"],
                            state_storage_pointers_unchanged=True, next_port=29000 + self.sequence)
            elif action == "pod_init":
                assert actor["group"] is None
                actor["group"] = deepcopy(fields["group"])
                self.record(worker, token, "pod_initialized", status="ready",
                            completed_step=actor["step"], generation=actor["group"]["generation"])
            elif action == "pod_reduce":
                assert actor["group"] == fields["group"]
                assert actor["root"] == fields["root_token"]
                assert actor["units"] == fields["assignments"][worker]
                actor["reducing"] = token
                self.record(worker, token, "pod_reduced", base_step=actor["step"],
                            generation=actor["group"]["generation"], status="reduced", local=self.local(actor))
            elif action == "pod_apply":
                assert actor["group"] == fields["group"] and actor["root"] == fields["root_token"]
                assert actor["released"] == fields["reduction_token"]
                key = worker, actor["step"]
                self.applications[key] += 1
                assert self.applications[key] == 1, "Optimizer applied more than once"
                actor["step"] += 1
                self.record(worker, token, "updated", completed_step=actor["step"], checkpoint_calls=0,
                    optimizer_applications=1, root_token=actor["root"], reduction_token=actor["released"],
                    generation=actor["group"]["generation"], grad_norm=0.125, local=self.local(actor))
                actor.update(units=None, released=None)
            elif action == "export":
                assert actor["units"] is None and actor["released"] is None
                assert fields["state_metadata"] == self._boundary_metadata(spec, actor["step"])
                self.record(worker, token, "exported", completed_step=actor["step"],
                    receipt=self.native_receipt(actor["step"]), next_port=29000 + self.sequence)
            elif action in ("inspect", "finish"):
                assert actor["units"] is None and actor["released"] is None
                self.record(worker, token, "inspected" if action == "inspect" else "finished",
                            completed_step=actor["step"], checkpoint_calls=0, **self.native_state())
                if action == "finish":
                    actor.update(finished=True, group=None)
            else:
                raise AssertionError(f"Unexpected command {action}")
        self.inject(action, occurrence, "after")
        return token


@pytest.fixture
def actors(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "time", Clock())
    api = Actors(tmp_path)
    controller = runtime.PodController(api, tmp_path, api.spec, api.members, {})
    return api, controller


def one_step(api, controller, members=None, group=None, step=0):
    members = api.members if members is None else members
    group = api._group(members, 0) if group is None else group
    return controller.run_step(members, group, step)


def assert_commit(api, members, commit, step):
    api.validate_coverage(commit["assignments"])
    assert commit["atomic_units"] == 64 and commit["samples"] == 256
    assert sum(commit["prepared_loss_contributions"].values()) == sum(range(1, 65)) / 64
    for member in members:
        worker = member["worker_id"]
        assert api.actors[worker]["step"] == step + 1
        assert api.applications[worker, step] == 1
    assert set(api.applications.values()) == {1}


def test_normal_two_updates_authorize_before_apply_and_cover_every_unit(actors):
    api, controller = actors
    members, group, commit = one_step(api, controller)
    assert_commit(api, members, commit, 0)
    assert commit["forward_recovery"] is None and not controller.events
    members, group, second = one_step(api, controller, members, group, 1)
    assert_commit(api, members, second, 1)
    assert commit["token"] != second["token"] and commit["root_token"] != second["root_token"]
    assert api.command_counts["pod_compute"] == 2 and api.command_counts["pod_apply"] == 2
    assert len(api.applications) == 12


@pytest.mark.parametrize("when", ["before", "after"])
def test_compute_pod_loss_then_replay_owner_loss_preserves_complete_ownership(actors, when):
    api, controller = actors
    api.lose_on("pod_compute", 1, ["w000005", "w000006"])
    api.lose_on("pod_replay", 1, ["w000001"], when=when)
    members, group, commit = one_step(api, controller)
    assert_commit(api, members, commit, 0)
    assert [m["worker_id"] for m in members] == ["w000002", "w000003", "w000004"]
    assert api.command_counts["pod_replay"] == 2
    assert api.command_counts["pod_compute"] == 1 and api.command_counts["pod_apply"] == 1
    replays = [entry[2]["replay"] for entry in api.log if entry[:2] == ("send", "pod_replay")]
    first_assigned_to_lost_owner = set(replays[0]["w000001"])
    assert first_assigned_to_lost_owner
    assert first_assigned_to_lost_owner.issubset({u for units in replays[1].values() for u in units})
    for member in members:
        worker = member["worker_id"]
        assert api.history[worker, 0] == commit["assignments"][worker]


def test_repeated_reduction_losses_abort_before_replay_and_do_not_recompute_healthy_work(actors):
    api, controller = actors
    api.lose_on("pod_reduce", 1, ["w000005", "w000006"])
    api.lose_on("pod_reduce", 2, ["w000001"])
    members, group, commit = one_step(api, controller)
    assert_commit(api, members, commit, 0)
    assert api.command_counts["pod_reduce"] == 3
    aborts = [(index, entry) for index, entry in enumerate(api.log)
              if entry[0] == "write" and "/abort/" in entry[1]]
    assert len(aborts) == 2 and aborts[0][1][2]["token"] != aborts[1][1][2]["token"]
    assert api.command_counts["pod_compute"] == 1 and api.command_counts["pod_apply"] == 1
    assert group["generation"] == 2


def test_loss_during_reduction_release_replays_without_a_second_optimizer_update(actors):
    api, controller = actors
    api.lose_on("release", 1, ["w000006"])
    members, group, commit = one_step(api, controller)
    assert_commit(api, members, commit, 0)
    assert api.command_counts["pod_reduce"] == 2 and api.command_counts["pod_apply"] == 1
    assert api.command_counts["pod_compute"] == 1


@pytest.mark.parametrize("when", ["before", "after"])
def test_partial_apply_then_inspection_loss_forward_commits_survivors_once(actors, when):
    api, controller = actors
    api.lose_on("pod_apply", 1, ["w000005", "w000006"], when=when)
    api.lose_on("inspect", 1, ["w000004"])
    members, group, commit = one_step(api, controller)
    assert_commit(api, members, commit, 0)
    assert len(members) == 3 and len(commit["prepared_members"]) == 6
    assert len(commit["assignments"]) == 6  # The authorized batch included the lost ranks.
    receipt = commit["forward_recovery"]
    assert receipt["completed_step"] == 1 and receipt["generation"] == 0
    assert {m["worker_id"] for m in receipt["members"]} == {m["worker_id"] for m in members}
    assert api._read(api.run / "pod" / "forward" / f"{commit['token']}.json") == receipt
    assert api.command_counts["pod_apply"] == 1 and api.command_counts["pod_replay"] == 0
    assert api.applications["w000006", 0] == (1 if when == "after" else 0)
    members, group, second = one_step(api, controller, members, group, 1)
    assert_commit(api, members, second, 1)


@pytest.mark.parametrize("field,value", [("model_sha256", "e" * 64),
    ("optimizer_sha256", "e" * 64), ("optimizer_config_sha256", "e" * 64)])
def test_forward_state_disagreement_fails_without_reapplying_optimizer(actors, field, value):
    api, controller = actors
    api.lose_on("pod_apply", 1, ["w000006"])
    api.mutations["w000001", "inspected"] = {field: value}
    with pytest.raises(PodRecoveryError, match="hashes disagree"):
        one_step(api, controller)
    assert api.command_counts["pod_apply"] == 1
    assert set(api.applications.values()) == {1}
    assert not any(path.parent == api.run / "pod" / "forward" for path in api.files)


@pytest.mark.parametrize("field,value", [("optimizer_applications", 2), ("completed_step", 0),
    ("root_token", "stale-root"), ("reduction_token", "stale-reduction"), ("generation", 999)])
def test_updated_report_must_match_exact_authorization(actors, field, value):
    api, controller = actors
    api.mutations["w000001", "updated"] = {field: value}
    with pytest.raises(runtime.PodRecoveryRequired, match="authorized optimizer application"):
        one_step(api, controller)
    assert api.command_counts["pod_apply"] == 1 and set(api.applications.values()) == {1}


@pytest.mark.parametrize("action", ["pod_compute", "pod_reduce", "pod_apply"])
def test_no_surviving_replica_requires_explicit_checkpoint_recovery(actors, action):
    api, controller = actors
    api.lose_on(action, 1, [m["worker_id"] for m in api.members])
    with pytest.raises(runtime.PodRecoveryRequired, match="No surviving current replica"):
        one_step(api, controller)
    assert not api.applications
    assert not any(path.parent == api.run / "pod" / "forward" for path in api.files)


def test_recovery_budget_exhaustion_never_authorizes_optimizer(actors):
    api, controller = actors
    controller.max_recoveries = 1
    api.lose_on("pod_compute", 1, ["w000006"])
    api.lose_on("pod_replay", 1, ["w000005"])
    with pytest.raises(runtime.PodRecoveryRequired, match="maximum pod-recovery attempts"):
        one_step(api, controller)
    assert not api.applications and api.command_counts["pod_apply"] == 0
    assert not any(path.parent == api.run / "pod" / "authorizations" for path in api.files)


def test_device_error_is_not_reclassified_as_a_recoverable_preemption(actors):
    api, controller = actors
    member = api.members[-1]
    api.files[api.run / "errors" / f"{member['worker_id']}-{member['session_id']}.json"] = {
        **member, "error": "nonfinite native gradient"}
    api.lose_on("unused", 1, [member["worker_id"]])
    api.inject("unused", 1, "before")
    with pytest.raises(runtime.PodRecoveryRequired, match="Device execution failed"):
        one_step(api, controller)
    assert api.command_counts["pod_compute"] == 0 and not api.applications


def call_boundary(controller, operation, members, group, *, export=True):
    if operation == "checkpoint":
        current, new_group, checkpoint, inspection = controller.checkpoint_boundary(
            members, group, 1, export=export)
        return {"members": current, "group": new_group, "checkpoint": checkpoint,
                "reports": inspection["members"]}
    assert operation == "finalize"
    current, checkpoint, finished = controller.finalize(members, group, 1, checkpoint=export)
    return {"members": current, "checkpoint": checkpoint, "reports": finished}


def assert_no_boundary_update(api, before):
    assert api.applications == before
    assert api.command_counts["pod_compute"] == 1
    assert api.command_counts["pod_reduce"] == 1
    assert api.command_counts["pod_apply"] == 1
    assert api.command_counts["pod_replay"] == 0
    assert all(actor["step"] == 1 for actor in api.actors.values())
    authorizations = [path for path in api.files if path.parent == api.run / "pod" / "authorizations"]
    assert len(authorizations) == 1


@pytest.mark.parametrize("operation", ["checkpoint", "finalize"])
@pytest.mark.parametrize("export", [False, True])
def test_healthy_committed_boundary_does_not_apply_another_update(actors, operation, export):
    api, controller = actors
    members, group, _ = one_step(api, controller)
    before = api.applications.copy()
    result = call_boundary(controller, operation, members, group, export=export)
    assert result["members"] == members
    assert (result["checkpoint"] is not None) == export
    assert set(result["reports"]) == {m["worker_id"] for m in members}
    assert api.command_counts["export"] == int(export)
    assert api.command_counts["finish"] == int(operation == "finalize")
    assert api.command_counts["pod_reset"] == 0
    if export:
        assert result["checkpoint"]["receipt"]["step"] == 1
    assert_no_boundary_update(api, before)


@pytest.mark.parametrize("operation", ["checkpoint", "finalize"])
def test_boundary_export_source_loss_retries_another_source_at_a_new_path(actors, operation):
    api, controller = actors
    members, group, _ = one_step(api, controller)
    before = api.applications.copy()
    api.lose_on("export", 1, ["w000001"])
    result = call_boundary(controller, operation, members, group)
    exports = [entry for entry in api.log if entry[:2] == ("send", "export")]
    assert len(exports) == 2
    assert exports[0][3] == ["w000001"] and exports[1][3] == ["w000002"]
    assert exports[0][2]["path"] != exports[1][2]["path"]
    assert result["checkpoint"]["path"] == exports[1][2]["path"]
    assert ("w000001", exports[0][2]["token"], "exported") not in api.reports
    assert {m["worker_id"] for m in result["members"]} == {m["worker_id"] for m in members[1:]}
    assert api.command_counts["pod_reset"] == api.command_counts["pod_init"] == 1
    assert_no_boundary_update(api, before)


@pytest.mark.parametrize("operation", ["checkpoint", "finalize"])
def test_boundary_export_loss_then_inspection_loss_shrinks_and_rebuilds(actors, operation):
    api, controller = actors
    members, group, _ = one_step(api, controller)
    before = api.applications.copy()
    api.lose_on("export", 1, ["w000001"])
    api.lose_on("inspect", 1, ["w000002", "w000003"])
    result = call_boundary(controller, operation, members, group)
    survivors = {"w000004", "w000005", "w000006"}
    assert {m["worker_id"] for m in result["members"]} == survivors
    assert set(result["reports"]) == survivors
    inits = [entry for entry in api.log if entry[:2] == ("send", "pod_init")]
    assert len(inits) == 1 and set(inits[0][3]) == survivors
    assert inits[0][2]["expected_step"] == 1
    assert inits[0][2]["group"]["generation"] == 1
    assert result["checkpoint"]["receipt"]["step"] == 1
    assert_no_boundary_update(api, before)
    if operation == "checkpoint":
        current, _, commit = one_step(api, controller, result["members"], result["group"], 1)
        assert_commit(api, current, commit, 1)


@pytest.mark.parametrize("operation", ["checkpoint", "finalize"])
@pytest.mark.parametrize("field", ["model_sha256", "optimizer_sha256"])
def test_boundary_checkpoint_hash_must_match_surviving_inspection(actors, operation, field):
    api, controller = actors
    members, group, _ = one_step(api, controller)
    before = api.applications.copy()
    api.mutations["w000001", "exported"] = {"receipt": {
        **api.native_receipt(1), field: "e" * 64}}
    with pytest.raises(runtime.PodRecoveryRequired, match="(?i)checkpoint.*differ"):
        call_boundary(controller, operation, members, group)
    assert api.command_counts["finish"] == 0
    assert_no_boundary_update(api, before)


@pytest.mark.parametrize("operation", ["checkpoint", "finalize"])
@pytest.mark.parametrize("field,value", [("model_sha256", "e" * 64),
    ("optimizer_sha256", "e" * 64), ("optimizer_config_sha256", "e" * 64),
    ("completed_step", 0), ("checkpoint_calls", 1)])
def test_boundary_inspection_mismatch_rejects_success(actors, operation, field, value):
    api, controller = actors
    members, group, _ = one_step(api, controller)
    before = api.applications.copy()
    api.mutations["w000001", "inspected"] = {field: value}
    with pytest.raises(runtime.PodRecoveryRequired):
        call_boundary(controller, operation, members, group)
    assert api.command_counts["finish"] == 0
    assert_no_boundary_update(api, before)


@pytest.mark.parametrize("when", ["before", "after"])
def test_final_finish_loss_accepts_valid_finished_survivors(actors, when):
    api, controller = actors
    members, group, _ = one_step(api, controller)
    before = api.applications.copy()
    api.lose_on("finish", 1, ["w000005", "w000006"], when=when)
    result = call_boundary(controller, "finalize", members, group)
    survivors = {m["worker_id"] for m in members[:4]}
    assert {m["worker_id"] for m in result["members"]} == survivors
    assert set(result["reports"]) == survivors
    assert all(api.actors[worker]["finished"] for worker in survivors)
    assert api.command_counts["finish"] == 1
    assert api._read(api.run / "pod" / "phase.json")["phase"] == "finished"
    assert_no_boundary_update(api, before)


@pytest.mark.parametrize("operation,action", [("checkpoint", "export"), ("checkpoint", "inspect"),
    ("finalize", "export"), ("finalize", "inspect"), ("finalize", "finish")])
def test_all_replicas_lost_at_boundary_requires_explicit_checkpoint_recovery(actors, operation, action):
    api, controller = actors
    members, group, _ = one_step(api, controller)
    before = api.applications.copy()
    api.lose_on(action, 1, [m["worker_id"] for m in members])
    with pytest.raises(runtime.PodRecoveryRequired, match="No surviving current replica"):
        call_boundary(controller, operation, members, group)
    assert api._read(api.run / "pod" / "phase.json")["phase"] != "finished"
    assert_no_boundary_update(api, before)


@pytest.mark.parametrize("operation", ["checkpoint", "finalize"])
@pytest.mark.parametrize("location", ["report", "receipt"])
@pytest.mark.parametrize("value", [0, 2, True, None, "missing"])
def test_export_requires_exact_integer_boundary_step_in_report_and_receipt(actors, operation, location, value):
    api, controller = actors
    members, group, _ = one_step(api, controller)
    before = api.applications.copy()
    if location == "report":
        if value == "missing":
            api.omissions["w000001", "exported"] = {"completed_step"}
        else:
            api.mutations["w000001", "exported"] = {"completed_step": value}
    else:
        receipt = api.native_receipt(1)
        if value == "missing":
            receipt.pop("step")
        else:
            receipt["step"] = value
        api.mutations["w000001", "exported"] = {"receipt": receipt}
    with pytest.raises(runtime.PodRecoveryRequired):
        call_boundary(controller, operation, members, group)
    assert api.command_counts["finish"] == 0
    assert_no_boundary_update(api, before)


@pytest.mark.parametrize("operation", ["checkpoint", "finalize"])
@pytest.mark.parametrize("field,value", [("model_sha256", "A" * 64),
    ("optimizer_sha256", "invalid"), ("manifest_sha256", None),
    ("payload_bytes_verified", False), ("payload_bytes_verified", 1)])
def test_export_requires_valid_digests_and_explicit_payload_verification(actors, operation, field, value):
    api, controller = actors
    members, group, _ = one_step(api, controller)
    before = api.applications.copy()
    api.mutations["w000001", "exported"] = {"receipt": {**api.native_receipt(1), field: value}}
    with pytest.raises(runtime.PodRecoveryRequired):
        call_boundary(controller, operation, members, group)
    assert api.command_counts["finish"] == 0
    assert_no_boundary_update(api, before)


@pytest.mark.parametrize("field,value", [("completed_step", 0), ("completed_step", True),
    ("checkpoint_calls", 1), ("checkpoint_calls", False), ("model_sha256", "bad"),
    ("optimizer_sha256", "A" * 64), ("optimizer_config_sha256", None),
    ("model_sha256", "e" * 64), ("optimizer_sha256", "e" * 64),
    ("optimizer_config_sha256", "e" * 64)])
def test_final_finished_reports_are_validated_again_after_inspection(actors, field, value):
    api, controller = actors
    members, group, _ = one_step(api, controller)
    before = api.applications.copy()
    api.mutations["w000001", "finished"] = {field: value}
    with pytest.raises(runtime.PodRecoveryRequired):
        call_boundary(controller, "finalize", members, group)
    assert api.command_counts["finish"] == 1
    assert api._read(api.run / "pod" / "phase.json")["phase"] != "finished"
    assert_no_boundary_update(api, before)


@pytest.mark.parametrize("field", ["model_sha256", "optimizer_sha256", "optimizer_config_sha256"])
def test_agreeing_finished_hashes_must_still_match_pre_finish_inspection(actors, field):
    api, controller = actors
    members, group, _ = one_step(api, controller)
    before = api.applications.copy()
    for member in members:
        api.mutations[member["worker_id"], "finished"] = {field: "e" * 64}
    with pytest.raises(runtime.PodRecoveryRequired):
        call_boundary(controller, "finalize", members, group)
    assert api._read(api.run / "pod" / "phase.json")["phase"] != "finished"
    assert_no_boundary_update(api, before)
