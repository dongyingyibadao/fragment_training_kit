"""Pure pod recovery contracts; no platform, device, liveness or authorization."""
from copy import deepcopy

import pytest

from fragment_training_kit import launch  # Select this standalone kit's vendor.
from jade_lotus.pod_recovery import PodRecoveryError, repair_ownership, validate_forward_update


def partition(count=4):
    return {f"w{rank + 1:06d}": list(range(rank, 64, count)) for rank in range(count)}


def test_whole_pod_loss_balances_missing_work_and_preserves_every_surviving_unit():
    original = partition(8)
    before = deepcopy(original)
    survivors = list(reversed(list(original)[:4]))
    mapping, replay = repair_ownership(original, survivors)
    assert list(mapping) == sorted(survivors) and list(replay) == sorted(survivors)
    assert sorted(unit for units in mapping.values() for unit in units) == list(range(64))
    assert {len(units) for units in mapping.values()} == {16}
    for worker in survivors:
        assert mapping[worker] == original[worker] + replay[worker]
        assert not set(original[worker]).intersection(replay[worker])
    assert original == before
    assert repair_ownership(original, sorted(survivors)) == (mapping, replay)


def test_replay_uses_current_least_total_load_with_stable_worker_id_ties():
    original = {"b": list(range(3, 10)), "a": [0, 1, 2], "lost": list(range(10, 64))}
    mapping, replay = repair_ownership(original, ["b", "a"])
    assert replay["a"][:5] == [10, 11, 12, 13, 14]
    assert replay["b"][0] == 15
    assert len(mapping["a"]) == len(mapping["b"]) == 32


def test_repeated_loss_replays_the_lost_owners_previous_replay_too():
    original = partition()
    first, first_replay = repair_ownership(original, list(original)[:3])
    lost_again = "w000001"
    second, second_replay = repair_ownership(first, ["w000002", "w000003"])
    assert first_replay[lost_again]
    assert sorted(unit for units in second_replay.values() for unit in units) == sorted(first[lost_again])
    for worker in second:
        assert second[worker] == first[worker] + second_replay[worker]
    assert sorted(unit for units in second.values() for unit in units) == list(range(64))


def test_no_loss_and_single_survivor_include_empty_replays_without_mutating_inputs():
    original = partition()
    assert repair_ownership(original, list(original)) == (original, {worker: [] for worker in original})
    mapping, replay = repair_ownership(original, ["w000002"])
    assert mapping["w000002"][:16] == original["w000002"]
    assert len(mapping["w000002"]) == 64 and len(replay["w000002"]) == 48


@pytest.mark.parametrize("original,survivors", [
    ({}, []), ({"a": list(range(63))}, ["a"]),
    ({"a": list(range(64)), "b": [0]}, ["a"]),
    ({"a": list(range(63)) + [True]}, ["a"]),
    ({"a": list(range(63)) + [64]}, ["a"]),
    ({"a": list(range(63)) + [-1]}, ["a"]),
    ({"a": "not units"}, ["a"]), ({"": list(range(64))}, [""]),
    ({"a": list(range(64))}, []), ({"a": list(range(64))}, ["a", "a"]),
    ({"a": list(range(64))}, ["new"]), ({"a": list(range(64))}, "a"),
    ({"a": list(range(64))}, [True]),
])
def test_invalid_partitions_or_new_members_fail_closed(original, survivors):
    with pytest.raises(PodRecoveryError):
        repair_ownership(original, survivors)


@pytest.fixture
def evidence():
    spec = {"run_id": "fixture-run", "config_sha256": "c" * 64}
    members = [{"worker_id": f"w{index:06d}", "session_id": f"session-{index}", "pid": index + 100}
               for index in (1, 2)]
    updates = {member["worker_id"]: {**member, "run_id": spec["run_id"], "token": "update-token",
        "phase": "updated", "generation": 3, "completed_step": 8, "checkpoint_calls": 0,
        "grad_norm": .125} for member in members}
    inspections = {member["worker_id"]: {**member, "run_id": spec["run_id"], "token": "inspect-token",
        "phase": "inspected", "completed_step": 8, "checkpoint_calls": 0,
        "model_sha256": "a" * 64, "optimizer_sha256": "b" * 64,
        "optimizer_config_sha256": "d" * 64} for member in members}
    return dict(spec=spec, members=members, update_token="update-token", base_step=7, generation=3,
                updates=updates, inspections=inspections, inspection_token="inspect-token")


def test_forward_gate_binds_survivor_evidence_without_claiming_commit_authorization(evidence):
    before = deepcopy(evidence)
    receipt = validate_forward_update(**evidence)
    assert receipt["completed_step"] == 8 and receipt["generation"] == 3
    assert receipt["all_survivor_reports_consistent"]
    assert receipt["authorizes_optimizer_or_commit"] is False
    assert "no independent byte verification" in receipt["scope"]
    assert len(receipt["evidence_sha256"]) == 64
    evidence["members"].reverse()
    assert validate_forward_update(**evidence) == receipt
    evidence["members"].reverse()
    assert evidence == before
    evidence["inspection_token"] = "fresh-inspect"
    for record in evidence["inspections"].values():
        record["token"] = "fresh-inspect"
    assert validate_forward_update(**evidence)["evidence_sha256"] != receipt["evidence_sha256"]


@pytest.mark.parametrize("collection", ["updates", "inspections"])
@pytest.mark.parametrize("field,value", [("run_id", "another-run"), ("worker_id", "w000003"),
    ("session_id", "replaced-session"), ("pid", 999), ("pid", True), ("token", "old-token"),
    ("phase", "other-phase"), ("completed_step", 7), ("completed_step", 9),
    ("completed_step", True), ("checkpoint_calls", 1), ("checkpoint_calls", False),
    ("base_step", 6), ("config_sha256", "f" * 64)])
def test_forward_gate_rejects_stale_or_malformed_reports(evidence, collection, field, value):
    evidence[collection]["w000001"][field] = value
    with pytest.raises(PodRecoveryError):
        validate_forward_update(**evidence)


@pytest.mark.parametrize("field,value", [("grad_norm", float("nan")), ("grad_norm", float("inf")),
    ("grad_norm", -.1), ("grad_norm", True), ("grad_norm", 10**1000), ("grad_norm", .12500000001),
    ("generation", 2), ("generation", True)])
def test_invalid_or_disagreeing_gradient_or_generation_is_not_forward_evidence(evidence, field, value):
    evidence["updates"]["w000001"][field] = value
    with pytest.raises(PodRecoveryError):
        validate_forward_update(**evidence)


@pytest.mark.parametrize("field", ["model_sha256", "optimizer_sha256", "optimizer_config_sha256"])
@pytest.mark.parametrize("value", ["e" * 64, "A" * 64, "bad", None])
def test_every_complete_state_digest_is_required_and_must_agree(evidence, field, value):
    evidence["inspections"]["w000001"][field] = value
    with pytest.raises(PodRecoveryError):
        validate_forward_update(**evidence)


@pytest.mark.parametrize("collection", ["updates", "inspections"])
@pytest.mark.parametrize("alteration", ["missing", "extra", "not_mapping"])
def test_report_sets_must_equal_the_exact_survivor_set(evidence, collection, alteration):
    if alteration == "missing":
        del evidence[collection]["w000002"]
    elif alteration == "extra":
        evidence[collection]["lost-worker"] = deepcopy(evidence[collection]["w000001"])
    else:
        evidence[collection] = list(evidence[collection].values())
    with pytest.raises(PodRecoveryError):
        validate_forward_update(**evidence)


@pytest.mark.parametrize("field,value", [("base_step", True), ("base_step", -1), ("generation", False),
    ("update_token", ""), ("inspection_token", "update-token"), ("members", []), ("members", "invalid")])
def test_invalid_forward_context_fails_closed(evidence, field, value):
    evidence[field] = value
    with pytest.raises(PodRecoveryError):
        validate_forward_update(**evidence)


def test_duplicate_member_and_wrong_inspection_generation_are_rejected(evidence):
    evidence["members"].append(deepcopy(evidence["members"][0]))
    with pytest.raises(PodRecoveryError):
        validate_forward_update(**evidence)
    evidence["members"].pop()
    evidence["inspections"]["w000001"]["generation"] = 2
    with pytest.raises(PodRecoveryError):
        validate_forward_update(**evidence)


def test_missing_required_fields_are_rejected(evidence):
    for collection, field in (("updates", "grad_norm"), ("updates", "generation"),
                              ("inspections", "optimizer_sha256"), ("inspections", "checkpoint_calls")):
        altered = deepcopy(evidence)
        del altered[collection]["w000001"][field]
        with pytest.raises(PodRecoveryError):
            validate_forward_update(**altered)
    evidence["spec"]["config_sha256"] = "unverified-config"
    with pytest.raises(PodRecoveryError):
        validate_forward_update(**evidence)
