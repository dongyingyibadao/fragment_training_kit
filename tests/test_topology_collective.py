"""CPU math/mock contracts; these tests do not initialize a distributed world.

They do not establish actual HCCL compatibility, device-stream behavior,
transport choice, performance, or recovery. Real-device acceptance is separate.
"""
import copy
from datetime import timedelta
import os
from types import SimpleNamespace

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import pytest
import torch

from fragment_training_kit.vendor.jade_lotus.topology_collective import RankTopology, TopologyCollective
from fragment_training_kit.examples.benchmark_topology_collective import byte_count, read_payload


class FakeWork:
    def __init__(self, fail=False):
        self.fail = fail

    def wait(self, *, timeout):
        assert isinstance(timeout, timedelta) and timeout.total_seconds() > 0
        if self.fail:
            raise TimeoutError("injected backend timeout after buffer modification")
        return True


class MathematicalBackend:
    """Validate phase inputs and supply the exact collective's known result.

    Each rank is evaluated independently against immutable original inputs.
    Assertions on intermediate data catch wrong shard/group/source selection,
    stage ordering, padding, or accidental division, rather than just filling
    the final output with a canned value.
    """

    ReduceOp = SimpleNamespace(SUM="SUM")

    def __init__(self, rank, topology, contributions, strategy):
        self.rank, self.topology = rank, topology
        self.contributions, self.strategy = contributions, strategy
        self.created, self.destroyed, self.calls = [], [], []
        self.bucket, self.fail, self.disagree = 0, False, False

    def is_initialized(self):
        return True

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return self.topology.world_size

    def get_backend(self):
        return "gloo"

    def new_group(self, *, ranks, backend, timeout):
        assert timeout.total_seconds() > 0
        self.created.append((tuple(ranks), backend))
        return tuple(ranks)

    def destroy_process_group(self, group):
        self.destroyed.append(group)

    def all_gather_object(self, output, local, *, group):
        assert group == tuple(range(self.topology.world_size))
        for index in range(len(output)):
            output[index] = copy.deepcopy(local)
        if self.disagree:
            output[-1]["descriptor"] = "different topology/epoch/schema"

    def _vector(self, rank, length):
        source = self.contributions[rank][self.bucket].detach().reshape(-1)
        result = torch.zeros(length, dtype=source.dtype)
        result[:source.numel()].copy_(source)
        return result

    def _sum(self, ranks, length):
        return torch.stack([self._vector(rank, length) for rank in ranks]).sum(0)

    def reduce_scatter_tensor(self, output, source, *, op, group, async_op):
        assert op == self.ReduceOp.SUM and async_op
        node_index, local = self.topology.locate(self.rank)
        assert group == self.topology.nodes[node_index]
        torch.testing.assert_close(source, self._vector(self.rank, source.numel()), rtol=0, atol=0)
        chunk = self._sum(group, source.numel()).chunk(len(group))[local]
        output.copy_(chunk)
        self.calls.append(("reduce_scatter", group))
        return FakeWork(self.fail)

    def all_reduce(self, tensor, *, op, group, async_op):
        assert op == self.ReduceOp.SUM and async_op
        node_index, local = self.topology.locate(self.rank)
        node = self.topology.nodes[node_index]
        if self.strategy == "flat":
            assert group == tuple(range(self.topology.world_size))
            torch.testing.assert_close(tensor, self._vector(self.rank, tensor.numel()), rtol=0, atol=0)
            value = self._sum(range(self.topology.world_size), tensor.numel())
            self.bucket += 1
        elif self.strategy == "sharded":
            assert group == tuple(ranks[local] for ranks in self.topology.nodes)
            length = tensor.numel() * len(node)
            own = self._sum(node, length).chunk(len(node))[local]
            torch.testing.assert_close(tensor, own, rtol=0, atol=0)
            value = self._sum(range(self.topology.world_size), length).chunk(len(node))[local]
            if len(node) == 1:
                self.bucket += 1
        else:
            assert self.rank in self.topology.leaders and group == self.topology.leaders
            torch.testing.assert_close(tensor, self._sum(node, tensor.numel()), rtol=0, atol=0)
            value = self._sum(range(self.topology.world_size), tensor.numel())
            if len(node) == 1:
                self.bucket += 1
        tensor.copy_(value)
        self.calls.append(("all_reduce", group))
        return FakeWork(self.fail)

    def all_gather_into_tensor(self, output, shard, *, group, async_op):
        assert async_op
        node_index, local = self.topology.locate(self.rank)
        assert group == self.topology.nodes[node_index]
        total = self._sum(range(self.topology.world_size), output.numel())
        torch.testing.assert_close(shard, total.chunk(len(group))[local], rtol=0, atol=0)
        output.copy_(total)
        self.calls.append(("all_gather", group))
        self.bucket += 1
        return FakeWork(self.fail)

    def reduce(self, tensor, *, dst, op, group, async_op):
        assert async_op and op == self.ReduceOp.SUM and dst == group[0]
        torch.testing.assert_close(tensor, self._vector(self.rank, tensor.numel()), rtol=0, atol=0)
        tensor.copy_(self._sum(group, tensor.numel()))
        self.calls.append(("reduce", group))
        return FakeWork(self.fail)

    def broadcast(self, tensor, *, src, group, async_op):
        assert async_op and src == group[0]
        expected_input = self._sum(range(self.topology.world_size) if self.rank == src else group,
                                   tensor.numel())
        torch.testing.assert_close(tensor, expected_input, rtol=0, atol=0)
        tensor.copy_(self._sum(range(self.topology.world_size), tensor.numel()))
        self.calls.append(("broadcast", group))
        self.bucket += 1
        return FakeWork(self.fail)


def contributions(world_size):
    return [[(torch.arange(15, dtype=torch.float64).reshape(3, 5) + rank * 3) / 16,
             torch.tensor([rank + 1, 2 * rank + 3, 7], dtype=torch.float32)]
            for rank in range(world_size)]


def test_topology_canonicalizes_without_confusing_rank_order_and_hardware():
    topology = RankTopology.build([[3, 1], [2, 0]], 4)
    assert topology.nodes == ((0, 2), (1, 3))
    assert topology.locate(3) == (1, 1)
    assert topology.group_schedule("sharded") == (
        ("node:0", (0, 2)), ("node:1", (1, 3)),
        ("stripe:0", (0, 1)), ("stripe:1", (2, 3)))
    assert RankTopology.from_node_ids(["A", "B", "A", "B"]) == topology


@pytest.mark.parametrize("nodes", [[], [[0], [0]], [[0]], [[0, 1, 2]], [[False], [1]], [[], [0, 1]]])
def test_invalid_partitions_are_rejected(nodes):
    with pytest.raises(ValueError):
        RankTopology.build(nodes, 2)


def test_uneven_hierarchy_has_explicit_preflight_fallback_only():
    topology = RankTopology.build([[0, 3], [1], [2]], 4)
    assert topology.select("auto") == "leader"
    assert topology.select("auto", "flat") == "flat"
    for strategy, fallback in [("sharded", "leader"), ("auto", "error")]:
        with pytest.raises(ValueError, match="equal node"):
            topology.select(strategy, fallback)


@pytest.mark.parametrize("nodes,strategy", [([[0, 2], [1, 3]], "sharded"),
                                            ([[0, 1], [2, 3]], "flat"),
                                            ([[0, 3], [1], [2]], "leader"),
                                            ([[0], [1], [2], [3]], "sharded")])
def test_sum_phase_math_padding_weights_and_source_protection(nodes, strategy):
    topology = RankTopology.build(nodes, 4)
    values = contributions(4)
    snapshots = [[tensor.clone() for tensor in rank] for rank in values]
    schedules = []
    for rank in range(4):
        backend = MathematicalBackend(rank, topology, values, strategy)
        reducer = TopologyCollective(topology, strategy=strategy, distributed=backend)
        prepared = reducer.prepare(values[rank])
        assert all(buffer.data_ptr() != original.data_ptr()
                   for buffer, original in zip(prepared.buffers, values[rank]))
        result = reducer.sum(prepared)
        for bucket, actual in enumerate(result.tensors):
            expected = torch.stack([row[bucket] for row in values]).sum(0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert result.communication_seconds >= 0 and result.strategy == strategy
        assert reducer.close()
        schedules.append(backend.created)
    assert all(schedule == schedules[0] for schedule in schedules)
    assert schedules[0][1:] == [(ranks, "gloo") for _, ranks in topology.group_schedule(strategy)]
    for before, after in zip(snapshots, values):
        for saved, original in zip(before, after):
            torch.testing.assert_close(saved, original, rtol=0, atol=0)


def test_nonuniform_sample_assignments_match_fixed_global_batch_autograd():
    # A local mean or node mean would give a different answer on this split.
    x = (torch.arange(256 * 3, dtype=torch.float64).reshape(256, 3) % 17 - 8) / 16
    y = torch.arange(256, dtype=torch.float64) % 5
    parameter = torch.tensor([.5, -.25, .125], dtype=torch.float64, requires_grad=True)
    expected, = torch.autograd.grad(((x @ parameter - y) ** 2).mean(), parameter)
    assignments = [range(0, 24), range(24, 160), range(160, 256)]
    local_grads = []
    for positions in assignments:
        indices = list(positions)
        local, = torch.autograd.grad(((x[indices] @ parameter - y[indices]) ** 2).sum() / 256, parameter)
        local_grads.append([local])
    topology = RankTopology.build([[0, 2], [1]], 3)
    for rank in range(3):
        backend = MathematicalBackend(rank, topology, local_grads, "leader")
        reducer = TopologyCollective(topology, distributed=backend)
        result = reducer.sum(reducer.prepare(local_grads[rank]))
        torch.testing.assert_close(result.tensors[0], expected, rtol=1e-14, atol=1e-14)


def test_partially_failed_sum_never_touches_original_or_falls_back():
    topology = RankTopology.build([[0, 1], [2, 3]], 4)
    values = contributions(4)
    backend = MathematicalBackend(0, topology, values, "sharded")
    reducer = TopologyCollective(topology, distributed=backend)
    originals = [tensor.clone() for tensor in values[0]]
    prepared = reducer.prepare(values[0])
    backend.fail = True
    with pytest.raises(TimeoutError, match="injected"):
        reducer.sum(prepared)
    assert reducer.failed and not reducer.close()
    assert [operation for operation, _ in backend.calls] == ["reduce_scatter"]
    for original, saved in zip(values[0], originals):
        torch.testing.assert_close(original, saved, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="unusable"):
        reducer.prepare(values[0])


@pytest.mark.parametrize("fault", ["stale", "consumed", "foreign"])
def test_epoch_or_buffer_misuse_fails_before_next_data_collective(fault):
    topology = RankTopology.build([[0], [1]], 2)
    values = contributions(2)
    backend = MathematicalBackend(0, topology, values, "flat")
    reducer = TopologyCollective(topology, strategy="flat", epoch="generation-7", distributed=backend)
    prepared = reducer.prepare(values[0])
    if fault == "foreign":
        prepared.owner = object()
    if fault == "consumed":
        prepared.consumed = True
    with pytest.raises(ValueError, match="preflight disagreement"):
        reducer.sum(prepared, epoch="generation-8" if fault == "stale" else None)
    assert reducer.failed and not backend.calls


def test_disagreed_topology_stops_before_data_group_creation():
    topology = RankTopology.build([[0, 1], [2, 3]], 4)
    backend = MathematicalBackend(0, topology, contributions(4), "sharded")
    backend.disagree = True
    with pytest.raises(ValueError, match="preflight disagreement"):
        TopologyCollective(topology, distributed=backend)
    assert backend.created == [((0, 1, 2, 3), "gloo")]


def test_noncontiguous_autograd_gradient_is_copied_without_storage_alias():
    topology = RankTopology.build([[0], [1]], 2)
    values = [[torch.arange(12, dtype=torch.float32).view(3, 4).T.requires_grad_()]
              for _ in range(2)]
    backend = MathematicalBackend(0, topology, values, "flat")
    reducer = TopologyCollective(topology, strategy="flat", distributed=backend)
    actual = reducer.sum(reducer.prepare(values[0])).tensors[0]
    assert actual.shape == values[0][0].shape and not actual.requires_grad
    torch.testing.assert_close(actual, values[0][0] * 2, rtol=0, atol=0)


def test_sharding_distributes_cross_node_bytes_and_leader_model_reports_limit():
    topology = RankTopology.build([range(start, start + 16) for start in range(0, 64, 16)], 64)
    sharded = topology.communication_model(1600, "sharded")
    leader = topology.communication_model(1600, "leader")
    assert sharded["cross_node_sent_bytes_per_rank"] == 150
    assert sharded["cross_node_sent_bytes_per_node"] == leader["cross_node_sent_bytes_per_leader"] == 2400
    assert leader["cross_node_sent_bytes_per_nonleader"] == 0
    assert "bottleneck" in leader["limit"]


def test_benchmark_preserves_payload_order_and_exact_byte_counts(tmp_path):
    path = tmp_path / "payload.json"
    path.write_text('{"buckets":[{"dtype":"bfloat16","bytes":34},{"dtype":"float32","bytes":20}]}')
    args = SimpleNamespace(payload_json=str(path), dtype="float32", bytes="1MiB")
    assert read_payload(args) == [{"dtype": "bfloat16", "bytes": 34, "numel": 17},
                                  {"dtype": "float32", "bytes": 20, "numel": 5}]
    assert byte_count("512MiB") == 536870912
    path.write_text('{"buckets":[{"dtype":"float32","bytes":3}]}')
    with pytest.raises(ValueError, match="element-aligned"):
        read_payload(args)
