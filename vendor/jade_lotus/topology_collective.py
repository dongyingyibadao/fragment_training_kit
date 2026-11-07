"""Independent, fixed-membership SUM collectives for topology experiments.

This module does not change the resident controller or its recovery protocol.
``node_ranks`` is an explicit grouping supplied by the caller; a group name is
not proof of a physical host, an HCCS domain, or a particular transport. In
particular, HCCL subgroups cannot repair missing cross-container device IPC.

Uniform groups use reduce-scatter locally, SUM between matching local ranks,
then all-gather locally. Uneven groups can use node reduce, leader SUM, and
broadcast, or explicitly select flat SUM. No stage averages by node size or by
world size. Inputs must already carry the training recipe's sample weights.

All global ranks must construct, prepare, reduce, and close in the same order.
Membership is immutable. After any error the instance is unusable; the caller
must fence the old epoch and exit/rebuild using its own recovery protocol. PG
timeouts bound cooperative backend waits, but cannot guarantee cancellation of
a hung driver/device synchronize. Use a separate process supervisor with a
hard deadline, as the accompanying benchmark does. There is no in-collective
fallback, membership discovery, shrink, or automatic recovery.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
import math
import time
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class RankTopology:
    """A complete partition of global ranks, with canonical group ordering."""

    nodes: tuple[tuple[int, ...], ...]

    @classmethod
    def build(cls, node_ranks: Iterable[Iterable[int]], world_size: int) -> "RankTopology":
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        nodes = []
        for group in node_ranks:
            ranks = tuple(group)
            if not ranks or any(isinstance(r, bool) or not isinstance(r, int) for r in ranks):
                raise ValueError("Each node must contain integer global ranks")
            nodes.append(tuple(sorted(ranks)))
        flat = [rank for group in nodes for rank in group]
        if sorted(flat) != list(range(world_size)):
            raise ValueError("node_ranks must partition every global rank exactly once")
        return cls(tuple(sorted(nodes)))

    @classmethod
    def from_node_ids(cls, node_ids: Sequence[str]) -> "RankTopology":
        groups: dict[str, list[int]] = {}
        for rank, node_id in enumerate(node_ids):
            if not isinstance(node_id, str) or not node_id:
                raise ValueError("node_ids must contain nonempty explicit identifiers")
            groups.setdefault(node_id, []).append(rank)
        return cls.build(groups.values(), len(node_ids))

    @property
    def world_size(self) -> int:
        return sum(map(len, self.nodes))

    @property
    def uniform(self) -> bool:
        return len({len(node) for node in self.nodes}) == 1

    @property
    def leaders(self) -> tuple[int, ...]:
        return tuple(node[0] for node in self.nodes)

    def locate(self, rank: int) -> tuple[int, int]:
        for index, node in enumerate(self.nodes):
            if rank in node:
                return index, node.index(rank)
        raise ValueError("Rank does not belong to this topology")

    def select(self, strategy: str = "auto", uneven_fallback: str = "leader") -> str:
        if strategy not in {"auto", "flat", "sharded", "leader"}:
            raise ValueError("strategy must be auto, flat, sharded, or leader")
        if uneven_fallback not in {"leader", "flat", "error"}:
            raise ValueError("uneven_fallback must be leader, flat, or error")
        if strategy in {"flat", "leader"}:
            return strategy
        if self.uniform:
            return "sharded"
        if strategy == "sharded" or uneven_fallback == "error":
            raise ValueError("Sharded hierarchy requires equal node group sizes")
        return uneven_fallback

    def group_schedule(self, strategy: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """Every rank calls new_group for this entire schedule, even nonmembers."""
        if strategy == "flat":
            return (("flat", tuple(range(self.world_size))),)
        schedule = [(f"node:{index}", node) for index, node in enumerate(self.nodes)]
        if strategy == "sharded":
            if not self.uniform:
                raise ValueError("Sharded hierarchy requires equal node group sizes")
            schedule.extend((f"stripe:{local}", tuple(node[local] for node in self.nodes))
                            for local in range(len(self.nodes[0])))
        elif strategy == "leader":
            schedule.append(("leaders", self.leaders))
        else:
            raise ValueError("Expected a resolved strategy")
        return tuple(schedule)

    def communication_model(self, tensor_bytes: int, strategy: str) -> dict[str, Any]:
        """Idealized sends, excluding padding, protocol bytes and retransmits.

        Ring all-reduce sends 2*(p-1)/p times the payload per participant.
        These are algorithmic models, not measured NIC, HCCS, or PCIe bytes.
        For sharded hierarchy the caller must use padded payload bytes here.
        """
        if tensor_bytes < 0:
            raise ValueError("tensor_bytes must be nonnegative")
        world, nodes = self.world_size, len(self.nodes)
        cross_per_node = 2 * (nodes - 1) / nodes * tensor_bytes
        report: dict[str, Any] = {"payload_bytes": tensor_bytes, "strategy": strategy,
                                  "scope": "Ideal ring sends; no link/path or bandwidth claim"}
        if strategy == "flat":
            report["sent_bytes_per_rank"] = 2 * (world - 1) / world * tensor_bytes
            report["cross_node_sent_bytes"] = None
            report["limit"] = "HCCL may itself choose a hierarchy; flat means one API-level AllReduce"
        elif strategy == "sharded" and self.uniform:
            local = len(self.nodes[0])
            report.update(local_sent_bytes_per_rank=2 * (local - 1) / local * tensor_bytes,
                          cross_node_sent_bytes_per_rank=cross_per_node / local,
                          cross_node_sent_bytes_per_node=cross_per_node,
                          limit="All local ranks share cross-node work; NIC sharing and topology may limit gains")
        elif strategy == "leader":
            report.update(cross_node_sent_bytes_per_leader=cross_per_node,
                          cross_node_sent_bytes_per_nonleader=0,
                          local_tree_sent_bytes_per_node=[2 * (len(n) - 1) * tensor_bytes for n in self.nodes],
                          limit="The leader carries the full cross-node tensor; one rank/NIC path can bottleneck")
        else:
            raise ValueError("Invalid strategy for this topology")
        return report


@dataclass
class PreparedSUM:
    """Independent reduction storage; never aliases any original local_grad."""

    owner: Any
    epoch: str
    buffers: tuple[Any, ...]
    shards: tuple[Any | None, ...]
    shapes: tuple[tuple[int, ...], ...]
    numels: tuple[int, ...]
    descriptor: str
    consumed: bool = False


@dataclass(frozen=True)
class SUMResult:
    tensors: tuple[Any, ...]
    communication_seconds: float
    strategy: str
    epoch: str


def synchronize_device(tensor: Any) -> None:
    """Complete device work, including work enqueued by asynchronous HCCL."""
    if tensor.device.type != "cpu":
        import torch
        getattr(torch, tensor.device.type).synchronize(tensor.device)


class TopologyCollective:
    """Fixed global-world SUM experiment with separately timed data transfer.

    A Gloo control group agrees on topology, tensor schemas, epoch and call
    sequence before data collectives. It never carries gradient payloads. All
    data groups use the default world's backend, normally HCCL on NPU. This
    requires a PyTorch build with Gloo available in addition to that backend.
    """

    def __init__(self, node_ranks: Iterable[Iterable[int]] | RankTopology, *,
                 strategy: str = "auto", uneven_fallback: str = "leader",
                 epoch: str = "benchmark-0", timeout_seconds: float = 120.0,
                 distributed: Any = None) -> None:
        if distributed is None:
            import torch.distributed as distributed
        self.dist = distributed
        if not self.dist.is_initialized():
            raise ValueError("Initialize the global distributed world first")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self.timeout = timedelta(seconds=timeout_seconds)
        self.rank, self.world_size = self.dist.get_rank(), self.dist.get_world_size()
        self.failed, self.closed, self.sequence = False, False, 0
        self._owner, self._created = object(), []
        self.control_group = self.dist.new_group(ranks=list(range(self.world_size)),
                                                backend="gloo", timeout=self.timeout)
        error, descriptor = None, None
        try:
            raw = node_ranks.nodes if isinstance(node_ranks, RankTopology) else node_ranks
            self.topology = RankTopology.build(raw, self.world_size)
            self.strategy = self.topology.select(strategy, uneven_fallback)
            if not isinstance(epoch, str) or not epoch:
                raise ValueError("epoch must be a nonempty immutable membership identifier")
            self.epoch = epoch
            descriptor = {"nodes": self.topology.nodes, "strategy": self.strategy,
                          "epoch": epoch, "timeout_seconds": timeout_seconds,
                          "backend": str(self.dist.get_backend())}
        except Exception as exc:
            error = str(exc)
        self._agree("construction", descriptor, error)
        self.backend = str(self.dist.get_backend())
        self.node_index, self.local_rank = self.topology.locate(self.rank)
        self.node_ranks = self.topology.nodes[self.node_index]
        self.groups = {}
        try:
            for name, ranks in self.topology.group_schedule(self.strategy):
                group = self.dist.new_group(ranks=list(ranks), backend=self.backend,
                                            timeout=self.timeout)
                self.groups[name] = group
                if self.rank in ranks:
                    self._created.append(group)
        except Exception:
            self.failed = True
            raise

    def _agree(self, phase: str, descriptor: Any, error: str | None = None) -> None:
        if self.failed or self.closed:
            raise RuntimeError("This collective epoch is unusable; exit and rebuild externally")
        local = {"phase": phase, "sequence": self.sequence,
                 "descriptor": descriptor, "error": error}
        reports = [None] * self.world_size
        try:
            self.dist.all_gather_object(reports, local, group=self.control_group)
            errors = [f"rank {rank}: {row['error']}" for rank, row in enumerate(reports) if row["error"]]
            canonical = [{k: v for k, v in row.items() if k != "error"} for row in reports]
            if errors or any(row != canonical[0] for row in canonical[1:]):
                raise ValueError("Collective preflight disagreement: " + ("; ".join(errors) or phase))
        except Exception:
            self.failed = True
            raise

    def prepare(self, local_grads: Sequence[Any], *, epoch: str | None = None) -> PreparedSUM:
        """Collectively copy already-weighted local contributions outside timing.

        Dense floating tensors must have identical shape/dtype order on every
        rank, including zero-valued contributions for ranks with no samples.
        The copy protects local_grad even if a later SUM partially overwrites
        its reduction buffer. Return values are one-use for this epoch.
        """
        import torch

        error, prepared, schema = None, None, None
        try:
            if epoch is not None and epoch != self.epoch:
                raise ValueError("Membership epoch changed; rebuild before using another membership")
            if not local_grads:
                raise ValueError("At least one nonempty gradient tensor is required")
            buffers, shards, shapes, numels, schema = [], [], [], [], []
            device = local_grads[0].device
            for gradient in local_grads:
                if (not isinstance(gradient, torch.Tensor) or gradient.layout != torch.strided
                        or not gradient.is_floating_point() or gradient.numel() == 0
                        or gradient.device != device):
                    raise ValueError("Gradients must be nonempty dense floating tensors on one device")
                elements = gradient.numel()
                local_size = len(self.node_ranks) if self.strategy == "sharded" else 1
                padded = ((elements + local_size - 1) // local_size) * local_size
                with torch.no_grad():
                    buffer = torch.empty(padded, dtype=gradient.dtype, device=device)
                    buffer[:elements].copy_(gradient.detach().reshape(-1))
                    buffer[elements:].zero_()
                    shard = (torch.empty(padded // local_size, dtype=gradient.dtype, device=device)
                             if self.strategy == "sharded" else None)
                buffers.append(buffer)
                shards.append(shard)
                shapes.append(tuple(gradient.shape))
                numels.append(elements)
                schema.append({"shape": list(gradient.shape), "dtype": str(gradient.dtype),
                               "device_type": device.type, "padded_numel": padded})
            prepared = PreparedSUM(self._owner, self.epoch, tuple(buffers), tuple(shards),
                                   tuple(shapes), tuple(numels), json.dumps(schema, sort_keys=True))
        except Exception as exc:
            error = str(exc)
        self._agree("prepare", {"epoch": self.epoch, "tensors": schema}, error)
        assert prepared is not None
        return prepared

    def _wait(self, work: Any) -> None:
        # Do not retry a backend collective or change groups after failure.
        if work.wait(timeout=self.timeout) is False:
            raise TimeoutError("Collective work did not complete within its process-group timeout")

    def _one(self, buffer: Any, shard: Any | None) -> None:
        dist = self.dist
        if self.strategy == "flat":
            if self.world_size > 1:
                self._wait(dist.all_reduce(buffer, op=dist.ReduceOp.SUM,
                                           group=self.groups["flat"], async_op=True))
            return
        node = self.groups[f"node:{self.node_index}"]
        local_size, node_count = len(self.node_ranks), len(self.topology.nodes)
        if self.strategy == "sharded":
            if local_size > 1:
                self._wait(dist.reduce_scatter_tensor(shard, buffer, op=dist.ReduceOp.SUM,
                                                      group=node, async_op=True))
            else:
                shard.copy_(buffer)
            if node_count > 1:
                self._wait(dist.all_reduce(shard, op=dist.ReduceOp.SUM,
                                           group=self.groups[f"stripe:{self.local_rank}"], async_op=True))
            if local_size > 1:
                self._wait(dist.all_gather_into_tensor(buffer, shard, group=node, async_op=True))
            else:
                buffer.copy_(shard)
            return
        leader = self.node_ranks[0]
        if local_size > 1:
            self._wait(dist.reduce(buffer, dst=leader, op=dist.ReduceOp.SUM, group=node, async_op=True))
        if self.rank == leader and node_count > 1:
            self._wait(dist.all_reduce(buffer, op=dist.ReduceOp.SUM,
                                       group=self.groups["leaders"], async_op=True))
        if local_size > 1:
            self._wait(dist.broadcast(buffer, src=leader, group=node, async_op=True))

    def sum(self, prepared: PreparedSUM, *, epoch: str | None = None) -> SUMResult:
        """SUM copied buffers and return complete tensors plus data-path time.

        The timer excludes input copies, allocations and Gloo agreement. It
        includes host submission, Work.wait and the final device synchronize.
        There is no per-bucket device-wide synchronize. Work.wait establishes
        the stream dependency between each local and inter-node stage.
        """
        error, descriptor = None, None
        try:
            if (prepared.owner is not self._owner or prepared.epoch != self.epoch
                    or prepared.consumed or (epoch is not None and epoch != self.epoch)):
                raise ValueError("Reduction buffer is consumed, foreign, or from a stale membership epoch")
            descriptor = {"epoch": self.epoch, "tensors": prepared.descriptor}
        except Exception as exc:
            error = str(exc)
        self._agree("sum", descriptor, error)
        prepared.consumed = True
        try:
            import torch
            synchronize_device(prepared.buffers[0])
            began = time.perf_counter()
            with torch.no_grad():
                for buffer, shard in zip(prepared.buffers, prepared.shards, strict=True):
                    self._one(buffer, shard)
                synchronize_device(prepared.buffers[0])
            elapsed = time.perf_counter() - began
            tensors = tuple(buffer[:numel].view(shape) for buffer, numel, shape
                            in zip(prepared.buffers, prepared.numels, prepared.shapes, strict=True))
            self.sequence += 1
            return SUMResult(tensors, elapsed, self.strategy, self.epoch)
        except Exception:
            self.failed = True
            raise

    def close(self) -> bool:
        """Close successful groups in reverse order; never block in failed cleanup.

        A failed backend can hang destroy_process_group too. In that case this
        returns False and the supervising process must terminate the worker.
        """
        if self.closed:
            return True
        if self.failed:
            return False
        for group in reversed(self._created):
            self.dist.destroy_process_group(group)
        self.dist.destroy_process_group(self.control_group)
        self.closed = True
        return True
