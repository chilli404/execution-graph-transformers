"""Raw PyTorch tensor-parallel benchmark: sequential vs parallel block execution.

Measures communication-computation overlap for a single Transformer block
sharded across 2 GPUs, with no serving-stack overhead.

Usage:
    torchrun --nproc_per_node=2 blackwell/bench_tp_raw.py
"""

import os
import torch
import torch.distributed as dist
import torch.nn.functional as F


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world


def rmsnorm(x):
    return F.rms_norm(x, (x.size(-1),))


class TPBlock:
    """Single Transformer block with manual tensor-parallel sharding."""

    def __init__(self, d_model, n_head, rank, world, device, dtype):
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.heads_per_rank = n_head // world
        self.rank = rank
        self.world = world
        shard = d_model // world  # per-rank hidden dim

        # Column-parallel: shard output dim (each rank holds shard rows)
        self.wq = torch.randn(shard, d_model, device=device, dtype=dtype) * 0.01
        self.wk = torch.randn(shard, d_model, device=device, dtype=dtype) * 0.01
        self.wv = torch.randn(shard, d_model, device=device, dtype=dtype) * 0.01
        self.mlp_up = torch.randn(4 * shard, d_model, device=device, dtype=dtype) * 0.01

        # Row-parallel: shard input dim (each rank holds shard columns)
        self.wo = torch.randn(d_model, shard, device=device, dtype=dtype) * 0.01
        self.mlp_down = torch.randn(d_model, 4 * shard, device=device, dtype=dtype) * 0.01

        self._attn_stream = torch.cuda.Stream(device=device)
        self._mlp_stream = torch.cuda.Stream(device=device)

    def _attention(self, x):
        """QKV project → SDPA → O project + all_reduce."""
        B, T, _ = x.shape
        H, Dh = self.heads_per_rank, self.head_dim
        q = F.linear(x, self.wq).view(B, T, H, Dh).transpose(1, 2)
        k = F.linear(x, self.wk).view(B, T, H, Dh).transpose(1, 2)
        v = F.linear(x, self.wv).view(B, T, H, Dh).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, H * Dh)
        out = F.linear(y, self.wo)           # row-parallel matmul
        dist.all_reduce(out)                  # TP all-reduce
        return out

    def _mlp(self, x):
        """MLP up (column-parallel) → ReLU² → down (row-parallel) + all_reduce."""
        h = F.linear(x, self.mlp_up)
        h = F.relu(h).square()
        out = F.linear(h, self.mlp_down)     # row-parallel matmul
        dist.all_reduce(out)                  # TP all-reduce
        return out

    def forward_sequential(self, x):
        n = rmsnorm(x)
        a = self._attention(n)
        x = x + a
        n2 = rmsnorm(x)
        m = self._mlp(n2)
        return x + m

    def forward_parallel(self, x):
        n = rmsnorm(x)
        a = self._attention(n)
        m = self._mlp(n)
        return x + a + m

    def forward_parallel_overlap(self, x):
        n = rmsnorm(x)
        current = torch.cuda.current_stream(x.device)
        self._attn_stream.wait_stream(current)
        self._mlp_stream.wait_stream(current)
        with torch.cuda.stream(self._attn_stream):
            a = self._attention(n)
        with torch.cuda.stream(self._mlp_stream):
            m = self._mlp(n)
        current.wait_stream(self._attn_stream)
        current.wait_stream(self._mlp_stream)
        a.record_stream(current)
        m.record_stream(current)
        return x + a + m


def bench(fn, x, warmup=50, repeats=200):
    """Time a function using CUDA events. Returns median ms."""
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]  # median


def main():
    rank, world = setup()
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    d_model = 4096
    n_head = 32
    seq_len = 512

    block = TPBlock(d_model, n_head, rank, world, device, dtype)

    if rank == 0:
        print(f"TP={world} on {torch.cuda.get_device_name(0)}")
        print(f"d_model={d_model}, n_head={n_head}, seq_len={seq_len}, dtype={dtype}")
        print(f"warmup=50, repeats=200\n")
        print(f"{'Batch':>5} | {'Sequential':>12} | {'Parallel':>12} | {'Par+Overlap':>12} | {'Par speedup':>11} | {'Overlap speedup':>15}")
        print(f"{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*11}-+-{'-'*15}")

    for B in [1, 4, 8, 16, 32]:
        x = torch.randn(B, seq_len, d_model, device=device, dtype=dtype)

        with torch.no_grad():
            t_seq = bench(block.forward_sequential, x)
            t_par = bench(block.forward_parallel, x)
            t_ovl = bench(block.forward_parallel_overlap, x)

        if rank == 0:
            par_sp = t_seq / t_par
            ovl_sp = t_seq / t_ovl
            print(f"{B:>5} | {t_seq:>10.3f}ms | {t_par:>10.3f}ms | {t_ovl:>10.3f}ms | {par_sp:>10.3f}x | {ovl_sp:>14.3f}x")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
