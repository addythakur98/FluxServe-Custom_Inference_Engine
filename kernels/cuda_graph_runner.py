from typing import Callable, Dict, List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None


def _default_width_buckets(max_blocks_per_seq: int) -> List[int]:
    """Powers of two from 8 blocks (128 tokens @ block_size 16) up to the max."""
    widths, w = [], 8
    while w < max_blocks_per_seq:
        widths.append(w)
        w *= 2
    widths.append(max_blocks_per_seq)
    return widths


class CUDAGraphRunner:
    """
    Manages CUDA Graph capture and replay for the Decode phase.

    Why CUDA Graphs for LLM Decode?
    In token-by-token generation, each step only generates 1 token per sequence.
    Kernel launch latencies from Python/PyTorch dominate the runtime.
    CUDA Graphs bake the entire forward pass (attention, MLP, norms)
    into a static execution graph, launching all kernels with near-zero CPU overhead.

    Graphs are captured on a 2-D grid of buckets:
        * batch size            (number of decode sequences)
        * block-table width     (KV context length, in blocks)
    Bucketing the width means short contexts only gather/attend over a short
    static window, while long contexts still get a graph instead of silently
    falling back to eager decode.

    ``model_forward_fn(tokens, positions, block_tables, seq_lens, slot_mapping)``
    must be free of host synchronisation and data-dependent shapes:
        tokens / positions: [bs, 1] long
        block_tables:       [bs, width] int32
        seq_lens:           [bs] int32
        slot_mapping:       [bs] long
    Padding rows (bucket size > live batch) write their KV into a reserved
    ``pad_block_id`` so they never corrupt real sequences.
    """
    def __init__(
        self,
        model_forward_fn: Callable,
        max_batch_size: int = 32,
        max_blocks_per_seq: int = 128,
        batch_buckets: Optional[List[int]] = None,
        width_buckets: Optional[List[int]] = None,
        device: str = "cuda",
        pad_block_id: int = 0,
        block_size: int = 16,
    ):
        self.model_forward_fn = model_forward_fn
        self.max_batch_size = max_batch_size
        self.max_blocks_per_seq = max_blocks_per_seq
        self.device = device
        self.pad_block_id = pad_block_id
        self.pad_slot = pad_block_id * block_size

        # Batch-size buckets
        buckets = batch_buckets or [1, 2, 4, 8, 16, 24, 32, 48, 64]
        self.batch_buckets = sorted(b for b in buckets if b <= max_batch_size)
        if not self.batch_buckets or self.batch_buckets[-1] < max_batch_size:
            self.batch_buckets.append(max_batch_size)

        # Block-table width (context length) buckets
        widths = width_buckets or _default_width_buckets(max_blocks_per_seq)
        self.width_buckets = sorted(set(w for w in widths if 0 < w <= max_blocks_per_seq))
        if not self.width_buckets or self.width_buckets[-1] < max_blocks_per_seq:
            self.width_buckets.append(max_blocks_per_seq)

        # Captured graphs: (bs, width) -> (torch.cuda.CUDAGraph, static_buffers)
        self.graphs: Dict[Tuple[int, int], Tuple] = {}
        self.is_captured = False
        self.enabled = True
        self.num_replays = 0
        self.num_eager = 0

    def can_use_cuda_graph(self) -> bool:
        return (
            torch is not None
            and torch.cuda.is_available()
            and str(self.device).startswith("cuda")
        )

    def _alloc_buffers(self, bs: int, width: int) -> dict:
        return {
            "tokens": torch.zeros((bs, 1), dtype=torch.long, device=self.device),
            "positions": torch.zeros((bs, 1), dtype=torch.long, device=self.device),
            "block_tables": torch.full(
                (bs, width), self.pad_block_id, dtype=torch.int32, device=self.device
            ),
            "seq_lens": torch.ones((bs,), dtype=torch.int32, device=self.device),
            "slot_mapping": torch.full((bs,), self.pad_slot, dtype=torch.long, device=self.device),
        }

    def _call(self, buf: dict):
        return self.model_forward_fn(
            buf["tokens"], buf["positions"], buf["block_tables"], buf["seq_lens"], buf["slot_mapping"]
        )

    @staticmethod
    def _pick(buckets: List[int], value: int) -> Optional[int]:
        return next((b for b in buckets if b >= value), None)

    def capture(self, warmup_iters: int = 2):
        """
        Pre-captures CUDA Graphs for every (batch size, width) bucket.
        Must be called once during engine initialization (after weights and
        KV caches are allocated). All graphs share one memory pool.

        If capture fails for any reason (e.g. an unsupported op, or an OOM on
        a small GPU) we transparently fall back to eager decode instead of
        crashing engine construction.
        """
        if not self.can_use_cuda_graph():
            print("[CUDA Graph] CUDA not available; skipping graph capture (eager decode).")
            return
        with torch.no_grad():
            self._capture_all(warmup_iters)

    def _capture_all(self, warmup_iters: int):
        print(f"[CUDA Graph] Capturing graphs for batch buckets {self.batch_buckets} "
              f"x width buckets {self.width_buckets} (blocks)")
        pool = None
        try:
            # Capture largest first so smaller graphs reuse its memory pool
            keys = sorted(
                ((bs, w) for bs in self.batch_buckets for w in self.width_buckets),
                key=lambda k: (k[0] * k[1], k[0]),
                reverse=True,
            )
            for bs, w in keys:
                buffers = self._alloc_buffers(bs, w)

                # 1. Warm up on a side stream (cuBLAS handles, allocator state)
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(warmup_iters):
                        self._call(buffers)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()

                # 2. Capture the graph
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    buffers["out"] = self._call(buffers)
                if pool is None:
                    pool = graph.pool()

                self.graphs[(bs, w)] = (graph, buffers)
        except Exception as exc:  # pragma: no cover - depends on CUDA runtime
            print(f"[CUDA Graph] Capture failed ({exc!r}); falling back to eager decode.")
            self.graphs.clear()
            self.is_captured = False
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            return

        torch.cuda.synchronize()
        self.is_captured = True
        print(f"[CUDA Graph] Capture complete ({len(self.graphs)} graphs).")

    def replay(
        self,
        input_tokens: "torch.Tensor",
        positions: "torch.Tensor",
        block_tables: "torch.Tensor",
        seq_lens: "torch.Tensor",
        slot_mapping: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Executes a decode step using the best-fit captured CUDA Graph.
        Pads the dynamic batch to the nearest (batch, width) bucket, copies
        data into static buffers, and replays the graph in a single driver
        call. Falls back to eager execution if no graph fits.
        """
        actual_bs = input_tokens.shape[0]
        width = block_tables.shape[1]

        key = (self._pick(self.batch_buckets, actual_bs), self._pick(self.width_buckets, width))

        if not self.enabled or not self.is_captured or key not in self.graphs:
            # Fallback to eager execution
            self.num_eager += 1
            return self.model_forward_fn(input_tokens, positions, block_tables, seq_lens, slot_mapping)

        bucket_bs = key[0]
        graph, buf = self.graphs[key]

        # Copy dynamic inputs into static graph memory buffers
        buf["tokens"][:actual_bs].copy_(input_tokens)
        buf["positions"][:actual_bs].copy_(positions)
        buf["block_tables"].fill_(self.pad_block_id)
        buf["block_tables"][:actual_bs, :width].copy_(block_tables)
        buf["seq_lens"][:actual_bs].copy_(seq_lens)
        buf["slot_mapping"][:actual_bs].copy_(slot_mapping)

        # Neutralise padding rows: attend to 1 token in the pad block and
        # write their (garbage) KV into the pad block only.
        if bucket_bs > actual_bs:
            buf["tokens"][actual_bs:].zero_()
            buf["positions"][actual_bs:].zero_()
            buf["seq_lens"][actual_bs:].fill_(1)
            buf["slot_mapping"][actual_bs:].fill_(self.pad_slot)

        # Replay the static graph
        graph.replay()
        self.num_replays += 1

        # Slice out valid outputs
        return buf["out"][:actual_bs]
