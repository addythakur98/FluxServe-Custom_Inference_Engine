"""
Custom LLM Inference Engine.

Unites all subsystems into a serving loop around a paged-KV model
(:class:`qwen2_model.Qwen2Model`):

    - Continuous Batching   (scheduler.ContinuousScheduler, iteration-level)
    - PagedAttention        (block_manager.BlockManager + paged KV caches)
    - Prefix Caching        (prefix_cache.PrefixCacheRadixTree, block reuse)
    - Prefill attention     (PyTorch SDPA, incl. prefill on cached context)
    - CUDA Graph Decode     (cuda_graph_runner.CUDAGraphRunner, bucketed)

Typical use::

    engine = CustomInferenceEngine(model, stop_token_ids=[tokenizer.eos_token_id])
    outputs = engine.generate([tokenizer.encode("Hello")], max_tokens=32)
    print(outputs[0].output_tokens, outputs[0].ttft_ms)
"""

import gc
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import torch

try:
    from .block_manager import BlockManager
    from ..kernels.cuda_graph_runner import CUDAGraphRunner
    from .scheduler import ContinuousScheduler, Sequence, ScheduleBatch
except ImportError:
    from core.block_manager import BlockManager
    from kernels.cuda_graph_runner import CUDAGraphRunner
    from core.scheduler import ContinuousScheduler, Sequence, ScheduleBatch


@dataclass
class RequestOutput:
    """Final result + timing of one request."""
    seq_id: str
    prompt_tokens: List[int]
    output_tokens: List[int]
    cached_tokens: int = 0
    num_preemptions: int = 0
    arrival_time: float = 0.0
    first_token_time: float = 0.0
    finish_time: float = 0.0
    token_times: List[float] = field(default_factory=list)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    @property
    def output_len(self) -> int:
        return len(self.output_tokens)

    @property
    def ttft_ms(self) -> float:
        """Time To First Token (includes queueing delay)."""
        return (self.first_token_time - self.arrival_time) * 1000

    @property
    def itl_ms(self) -> List[float]:
        """Inter-token latencies between consecutive generated tokens."""
        t = self.token_times
        return [(t[i] - t[i - 1]) * 1000 for i in range(1, len(t))]

    @property
    def tpot_ms(self) -> float:
        """Time Per Output Token (excluding the first token)."""
        if self.output_len <= 1:
            return 0.0
        return (self.finish_time - self.first_token_time) * 1000 / (self.output_len - 1)

    @property
    def e2e_ms(self) -> float:
        return (self.finish_time - self.arrival_time) * 1000


class CustomInferenceEngine:
    """
    End-to-end inference engine: continuous batching over a paged KV cache with
    radix-tree prefix caching and CUDA-Graph decode.
    """
    def __init__(
        self,
        model: "torch.nn.Module",
        num_blocks: int = 1024,
        block_size: int = 16,
        max_batch_size: int = 32,
        max_batched_tokens: int = 4096,
        max_model_len: int = 2048,
        stop_token_ids: Optional[Iterable[int]] = None,
        enable_prefix_caching: bool = True,
        use_cuda_graph: bool = True,
        graph_max_blocks_per_seq: Optional[int] = None,
        temperature: float = 0.0,
    ):
        self.model = model.eval()
        param = next(model.parameters())
        self.device = param.device
        self.dtype = param.dtype
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.temperature = temperature
        self.stop_token_ids = set(stop_token_ids or [])
        self.max_model_len = min(max_model_len, model.config.max_model_len)
        self.max_blocks_per_seq = math.ceil(self.max_model_len / block_size)

        # 1. Paged KV Memory Allocator & Prefix Cache
        self.block_manager = BlockManager(
            num_blocks=num_blocks, block_size=block_size,
            enable_prefix_caching=enable_prefix_caching,
        )

        # 2. Continuous Batching Scheduler
        self.scheduler = ContinuousScheduler(
            block_manager=self.block_manager,
            max_batch_size=max_batch_size,
            max_batched_tokens=max_batched_tokens,
            eos_token_id=self.stop_token_ids,
        )

        # 3. Physical KV caches: one extra "pad" block absorbs writes from
        #    CUDA-graph padding rows.
        self.pad_block_id = num_blocks
        self.key_caches, self.value_caches = model.allocate_kv_caches(num_blocks + 1, block_size)

        # 4. CUDA Graph Runner for the decode phase.
        #    The graph uses a static block-table width; cap it so short/medium
        #    contexts do not gather the entire max_model_len worth of KV every
        #    decode step. Longer contexts automatically fall back to eager
        #    PagedAttention (still correct, just without graph launch savings).
        if graph_max_blocks_per_seq is None:
            graph_max_blocks_per_seq = min(self.max_blocks_per_seq, 64)
        graph_max_blocks_per_seq = max(1, min(graph_max_blocks_per_seq, self.max_blocks_per_seq))
        self.cuda_graph_runner = CUDAGraphRunner(
            model_forward_fn=self._decode_forward,
            max_batch_size=max_batch_size,
            max_blocks_per_seq=graph_max_blocks_per_seq,
            device=str(self.device),
            pad_block_id=self.pad_block_id,
            block_size=block_size,
        )
        self.cuda_graph_runner.enabled = bool(use_cuda_graph)
        if self.cuda_graph_runner.enabled and self.device.type == "cuda":
            self.cuda_graph_runner.capture()

        self._next_id = 0
        self.num_steps = 0
        self.num_prefill_tokens_computed = 0
        self.num_prefill_tokens_cached = 0
        self.num_decode_tokens = 0
        self.reset_kv_accounting()

    # ------------------------------------------------------------------
    # Request API
    # ------------------------------------------------------------------
    def add_request(self, prompt_tokens: List[int], max_tokens: int = 32) -> str:
        prompt_tokens = list(prompt_tokens)
        if not prompt_tokens:
            raise ValueError("Empty prompt")
        if len(prompt_tokens) >= self.max_model_len:
            raise ValueError(
                f"Prompt length {len(prompt_tokens)} exceeds max_model_len={self.max_model_len}"
            )
        max_tokens = max(1, min(max_tokens, self.max_model_len - len(prompt_tokens)))
        needed_blocks = math.ceil((len(prompt_tokens) + max_tokens) / self.block_size)
        if needed_blocks > self.num_blocks:
            raise ValueError(f"Request needs {needed_blocks} KV blocks but only {self.num_blocks} exist")

        seq_id = f"seq_{self._next_id}"
        self._next_id += 1
        seq = Sequence(seq_id=seq_id, prompt_tokens=prompt_tokens, max_tokens=max_tokens)
        seq.arrival_time = time.perf_counter()
        self.scheduler.add_sequence(seq)
        return seq_id

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished()

    def generate(self, prompts: List[List[int]], max_tokens: int = 32) -> List[RequestOutput]:
        """Submit all prompts at once and run continuous batching to completion."""
        ids = [self.add_request(p, max_tokens) for p in prompts]
        results: Dict[str, RequestOutput] = {}
        while self.has_unfinished_requests():
            for out in self.step():
                results[out.seq_id] = out
        return [results[i] for i in ids]

    # ------------------------------------------------------------------
    # Model execution
    # ------------------------------------------------------------------
    def _decode_forward(
        self,
        tokens: "torch.Tensor",
        positions: "torch.Tensor",
        block_tables: "torch.Tensor",
        seq_lens: "torch.Tensor",
        slot_mapping: "torch.Tensor",
    ) -> "torch.Tensor":
        """Captured inside CUDA Graph for decode acceleration."""
        return self.model.forward_decode(
            tokens, positions, self.key_caches, self.value_caches,
            block_tables, seq_lens, slot_mapping,
        )

    def _sample(self, logits: "torch.Tensor") -> List[int]:
        """logits: [num_seqs, vocab] -> sampled token ids."""
        if self.temperature <= 0:
            return torch.argmax(logits, dim=-1).tolist()
        probs = torch.softmax(logits.float() / self.temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1).tolist()

    def _run_prefill(self, seq: Sequence) -> None:
        """Prefill one sequence (FlashAttention path) on top of any cached prefix."""
        tokens = seq.all_tokens           # prompt (+ outputs if re-prefilled after preemption)
        ctx = seq.cached_tokens
        table = self.block_manager.get_block_table(seq.seq_id)
        bs = self.block_size
        dev = self.device

        slots = [table[p // bs] * bs + p % bs for p in range(ctx, len(tokens))]
        input_ids = torch.tensor([tokens[ctx:]], dtype=torch.long, device=dev)
        positions = torch.arange(ctx, len(tokens), dtype=torch.long, device=dev).unsqueeze(0)
        slot_t = torch.tensor(slots, dtype=torch.long, device=dev)
        bt = torch.tensor([table], dtype=torch.int32, device=dev)

        logits = self.model.forward_prefill(
            input_ids, positions,
            key_caches=self.key_caches, value_caches=self.value_caches,
            slot_mapping=slot_t, block_table=bt, context_len=ctx,
            last_token_only=True,
        )
        next_token = self._sample(logits[:, -1, :])[0]

        # Publish full prompt blocks so later requests can share them
        self.block_manager.cache_prefix(seq.seq_id, seq.prompt_tokens)

        seq.output_tokens.append(next_token)
        now = time.perf_counter()
        if seq.first_token_time is None:
            seq.first_token_time = now
        seq.token_times.append(now)
        self.num_prefill_tokens_computed += len(tokens) - ctx
        self.num_prefill_tokens_cached += ctx

    def _run_decode(self, batch: ScheduleBatch) -> None:
        """Batched single-token decode (PagedAttention + CUDA Graph path)."""
        seqs = batch.decode_seqs
        dev = self.device

        tables = [self.block_manager.get_block_table(s.seq_id) for s in seqs]
        width = max(len(t) for t in tables)
        bt = torch.tensor(
            [t + [self.pad_block_id] * (width - len(t)) for t in tables],
            dtype=torch.int32, device=dev,
        )
        tokens = torch.tensor([[s.output_tokens[-1]] for s in seqs], dtype=torch.long, device=dev)
        positions = torch.tensor([[s.total_len - 1] for s in seqs], dtype=torch.long, device=dev)
        seq_lens = torch.tensor([s.total_len for s in seqs], dtype=torch.int32, device=dev)
        slots = torch.tensor(batch.decode_slots, dtype=torch.long, device=dev)

        logits = self.cuda_graph_runner.replay(tokens, positions, bt, seq_lens, slots)
        next_tokens = self._sample(logits[:, -1, :])

        now = time.perf_counter()
        for s, tok in zip(seqs, next_tokens):
            s.output_tokens.append(tok)
            s.token_times.append(now)
        self.num_decode_tokens += len(seqs)

    @staticmethod
    def _to_output(seq: Sequence) -> RequestOutput:
        return RequestOutput(
            seq_id=seq.seq_id,
            prompt_tokens=seq.prompt_tokens,
            output_tokens=list(seq.output_tokens),
            cached_tokens=seq.cached_tokens,
            num_preemptions=seq.num_preemptions,
            arrival_time=seq.arrival_time,
            first_token_time=seq.first_token_time,
            finish_time=seq.finish_time,
            token_times=list(seq.token_times),
        )

    def step(self) -> List[RequestOutput]:
        """
        Executes one iteration of continuous batching:
        - Prefill newly admitted prompts (with Prefix Caching + FlashAttention)
        - Decode active sequences (with PagedAttention + CUDA Graph replay)
        Returns:
            outputs of requests that finished during this step
        """
        with torch.no_grad():
            batch: ScheduleBatch = self.scheduler.schedule()
            if batch.is_empty:
                if self.scheduler.has_unfinished():
                    raise RuntimeError("Scheduler made no progress (insufficient KV blocks?)")
                return []
            self._account_kv(batch.prefill_seqs + batch.decode_seqs)

            for seq in batch.prefill_seqs:
                self._run_prefill(seq)
            if batch.decode_seqs:
                self._run_decode(batch)
            self.num_steps += 1

            finished: List[RequestOutput] = []
            now = time.perf_counter()
            for seq in batch.prefill_seqs + batch.decode_seqs:
                if seq.is_finished(self.stop_token_ids):
                    seq.finish_time = now
                    self.scheduler.finish_sequence(seq)
                    finished.append(self._to_output(seq))
            return finished

    # ------------------------------------------------------------------
    # Introspection & Memory / Capacity Profiling
    # ------------------------------------------------------------------
    def set_cuda_graph_enabled(self, enabled: bool) -> None:
        """Dynamically enable or disable CUDA Graph decode replay without reallocating weights/KV."""
        self.cuda_graph_runner.enabled = bool(enabled)
        if self.cuda_graph_runner.enabled and not self.cuda_graph_runner.is_captured and self.device.type == "cuda":
            self.cuda_graph_runner.capture()
        self.cuda_graph_runner.num_replays = 0
        self.cuda_graph_runner.num_eager = 0

    def clear_prefix_cache(self) -> None:
        self.block_manager.clear_prefix_cache()

    # ------------------------------------------------------------------
    # KV-cache memory accounting
    # ------------------------------------------------------------------
    def reset_kv_accounting(self) -> None:
        self.kv_slots_used = 0           # token slots actually holding KV
        self.kv_slots_paged = 0          # slots reserved by the paged allocator (per sequence)
        self.kv_slots_paged_unique = 0   # same, counting prefix-shared blocks once
        self.kv_slots_static = 0         # slots a static allocator reserves (prompt + max_tokens)
        self.kv_slots_maxlen = 0         # slots a static allocator reserves (max_model_len)
        self.kv_samples = 0

    def _account_kv(self, seqs: List[Sequence]) -> None:
        """
        Sample KV-slot usage for the sequences scheduled this step (summed over
        steps, i.e. time-weighted). After this step each sequence's KV holds
        ``total_len`` tokens. Baselines: a static, contiguous allocator that
        reserves either ``prompt_len + max_tokens`` or ``max_model_len`` slots
        per request up front.
        """
        bs = self.block_size
        bm = self.block_manager
        unique_blocks = set()
        for s in seqs:
            table = bm.get_block_table(s.seq_id)
            self.kv_slots_used += s.total_len
            self.kv_slots_paged += len(table) * bs
            self.kv_slots_static += len(s.prompt_tokens) + s.max_tokens
            self.kv_slots_maxlen += self.max_model_len
            unique_blocks.update(table)
        self.kv_slots_paged_unique += len(unique_blocks) * bs
        self.kv_samples += 1

    def kv_memory_stats(self) -> Dict[str, float]:
        """
        Time-weighted KV-cache waste (reserved slots not holding a live token):
            paged_waste_pct    : internal fragmentation of the last block per sequence
            static_waste_pct   : waste of a prompt+max_tokens contiguous reservation
            maxlen_waste_pct   : waste of a max_model_len contiguous reservation
            waste_reduction_pct / waste_reduction_vs_maxlen_pct:
                                 1 - paged_waste_slots / baseline_waste_slots (negative = paged wastes more)
            prefix_sharing_savings_pct: reserved slots saved by sharing prefix blocks
        """
        used, paged, static = self.kv_slots_used, self.kv_slots_paged, self.kv_slots_static
        maxlen = self.kv_slots_maxlen
        paged_waste, static_waste, maxlen_waste = paged - used, static - used, maxlen - used
        return {
            "samples": self.kv_samples,
            "slots_used": used,
            "slots_paged": paged,
            "slots_paged_unique": self.kv_slots_paged_unique,
            "slots_static": static,
            "slots_maxlen": maxlen,
            "paged_waste_pct": 100.0 * paged_waste / paged if paged else 0.0,
            "static_waste_pct": 100.0 * static_waste / static if static else 0.0,
            "maxlen_waste_pct": 100.0 * maxlen_waste / maxlen if maxlen else 0.0,
            "waste_reduction_pct": 100.0 * (1 - paged_waste / static_waste) if static_waste > 0 else 0.0,
            "waste_reduction_vs_maxlen_pct": 100.0 * (1 - paged_waste / maxlen_waste) if maxlen_waste > 0 else 0.0,
            "prefix_sharing_savings_pct": 100.0 * (1 - self.kv_slots_paged_unique / paged) if paged else 0.0,
        }

    def release_memory(self) -> None:
        """
        Explicitly release all CUDA Graph pools, static buffers, paged KV cache tensors,
        and model weights from GPU VRAM so external baselines (e.g., HF / vLLM) can run
        in the same notebook session without OutOfMemoryError.
        """
        if hasattr(self, "cuda_graph_runner") and self.cuda_graph_runner is not None:
            self.cuda_graph_runner.graphs.clear()
            self.cuda_graph_runner.is_captured = False
            self.cuda_graph_runner.model_forward_fn = None
        if hasattr(self, "key_caches"):
            self.key_caches.clear()
        if hasattr(self, "value_caches"):
            self.value_caches.clear()
        if hasattr(self, "model") and self.model is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def get_memory_profile(
        self,
        avg_prompt_len: int = 128,
        max_new_tokens: int = 64,
        shared_prefix_tokens: int = 0,
    ) -> Dict[str, float]:
        """Compute detailed GPU memory breakdown and max concurrent sequence limits before preemption."""
        model_weight_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e6
        kv_cache_mb = sum(t.numel() * t.element_size() for t in self.key_caches + self.value_caches) / 1e6

        peak_alloc_mb = 0.0
        peak_reserved_mb = 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            peak_alloc_mb = torch.cuda.max_memory_allocated(self.device) / 1e6
            peak_reserved_mb = torch.cuda.max_memory_reserved(self.device) / 1e6

        # Unshared capacity: blocks per full sequence (prompt + output)
        total_toks_per_seq = max(1, avg_prompt_len + max_new_tokens)
        blocks_per_seq_unshared = max(1, math.ceil(total_toks_per_seq / self.block_size))
        max_seqs_unshared = self.num_blocks // blocks_per_seq_unshared

        # Prefix-shared capacity: shared prefix blocks allocated once in Radix tree
        shared_blocks = min(self.num_blocks - 1, max(0, shared_prefix_tokens // self.block_size))
        unshared_toks = max(1, (avg_prompt_len - shared_blocks * self.block_size) + max_new_tokens)
        blocks_per_seq_shared = max(1, math.ceil(unshared_toks / self.block_size))
        max_seqs_shared = max(0, (self.num_blocks - shared_blocks) // blocks_per_seq_shared)

        return {
            "model_weight_memory_mb": model_weight_mb,
            "kv_cache_memory_mb": kv_cache_mb,
            "peak_allocated_memory_mb": peak_alloc_mb,
            "peak_reserved_memory_mb": peak_reserved_mb,
            "blocks_per_seq_unshared": blocks_per_seq_unshared,
            "max_concurrent_seqs_unshared": max_seqs_unshared,
            "max_concurrent_seqs_prefix_shared": max_seqs_shared,
        }

    def get_stats(self) -> Dict[str, float]:
        bm = self.block_manager
        mem = self.get_memory_profile()
        return {
            "steps": self.num_steps,
            "prefill_tokens_computed": self.num_prefill_tokens_computed,
            "prefill_tokens_cached": self.num_prefill_tokens_cached,
            "decode_tokens": self.num_decode_tokens,
            "prefix_cache_hit_rate": bm.prefix_hit_rate,
            "prefix_cached_blocks": bm.prefix_cache.num_cached_blocks,
            "evicted_blocks": bm.num_evicted_blocks,
            "kv_block_usage": bm.usage(),
            "cuda_graph_replays": self.cuda_graph_runner.num_replays,
            "eager_decode_steps": self.cuda_graph_runner.num_eager,
            "kv_cache_mb": mem["kv_cache_memory_mb"],
            "model_weight_mb": mem["model_weight_memory_mb"],
            "peak_allocated_mb": mem["peak_allocated_memory_mb"],
            "peak_reserved_mb": mem["peak_reserved_memory_mb"],
            "max_concurrent_seqs_unshared": mem["max_concurrent_seqs_unshared"],
        }
