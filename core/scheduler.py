from enum import Enum
from typing import List, Optional

try:
    from .block_manager import BlockManager
except ImportError:
    from core.block_manager import BlockManager


class SequenceStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


class Sequence:
    """Represents a generation request."""
    def __init__(self, seq_id: str, prompt_tokens: List[int], max_tokens: int = 128):
        self.seq_id = seq_id
        self.prompt_tokens = prompt_tokens
        self.output_tokens: List[int] = []
        self.status = SequenceStatus.WAITING
        self.max_tokens = max_tokens
        self.cached_tokens = 0  # number of tokens reused via Prefix Caching
        self.num_preemptions = 0

        # Timing hooks filled in by the engine (perf_counter seconds)
        self.arrival_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.token_times: List[float] = []
        self.finish_time: Optional[float] = None

    @property
    def total_len(self) -> int:
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def all_tokens(self) -> List[int]:
        return self.prompt_tokens + self.output_tokens

    def is_finished(self, eos_token_id = None) -> bool:
        if len(self.output_tokens) >= self.max_tokens:
            return True
        if eos_token_id is not None and self.output_tokens:
            last = self.output_tokens[-1]
            if isinstance(eos_token_id, (set, list, tuple)):
                if last in eos_token_id:
                    return True
            elif last == eos_token_id:
                return True
        return False


class ScheduleBatch:
    """Batch of requests scheduled for the current iteration."""
    def __init__(self):
        self.prefill_seqs: List[Sequence] = []
        self.decode_seqs: List[Sequence] = []
        # physical KV slot for the token each decode sequence processes
        self.decode_slots: List[int] = []

    @property
    def is_empty(self) -> bool:
        return len(self.prefill_seqs) == 0 and len(self.decode_seqs) == 0


class ContinuousScheduler:
    """
    Iteration-level Continuous Batching Scheduler.
    Interleaves Prefill and Decode across requests dynamically.

    Preemption policy is *recompute*: a preempted sequence releases its KV
    blocks and is re-queued; on re-admission its prompt + already generated
    tokens are prefilled again (prefix caching makes this cheap).
    """
    def __init__(
        self,
        block_manager: BlockManager,
        max_batch_size: int = 32,
        max_batched_tokens: int = 2048,
        eos_token_id: Optional[int] = 2
    ):
        self.block_manager = block_manager
        self.max_batch_size = max_batch_size
        self.max_batched_tokens = max_batched_tokens
        self.eos_token_id = eos_token_id

        self.waiting_queue: List[Sequence] = []
        self.running_queue: List[Sequence] = []
        self.last_batch: ScheduleBatch = ScheduleBatch()

    def add_sequence(self, seq: Sequence):
        self.waiting_queue.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting_queue or self.running_queue)

    def finish_sequence(self, seq: Sequence):
        """Mark a sequence finished and release its KV blocks."""
        seq.status = SequenceStatus.FINISHED
        self.block_manager.free_sequence(seq.seq_id)
        if seq in self.running_queue:
            self.running_queue.remove(seq)

    def _preempt(self, seq: Sequence):
        print(f"[Scheduler] KV memory constrained; preempting {seq.seq_id}")
        seq.status = SequenceStatus.WAITING
        seq.num_preemptions += 1
        self.block_manager.free_sequence(seq.seq_id)
        self.waiting_queue.insert(0, seq)

    def schedule(self) -> ScheduleBatch:
        """
        Plans the next iteration step:
        1. Keeps running decode sequences (if memory permits).
        2. Admits waiting prompt sequences (Prefill) with Prefix Caching.
        """
        batch = ScheduleBatch()
        curr_tokens = 0

        # 1. Schedule RUNNING decode sequences (1 token each)
        remaining_running = []
        for seq in self.running_queue:
            if seq.is_finished(self.eos_token_id):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.free_sequence(seq.seq_id)
                continue

            # The token fed this step is output_tokens[-1] at position total_len-1;
            # make sure a KV slot exists for it.
            try:
                slot = self.block_manager.append_slot(seq.seq_id, seq.total_len - 1)
                batch.decode_seqs.append(seq)
                batch.decode_slots.append(slot)
                curr_tokens += 1
                remaining_running.append(seq)
            except RuntimeError:
                # KV Cache full: preempt (recompute later)
                self._preempt(seq)

        self.running_queue = remaining_running

        # 2. Admit WAITING sequences for Prefill if slots and tokens remain
        while self.waiting_queue and len(batch.decode_seqs) + len(batch.prefill_seqs) < self.max_batch_size:
            seq = self.waiting_queue[0]
            prefill_tokens = seq.all_tokens

            # Conservative token-budget check (before prefix-cache discount)
            if batch.prefill_seqs and curr_tokens + len(prefill_tokens) > self.max_batched_tokens:
                break

            try:
                cached_tokens = self.block_manager.allocate_for_prompt(seq.seq_id, prefill_tokens)
            except RuntimeError:
                # Insufficient free blocks right now
                break

            seq.cached_tokens = cached_tokens
            self.waiting_queue.pop(0)
            seq.status = SequenceStatus.RUNNING
            batch.prefill_seqs.append(seq)
            self.running_queue.append(seq)
            curr_tokens += len(prefill_tokens) - cached_tokens

        self.last_batch = batch
        return batch
