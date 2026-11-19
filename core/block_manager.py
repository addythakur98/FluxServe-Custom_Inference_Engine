from typing import Dict, List

try:
    from .prefix_cache import Block, PrefixCacheRadixTree
except ImportError:
    from core.prefix_cache import Block, PrefixCacheRadixTree


class BlockManager:
    """
    Manages physical KV cache memory blocks and per-sequence block tables.
    Supports Paged KV cache allocation, block recycling, and Prefix Caching.

    Reference counting:
        * every sequence whose block table contains a block holds 1 reference
        * the prefix-cache radix tree holds 1 reference on each cached block
    A block returns to the free pool only when its ref_count drops to 0.
    When the free pool is exhausted, least-recently-used cached prefix blocks
    that no running sequence is using are evicted from the radix tree.
    """
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching

        # Physical block pool
        self.blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: List[int] = list(range(num_blocks))

        # Sequence logical -> physical block mapping: seq_id -> List[block_id]
        self.block_tables: Dict[str, List[int]] = {}

        # Prefix cache radix tree
        self.prefix_cache = PrefixCacheRadixTree(block_size=block_size)

        # Statistics
        self.prefix_query_tokens = 0
        self.prefix_hit_tokens = 0
        self.num_evicted_blocks = 0

    # ------------------------------------------------------------------
    # Eviction & block pool
    # ------------------------------------------------------------------
    def _evict(self, num_needed: int) -> None:
        """Evict LRU prefix-cache leaves until ``num_needed`` blocks are free."""
        while len(self.free_block_ids) < num_needed:
            evicted_any = False
            for leaf in self.prefix_cache.evictable_leaves():
                if all(self.blocks[b].ref_count == 1 for b in leaf.block_ids):
                    for b_id in self.prefix_cache.remove_leaf(leaf):
                        self._release(b_id)
                        self.num_evicted_blocks += 1
                    evicted_any = True
                    break
            if not evicted_any:
                return

    def _pop_free_block(self) -> int:
        if not self.free_block_ids:
            self._evict(1)
        if not self.free_block_ids:
            raise RuntimeError("Out of KV cache memory")
        b_id = self.free_block_ids.pop()
        self.blocks[b_id].ref_count = 1
        return b_id

    def _release(self, b_id: int) -> None:
        block = self.blocks[b_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_block_ids.append(b_id)

    # ------------------------------------------------------------------
    # Allocation API
    # ------------------------------------------------------------------
    def allocate_for_prompt(self, seq_id: str, prompt_tokens: List[int]) -> int:
        """
        Allocate blocks for a new prompt sequence.
        1. Checks prefix cache for matching shared blocks.
        2. Allocates new blocks for the remaining unshared tokens.
        Returns:
            cached_token_count: number of prefix tokens reused (no need to prefill)
        """
        if self.enable_prefix_caching:
            matched_blocks, cached_tokens = self.prefix_cache.match_prefix(prompt_tokens)
        else:
            matched_blocks, cached_tokens = [], 0

        # Always recompute at least the final prompt token so we get logits
        # for the first generated token (and never write into a shared block).
        if cached_tokens >= len(prompt_tokens) and matched_blocks:
            matched_blocks = matched_blocks[:-1]
            cached_tokens -= self.block_size

        remaining_tokens = len(prompt_tokens) - cached_tokens
        num_new_blocks = (remaining_tokens + self.block_size - 1) // self.block_size

        # Pin matched blocks first so eviction cannot reclaim them
        for b_id in matched_blocks:
            self.blocks[b_id].ref_count += 1

        if len(self.free_block_ids) < num_new_blocks:
            self._evict(num_new_blocks)
        if len(self.free_block_ids) < num_new_blocks:
            # Rollback ref counts if allocation fails
            for b_id in matched_blocks:
                self._release(b_id)
            raise RuntimeError(
                f"Out of memory: requires {num_new_blocks} blocks, {len(self.free_block_ids)} free"
            )

        new_blocks = [self._pop_free_block() for _ in range(num_new_blocks)]
        self.block_tables[seq_id] = matched_blocks + new_blocks

        self.prefix_query_tokens += len(prompt_tokens)
        self.prefix_hit_tokens += cached_tokens
        return cached_tokens

    def cache_prefix(self, seq_id: str, prompt_tokens: List[int]) -> int:
        """
        Publish the full blocks of a prefilled prompt into the radix tree so
        later requests sharing the prefix can reuse them.
        Returns the number of newly cached blocks.
        """
        if not self.enable_prefix_caching:
            return 0
        table = self.block_tables.get(seq_id)
        if not table:
            return 0
        num_full = len(prompt_tokens) // self.block_size
        if num_full == 0:
            return 0
        tokens = prompt_tokens[: num_full * self.block_size]
        newly = self.prefix_cache.insert(tokens, table[:num_full])
        for b_id in newly:
            self.blocks[b_id].ref_count += 1  # radix tree's reference
        return len(newly)

    def append_slot(self, seq_id: str, position: int) -> int:
        """
        Ensure a KV slot exists for the token at ``position`` (0-indexed).
        If the position falls into a block not yet allocated, a new physical
        block is appended to the sequence's block table.
        Returns:
            physical_slot_index: (block_id * block_size + offset)
        """
        table = self.block_tables[seq_id]
        logical_block = position // self.block_size
        offset = position % self.block_size

        while logical_block >= len(table):
            try:
                table.append(self._pop_free_block())
            except RuntimeError:
                raise RuntimeError(f"Out of KV cache memory for sequence {seq_id}")

        return table[logical_block] * self.block_size + offset

    def free_sequence(self, seq_id: str):
        """Release blocks held by a completed sequence."""
        if seq_id not in self.block_tables:
            return
        for b_id in self.block_tables.pop(seq_id):
            self._release(b_id)

    def get_block_table(self, seq_id: str) -> List[int]:
        return self.block_tables.get(seq_id, [])

    def clear_prefix_cache(self) -> None:
        """Evict all unpinned prefix-cache entries and reset hit/miss counters."""
        self._evict(self.num_blocks)
        self.prefix_query_tokens = 0
        self.prefix_hit_tokens = 0

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    @property
    def prefix_hit_rate(self) -> float:
        if self.prefix_query_tokens == 0:
            return 0.0
        return self.prefix_hit_tokens / self.prefix_query_tokens

    def usage(self) -> float:
        """Fraction of physical blocks currently in use (incl. cached)."""
        return 1.0 - len(self.free_block_ids) / self.num_blocks
