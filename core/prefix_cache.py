import time
from typing import Dict, List, Optional, Tuple


class Block:
    """A physical block in the paged KV cache (reference counted)."""
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0


class RadixNode:
    """Node in the (block-granular) Radix Tree for Prefix Caching.

    Every edge stores exactly one KV block worth of tokens (``block_size``
    tokens), so each node owns exactly one physical block id.
    """
    def __init__(self, tokens: Tuple[int, ...] = ()):
        self.tokens: Tuple[int, ...] = tokens  # token chunk for this edge
        self.block_ids: List[int] = []         # Physical KV blocks storing these tokens
        # full token chunk of child -> RadixNode (keyed by the whole chunk so
        # two chunks sharing a first token never collide)
        self.children: Dict[Tuple[int, ...], 'RadixNode'] = {}
        self.parent: Optional['RadixNode'] = None
        self.last_accessed: float = time.time()

    def is_leaf(self) -> bool:
        return not self.children


class PrefixCacheRadixTree:
    """
    Radix Tree-based Prefix Cache.
    Enables automatic KV cache sharing across requests with common prefixes
    (e.g., system prompts, multi-turn chat dialogues, few-shot examples).

    Only *full* blocks are cached. The tree does not manage reference counts
    itself; :class:`BlockManager` holds one reference per cached block and
    uses :meth:`evictable_leaves` / :meth:`remove_leaf` for LRU eviction.
    """
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.root = RadixNode()
        self.num_cached_blocks = 0

    def _chunks(self, token_ids: List[int]):
        bs = self.block_size
        for i in range(0, len(token_ids) - bs + 1, bs):
            yield tuple(token_ids[i:i + bs])

    def match_prefix(self, token_ids: List[int]) -> Tuple[List[int], int]:
        """
        Match incoming token sequence against cached prefix tree.
        Returns:
            matched_block_ids: list of physical block IDs reused from cache
            matched_token_count: number of tokens covered by reused blocks
        """
        curr = self.root
        matched_blocks: List[int] = []
        now = time.time()
        for chunk in self._chunks(token_ids):
            child = curr.children.get(chunk)
            if child is None:
                break
            matched_blocks.extend(child.block_ids)
            child.last_accessed = now
            curr = child

        tokens_covered = len(matched_blocks) * self.block_size
        return matched_blocks, tokens_covered

    def insert(self, token_ids: List[int], block_ids: List[int]) -> List[int]:
        """
        Insert completed prefix blocks into the Radix Tree.
        Each block corresponds to exactly block_size tokens.

        Returns:
            The subset of ``block_ids`` that were newly added to the tree
            (blocks whose chunk was already cached are not re-inserted).
        """
        assert len(token_ids) == len(block_ids) * self.block_size
        curr = self.root
        newly_inserted: List[int] = []
        now = time.time()

        for chunk, block_id in zip(self._chunks(token_ids), block_ids):
            child = curr.children.get(chunk)
            if child is None:
                child = RadixNode(tokens=chunk)
                child.block_ids = [block_id]
                child.parent = curr
                curr.children[chunk] = child
                newly_inserted.append(block_id)
                self.num_cached_blocks += 1
            child.last_accessed = now
            curr = child

        return newly_inserted

    # ------------------------------------------------------------------
    # Eviction helpers
    # ------------------------------------------------------------------
    def evictable_leaves(self) -> List[RadixNode]:
        """All leaf nodes (excluding root), oldest access first."""
        leaves: List[RadixNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node is not self.root and node.is_leaf():
                leaves.append(node)
            stack.extend(node.children.values())
        leaves.sort(key=lambda n: n.last_accessed)
        return leaves

    def remove_leaf(self, node: RadixNode) -> List[int]:
        """Detach a leaf node and return the block ids it owned."""
        assert node.is_leaf() and node.parent is not None
        del node.parent.children[node.tokens]
        node.parent = None
        self.num_cached_blocks -= len(node.block_ids)
        return node.block_ids
