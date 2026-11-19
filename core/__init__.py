"""Core inference engine subsystems: Engine loop, Continuous Scheduler, Paged BlockManager, and Radix Prefix Cache."""

from .prefix_cache import Block, RadixNode, PrefixCacheRadixTree
from .block_manager import BlockManager
from .scheduler import ContinuousScheduler, ScheduleBatch, Sequence, SequenceStatus

try:
    from .engine import CustomInferenceEngine, RequestOutput
except ImportError:  # torch optional for pure-Python scheduler/cache tests
    CustomInferenceEngine = None  # type: ignore[assignment,misc]
    RequestOutput = None  # type: ignore[assignment,misc]

__all__ = [
    "Block",
    "RadixNode",
    "PrefixCacheRadixTree",
    "BlockManager",
    "ContinuousScheduler",
    "ScheduleBatch",
    "Sequence",
    "SequenceStatus",
    "CustomInferenceEngine",
    "RequestOutput",
]
