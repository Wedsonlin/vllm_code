"""B4: refcounted block allocate/free."""
from __future__ import annotations
from typing import Dict, List, Optional, Set


class BlockPoolExhausted(Exception):
    pass


class BlockPool:
    def __init__(self, num_blocks: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        self.num_blocks = num_blocks
        self._free: List[int] = list(range(num_blocks))
        self._refcount: Dict[int, int] = {}

    def allocate(self) -> int:
        """Return a free block id with refcount=1."""
        if not self._free:
            raise BlockPoolExhausted("no free blocks")
        bid = self._free.pop()
        self._refcount[bid] = 1
        return bid

    def retain(self, block_id: int) -> None:
        if block_id not in self._refcount:
            raise KeyError(block_id)
        self._refcount[block_id] += 1

    def free(self, block_id: int) -> None:
        if block_id not in self._refcount:
            raise KeyError(block_id)
        self._refcount[block_id] -= 1
        if self._refcount[block_id] == 0:
            del self._refcount[block_id]
            self._free.append(block_id)

    @property
    def free_count(self) -> int:
        return len(self._free)

    def refcount(self, block_id: int) -> int:
        return self._refcount.get(block_id, 0)
