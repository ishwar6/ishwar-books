"""The KV cache manager of vLLM (§4.2 to §4.4): physical blocks, reference counts, block tables, copy-on-write.

Two managers share one interface, so the engine can run with either:
    BlockManager       paged: blocks on demand, any free block anywhere, sharing with copy-on-write (vLLM)
    ContiguousManager  the baseline of §3.1: one contiguous chunk per sequence, booked for the maximum length

A "slot" is the place of one token's keys and values: slot = physical_block * block_size + offset.
"""
from dataclasses import dataclass, field


def cdiv(a, b):
    """Integer division that rounds up: how many blocks of size b hold a tokens."""
    return -(-a // b)


class BlockAllocator:
    """Hands out physical KV blocks by number and counts how many logical blocks point at each one (§4.4)."""

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.free_list = list(range(num_blocks))[::-1]   # pop() hands out block 0 first, then 1, 2, ...
        self.ref_count = [0] * num_blocks

    def allocate(self):
        if not self.free_list:
            raise MemoryError('no free KV block')
        block = self.free_list.pop()
        self.ref_count[block] = 1
        return block

    def share(self, block):
        """One more logical block points at this physical block."""
        self.ref_count[block] += 1

    def free(self, block):
        """One logical block lets go; the physical block returns to the free list when nobody points at it."""
        self.ref_count[block] -= 1
        if self.ref_count[block] == 0:
            self.free_list.append(block)

    @property
    def num_free(self):
        return len(self.free_list)


@dataclass
class BlockTable:
    """The block table of one sequence (Figure 6): logical block j lives in physical block blocks[j]."""
    blocks: list = field(default_factory=list)
    num_tokens: int = 0                          # tokens whose keys and values are stored

    def filled(self, j, block_size):
        """The "# filled" column of Figure 6: how many slots of logical block j hold a token."""
        return min(block_size, self.num_tokens - j * block_size)

    def slot(self, pos, block_size):
        """Where token number pos lives: its physical block times the block size, plus its offset."""
        return self.blocks[pos // block_size] * block_size + pos % block_size


class BlockManager:
    """Paged KV cache manager (vLLM). Blocks are booked only when a token needs one (§4.3)."""
    paged = True

    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)
        self.tables = {}                         # seq_id -> BlockTable
        self.copies = []                         # (src, dst) block copies that copy-on-write asks for

    # ---- questions the scheduler asks
    def can_allocate(self, lengths):
        """Is there room for the first len tokens of each of these sequences?"""
        return sum(cdiv(n, self.block_size) for n in lengths) <= self.allocator.num_free

    def needs_block(self, seq_id):
        """Does the next token of this sequence need a fresh block (last block full, or shared)?"""
        t = self.tables[seq_id]
        if t.num_tokens % self.block_size == 0:
            return True
        return self.allocator.ref_count[t.blocks[-1]] > 1

    def can_append(self, seq_ids):
        return sum(self.needs_block(s) for s in seq_ids) <= self.allocator.num_free

    # ---- the three operations of §5.2 (fork, append, free) plus the first allocation
    def allocate(self, seq_id, num_tokens):
        """Prompt phase (§4.3, step 1): book only the blocks that the prompt fills. Returns the prompt's slots."""
        t = BlockTable([self.allocator.allocate() for _ in range(cdiv(num_tokens, self.block_size))], num_tokens)
        self.tables[seq_id] = t
        return [t.slot(p, self.block_size) for p in range(num_tokens)]

    def append_slot(self, seq_id):
        """Find the slot of the next token (§4.3, steps 2 and 3; §4.4, copy-on-write). Returns the slot."""
        t, B = self.tables[seq_id], self.block_size
        if t.num_tokens % B == 0:                          # last block full: map a new logical block
            t.blocks.append(self.allocator.allocate())
        else:
            last = t.blocks[-1]
            if self.allocator.ref_count[last] > 1:         # the last block is shared: copy it before writing
                new = self.allocator.allocate()
                self.copies.append((last, new))            # the block engine copies the data later
                self.allocator.free(last)                  # ref count of the old block goes down by one
                t.blocks[-1] = new
        slot = t.slot(t.num_tokens, B)
        t.num_tokens += 1
        return slot

    def fork(self, parent_id, child_id):
        """A new sequence that shares every block of its parent (parallel sampling, §4.4)."""
        p = self.tables[parent_id]
        for b in p.blocks:
            self.allocator.share(b)
        self.tables[child_id] = BlockTable(list(p.blocks), p.num_tokens)

    def free(self, seq_id):
        for b in self.tables.pop(seq_id).blocks:
            self.allocator.free(b)

    # ---- numbers for the memory report
    def blocks_held(self):
        return self.allocator.num_blocks - self.allocator.num_free

    def blocks_without_sharing(self):
        return sum(cdiv(t.num_tokens, self.block_size) for t in self.tables.values())

    def tokens_stored(self):
        return sum(t.num_tokens for t in self.tables.values())


class ContiguousManager:
    """The baseline of §3.1: each sequence books one contiguous chunk for max_len tokens when it starts.
    Nothing is shared: a forked sequence gets its own chunk and a full copy of its parent's data."""
    paged = False

    def __init__(self, num_blocks, block_size, max_len):
        self.block_size = block_size
        self.chunk = cdiv(max_len, block_size)            # blocks per sequence, booked up front
        self.used = [False] * num_blocks
        self.tables = {}
        self.copies = []

    def _find_chunks(self, k):
        """First fit: the start of k free runs of self.chunk blocks each, or None."""
        used, starts, i = list(self.used), [], 0
        while len(starts) < k and i + self.chunk <= len(used):
            if not any(used[i:i + self.chunk]):
                starts.append(i)
                used[i:i + self.chunk] = [True] * self.chunk
                i += self.chunk
            else:
                i += 1
        return starts if len(starts) == k else None

    def can_allocate(self, lengths):
        return self._find_chunks(len(lengths)) is not None

    def can_append(self, seq_ids):
        return True                                       # the room was booked at the start

    def allocate(self, seq_id, num_tokens):
        start = self._find_chunks(1)[0]
        self.used[start:start + self.chunk] = [True] * self.chunk
        t = BlockTable(list(range(start, start + self.chunk)), num_tokens)
        self.tables[seq_id] = t
        return [t.slot(p, self.block_size) for p in range(num_tokens)]

    def append_slot(self, seq_id):
        t = self.tables[seq_id]
        assert t.num_tokens < self.chunk * self.block_size, 'sequence longer than its booked chunk'
        slot = t.slot(t.num_tokens, self.block_size)
        t.num_tokens += 1
        return slot

    def fork(self, parent_id, child_id):
        p = self.tables[parent_id]
        self.allocate(child_id, p.num_tokens)
        self.copies += list(zip(p.blocks, self.tables[child_id].blocks))   # copy the whole chunk

    def free(self, seq_id):
        t = self.tables.pop(seq_id)
        self.used[t.blocks[0]:t.blocks[0] + self.chunk] = [False] * self.chunk

    def blocks_held(self):
        return sum(self.used)

    def blocks_without_sharing(self):
        return self.blocks_held()

    def tokens_stored(self):
        return sum(t.num_tokens for t in self.tables.values())
