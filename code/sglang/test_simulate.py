"""Self-tests for the caches in simulate.py. Run: python -m unittest discover -s code/sglang -p 'test_*.py'"""
import random, unittest
from simulate import RadixLRU, BlockLRU


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n


class Caches(unittest.TestCase):
    def test_unbounded_caches_match_brute_force(self):
        rng = random.Random(1)
        for B in (1, 4, 16):
            r, h, seen = RadixLRU(10**9), BlockLRU(10**9, B), []
            for _ in range(300):
                p = [rng.randrange(4) for _ in range(rng.randrange(0, 40))]
                best = max((lcp(p, q) for q in seen), default=0)
                self.assertEqual(r.match(p), best)
                self.assertEqual(h.match(p), best // B * B)
                r.insert(p); h.insert(p); seen.append(p)

    def test_tree_size_is_sum_of_edges(self):
        r = RadixLRU(10**9)
        r.insert([1, 2, 3, 4]); r.insert([1, 2, 5]); r.insert([9])
        self.assertEqual(r.size, 6)              # 1 2 | 3 4 | 5 | 9

    def test_leaf_first_lru_keeps_shared_parent(self):
        r = RadixLRU(10)
        for s in ([1, 2, 3, 4, 5, 6, 20, 21], [1, 2, 3, 4, 5, 6, 30, 31], [1, 2, 3, 4, 5, 6, 40, 41]):
            r.match(s); r.insert(s)
        self.assertEqual(r.match([1, 2, 3, 4, 5, 6, 20, 21]), 6)   # q1's leaf was the oldest: evicted
        self.assertEqual(r.match([1, 2, 3, 4, 5, 6, 30, 31]), 8)
        self.assertLessEqual(r.size, 10)

    def test_running_request_is_never_evicted(self):
        r = RadixLRU(5)
        r.insert(list(range(8)))                 # larger than capacity: its own path is locked
        self.assertEqual(r.match(list(range(8))), 8)

    def test_block_cache_needs_preceding_context(self):
        h = BlockLRU(10**9, 2); h.insert([1, 2, 3, 4])
        self.assertEqual(h.match([9, 9, 3, 4]), 0)

    def test_block_eviction_drops_tail_blocks_first(self):
        h = BlockLRU(6, 2); h.insert([1, 2, 3, 4, 5, 6])      # 3 blocks fit exactly
        h.insert([7, 8])                                      # one block over: evict the old request's last block
        self.assertEqual(h.match([1, 2, 3, 4, 5, 6]), 4)


if __name__ == '__main__':
    unittest.main()
