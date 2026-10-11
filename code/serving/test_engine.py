"""Unit tests for the engine simulator. Run: python3 -m unittest discover -s code/serving -p 'test_*.py'"""
import unittest
from engine import Engine, LinearCost, Request, poisson_workload, run, summarize, pct

COST = LinearCost(10.0, 0.04, 4e-6, 3e-4)


class EngineTest(unittest.TestCase):
    def test_every_request_finishes_with_its_tokens(self):
        reqs = poisson_workload(6, 400, seed=1)
        e = Engine(COST, budget=512, max_seqs=64)
        run(reqs, [e])
        for r in reqs:
            self.assertIsNotNone(r.finish)
            self.assertEqual(r.generated, r.output)
            self.assertEqual(len(r.token_times), r.output)
            self.assertLessEqual(r.arrival, r.first_token)
            self.assertEqual(r.token_times, sorted(r.token_times))

    def test_single_request_timing(self):
        r = Request(0, 0.0, 100, 3)
        e = Engine(COST, budget=4096)
        run([r], [e])
        prefill = COST(100, 100 * 101 // 2, 100)
        self.assertAlmostEqual(r.first_token, prefill)
        self.assertAlmostEqual(r.finish, prefill + COST(1, 0, 102) + COST(1, 0, 103))

    def test_chunked_prefill_splits_long_prompt(self):
        r = Request(0, 0.0, 1000, 2)
        e = Engine(COST, budget=256)
        run([r], [e])
        self.assertEqual(e.steps, 4 + 1)      # ceil(1000 / 256) prefill steps, then one decode step

    def test_prefix_cache_hit_skips_prefill(self):
        a = Request(0, 0.0, 600, 2, prefix=7, prefix_len=500)
        b = Request(1, 10_000.0, 600, 2, prefix=7, prefix_len=500)
        e = Engine(COST, budget=4096, cache_capacity=1000)
        run([a, b], [e])
        self.assertEqual(a.cached, 0)
        self.assertEqual(b.cached, 500)
        self.assertLess(b.first_token - b.arrival, a.first_token - a.arrival)

    def test_preemption_under_tiny_memory_still_finishes(self):
        reqs = poisson_workload(20, 200, seed=2, output=(200, 0.5, 50, 400))
        e = Engine(COST, budget=512, max_seqs=64, kv_capacity=6000)
        run(reqs, [e])
        self.assertGreater(e.preemptions, 0)
        self.assertTrue(all(r.finish is not None and r.generated == r.output for r in reqs))

    def test_percentile(self):
        self.assertEqual(pct([1, 2, 3, 4, 5], 50), 3)
        self.assertAlmostEqual(pct(list(range(101)), 99), 99)


if __name__ == '__main__':
    unittest.main()
