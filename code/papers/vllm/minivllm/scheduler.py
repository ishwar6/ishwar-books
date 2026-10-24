"""Iteration-level scheduling with first-come-first-serve order and preemption by recomputation (§4.5)."""
from collections import deque
from dataclasses import dataclass, field

from .sequence import WAITING, RUNNING


@dataclass
class SchedulerOutput:
    prefill: list = field(default_factory=list)      # [(seq, slots)]: sequences whose tokens all go in this step
    decode: list = field(default_factory=list)       # [(seq, slot)]: sequences that feed one token
    copies: list = field(default_factory=list)       # [(src, dst)] block copies for copy-on-write
    preempted: list = field(default_factory=list)    # groups that lost their blocks in this step


class Scheduler:
    def __init__(self, block_manager, max_num_seqs=256, max_prefill_tokens=4096):
        self.bm = block_manager
        self.max_num_seqs, self.max_prefill_tokens = max_num_seqs, max_prefill_tokens
        self.waiting = deque()                        # groups in arrival order
        self.running = []
        self.num_preemptions = 0

    def add(self, group):
        self.waiting.append(group)

    def has_work(self):
        return bool(self.waiting or self.running)

    def preempt(self, group):
        """Recomputation: free every block of the group (all or nothing), and queue it again in arrival order.
        Later its prompt plus the tokens it already wrote are computed again as one longer prompt."""
        for s in group.unfinished():
            self.bm.free(s.seq_id)
            s.num_cached, s.status = 0, WAITING
        self.running.remove(group)
        i = 0
        while i < len(self.waiting) and self.waiting[i].arrival < group.arrival:
            i += 1
        self.waiting.insert(i, group)                 # it arrived early, so it goes near the front
        self.num_preemptions += 1

    def schedule(self):
        """Pick the work of one iteration. Running requests first (oldest first), then new ones (FCFS)."""
        out = SchedulerOutput()
        # 1. Each running sequence needs a slot for the token it feeds now. When the blocks run out,
        #    preempt the request that arrived last, until the older ones fit.
        self.running.sort(key=lambda g: g.arrival)
        queue = list(self.running)
        while queue:
            g = queue.pop(0)
            seq_ids = [s.seq_id for s in g.unfinished()]
            while not self.bm.can_append(seq_ids):
                victim = queue.pop() if queue else g
                self.preempt(victim)
                out.preempted.append(victim)
                if victim is g:
                    break
            else:
                out.decode += [(s, self.bm.append_slot(s.seq_id)) for s in g.unfinished()]
        # 2. Admit waiting requests in arrival order while their tokens fit. Not in a step that preempted.
        budget = self.max_prefill_tokens
        num_seqs = sum(len(g.unfinished()) for g in self.running)
        while self.waiting and not out.preempted:
            g = self.waiting[0]
            lengths = [len(s.token_ids) for s in g.unfinished()]
            if not g.first_step and g.params.n > 1:   # samples not forked yet: count them for a non-sharing manager
                lengths += lengths[:1] * (g.params.n - 1)
            if (num_seqs + len(lengths) > self.max_num_seqs or sum(lengths) > budget
                    or not self.bm.can_allocate(lengths)):
                break
            self.waiting.popleft()
            budget -= sum(len(s.token_ids) for s in g.unfinished())
            num_seqs += len(lengths)
            for s in g.unfinished():
                out.prefill.append((s, self.bm.allocate(s.seq_id, len(s.token_ids))))
                s.status = RUNNING
            self.running.append(g)
        out.copies, self.bm.copies = self.bm.copies, []
        return out
