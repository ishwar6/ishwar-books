"""The engine: one loop that schedules, runs the model on the paged cache, samples, and frees finished work."""
import time

import torch

from .block_manager import BlockManager, ContiguousManager
from .kv_cache import PagedKVCache
from .model import Batch, MiniOPT
from .scheduler import Scheduler
from .sequence import FINISHED, RUNNING, Sequence, SequenceGroup


class LLMEngine:
    def __init__(self, hf_model, num_blocks, block_size=16, manager='paged', max_len=512, max_num_seqs=256,
                 eos_id=2):
        self.model = MiniOPT(hf_model)
        m = self.model
        self.cache = PagedKVCache(m.n_layers, num_blocks, block_size, m.h, m.dh, dtype=m.embed.dtype)
        if manager == 'paged':
            self.bm = BlockManager(num_blocks, block_size)
        else:
            self.bm = ContiguousManager(num_blocks, block_size, max_len)
        self.scheduler = Scheduler(self.bm, max_num_seqs)
        self.eos_id, self.next_id, self.step_no = eos_id, 0, 0
        self.groups, self.stats = [], []

    def add_request(self, prompt_ids, params):
        g = SequenceGroup(len(self.groups), params, arrival=len(self.groups))
        g.seqs.append(self._new_seq(prompt_ids, len(prompt_ids), g))
        self.groups.append(g)
        self.scheduler.add(g)
        return g

    def _new_seq(self, token_ids, prompt_len, group):
        s = Sequence(self.next_id, token_ids, prompt_len, group)
        s.gen = torch.Generator().manual_seed(group.params.seed * 1000 + len(group.seqs))
        self.next_id += 1
        return s

    def _batch(self, out):
        """Join the new tokens of all scheduled sequences into one list (§4.3)."""
        ids, pos, slots, prefills, last, order = [], [], [], [], [], []
        for seq, s_slots in out.prefill:                       # all tokens: the prompt (or prompt + output, if recomputed)
            n, start = len(seq.token_ids), len(ids)
            ids += seq.token_ids
            pos += range(n)
            slots += s_slots
            prefills.append((start, start + n, torch.tensor(self.bm.tables[seq.seq_id].blocks)))
            last.append(start + n - 1)
            order.append(seq)
        dec_idx, tables, lens = [], [], []
        for seq, slot in out.decode:                           # one token: the one written in the last step
            dec_idx.append(len(ids))
            last.append(len(ids))
            ids.append(seq.token_ids[-1])
            pos.append(len(seq.token_ids) - 1)
            slots.append(slot)
            tables.append(self.bm.tables[seq.seq_id].blocks)
            lens.append(len(seq.token_ids))
            order.append(seq)
        width = max((len(t) for t in tables), default=1)
        tables = [t + [0] * (width - len(t)) for t in tables]   # pad; the mask hides the padding
        T = torch.tensor
        return Batch(T(ids), T(pos), T(slots), prefills, T(dec_idx, dtype=torch.long),
                     T(tables, dtype=torch.long).view(-1, width), T(lens, dtype=torch.long), T(last)), order

    def _sample(self, logits, seq):
        p = seq.group.params
        if p.temperature == 0:
            token = int(logits.argmax())
        else:
            probs = torch.softmax(logits / p.temperature, -1)
            token = int(torch.multinomial(probs, 1, generator=seq.gen))
        seq.logprobs.append(float(torch.log_softmax(logits, -1)[token]))
        seq.token_ids.append(token)
        if len(seq.output_ids) >= p.max_tokens or (token == self.eos_id and not p.ignore_eos):
            seq.status = FINISHED
            self.bm.free(seq.seq_id)                          # its blocks go back to the free list at once

    def step(self):
        out = self.scheduler.schedule()
        self.cache.copy_blocks(out.copies)                    # copy-on-write copies, before anyone writes
        if not out.prefill and not out.decode:
            if self.scheduler.has_work():
                raise RuntimeError('a request does not fit in the KV cache even when it runs alone')
            return out
        batch, order = self._batch(out)
        logits = self.model.forward(batch, self.cache)        # (sequences, vocab)
        self.last_logits = {seq.seq_id: logits[row] for row, seq in enumerate(order)}   # kept for the checks
        for row, seq in enumerate(order):
            seq.num_cached = len(seq.token_ids)
            g = seq.group
            if g.first_step is None:                           # first step of a request with n samples:
                g.first_step = self.step_no                    # fork n - 1 children that share every block
                for _ in range(g.params.n - 1):
                    child = self._new_seq(seq.token_ids, seq.prompt_len, g)
                    child.num_cached, child.status = seq.num_cached, RUNNING
                    self.bm.fork(seq.seq_id, child.seq_id)
                    g.seqs.append(child)
                    self._sample(logits[row], child)
            self._sample(logits[row], seq)
        for g in list(self.scheduler.running):
            if not g.unfinished():
                g.finish_step = self.step_no
                self.scheduler.running.remove(g)
        self.stats.append(dict(step=self.step_no, prefill_tokens=sum(len(s.token_ids) - 1 for s, _ in out.prefill),
                               decode=len(out.decode), running=sum(len(g.unfinished()) for g in self.scheduler.running),
                               blocks_held=self.bm.blocks_held(), blocks_no_sharing=self.bm.blocks_without_sharing(),
                               tokens=self.bm.tokens_stored(), preempted=len(out.preempted), copies=len(out.copies)))
        self.step_no += 1
        return out

    def run(self):
        """Step until every request is done. Returns the wall time in seconds."""
        t0 = time.perf_counter()
        while self.scheduler.has_work():
            self.step()
        return time.perf_counter() - t0
