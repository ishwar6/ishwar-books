"""Requests and sequences. One request (a sequence group) holds one prompt and its n samples (§4.4, §4.5)."""
from dataclasses import dataclass

WAITING, RUNNING, FINISHED = 'waiting', 'running', 'finished'


@dataclass
class SamplingParams:
    n: int = 1                       # number of samples from one prompt (parallel sampling)
    max_tokens: int = 16             # stop after this many new tokens
    temperature: float = 0.0         # 0 means greedy: always take the most likely token
    ignore_eos: bool = False         # True: keep writing past </s> (to replay known output lengths)
    seed: int = 0


class Sequence:
    """One prompt plus the tokens written so far, and how many of them have keys and values in the cache."""

    def __init__(self, seq_id, token_ids, prompt_len, group):
        self.seq_id = seq_id
        self.token_ids = list(token_ids)
        self.prompt_len = prompt_len
        self.group = group
        self.num_cached = 0          # tokens whose keys and values are stored in the KV cache
        self.status = WAITING
        self.logprobs = []           # log-probability of each written token, for the checks

    @property
    def output_ids(self):
        return self.token_ids[self.prompt_len:]


class SequenceGroup:
    """One request. Its sequences are scheduled and preempted together (gang scheduling, §4.5)."""

    def __init__(self, request_id, params, arrival):
        self.request_id, self.params, self.arrival = request_id, params, arrival
        self.seqs = []
        self.first_step = self.finish_step = None

    def unfinished(self):
        return [s for s in self.seqs if s.status != FINISHED]
