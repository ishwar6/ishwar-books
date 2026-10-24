"""OPT written out by hand, reading and writing its keys and values through the paged KV cache.

The weights are copied from the transformers model facebook/opt-125m. (Larger OPT models work too, except
opt-350m, which puts its layer norm in another place.) The rest is plain tensor arithmetic, so each step is visible.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .attention import paged_attention, paged_attention_batched

POS_OFFSET = 2                                           # OPT's learned position table starts at row 2


@dataclass
class Batch:
    """The input of one iteration (§4.3): the new tokens of every scheduled sequence, joined into one list."""
    input_ids: torch.Tensor                              # (T,) token ids
    positions: torch.Tensor                              # (T,) position of each token in its own sequence
    slots: torch.Tensor                                  # (T,) where each token's key and value go in the cache
    prefills: list                                       # [(start, end, block_table tensor)] one per prompt
    decode_idx: torch.Tensor                             # (S,) index into the T tokens of each one-token sequence
    decode_tables: torch.Tensor                          # (S, max_blocks) their block tables, padded
    decode_lens: torch.Tensor                            # (S,) their context lengths, the new token included
    last_idx: torch.Tensor                               # (R,) the token whose logits we need, one per sequence


class MiniOPT:
    def __init__(self, hf_model):
        c = hf_model.config
        assert c.do_layer_norm_before and c.word_embed_proj_dim == c.hidden_size, 'needs a pre-norm OPT'
        sd = {k: v.detach() for k, v in hf_model.state_dict().items()}
        self.n_layers, self.d, self.h = c.num_hidden_layers, c.hidden_size, c.num_attention_heads
        self.dh = self.d // self.h
        self.embed = sd['model.decoder.embed_tokens.weight']          # (vocab, d), also the output layer
        self.pos_embed = sd['model.decoder.embed_positions.weight']   # (2050, d)
        self.final_ln = (sd['model.decoder.final_layer_norm.weight'], sd['model.decoder.final_layer_norm.bias'])
        self.layers = []
        for i in range(self.n_layers):
            g = lambda name: sd[f'model.decoder.layers.{i}.{name}']
            self.layers.append({n: g(n) for n in [
                'self_attn_layer_norm.weight', 'self_attn_layer_norm.bias',
                'self_attn.q_proj.weight', 'self_attn.q_proj.bias', 'self_attn.k_proj.weight', 'self_attn.k_proj.bias',
                'self_attn.v_proj.weight', 'self_attn.v_proj.bias', 'self_attn.out_proj.weight', 'self_attn.out_proj.bias',
                'final_layer_norm.weight', 'final_layer_norm.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']})

    @torch.no_grad()
    def forward(self, batch, cache):
        """Run one iteration. Returns the logits (R, vocab) of the last token of each scheduled sequence."""
        T, H, D = batch.input_ids.shape[0], self.h, self.dh
        x = self.embed[batch.input_ids] + self.pos_embed[batch.positions + POS_OFFSET]    # (T, d)
        for l, W in enumerate(self.layers):
            # ---- attention: the only place where the tokens of one sequence meet
            a = F.layer_norm(x, (self.d,), W['self_attn_layer_norm.weight'], W['self_attn_layer_norm.bias'])
            q = F.linear(a, W['self_attn.q_proj.weight'], W['self_attn.q_proj.bias']).view(T, H, D)
            k = F.linear(a, W['self_attn.k_proj.weight'], W['self_attn.k_proj.bias']).view(T, H, D)
            v = F.linear(a, W['self_attn.v_proj.weight'], W['self_attn.v_proj.bias']).view(T, H, D)
            cache.write(l, batch.slots, k, v)                        # store the new keys and values first
            k_blocks, v_blocks = cache.blocks(l)
            o = torch.empty_like(q)
            for start, end, table in batch.prefills:                 # prompt phase: many tokens of one sequence
                o[start:end] = paged_attention(q[start:end], k_blocks, v_blocks, table, batch.positions[start:end])
            if len(batch.decode_idx):                                # generation phase: one token per sequence
                o[batch.decode_idx] = paged_attention_batched(q[batch.decode_idx], k_blocks, v_blocks,
                                                              batch.decode_tables, batch.decode_lens)
            x = x + F.linear(o.reshape(T, self.d), W['self_attn.out_proj.weight'], W['self_attn.out_proj.bias'])
            # ---- feed-forward: each token on its own (position-wise)
            f = F.layer_norm(x, (self.d,), W['final_layer_norm.weight'], W['final_layer_norm.bias'])
            f = F.linear(F.relu(F.linear(f, W['fc1.weight'], W['fc1.bias'])), W['fc2.weight'], W['fc2.bias'])
            x = x + f
        x = F.layer_norm(x[batch.last_idx], (self.d,), *self.final_ln)
        return x @ self.embed.T                                      # (R, vocab): the output layer reuses the embedding
