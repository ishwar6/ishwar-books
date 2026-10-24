"""PagedAttention (§4.1): attention over keys and values that sit in blocks anywhere in memory.

Three versions, all reading the same paged cache:
    paged_attention_eq4      Eq. 4 word for word: one query, a loop over the blocks of the block table
    paged_attention          many queries of one sequence (the prompt phase), causal
    paged_attention_batched  one query for each of many sequences (the generation phase), padded block tables

Shapes: k_blocks, v_blocks (num_blocks, B, H, D) for one layer; B block size, H heads, D head size.
"""
import torch


def paged_attention_eq4(q, k_blocks, v_blocks, block_table, context_len):
    """Eq. 4 for one query token i: q (H, D), the sequence's block table, tokens 1..context_len. Returns (H, D)."""
    B, D = k_blocks.shape[1], q.shape[-1]
    n_blocks = -(-context_len // B)                         # ceil(i / B) blocks hold the context
    scores = []
    for j in range(n_blocks):
        K_j = k_blocks[block_table[j]]                      # fetch block j from wherever it sits: (B, H, D)
        n_j = min(B, context_len - j * B)                   # the last block may be only partly filled
        scores.append(torch.einsum('hd,bhd->hb', q, K_j[:n_j]) / D ** 0.5)   # q_i^T K_j / sqrt(d): (H, n_j)
    # The denominator of Eq. 4 sums exp(...) over all blocks; we subtract the largest score first so that
    # exp() cannot overflow. This changes no result: it cancels between the top and the bottom of the fraction.
    m = torch.cat(scores, -1).max(-1, keepdim=True).values  # (H, 1)
    total = sum(torch.exp(s - m).sum(-1, keepdim=True) for s in scores)
    o = torch.zeros_like(q)
    for j, s in enumerate(scores):
        A_ij = torch.exp(s - m) / total                     # the attention scores on block j: (H, n_j)
        V_j = v_blocks[block_table[j]][:s.shape[-1]]        # (n_j, H, D)
        o += torch.einsum('hb,bhd->hd', A_ij, V_j)          # o_i = sum_j V_j A_ij^T
    return o


def gather(blocks, block_table, n):
    """Read the first n tokens of a sequence in logical order from its scattered blocks: (n, H, D)."""
    return blocks[block_table].flatten(0, 1)[:n]


def paged_attention(q, k_blocks, v_blocks, block_table, positions):
    """All new tokens of one sequence (prompt phase or recomputation): q (t, H, D) at token positions (t,).
    Each query sees the keys at positions up to its own (the causal mask). Returns (t, H, D)."""
    n = int(positions[-1]) + 1
    K, V = gather(k_blocks, block_table, n), gather(v_blocks, block_table, n)   # (n, H, D)
    s = torch.einsum('thd,nhd->htn', q, K) / q.shape[-1] ** 0.5                 # (H, t, n)
    hide = torch.arange(n)[None, :] > positions[:, None]                        # (t, n): keys after the query
    a = torch.softmax(s.masked_fill(hide, float('-inf')), -1)
    return torch.einsum('htn,nhd->thd', a, V)


def paged_attention_batched(q, k_blocks, v_blocks, block_tables, context_lens):
    """One new token for each of S sequences: q (S, H, D), block_tables (S, max_blocks) padded with any block,
    context_lens (S,). The padding is hidden by the mask, as the CUDA kernel skips it. Returns (S, H, D)."""
    S, B = block_tables.shape[0], k_blocks.shape[1]
    K = k_blocks[block_tables].flatten(1, 2)                 # (S, max_blocks * B, H, D)
    V = v_blocks[block_tables].flatten(1, 2)
    s = torch.einsum('shd,snhd->shn', q, K) / q.shape[-1] ** 0.5
    hide = torch.arange(K.shape[1])[None, :] >= context_lens[:, None]           # (S, n): beyond the context
    a = torch.softmax(s.masked_fill(hide[:, None, :], float('-inf')), -1)
    return torch.einsum('shn,snhd->shd', a, V)
