"""The paged KV cache: one big tensor cut into fixed-size blocks (§4.2), plus the block copy of copy-on-write."""
import torch


class PagedKVCache:
    """GPU memory for keys and values, booked once at start-up and cut into num_blocks physical blocks.

    Shape: (layers, 2, num_blocks, block_size, heads, head_dim). Index 0 of the second axis holds keys, 1 values.
    Every layer uses the same block numbers, so one block table per sequence serves all layers."""

    def __init__(self, num_layers, num_blocks, block_size, num_heads, head_dim, dtype=torch.float32, device='cpu'):
        self.num_blocks, self.block_size = num_blocks, block_size
        self.data = torch.zeros(num_layers, 2, num_blocks, block_size, num_heads, head_dim, dtype=dtype, device=device)

    def blocks(self, layer):
        """Keys and values of one layer, each (num_blocks, block_size, heads, head_dim)."""
        return self.data[layer, 0], self.data[layer, 1]

    def write(self, layer, slots, k, v):
        """Store the new keys and values (t, heads, head_dim) at their slots (t,) = block * block_size + offset."""
        keys, values = self.blocks(layer)
        keys.view(-1, *keys.shape[2:])[slots] = k             # view the blocks as one long list of slots
        values.view(-1, *values.shape[2:])[slots] = v

    def copy_blocks(self, pairs):
        """Copy-on-write (§4.4): copy whole blocks src -> dst, in every layer, keys and values."""
        if pairs:
            src = torch.tensor([s for s, _ in pairs])
            dst = torch.tensor([d for _, d in pairs])
            self.data[:, :, dst] = self.data[:, :, src]       # one batched copy, as in the fused kernel of §5.1

    def nbytes(self):
        return self.data.numel() * self.data.element_size()

    def bytes_per_block(self):
        return self.nbytes() // self.num_blocks
