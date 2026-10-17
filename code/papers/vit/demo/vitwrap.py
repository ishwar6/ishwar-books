"""ViT written out step by step (Eq. 1 to 4 of the paper, Appendix A for attention), on the weights of a Hugging Face
ViTForImageClassification. It gives the same logits as the library and also returns what the demo shows:
the patch embeddings after the linear projection, the token sequence z_0 after the position embeddings are added,
and the attention probabilities of every layer and head.

Inputs:  pixels    (1, 3, 224, 224) float, RGB values 0 to 255 (the picture after it is resized to 224 x 224)
         pos_scale (1,) float: 1 adds the position embeddings (the real model), 0 leaves them out (the experiment)
Outputs: patch_embeddings (1, 196, D), tokens (1, 197, D) = z_0, attentions (L, H, 197, 197), logits (1, 1000)"""
import math
import torch
from torch import nn
from transformers import ViTForImageClassification, AutoImageProcessor


class ViTSteps(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.m = ViTForImageClassification.from_pretrained(name, attn_implementation='eager').eval()
        proc = AutoImageProcessor.from_pretrained(name)
        c = self.m.config
        self.D, self.H, self.L, self.P = c.hidden_size, c.num_attention_heads, c.num_hidden_layers, c.patch_size
        self.register_buffer('mean', torch.tensor(proc.image_mean).view(1, 3, 1, 1) * 255)
        self.register_buffer('std', torch.tensor(proc.image_std).view(1, 3, 1, 1) * 255)

    def forward(self, pixels, pos_scale):
        v, emb = self.m.vit, self.m.vit.embeddings
        x = (pixels - self.mean) / self.std                                  # scaled to [-1, 1] for these models
        e = emb.patch_embeddings.projection(x).flatten(2).transpose(1, 2)    # (1, 196, D): x_p E, Eq. 1
        z = torch.cat([emb.cls_token, e], 1) + pos_scale.view(1, 1, 1) * emb.position_embeddings   # z_0, Eq. 1
        z0 = z
        H, Dh = self.H, self.D // self.H
        atts = []
        for layer in v.layers:
            a = layer.attention
            h = layer.layernorm_before(z)                                    # LN, Eq. 2
            q, k, w = (p(h).view(1, -1, H, Dh).transpose(1, 2) for p in (a.q_proj, a.k_proj, a.v_proj))   # Eq. 5
            A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(Dh), -1)   # Eq. 6: (1, H, 197, 197)
            atts.append(A[0])
            sa = (A @ w).transpose(1, 2).reshape(1, -1, self.D)              # Eq. 7 per head, Eq. 8 concatenation
            z = z + a.o_proj(sa)                                             # Eq. 2: residual
            z = z + layer.mlp.fc2(layer.mlp.activation_fn(layer.mlp.fc1(layer.layernorm_after(z))))   # Eq. 3
        y = v.layernorm(z)[:, 0]                                             # Eq. 4: LN of the [class] token
        return e, z0, torch.stack(atts), self.m.classifier(y)
