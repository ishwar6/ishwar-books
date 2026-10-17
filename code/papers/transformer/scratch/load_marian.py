"""Put the weights of the released translation model Helsinki-NLP/opus-mt-en-de into OUR Transformer (model.py).

Marian is the paper's base model with four differences. We handle each one in the open:
  1. Biases: Marian's W^Q, W^K, W^V, W^O have a bias vector. Our MultiHeadAttention has them too (Config.bias=True).
  2. final_logits_bias: Marian adds one fixed number per vocabulary entry to the logits. We copy it into
     Transformer.logits_bias (the paper has no such term; it is zero in our default model).
  3. Swish instead of ReLU in the feed-forward network: Config.activation='swish'.
  4. Position vectors with all sines first, then all cosines: Config.pos_layout='sin_then_cos'.
     These vectors are not learned, so there is nothing to copy. We check that our table equals Marian's.
Everything else (post-norm, sqrt(d_model) scale, one shared embedding table, 6 + 6 layers) is as in the paper.

    python load_marian.py       # loads, checks that every tensor was used, translates two sentences
"""
import re

import torch

from model import Config, Transformer, sinusoid_table, greedy_decode

NAME = 'Helsinki-NLP/opus-mt-en-de'

# Marian's name for a part of one layer -> our name for the same part.
LAYER_PARTS = {
    'self_attn.q_proj': 'self_attn.W_q',
    'self_attn.k_proj': 'self_attn.W_k',
    'self_attn.v_proj': 'self_attn.W_v',
    'self_attn.out_proj': 'self_attn.W_o',
    'encoder_attn.q_proj': 'cross_attn.W_q',          # decoder only: queries from the decoder
    'encoder_attn.k_proj': 'cross_attn.W_k',          # keys and values from the encoder output
    'encoder_attn.v_proj': 'cross_attn.W_v',
    'encoder_attn.out_proj': 'cross_attn.W_o',
    'self_attn_layer_norm': 'add_norm_1.norm',        # after self-attention
    'fc1': 'ffn.W_1',
    'fc2': 'ffn.W_2',
}


def marian_config(hf_config):
    return Config(vocab_size=hf_config.vocab_size, d_model=hf_config.d_model, n_heads=hf_config.encoder_attention_heads,
                  d_ff=hf_config.encoder_ffn_dim, n_encoder_layers=hf_config.encoder_layers,
                  n_decoder_layers=hf_config.decoder_layers, dropout=0.0,
                  activation='swish',               # difference 3
                  pos_layout='sin_then_cos',        # difference 4
                  max_len=hf_config.max_position_embeddings, pad_id=hf_config.pad_token_id,
                  bias=True,                        # difference 1
                  final_logits_bias=True)           # difference 2


def our_name(marian_name):
    """Map one Marian tensor name to our tensor name. Returns None for tensors we do not copy (with the reason)."""
    if marian_name == 'model.shared.weight':
        return 'embedding.table.weight'                       # the one shared table (§3.4)
    if marian_name in ('model.encoder.embed_tokens.weight', 'model.decoder.embed_tokens.weight', 'lm_head.weight'):
        return None                                           # the same table again (tied); checked below
    if marian_name.endswith('embed_positions.weight'):
        return None                                           # fixed sinusoids, not learned; checked below
    if marian_name == 'final_logits_bias':
        return 'logits_bias'                                  # difference 2
    m = re.match(r'model\.(encoder|decoder)\.layers\.(\d+)\.(.+)\.(weight|bias)$', marian_name)
    if not m:
        raise KeyError(marian_name)
    stack, i, part, kind = m.groups()
    if part == 'final_layer_norm':                            # the norm after the feed-forward network
        part_ours = 'add_norm_2.norm' if stack == 'encoder' else 'add_norm_3.norm'
    elif part == 'encoder_attn_layer_norm':                   # decoder only: the norm after cross-attention
        part_ours = 'add_norm_2.norm'
    else:
        part_ours = LAYER_PARTS[part]
    return f'{stack}.layers.{i}.{part_ours}.{kind}'


def load_marian(name=NAME, return_hf=False):
    from transformers import MarianMTModel, MarianTokenizer
    hf = MarianMTModel.from_pretrained(name).eval()
    tok = MarianTokenizer.from_pretrained(name)
    cfg = marian_config(hf.config)
    model = Transformer(cfg).eval()

    hf_sd = hf.state_dict()
    ours = dict(model.named_parameters())
    ours.update(dict(model.named_buffers()))
    filled = set()
    with torch.no_grad():
        for k, v in hf_sd.items():
            target = our_name(k)
            if target is None:
                continue
            t = ours[target]
            src = v.reshape(t.shape)                          # final_logits_bias is (1, vocab); ours is (vocab,)
            assert src.shape == t.shape, (k, target)
            t.copy_(src)                                      # nn.Linear stores W as (out, in) in both models
            filled.add(target)

    # Every learned tensor of our model must have been filled exactly from Marian.
    missing = [n for n, _ in model.named_parameters() if n not in filled]
    assert not missing, missing
    # The tied copies really are the same table.
    shared = hf_sd['model.shared.weight']
    for k in ('model.encoder.embed_tokens.weight', 'model.decoder.embed_tokens.weight', 'lm_head.weight'):
        assert torch.equal(hf_sd[k], shared), k
    # Marian's fixed position table equals our sin_then_cos table.
    pe = sinusoid_table(cfg.max_len, cfg.d_model, 'sin_then_cos')
    pos_diff = (hf_sd['model.encoder.embed_positions.weight'] - pe).abs().max().item()
    info = {'copied_tensors': len(filled), 'marian_tensors': len(hf_sd), 'position_table_max_diff': pos_diff,
            'start_id': hf.config.decoder_start_token_id, 'eos_id': hf.config.eos_token_id}
    return (model, tok, info, hf) if return_hf else (model, tok, info)


def translate(model, tok, sentences, info, max_len=64):
    src = tok(sentences, return_tensors='pt', padding=True).input_ids          # (b, n)
    out = greedy_decode(model, src, info['start_id'], info['eos_id'], max_len)
    return tok.batch_decode(out, skip_special_tokens=True)


if __name__ == '__main__':
    model, tok, info = load_marian()
    print(info)
    print(translate(model, tok, ['The cat sat on the mat.', 'Attention is all you need.'], info))
