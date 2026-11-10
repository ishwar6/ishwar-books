"""Part 2: the DeepSeek-V3 router, from scratch, checked against transformers.

1. A hand-sized example (8 experts on 4 nodes, pick 4 experts from at most 2 nodes) you can check on paper.
2. route_v3(): our NumPy version of V3's routing (sigmoid affinity, bias for choice only, node-limited top-8 of 256,
   normalised gates times 2.5), with the real V3 routing shape, against transformers' DeepseekV3TopkRouter.
3. route_v2(): V2's softmax + device-limited greedy routing on a tiny random config, against DeepseekV2TopkRouter.
4. One token, followed step by step (saved for the routing figure).
Random weights: the picks show the mechanism, not what the trained model would choose.
"""
import numpy as np
import torch
from transformers import DeepseekV2Config, DeepseekV3Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2TopkRouter
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3TopkRouter
from common import Log, save

log = Log('part2_router')


def route_v3(u, centroids, bias, n_group, topk_group, top_k, scale):
    """u: (T, d) token vectors; centroids: (E, d); bias: (E,). Returns expert ids (T, top_k) and gates (T, top_k)."""
    s = 1 / (1 + np.exp(-(u @ centroids.T)))            # Eq. 15: affinity s_i,t = sigmoid(u_t . e_i), shape (T, E)
    choice = s + bias                                    # Eq. 16: the bias is added for choosing only
    T, E = s.shape
    per = choice.reshape(T, n_group, E // n_group)       # experts grouped by node: (T, 8 nodes, 32 experts)
    k_per_node = top_k // topk_group                     # K_r / M = 8 / 4 = 2
    node_score = np.sort(per, axis=-1)[:, :, -k_per_node:].sum(-1)   # sum of the 2 best experts on each node
    nodes = np.argsort(-node_score, axis=-1)[:, :topk_group]          # keep the 4 best nodes
    allowed = np.zeros((T, n_group), bool)
    np.put_along_axis(allowed, nodes, True, axis=1)
    masked = np.where(np.repeat(allowed, E // n_group, axis=1), choice, -np.inf)
    ids = np.argsort(-masked, axis=-1)[:, :top_k]       # Eq. 14/16: top-8 among the allowed experts
    g = np.take_along_axis(s, ids, axis=1)               # gates come from s, NOT from s + b
    g = g / g.sum(-1, keepdims=True)                     # Eq. 13: normalise the 8 gates to sum to 1
    return ids, g * scale, dict(s=s, node_score=node_score, nodes=nodes)


def route_v2(u, centroids, n_group, topk_group, top_k, scale):
    """V2: softmax affinity, keep the topk_group devices with the highest single score, then top-k."""
    z = u @ centroids.T
    s = np.exp(z - z.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
    T, E = s.shape
    dev_score = s.reshape(T, n_group, E // n_group).max(-1)
    devs = np.argsort(-dev_score, axis=-1)[:, :topk_group]
    allowed = np.zeros((T, n_group), bool)
    np.put_along_axis(allowed, devs, True, axis=1)
    masked = np.where(np.repeat(allowed, E // n_group, axis=1), s, 0.0)
    ids = np.argsort(-masked, axis=-1)[:, :top_k]
    return ids, np.take_along_axis(s, ids, axis=1) * scale   # V2 does not renormalise


def same(ids_a, w_a, ids_b, w_b):
    """Compare two routings as sets per token (torch.topk(sorted=False) returns any order)."""
    oa, ob = np.argsort(ids_a, 1), np.argsort(ids_b, 1)
    ia, ib = np.take_along_axis(ids_a, oa, 1), np.take_along_axis(ids_b, ob, 1)
    wa, wb = np.take_along_axis(w_a, oa, 1), np.take_along_axis(w_b, ob, 1)
    return (ia == ib).all(), float(np.abs(wa - wb).max())


# ---------------------------------------------------------------- 0. how many expert teams a token can pick
from math import comb
log('== 0. Possible sets of experts for one token ==')
for n, k, what in [(16, 2, '16 experts, top-2 (the DeepSeekMoE paper example)'), (64, 8, 'each split in 4: 64 experts, top-8'),
                   (256, 8, 'DeepSeek-V3: 256 routed experts, top-8')]:
    log(f'  C({n},{k}) = {comb(n, k):,}   {what}')
log('')

# ---------------------------------------------------------------- 1. a hand-sized example
log('== 1. Hand example: 12 experts on 4 nodes (3 per node), pick K = 4 experts from at most M = 2 nodes ==')
s = np.array([[0.90, 0.10, 0.30, 0.60, 0.55, 0.05, 0.20, 0.15, 0.25, 0.85, 0.05, 0.10]])
log('  affinities s:', ' '.join(f'e{i}={v:.2f}' for i, v in enumerate(s[0])))
logit = np.log(s / (1 - s))                              # undo the sigmoid so route_v3 sees these exact scores
ids, g, info = route_v3(logit, np.eye(12), np.zeros(12), n_group=4, topk_group=2, top_k=4, scale=1.0)
log('  node scores (sum of the best K/M = 2 per node):', ' '.join(f'n{i}={v:.2f}' for i, v in enumerate(info['node_score'][0])))
log('  nodes kept:', sorted(info['nodes'][0].tolist()))
log('  experts chosen:', ids[0].tolist(), ' gates:', ' '.join(f'{v:.3f}' for v in g[0]))
plain = np.argsort(-s[0])[:4]
log('  without the node limit the top-4 would be', sorted(plain.tolist()), 'on nodes', sorted({int(e) // 3 for e in plain}))

# ---------------------------------------------------------------- 2. V3 routing, real shape, vs transformers
log('\n== 2. V3 router (256 experts, 8 nodes, 4 nodes kept, top-8, scale 2.5) vs transformers ==')
torch.manual_seed(0)
d, E, T = 7168, 256, 4096
cfg3 = DeepseekV3Config(hidden_size=d, n_routed_experts=E, num_experts_per_tok=8, n_group=8, topk_group=4,
                        routed_scaling_factor=2.5, norm_topk_prob=True)
gate = DeepseekV3TopkRouter(cfg3)
torch.nn.init.normal_(gate.weight, std=0.006)            # V3 initialises weights with std 0.006
gate.e_score_correction_bias.copy_(torch.randn(E) * 0.01)
u = torch.randn(T, d)
_, w_lib, i_lib = gate(u)
ids, g, info = route_v3(u.double().numpy(), gate.weight.detach().double().numpy(), gate.e_score_correction_bias.double().numpy(),
                        8, 4, 8, 2.5)
ok, err = same(ids, g, i_lib.numpy(), w_lib.detach().double().numpy())
log(f'  {T} random tokens: same 8 experts for every token: {ok}; largest gate difference: {err:.1e}')
log(f'  gates sum to {w_lib.sum(-1).mean().item():.4f} (= 2.5, the routed_scaling_factor)')
nodes_used = np.array([len(set((row // 32).tolist())) for row in ids])
log(f'  nodes touched per token: max {nodes_used.max()}, mean {nodes_used.mean():.2f} (limit 4)')
ids_nb, _, _ = route_v3(u.double().numpy(), gate.weight.detach().double().numpy(), np.zeros(E), 8, 8, 8, 2.5)
nodes_free = np.array([len(set((row // 32).tolist())) for row in ids_nb])
log(f'  same tokens with no node limit: mean {nodes_free.mean():.2f} nodes, {100 * (nodes_free > 4).mean():.0f}% of tokens need more than 4')

# ---------------------------------------------------------------- 3. V2 routing on a tiny config
log('\n== 3. V2 router on a tiny random config (16 experts, 4 devices, 2 devices kept, top-4) vs transformers ==')
cfg2 = DeepseekV2Config(hidden_size=64, n_routed_experts=16, num_experts_per_tok=4, n_group=4, topk_group=2,
                        topk_method='group_limited_greedy', routed_scaling_factor=1.0)
gate2 = DeepseekV2TopkRouter(cfg2)
torch.nn.init.normal_(gate2.weight, std=0.3)
u2 = torch.randn(1000, 64)
_, w2_lib, i2_lib = gate2(u2)
ids2, g2 = route_v2(u2.double().numpy(), gate2.weight.detach().double().numpy(), 4, 2, 4, 1.0)
ok2, err2 = same(ids2, g2, i2_lib.numpy(), w2_lib.detach().double().numpy())
log(f'  1000 random tokens: same 4 experts for every token: {ok2}; largest gate difference: {err2:.1e}')

# ---------------------------------------------------------------- 4. one token, step by step
log('\n== 4. One token through the V3 router ==')
t = 7
s_t = info['s'][t]
log(f'  256 affinities: min {s_t.min():.3f}, max {s_t.max():.3f} (random weights keep them near 0.5)')
log('  node scores:', ' '.join(f'{v:.3f}' for v in info['node_score'][t]))
log('  nodes kept:', sorted(info['nodes'][t].tolist()))
order = np.argsort(ids[t])
log('  experts:', ' '.join(f'{e}(node {e // 32})' for e in ids[t][order]))
log('  gates  :', ' '.join(f'{v:.3f}' for v in g[t][order]), f' sum {g[t].sum():.3f}')

save('part2_router', dict(
    hand=dict(s=s[0].tolist(), node_score=[float(x) for x in route_v3(logit, np.eye(12), np.zeros(12), 4, 2, 4, 1.0)[2]['node_score'][0]],
              chosen=sorted(route_v3(logit, np.eye(12), np.zeros(12), 4, 2, 4, 1.0)[0][0].tolist())),
    v3_match=bool(ok), v3_err=err, v2_match=bool(ok2), v2_err=err2,
    nodes_mean=float(nodes_used.mean()), nodes_free_mean=float(nodes_free.mean()), over4=float((nodes_free > 4).mean()),
    token=dict(s=s_t.tolist(), bias=gate.e_score_correction_bias.tolist(), node_score=info['node_score'][t].tolist(),
               nodes=sorted(info['nodes'][t].tolist()), experts=ids[t][order].tolist(), gates=g[t][order].tolist())))
