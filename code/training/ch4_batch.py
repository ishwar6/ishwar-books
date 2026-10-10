"""Chapter 4: why bigger batches give cleaner gradients. For the trained 12M TinyStories GPT, compute the gradient on two
independent random batches of B windows and measure how much they agree (cosine similarity), for B = 1 ... 128."""
import torch
from common import Log, save
from ch4_gpt import GPT, get_tokenizer, get_tokens, batch

log = Log('ch4_batch')
dev = 'mps'
tok = get_tokenizer(4096)
train = get_tokens('train', tok)
m = GPT(4096).to(dev)
m.load_state_dict(torch.load('/Users/admin/.cache/tinystories/ch4_tinygpt.pt'))


def grad(B, g):
    m.zero_grad(set_to_none=True)
    x, y = batch(train, B, 256, g, dev)
    m(x, y)[1].backward()
    return torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]).clone()


g = torch.Generator().manual_seed(7)
out = []
for B in [1, 2, 4, 8, 16, 32, 64, 128]:
    cos = []
    for _ in range(6):
        a, b = grad(B, g), grad(B, g)
        cos.append(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
    c = sum(cos) / len(cos)
    log(f'batch of {B:4d} windows ({B * 256:6,} tokens): cosine(grad1, grad2) = {c:.3f}')
    out.append({'B': B, 'tokens': B * 256, 'cos': c})
save('ch4_batch', out)
