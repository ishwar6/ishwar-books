"""Deterministic CPU teaching experiments; neither a serving engine nor a GPU benchmark.
Run: python experiments.py (requires numpy). Results are written beside this file.
"""
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict
import hashlib
import json
import platform
import sys
import numpy as np

HERE = Path(__file__).resolve().parent

@dataclass
class Node:
    edge: tuple = ()
    children: dict = field(default_factory=dict)

class Radix:
    """Compressed token trie. No tensors, eviction, locks, namespaces or page alignment."""
    def __init__(self):
        self.root = Node()

    def match(self, tokens):
        tokens = tuple(tokens)
        node, used = self.root, 0
        while used < len(tokens) and tokens[used] in node.children:
            child = node.children[tokens[used]]
            n = 0
            while n < min(len(child.edge), len(tokens)-used) and child.edge[n] == tokens[used+n]:
                n += 1
            used += n
            if n < len(child.edge):
                break
            node = child
        return used

    def insert(self, tokens):
        tail, node = tuple(tokens), self.root
        while tail:
            child = node.children.get(tail[0])
            if child is None:
                node.children[tail[0]] = Node(tail)
                return
            n = 0
            while n < min(len(tail), len(child.edge)) and tail[n] == child.edge[n]:
                n += 1
            if n < len(child.edge):
                parent = Node(child.edge[:n])
                node.children[tail[0]] = parent
                child.edge = child.edge[n:]
                parent.children[child.edge[0]] = child
                child = parent
            node, tail = child, tail[n:]

class BlockHash:
    """Full-block prefix index; illustrative SHA256 chain, not vLLM's serialization."""
    def __init__(self, size=16):
        self.size, self.blocks = size, set()

    def keys(self, tokens):
        parent = b''
        for i in range(0, len(tokens)-self.size+1, self.size):
            payload = json.dumps(list(tokens[i:i+self.size]), separators=(',', ':')).encode()
            parent = hashlib.sha256(parent + payload).digest()
            yield parent

    def match(self, tokens):
        count = 0
        for key in self.keys(tokens):
            if key not in self.blocks:
                break
            count += self.size
        return count

    def insert(self, tokens):
        self.blocks.update(self.keys(tokens))

def workload(shared, group, suffix):
    return [list(range(shared)) + [10000+i%4]*group + [20000+i]*suffix for i in range(64)]

def count_work(requests, index):
    hits = []
    for p in requests:
        hits.append(index.match(p))
        index.insert(p)
    total = sum(map(len, requests))
    return {'prompt_tokens': total, 'cached_tokens': sum(hits), 'computed_tokens': total-sum(hits),
            'hit_rate': sum(hits)/total, 'per_request_hits': hits}

def attention_check():
    """Two causal attention layers, float64, random fixed weights, CPU. No learned model."""
    rng = np.random.default_rng(7)
    d = 16
    embed = rng.normal(size=(100, d))
    pos = rng.normal(size=(20, d))*.1
    weights = [[rng.normal(size=(d,d))/np.sqrt(d) for _ in range(4)] for _ in range(2)]
    def forward(ids, cached=None):
        offset = 0 if cached is None else cached[0][0].shape[0]
        x = embed[ids] + pos[offset:offset+len(ids)]
        caches = []
        for layer,(wq,wk,wv,wo) in enumerate(weights):
            q,k,v = x@wq, x@wk, x@wv
            if cached is not None:
                k = np.concatenate([cached[layer][0], k]); v = np.concatenate([cached[layer][1], v])
            scores = q@k.T/np.sqrt(d)
            allowed = np.arange(len(k))[None,:] <= (offset+np.arange(len(ids)))[:,None]
            scores = np.where(allowed,scores,-np.inf)
            a = np.exp(scores-np.max(scores,axis=1,keepdims=True)); a /= a.sum(axis=1,keepdims=True)
            x = x + a@v@wo
            caches.append((k,v))
        return x,caches
    prefix, tail = [1,2,3,4,5,6], [7,8,9]
    full,_ = forward(prefix+tail)
    _,kv = forward(prefix)
    reused,_ = forward(tail,kv)
    correct = float(np.max(np.abs(full[len(prefix):]-reused)))
    changed,_ = forward([99]+prefix[1:]+tail)
    wrong = float(np.max(np.abs(changed[len(prefix):]-reused)))
    assert correct < 1e-12 and wrong > 1e-3
    return {'max_abs_error_valid_reuse':correct,'max_abs_error_wrong_prefix':wrong,'layers':2,'width':d,'dtype':'float64'}

def scheduling():
    # Four tenants, only two whole prefix groups fit. Explicit whole-group LRU model.
    def run(order):
        cache = OrderedDict(); misses=0
        for tenant in order:
            if tenant not in cache:
                misses += 1
                if len(cache)==2: cache.popitem(last=False)
            else: cache.pop(tenant)
            cache[tenant]=True
        return {'misses':misses,'hits':len(order)-misses,'computed_tokens':misses*1024+len(order)*64}
    return {'arrival_order':run(list(range(4))*8),'grouped_order':run([g for g in range(4) for _ in range(8)])}

def verify_indexes():
    rng=np.random.default_rng(11)
    for size in [1,4,16]:
        r,h,seen=Radix(),BlockHash(size),[]
        for _ in range(200):
            p=tuple(int(x) for x in rng.integers(0,4,size=int(rng.integers(0,35))))
            def lcp(q):
                n=0
                for a,b in zip(p,q):
                    if a!=b: break
                    n+=1
                return n
            expected=max([lcp(q) for q in seen],default=0)
            assert r.match(p)==expected
            assert h.match(p)==expected//size*size
            r.insert(p); h.insert(p); seen.append(p)
    h=BlockHash(2); h.insert([1,2,3,4]); assert h.match([9,9,3,4])==0
    r=Radix(); r.insert([1,2,3,4]); r.insert([1,2,5]); assert r.match([1,2,3,9])==3
    r.insert([1,2]); assert r.match([1,2])==2
    return '600 randomized request checks plus edge-splitting and context-identity checks passed'

def main():
    results={'kind':'CPU teaching experiments; no SGLang/vLLM runtime benchmark',
             'environment':{'python':sys.version.split()[0],'numpy':np.__version__,'platform':platform.platform()},
             'validation':verify_indexes(),'attention':attention_check()}
    for label,args in [('aligned',(1024,256,64)),('unaligned',(1027,257,61))]:
        reqs=workload(*args)
        results[label]={'radix':count_work(reqs,Radix()),'blocks16':count_work(reqs,BlockHash(16))}
    results['scheduling']=scheduling()
    results['arithmetic']={'kv_bytes_per_token':2*32*8*128*2,'shared_prefix_tokens':8192,
        'prefix_gib':2*32*8*128*2*8192/2**30,'serial_100_steps_ms':100*(2+8),
        'overlap_100_steps_ms':2+8+99*max(2,8),'transfer_1gib_at_25GBps_ms':2**30/(25*10**9)*1000}
    out=HERE/'results'; out.mkdir(exist_ok=True)
    (out/'experiments.json').write_text(json.dumps(results,indent=2)+'\n')
    summary={k:v for k,v in results.items() if k not in ['aligned','unaligned']}
    for k in ['aligned','unaligned']:
        summary[k]={a:{b:c for b,c in v.items() if b!='per_request_hits'} for a,v in results[k].items()}
    log=json.dumps(summary,indent=2)+'\n'
    (out/'experiments_stdout.txt').write_text(log); print(log,end='')
if __name__=='__main__': main()
