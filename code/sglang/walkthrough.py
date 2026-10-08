"""The small examples printed in the article. Run: python code/sglang/walkthrough.py."""
from pathlib import Path
import json
import math
from experiments import Radix, BlockHash

def main():
    # Six shared token IDs stand for instructions + the handbook.
    prefix = [10, 11, 12, 13, 14, 15]
    prompts = [prefix + [20, 21], prefix + [30, 31], prefix + [20, 40]]
    lines=[]; results={}
    for name,cache in [('radix',Radix()),('blocks of 4',BlockHash(4))]:
        lines.append(name); rows=[]
        for number,prompt in enumerate(prompts,1):
            reused=cache.match(prompt); new=len(prompt)-reused
            lines.append(f'request {number}: reuse {reused}, compute {new}')
            rows.append({'request':number,'reused':reused,'computed':new})
            cache.insert(prompt)
        results[name]=rows
    assert [x['reused'] for x in results['radix']]==[0,6,7]
    assert [x['reused'] for x in results['blocks of 4']]==[0,4,4]
    # Same two stored keys/values; a new query changes the attention weights.
    attention=[]
    for q in [0.0,math.log(3)]:
        keys,values=[0.0,1.0],[2.0,6.0]
        scores=[q*k for k in keys] # d_k=1, so sqrt(d_k)=1
        exp=[math.exp(s) for s in scores]
        weights=[v/sum(exp) for v in exp]
        output=sum(a*v for a,v in zip(weights,values))
        attention.append({'query':q,'weights':weights,'output':output})
    assert math.isclose(attention[0]['output'],4)
    assert math.isclose(attention[1]['output'],5)
    results['attention']=attention
    lines.extend(['attention with the same K and V:', 'query 0: weights [0.50, 0.50], output 4.00',
                  f'query ln(3): weights [{attention[1]["weights"][0]:.2f}, {attention[1]["weights"][1]:.2f}], output {attention[1]["output"]:.2f}'])
    results['latency_example']={'cold_ms':100+900,'warm_ms':20+900,'speedup':1000/920}
    out=Path(__file__).resolve().parent/'results'
    (out/'walkthrough.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'walkthrough_stdout.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
if __name__=='__main__':main()
